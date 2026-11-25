#!/usr/bin/env python
"""Build pairs, check pair validity, find duplicate clusters, make the fixed split, pre-resize all images.
One pass over the 90,704 JPEGs of the mirror (Kaggle CPU notebook, ~20-30 min on 4 vCPU).

  python scripts/prepare_data.py --out /kaggle/working/prepare --resized /kaggle/working/upmc256.zip

Per image (one decode): md5 of the file, size, pixel std, channel sums, 64-bit pHash, and a copy resized to a
shorter side of 256 px (JPEG q95) written into upmc256.zip as upmc256/images/<split>/<label>/<id>.jpg.

Pair validity (proposal 2.3):
  1 image valid   decodes, shorter side >= 64 px, pixel std > 2            -> else image-missing
  2 text valid    a title row exists and has >= 1 content word (src/data/text.py) -> else text-missing
  3 same item     image and title share <id> and label (checked; mismatches are reported and dropped)
Duplicate clusters (proposal 3.3): items are linked when their files are identical (md5) or their pHashes differ
by <= max_hamming bits (valid images only); clusters = connected components.
  test_dup     test item whose cluster contains an official-train item
  cross_label  item whose cluster carries >= 2 labels
Split (proposal 3.1): official train -> StratifiedGroupKFold(n_splits=round(N / val_size), shuffle, seed 42),
class = stratum, cluster = group, fold 0 = val. Official test untouched.

Writes <out>/data/splits/{train,val,test}.csv + split.json and <out>/data/meta/{image_meta,text_meta}.csv.gz
(no titles, no pixels) -> copy both folders into the repo and commit them.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from src.data.phash import near_duplicate_pairs, phash
from src.data.text import ClassLexicon, clean_title, content_words, normalize, site_name
from src.data.upmc import fingerprint, read_titles
from src.utils import paths
from src.utils.io import save_json

MIN_SIDE, MIN_STD = 64, 2.0


def list_images(root: Path) -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        for label in sorted(os.listdir(root / "images" / split)):
            d = root / "images" / split / label
            if not d.is_dir():
                continue
            for e in os.scandir(d):
                if e.is_file() and e.name.lower().endswith((".jpg", ".jpeg")):
                    rows.append((e.name.rsplit(".", 1)[0], split, label, e.path))
    return pd.DataFrame(rows, columns=["id", "official_split", "label", "path"])


def process(args):
    path, short_side, quality, resize = args
    rec = {"ok": False, "w": 0, "h": 0, "nbytes": 0, "md5": "", "phash": "", "std": np.nan,
           "sum_r": 0.0, "sum_g": 0.0, "sum_b": 0.0, "sumsq_r": 0.0, "sumsq_g": 0.0, "sumsq_b": 0.0, "n_px": 0,
           "rw": 0, "rh": 0, "err": ""}
    out = None
    try:
        data = Path(path).read_bytes()
        rec["nbytes"], rec["md5"] = len(data), hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            rec["w"], rec["h"] = w, h
            s = min(w, h)
            if s > short_side:                           # JPEG draft: decode at a reduced scale >= target
                im.draft("RGB", (int(w * short_side / s) + 1, int(h * short_side / s) + 1))
            im = im.convert("RGB")
            if min(im.size) > short_side:
                k = short_side / min(im.size)
                im = im.resize((max(1, round(im.size[0] * k)), max(1, round(im.size[1] * k))), Image.BICUBIC)
        x = np.asarray(im, dtype=np.float64) / 255.0
        rec["std"] = float((np.asarray(im.convert("L"), dtype=np.float64)).std())
        for i, c in enumerate("rgb"):
            rec[f"sum_{c}"] = float(x[..., i].sum())
            rec[f"sumsq_{c}"] = float((x[..., i] ** 2).sum())
        rec["n_px"] = int(x.shape[0] * x.shape[1])
        rec["rw"], rec["rh"] = im.size
        rec["phash"] = f"{phash(im):016x}"
        if resize:
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality)
            out = buf.getvalue()
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["err"] = f"{type(e).__name__}: {e}"[:200]
    return rec, out


def clusters(meta: pd.DataFrame, max_hamming: int, log):
    n = len(meta)
    rows, cols = [], []
    # exact duplicates: chain the members of every md5 group
    dup = meta[(meta["md5"] != "") & meta["md5"].duplicated(keep=False)]
    for _, idx in dup.groupby("md5").groups.items():          # index labels = positions (RangeIndex)
        idx = np.asarray(idx)
        if len(idx) > 1:
            rows += list(idx[:-1])
            cols += list(idx[1:])
    n_md5_edges = len(rows)
    valid = np.flatnonzero(meta["has_img"].to_numpy())
    h = np.array([int(v, 16) for v in meta["phash"].to_numpy()[valid]], dtype=np.uint64)
    t0 = time.time()
    i, j, d = near_duplicate_pairs(h, h, max_dist=max_hamming, same_set=True, chunk=256)
    log(f"  pHash all-pairs over {len(h):,} valid images: {len(i):,} pairs <= {max_hamming} bits ({time.time() - t0:.0f}s)")
    rows += list(valid[i])
    cols += list(valid[j])
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    return comp, {"md5_edges": n_md5_edges, "phash_pairs": int(len(i)),
                  "phash_pairs_by_distance": {int(k): int(v) for k, v in zip(*np.unique(d, return_counts=True))}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upmc-root", default="auto")
    p.add_argument("--out", default="outputs/prepare")
    p.add_argument("--resized", default="outputs/upmc256.zip", help="zip for the pre-resized copy ('' = skip)")
    p.add_argument("--short-side", type=int, default=256)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-hamming", type=int, default=4)
    args = p.parse_args()
    t_start = time.time()
    out = Path(args.out)
    (out / "data/splits").mkdir(parents=True, exist_ok=True)
    (out / "data/meta").mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    root = paths.upmc_root(args.upmc_root)
    log(f"mirror: {root}")
    imgs = list_images(root)
    titles = read_titles(root)
    log(f"images: {len(imgs):,} | title rows: {len(titles):,}")

    # ---- one pass over the images
    t0 = time.time()
    jobs = [(pth, args.short_side, args.quality, bool(args.resized)) for pth in imgs["path"]]
    recs = []
    zf = zipfile.ZipFile(args.resized, "w", zipfile.ZIP_STORED) if args.resized else None
    with Pool(args.workers) as pool:
        for k, (rec, data) in enumerate(pool.imap(process, jobs, chunksize=32)):
            recs.append(rec)
            if zf is not None and data is not None:
                r = imgs.iloc[k]
                zf.writestr(f"{paths.RESIZED_NAME}/images/{r.official_split}/{r.label}/{r.id}.jpg", data)
            if (k + 1) % 10000 == 0:
                log(f"  {k + 1:,}/{len(jobs):,} images ({time.time() - t0:.0f}s)")
    if zf is not None:
        zf.close()
    log(f"image pass: {time.time() - t0:.0f}s" + (f" | resized copy -> {args.resized} "
                                                  f"({os.path.getsize(args.resized) / 1e9:.2f} GB)" if args.resized else ""))
    meta = pd.concat([imgs.drop(columns="path"), pd.DataFrame(recs)], axis=1)
    meta["has_img"] = meta["ok"] & (meta[["w", "h"]].min(axis=1) >= MIN_SIDE) & (meta["std"] > MIN_STD)

    # ---- pairs (criterion 3) and text validity (criterion 2)
    t = titles.set_index("id")
    meta["has_title_row"] = meta["id"].isin(t.index)
    label_mismatch = meta["has_title_row"] & (meta["label"] != meta["id"].map(t["label"]))
    split_mismatch = meta["has_title_row"] & (meta["official_split"] != meta["id"].map(t["official_split"]))
    orphan_titles = int((~titles["id"].isin(meta["id"])).sum())
    log(f"pairs: title rows without image {orphan_titles} | label mismatches {int(label_mismatch.sum())} | "
        f"split mismatches {int(split_mismatch.sum())}")
    meta.loc[label_mismatch | split_mismatch, "has_title_row"] = False      # never pair across labels / splits
    raw = meta["id"].map(t["title"]).where(meta["has_title_row"], "")
    clean = raw.map(clean_title)
    n_content = clean.map(lambda s: len(content_words(s)))
    meta["has_txt"] = meta["has_title_row"] & (n_content > 0)

    # ---- duplicate clusters
    comp, link_stats = clusters(meta, args.max_hamming, log)
    meta["cluster_id"] = comp
    csize = meta.groupby("cluster_id")["id"].transform("size")
    clabels = meta.groupby("cluster_id")["label"].transform("nunique")
    has_train = meta.assign(t=meta["official_split"] == "train").groupby("cluster_id")["t"].transform("any")
    meta["cluster_size"] = csize
    meta["cross_label"] = clabels > 1
    meta["test_dup"] = (meta["official_split"] == "test") & has_train & (csize > 1)
    # renumber clusters 0..C-1 by first appearance (stable, independent of scipy internals)
    meta["cluster_id"] = pd.factorize(meta["cluster_id"])[0]

    # ---- split
    classes = sorted(meta["label"].unique())
    tr = meta[meta["official_split"] == "train"].reset_index(drop=True)
    n_splits = max(2, round(len(tr) / args.val_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.split_seed)
    _, va_idx = next(sgkf.split(np.zeros(len(tr)), tr["label"], groups=tr["cluster_id"]))
    is_val = np.zeros(len(tr), bool)
    is_val[va_idx] = True
    cols = ["id", "label", "official_split", "has_img", "has_txt", "cluster_id", "test_dup", "cross_label"]
    frames = {"train": tr[~is_val][cols], "val": tr[is_val][cols],
              "test": meta[meta["official_split"] == "test"][cols]}
    leak = set(frames["train"]["cluster_id"]) & set(frames["val"]["cluster_id"])
    assert not leak, "a duplicate cluster is split between train and val"
    fps = {}
    for s, df in frames.items():
        df = df.sort_values("id").reset_index(drop=True)
        frames[s] = df
        df.to_csv(out / "data/splits" / f"{s}.csv", index=False)
        fps[s] = fingerprint(df["id"])

    def counts(df):
        hi, ht = df["has_img"].astype(bool), df["has_txt"].astype(bool)
        return {"items": int(len(df)), "complete": int((hi & ht).sum()), "text_missing": int((hi & ~ht).sum()),
                "image_missing": int((~hi & ht).sum()), "both_missing": int((~hi & ~ht).sum()),
                "test_dup": int(df["test_dup"].sum()), "cross_label": int(df["cross_label"].sum())}

    big = meta.groupby("cluster_id").size().sort_values(ascending=False)
    info = {
        "dataset": "upmc_food101", "source": "Kaggle gianmarco96/upmcfood101", "classes": classes,
        "split_unit": "item (one web page = image + title, id = file stem)",
        "protocol": "official train -> StratifiedGroupKFold fold 0 = val (stratum: class, group: image-duplicate "
                    "cluster); official test untouched",
        "split_seed": args.split_seed, "n_splits": n_splits, "val_fold": 0,
        "validity": {"min_side": MIN_SIDE, "min_std": MIN_STD,
                     "text": "title row exists and has >= 1 content word (src/data/text.py: GENERIC_WORDS)"},
        "duplicates": {"max_hamming": args.max_hamming, **link_stats,
                       "clusters_with_2plus_items": int((big > 1).sum()),
                       "items_in_2plus_clusters": int(big[big > 1].sum()),
                       "largest_clusters": [int(v) for v in big.head(10)],
                       "cross_label_clusters": int(meta[meta["cross_label"]]["cluster_id"].nunique())},
        "pairing": {"title_rows_without_image": orphan_titles, "label_mismatch": int(label_mismatch.sum()),
                    "split_mismatch": int(split_mismatch.sum()),
                    "images_without_title_row": int((~meta["has_title_row"]).sum())},
        "counts": {s: counts(df) for s, df in frames.items()},
        "class_counts": {s: df.groupby("label").size().reindex(classes, fill_value=0).tolist() for s, df in frames.items()},
        "fingerprints": fps,
        "created_with": "scripts/prepare_data.py",
    }
    save_json(info, out / "data/splits/split.json")

    # ---- per-item metadata without titles / pixels (EDA input, committed)
    img_cols = ["id", "official_split", "label", "ok", "err", "w", "h", "nbytes", "md5", "phash", "std", "n_px",
                "sum_r", "sum_g", "sum_b", "sumsq_r", "sumsq_g", "sumsq_b", "has_img", "cluster_id", "cluster_size",
                "test_dup", "cross_label"]
    meta[img_cols].to_csv(out / "data/meta/image_meta.csv.gz", index=False)
    lex = ClassLexicon(classes)
    mentioned = clean.map(lex.mentioned_classes)
    text_meta = pd.DataFrame({
        "id": meta["id"], "official_split": meta["official_split"], "label": meta["label"],
        "has_title_row": meta["has_title_row"], "has_txt": meta["has_txt"],
        "n_chars": clean.str.len(), "n_words": clean.str.split().str.len().fillna(0).astype(int),
        "n_content_words": n_content, "site": clean.map(site_name),
        "mention_true_exact": [lex.mentions_exact(s, c) if s else False for s, c in zip(clean, meta["label"])],
        "mention_true": [c in m for c, m in zip(meta["label"], mentioned)],
        "mention_other": [bool(m - {c}) for c, m in zip(meta["label"], mentioned)],
        "n_classes_mentioned": mentioned.map(len),
        "k0_pred": clean.map(lambda s: lex.keyword_rule(s) or ""),
        "norm_md5": clean.map(lambda s: hashlib.md5(normalize(s).encode()).hexdigest() if s else ""),
    })
    text_meta.to_csv(out / "data/meta/text_meta.csv.gz", index=False)

    for s, c in info["counts"].items():
        log(f"{s:5s}: {c}")
    log(f"duplicates: {json.dumps(info['duplicates'])}")
    log(f"total {time.time() - t_start:.0f}s -> {out}")
    (out / "prepare_log.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
