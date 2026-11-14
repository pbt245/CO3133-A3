#!/usr/bin/env python
"""Read-only inventory of the UPMC Food-101 Kaggle mirror (gianmarco96/upmcfood101).

Verifies the facts the M1 proposal takes from the literature: counts per split and class,
CSV layout, what the text field contains, image-text matching, decode failures, image
geometry, exact duplicates, and how often the text names its own class.

CPU only, no internet needed, standard Kaggle packages only (numpy, pandas, Pillow).

Kaggle:  notebook `co3133-a3-inventory`, inputs `upmcfood101` + `co3133-a3-code` (scripts/make_kaggle_bundle.py),
         after the setup cell:  !python scripts/inventory_upmc.py   (dataset root is found under /kaggle/input)
Local:   python scripts/inventory_upmc.py --root /path/to/upmcfood101 --out outputs/inventory

Outputs (in --out, default /kaggle/working/inventory or ./outputs/inventory):
  summary.json          every number printed below
  image_index.csv.gz    one row per image file (size, mode, bytes, md5, grey std, decode error)
  text_index.csv.gz     one row per CSV row (id, label, length, matching flags; no text)
"""
import argparse
import collections
import hashlib
import io
import json
import os
import platform
import re
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")
IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
IMG_RE = r"\.(jpe?g|png|gif|bmp|webp|tiff?)$"
PCTS = [0, 1, 5, 25, 50, 75, 95, 99, 100]
SUMMARY = {}


def sec(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def pct(values):
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return {}
    return {f"p{p}": round(float(np.percentile(v, p)), 2) for p in PCTS} | {"mean": round(float(v.mean()), 2)}


def show_pct(name, d):
    print(f"  {name:<22}" + "  ".join(f"{k}={v:g}" for k, v in d.items()))


def find_root(start="/kaggle/input"):
    """First directory (depth <= 5) that has both an images* and a texts* subfolder."""
    if not os.path.isdir(start):
        return None
    for dirpath, dirnames, _ in os.walk(start):
        if dirpath[len(start):].count(os.sep) > 5:
            dirnames[:] = []
            continue
        low = {d.lower(): d for d in dirnames}
        if any(k.startswith("image") for k in low) and any(k.startswith("text") for k in low):
            return dirpath
    return None


def subdir(root, prefix):
    for d in sorted(os.listdir(root)):
        if d.lower().startswith(prefix) and os.path.isdir(os.path.join(root, d)):
            return os.path.join(root, d)
    return None


# --------------------------------------------------------------------------- tree
def tree(root):
    sec(f"1. Directory tree  ({root})")
    total_files, total_bytes = 0, 0
    per_dir = collections.OrderedDict()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        nbytes = 0
        for f in filenames:
            try:
                nbytes += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
        total_files += len(filenames)
        total_bytes += nbytes
        top = rel.split(os.sep)[:2]
        key = "/".join(top) if rel != "." else "."
        e = per_dir.setdefault(key, {"files": 0, "bytes": 0, "subdirs": 0, "examples": []})
        e["files"] += len(filenames)
        e["bytes"] += nbytes
        if depth <= 2:
            e["subdirs"] += len(dirnames)
        if len(e["examples"]) < 3:
            e["examples"] += [os.path.join(rel, f) for f in sorted(filenames)[: 3 - len(e["examples"])]]
    for k, e in per_dir.items():
        print(f"  {k:<28} files={e['files']:>7}  size={e['bytes'] / 2**30:6.2f} GB  "
              f"subdirs(lvl<=2)={e['subdirs']:>4}  e.g. {e['examples'][:2]}")
    print(f"  TOTAL files={total_files}  size={total_bytes / 2**30:.2f} GB")
    SUMMARY["tree"] = {"root": root, "total_files": total_files,
                       "total_gb": round(total_bytes / 2**30, 3),
                       "dirs": {k: {kk: vv for kk, vv in e.items()} for k, e in per_dir.items()}}


# --------------------------------------------------------------------------- images
def list_images(img_root):
    rows, other = [], collections.Counter()
    for dirpath, _, filenames in os.walk(img_root):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            full = os.path.join(dirpath, f)
            parts = os.path.relpath(full, img_root).split(os.sep)
            if ext not in IMG_EXT:
                other[ext or "<none>"] += 1
                continue
            rows.append({"path": full, "rel": "/".join(parts), "file": f,
                         "split": parts[0] if len(parts) > 1 else "",
                         "label": parts[-2] if len(parts) >= 3 else ""})
    return pd.DataFrame(rows), other


def probe(path):
    r = {"path": path, "ok": False, "w": -1, "h": -1, "mode": "", "fmt": "",
         "nbytes": -1, "md5": "", "std": -1.0, "err": ""}
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        r["nbytes"], r["md5"] = len(data), hashlib.md5(data).hexdigest()
        im = Image.open(io.BytesIO(data))
        r["w"], r["h"] = im.size
        r["mode"], r["fmt"] = im.mode, im.format or ""
        try:  # fast reduced JPEG decode; full decode still detects truncation/corruption
            im.draft("L", (max(1, im.size[0] // 8), max(1, im.size[1] // 8)))
        except Exception:
            pass
        g = np.asarray(im.convert("L"), dtype=np.float32)
        r["std"], r["ok"] = float(g.std()), True
    except Exception as e:  # noqa: BLE001
        r["err"] = f"{type(e).__name__}: {str(e)[:80]}"
    return r


def images_report(img_df, other, workers):
    sec("2. Images")
    print(f"  image files: {len(img_df)}   non-image files: {dict(other)}")
    t0 = time.time()
    res = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(probe, img_df["path"].tolist(), chunksize=64), 1):
            res.append(r)
            if i % 10000 == 0:
                print(f"    probed {i}/{len(img_df)}  ({time.time() - t0:.0f}s)", flush=True)
    df = img_df.merge(pd.DataFrame(res), on="path")
    print(f"  probed all images in {time.time() - t0:.0f}s with {workers} workers")
    out = {}
    for split, g in df.groupby("split"):
        ok = g[g.ok]
        cls = g.groupby("label").size()
        minside = np.minimum(ok.w, ok.h)
        e = {
            "n": len(g), "decoded_ok": int(g.ok.sum()), "failed": int((~g.ok).sum()),
            "fail_types": collections.Counter(s.split(":")[0] for s in g.err[g.err != ""]).most_common(5),
            "classes": int(cls.size), "per_class_min": int(cls.min()), "per_class_max": int(cls.max()),
            "per_class_mean": round(float(cls.mean()), 1),
            "smallest_classes": cls.nsmallest(5).to_dict(),
            "formats": collections.Counter(ok.fmt).most_common(6),
            "modes": collections.Counter(ok["mode"]).most_common(6),
            "width": pct(ok.w), "height": pct(ok.h), "min_side": pct(minside),
            "aspect_w_over_h": pct(ok.w / ok.h.clip(lower=1)), "kbytes": pct(ok.nbytes / 1024),
            "min_side_lt_64": int((minside < 64).sum()), "grey_std_lt_2": int((ok["std"] < 2).sum()),
        }
        out[split] = e
        print(f"\n  [split={split!r}] n={e['n']} ok={e['decoded_ok']} failed={e['failed']} {e['fail_types']}")
        print(f"  classes={e['classes']} per-class min/mean/max={e['per_class_min']}/"
              f"{e['per_class_mean']}/{e['per_class_max']}  smallest={e['smallest_classes']}")
        print(f"  formats={e['formats']}  modes={e['modes']}")
        for k in ("width", "height", "min_side", "aspect_w_over_h", "kbytes"):
            show_pct(k, e[k])
        print(f"  min side < 64 px: {e['min_side_lt_64']}   near-constant (grey std < 2): {e['grey_std_lt_2']}")
    bad = df[~df.ok].head(5)
    if len(bad):
        print("\n  examples of failures:")
        for _, r in bad.iterrows():
            print(f"    {r.rel}: {r.err}")
    SUMMARY["images"] = out
    return df


# --------------------------------------------------------------------------- texts
def read_csv_any(path):
    for kw in ({"engine": "c"}, {"engine": "python"}):
        try:
            return pd.read_csv(path, header=None, dtype=str, keep_default_na=False,
                               on_bad_lines="skip", **kw), kw["engine"]
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"cannot parse {path}: {err}")


def infer_roles(df, labels):
    st = {}
    for c in df.columns:
        s = df[c].astype(str)
        st[c] = {"img_ext_rate": round(float(s.str.lower().str.strip().str.contains(IMG_RE).mean()), 3),
                 "label_rate": round(float(s.str.strip().isin(labels).mean()), 3) if labels else 0.0,
                 "nunique": int(s.nunique()), "mean_len": round(float(s.str.len().mean()), 1)}
    cols = list(df.columns)
    fn = max(cols, key=lambda c: st[c]["img_ext_rate"])
    fn = fn if st[fn]["img_ext_rate"] > 0.5 else None
    rest = [c for c in cols if c != fn]
    lab = max(rest, key=lambda c: (st[c]["label_rate"], -st[c]["nunique"])) if rest else None
    rest = [c for c in rest if c != lab]
    txt = max(rest, key=lambda c: st[c]["mean_len"]) if rest else None
    return {"file": fn, "label": lab, "text": txt}, st


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def texts_report(txt_root, labels):
    sec(f"3. Text files  ({txt_root})")
    files = sorted(os.listdir(txt_root))
    print(f"  files in text folder: {files}")
    names = {lab: lab.replace("_", " ").lower() for lab in labels}
    frames, out = [], {}
    for f in files:
        p = os.path.join(txt_root, f)
        if not f.lower().endswith((".csv", ".tsv", ".txt")) or os.path.isdir(p):
            continue
        df, engine = read_csv_any(p)
        roles, st = infer_roles(df, labels)
        first = df.iloc[0].tolist()
        header = False
        if roles["file"] is not None and not re.search(IMG_RE, str(first[roles["file"]]).lower().strip()):
            header = True
        if roles["label"] is not None and labels and str(first[roles["label"]]).strip() not in labels \
                and st[roles["label"]]["label_rate"] > 0.9:
            header = True
        if header:
            df = df.iloc[1:].reset_index(drop=True)
        split = "train" if "train" in f.lower() else "test" if "test" in f.lower() else f
        print(f"\n  --- {f}  (split={split}, parser={engine}, rows={len(df)}, columns={df.shape[1]}, "
              f"header_row={'yes: ' + str(first) if header else 'no'})")
        for c, s in st.items():
            role = [k for k, v in roles.items() if v == c]
            print(f"    col {c}: role={role or '-'} {s}")
        print("    first 3 rows (text cut to 120 chars):")
        for _, r in df.head(3).iterrows():
            print("     ", [str(x)[:120] for x in r.tolist()])
        if roles["text"] is None:
            print("    !! no text column identified")
            continue
        t = df[roles["text"]].astype(str)
        lab = df[roles["label"]].astype(str).str.strip() if roles["label"] is not None else pd.Series([""] * len(df))
        fn = df[roles["file"]].astype(str).str.strip() if roles["file"] is not None else pd.Series([""] * len(df))
        low = t.str.lower()
        words = t.str.split().str.len().fillna(0).astype(int)
        own = np.array([names.get(l, "\0") in s for l, s in zip(lab, low)])
        # keyword rule: predict the class whose name is mentioned most often (rough leakage indicator)
        name_list = list(names.items())
        kw_pred, any_other = [], []
        for l, s in zip(lab, low):
            counts = [(s.count(n), k) for k, n in name_list]
            best = max(counts)
            kw_pred.append(best[1] if best[0] > 0 else None)
            any_other.append(any(c > 0 and k != l for c, k in counts))
        kw_pred = pd.Series(kw_pred)
        nonascii = t.map(lambda s: sum(ord(ch) > 127 for ch in s) / max(1, len(s)))
        nt = t.map(norm)
        e = {"rows": len(df), "header_row": header, "roles": roles,
             "empty": int((t.str.strip() == "").sum()), "words_lt_5": int((words < 5).sum()),
             "words": pct(words), "chars": pct(t.str.len()),
             "labels": int(lab.nunique()), "labels_not_in_image_folders": sorted(set(lab) - set(labels))[:10],
             "rows_per_label_min": int(lab.value_counts().min()), "rows_per_label_max": int(lab.value_counts().max()),
             "contains_own_class_name": round(float(own.mean()), 4),
             "contains_other_class_name": round(float(np.mean(any_other)), 4),
             "keyword_rule_acc": round(float((kw_pred == lab.values).mean()), 4),
             "keyword_rule_no_mention": round(float(kw_pred.isna().mean()), 4),
             "nonascii_gt_20pct": int((nonascii > 0.2).sum()),
             "duplicate_file_ids": int(fn.duplicated().sum()) if roles["file"] is not None else None,
             "exact_dup_texts_rows": int((nt.duplicated(keep=False) & (nt != "")).sum()),
             "exact_dup_texts_cross_label": int(pd.DataFrame({"t": nt, "l": lab})[nt != ""]
                                                .groupby("t").l.nunique().gt(1).sum())}
        out[split] = e
        print(f"    empty={e['empty']}  words<5={e['words_lt_5']}  labels={e['labels']} "
              f"rows/label min..max={e['rows_per_label_min']}..{e['rows_per_label_max']}  "
              f"labels missing from image folders={e['labels_not_in_image_folders']}")
        show_pct("words", e["words"])
        show_pct("chars", e["chars"])
        print(f"    contains own class name: {e['contains_own_class_name']:.1%}   "
              f"contains another class name: {e['contains_other_class_name']:.1%}")
        print(f"    keyword rule (most-mentioned class name) accuracy: {e['keyword_rule_acc']:.1%}  "
              f"(no class name at all: {e['keyword_rule_no_mention']:.1%})")
        print(f"    mostly non-ASCII texts: {e['nonascii_gt_20pct']}   duplicate file ids: {e['duplicate_file_ids']}   "
              f"rows in exact-duplicate texts: {e['exact_dup_texts_rows']} "
              f"(duplicate texts with >1 label: {e['exact_dup_texts_cross_label']})")
        rng = np.random.default_rng(0)
        print("    5 random texts (first 400 chars):")
        for i in rng.choice(len(df), size=min(5, len(df)), replace=False):
            print(f"      [{lab.iloc[i]} | {words.iloc[i]} words] {t.iloc[i][:400]!r}")
        print("    3 shortest non-empty texts:")
        for i in words[words > 0].nsmallest(3).index:
            print(f"      [{lab.iloc[i]}] {t.iloc[i][:200]!r}")
        frames.append(pd.DataFrame({"csv": f, "split": split, "file": fn, "label": lab,
                                    "n_words": words, "n_chars": t.str.len(), "own_name": own,
                                    "kw_pred": kw_pred, "norm_md5": nt.map(lambda s: hashlib.md5(s.encode()).hexdigest())}))
    SUMMARY["texts"] = out
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- matching / duplicates
def matching_report(img, txt):
    sec("4. Image <-> text matching (by file name)")
    out = {}
    img_by_split = {s: g for s, g in img.groupby("split")}
    for split, g in txt.groupby("split"):
        ig = img_by_split.get(split)
        if ig is None:
            print(f"  [{split}] no image folder named {split!r}; image splits = {list(img_by_split)}")
            continue
        base_img = ig.drop_duplicates("file").set_index("file")
        base = g.file.map(os.path.basename)
        found = base.isin(base_img.index)
        other = set().union(*(set(v.file) for s, v in img_by_split.items() if s != split))
        lab_mismatch = int((base_img.loc[base[found], "label"].values != g.label[found].values).sum())
        img_ok = base_img.loc[base[found], "ok"].values
        txt_files = set(base)
        e = {"csv_rows": len(g), "with_image": int(found.sum()),
             "image_in_other_split_only": int((~found & base.isin(other)).sum()),
             "no_image": int((~found & ~base.isin(other)).sum()),
             "images_without_csv_row": int((~ig.file.isin(txt_files)).sum()),
             "label_mismatch_csv_vs_folder": lab_mismatch,
             "matched_but_image_broken": int((~img_ok).sum())}
        tf = g[found].copy()
        ii = base_img.loc[base[found]]
        img_valid = ii.ok.values & (np.minimum(ii.w.values, ii.h.values) >= 64) & (ii["std"].values > 2)
        txt_valid = tf.n_words.values >= 5
        e["complete_pairs"] = int((img_valid & txt_valid).sum())
        e["text_missing_only"] = int((img_valid & ~txt_valid).sum())
        e["image_invalid_only"] = int((~img_valid & txt_valid).sum())
        e["both_invalid"] = int((~img_valid & ~txt_valid).sum())
        e["images_without_text_row"] = e["images_without_csv_row"]
        out[split] = e
        print(f"  [{split}] " + "  ".join(f"{k}={v}" for k, v in e.items()))
        if e["no_image"]:
            print(f"    e.g. CSV ids without image: {base[~found].head(3).tolist()}")
        if e["images_without_csv_row"]:
            print(f"    e.g. images without CSV row: {ig.file[~ig.file.isin(txt_files)].head(3).tolist()}")
    same = set(img_by_split.get("train", pd.DataFrame(columns=["file"])).file) & \
        set(img_by_split.get("test", pd.DataFrame(columns=["file"])).file)
    print(f"  same file name in train and test image folders: {len(same)}")
    out["same_filename_train_and_test"] = len(same)
    SUMMARY["matching"] = out

    sec("5. Exact duplicates (md5 of image bytes, normalised text)")
    d = {}
    ok = img[img.ok]
    grp = ok.groupby("md5")
    sizes = grp.size()
    dup = sizes[sizes > 1]
    multi_split = grp.split.nunique()
    multi_label = grp.label.nunique()
    d["image_dup_groups"] = int(dup.size)
    d["images_in_dup_groups"] = int(dup.sum())
    d["image_dup_groups_cross_split"] = int((multi_split > 1).sum())
    d["image_dup_groups_cross_label"] = int((multi_label > 1).sum())
    for md5 in dup.index[:3]:
        print(f"    e.g. {ok[ok.md5 == md5].rel.tolist()[:4]}")
    if len(txt):
        tg = txt[txt.n_words > 0].groupby("norm_md5")
        d["text_dup_groups"] = int((tg.size() > 1).sum())
        d["text_dup_groups_cross_split"] = int((tg.split.nunique() > 1).sum())
        d["text_dup_groups_cross_label"] = int((tg.label.nunique() > 1).sum())
    print("  " + "  ".join(f"{k}={v}" for k, v in d.items()))
    SUMMARY["duplicates"] = d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="dataset root containing images/ and texts/")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args, _ = ap.parse_known_args()  # tolerate ipykernel's own argv inside notebooks
    root = args.root or find_root()
    if root is None:
        sys.exit("dataset root not found: pass --root (folder that contains images/ and texts/)")
    out = args.out or ("/kaggle/working/inventory" if os.path.isdir("/kaggle/working") else "outputs/inventory")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sec("0. Environment")
    print(f"  python {platform.python_version()}  pandas {pd.__version__}  numpy {np.__version__}  "
          f"Pillow {Image.__version__}  cpus={os.cpu_count()}  platform={platform.platform()}")
    SUMMARY["env"] = {"python": platform.python_version(), "pandas": pd.__version__,
                      "numpy": np.__version__, "pillow": Image.__version__, "cpus": os.cpu_count()}
    tree(root)
    img_root, txt_root = subdir(root, "image"), subdir(root, "text")
    print(f"\n  images folder: {img_root}\n  texts folder:  {txt_root}")
    img_df, other = list_images(img_root)
    if img_df.empty:
        sys.exit("no image files found")
    labels = set(img_df.label) - {""}
    img = images_report(img_df, other, args.workers)
    txt = texts_report(txt_root, labels)
    if len(txt):
        matching_report(img, txt)
    img.drop(columns=["path"]).to_csv(os.path.join(out, "image_index.csv.gz"), index=False)
    if len(txt):
        txt.to_csv(os.path.join(out, "text_index.csv.gz"), index=False)
    SUMMARY["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(SUMMARY, fh, indent=1, default=str)
    sec(f"Done in {SUMMARY['runtime_s']}s. Files written to {out}: {sorted(os.listdir(out))}")


if __name__ == "__main__":
    main()
