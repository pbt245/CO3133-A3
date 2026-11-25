#!/usr/bin/env python
"""Aggregate every evaluated run into the comparison of the report.

  python scripts/compare.py                    # -> outputs/comparison/

summary.md / summary.csv   per experiment: mean +- std over seeds (accuracy, macro-F1, top-5, ECE, dedup test,
                           class-name subsets, missing-text set, modality removal, cost)
decisions.json             decision rule (proposal 4): mean macro-F1 difference > larger seed std AND
                           McNemar p < 0.05 on seed 42
support_conflict.csv       preliminary support / conflict groups (proposal 5.5) on seed 42: UA vs UB correctness,
                           fusion accuracy per group, which modality the fused prediction followed
*.png                      macro-F1 bars, validation curves, support / conflict bars
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.engine.metrics import mcnemar_test
from src.utils.io import load_json, resolve_path, save_json
from src.viz.plots import _save, plt

ORDER = ["k0_keyword", "ua_resnet50", "ub_distilbert", "f0_late_fusion", "f1_concat"]
LABEL = {"k0_keyword": "K0 keyword rule", "ua_resnet50": "UA image (ResNet-50)", "ub_distilbert": "UB text (DistilBERT)",
         "f0_late_fusion": "F0 late fusion", "f1_concat": "F1 concat fusion"}
PAIRS = [("ub_distilbert", "k0_keyword"), ("ub_distilbert", "ua_resnet50"), ("f0_late_fusion", "ub_distilbert"),
         ("f1_concat", "ub_distilbert"), ("f1_concat", "ua_resnet50"), ("f1_concat", "f0_late_fusion")]


def order_key(name):
    return (ORDER.index(name) if name in ORDER else 99, name)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def pm(mean, std, scale=100.0, digits=2):
    if mean is None or not np.isfinite(mean):
        return "-"
    if std is None or not np.isfinite(std):
        return f"{scale * mean:.{digits}f}"
    return f"{scale * mean:.{digits}f} ± {scale * std:.{digits}f}"


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return np.nan
        d = d[k]
    return d


def collect(runs: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs.glob("*/seed*/eval/test_metrics.json")):
        m = load_json(f)
        run_dir = f.parent.parent
        s = load_json(run_dir / "train_summary.json") if (run_dir / "train_summary.json").exists() else {}
        rows.append({
            "experiment": run_dir.parent.name, "seed": int(m["seed"]), "run_dir": str(run_dir),
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "top5": m["top5_accuracy"], "ece": m["ece"],
            "n_test": m["n_samples"],
            "dedup_acc": g(m, "subsets", "dedup", "accuracy"), "dedup_f1": g(m, "subsets", "dedup", "macro_f1"),
            "n_dedup": g(m, "subsets", "dedup", "n_samples"),
            "mention_acc": g(m, "subsets", "mention_true", "accuracy"),
            "no_mention_acc": g(m, "subsets", "no_mention_true", "accuracy"),
            "n_no_mention": g(m, "subsets", "no_mention_true", "n_samples"),
            "missing_text_acc": g(m, "missing_text", "as_is", "accuracy"),
            "missing_text_zero_acc": g(m, "missing_text", "text_zero_filled", "accuracy"),
            "n_missing_text": g(m, "missing_text", "as_is", "n_samples"),
            "img_removed_acc": g(m, "modality_removal", "image_removed", "accuracy"),
            "txt_removed_acc": g(m, "modality_removal", "text_removed", "accuracy"),
            "val_f1": g(m, "val", "macro_f1"), "best_epoch": s.get("best_epoch", np.nan),
            "epochs_run": s.get("epochs_run", np.nan), "train_min": s.get("train_time_s", np.nan) / 60,
            "params_m": g(m, "parameters", "total") / 1e6,
            "throughput": g(m, "inference", "throughput_items_per_s"),
            "latency_ms": g(m, "inference", "latency_ms_bs1_mean"),
            "weight_image": m.get("weight_image", np.nan),
        })
    return pd.DataFrame(rows)


def preds(run_dir, name="test"):
    """(ids, y_true, y_pred); the .npz files are not committed, so fall back to the CSV."""
    f = Path(run_dir) / "eval" / f"predictions_{name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return z["ids"], z["y_true"], z["probs"].astype(np.float32).argmax(1)
    d = pd.read_csv(f.with_suffix(".csv"), dtype={"id": str})
    return d["id"].to_numpy(), d["y_true"].to_numpy(), d["y_pred"].to_numpy()


def decide(df, agg, a, b):
    if a not in agg.index or b not in agg.index:
        return None
    ra, rb = df[(df.experiment == a) & (df.seed == 42)], df[(df.experiment == b) & (df.seed == 42)]
    mc = None
    if len(ra) and len(rb):
        ia, y, pa = preds(ra.run_dir.iloc[0])
        ib, _, pb = preds(rb.run_dir.iloc[0])
        assert (ia == ib).all()
        mc = mcnemar_test(y, pa, pb)
    diff = agg.loc[a, "macro_f1_mean"] - agg.loc[b, "macro_f1_mean"]
    std = np.nanmax([agg.loc[a, "macro_f1_std"], agg.loc[b, "macro_f1_std"], 0.0])
    big = abs(diff) > std
    sig = mc is not None and mc["p_value"] < 0.05
    verdict = (f"{a} better" if diff > 0 else f"{b} better") if (big and sig) else "no decision (rule not met)"
    return {"a": a, "b": b, "diff_macro_f1_pp": 100 * diff, "larger_std_pp": 100 * std, "diff_exceeds_std": bool(big),
            "mcnemar_seed42": mc, "verdict": verdict,
            "seeds": [int(agg.loc[a, "n_seeds"]), int(agg.loc[b, "n_seeds"])]}


def test_flags(meta_dir: Path) -> pd.DataFrame:
    """Per test item, from the committed data/meta (no titles needed):
    mention_true  the title mentions the true class (same lexicon as src/data/text.py)
    title_seen    the normalised title occurs verbatim in train/val (several images of one web page, or a
                  generic title: the image-duplicate grouping cannot see this, proposal 3.3 item 2)"""
    tm = pd.read_csv(meta_dir / "text_meta.csv.gz", dtype={"id": str}, keep_default_na=False)
    tm["mention_true"] = tm["mention_true"].astype(str).str.lower() == "true"
    trainval = set(pd.read_csv(meta_dir.parent / "splits" / "train.csv", dtype={"id": str})["id"]) | \
        set(pd.read_csv(meta_dir.parent / "splits" / "val.csv", dtype={"id": str})["id"])
    seen = set(tm.loc[tm["id"].isin(trainval) & (tm["norm_md5"] != ""), "norm_md5"])
    tm["title_seen"] = tm["norm_md5"].isin(seen) & (tm["norm_md5"] != "")
    return tm.set_index("id")[["mention_true", "title_seen"]]


def title_unseen(df, flags) -> pd.DataFrame:
    """Accuracy / macro-F1 of every run on test items whose title never occurs in train/val."""
    from src.engine.metrics import classification_metrics
    rows = []
    for r in df.itertuples():
        ids, y, p = preds(r.run_dir)
        seen = flags.loc[list(ids), "title_seen"].to_numpy()
        k = int(max(y.max(), p.max()) + 1)
        for name, m in (("seen", seen), ("unseen", ~seen)):
            cm = classification_metrics(y[m], p[m], max(k, 101))
            rows.append({"experiment": r.experiment, "seed": r.seed, "subset": name, "n": int(m.sum()),
                         "accuracy": cm["accuracy"], "macro_f1": cm["macro_f1"]})
    return pd.DataFrame(rows)


def support_conflict(df, out: Path, flags=None):
    def run(e):
        r = df[(df.experiment == e) & (df.seed == 42)]
        return r.run_dir.iloc[0] if len(r) else None
    if run("ua_resnet50") is None or run("ub_distilbert") is None:
        return None, None
    ids, y, pa = preds(run("ua_resnet50"))
    _, _, pb = preds(run("ub_distilbert"))
    group = np.where(pa == y, np.where(pb == y, "support", "conflict_I"),
                     np.where(pb == y, "conflict_T", np.where(pa == pb, "both_wrong_agree", "both_wrong_disagree")))
    mention = flags.loc[list(ids), "mention_true"].to_numpy() if flags is not None else None
    seen = flags.loc[list(ids), "title_seen"].to_numpy() if flags is not None else None
    rows = []
    fused = {e: preds(run(e))[2] for e in ("f0_late_fusion", "f1_concat") if run(e) is not None}
    names = ["support", "conflict_I", "conflict_T", "both_wrong_agree", "both_wrong_disagree"]
    subsets = [("all", np.ones(len(y), bool))]
    if mention is not None:
        subsets += [("title mentions class", mention), ("no class mention", ~mention),
                    ("title seen in train", seen), ("title unseen", ~seen)]
    for sname, smask in subsets:
        for gname in names:
            m = (group == gname) & smask
            r = {"subset": sname, "group": gname, "n": int(m.sum()), "share": float(m.sum() / max(1, smask.sum()))}
            for e, pf in fused.items():
                r[f"{e}_acc"] = float((pf[m] == y[m]).mean()) if m.any() else np.nan
                r[f"{e}_follows_image"] = float((pf[m] == pa[m]).mean()) if m.any() else np.nan
                r[f"{e}_follows_text"] = float((pf[m] == pb[m]).mean()) if m.any() else np.nan
            rows.append(r)
    sc = pd.DataFrame(rows)
    sc.to_csv(out / "support_conflict.csv", index=False)
    a = sc[sc.subset == "all"].set_index("group")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(names))
    for k, e in enumerate(fused):
        ax.bar(x + (k - 0.5) * 0.38, a.loc[names, f"{e}_acc"], 0.38, label=LABEL.get(e, e))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={a.loc[n, 'n']:,})" for n in names], fontsize=8)
    ax.set_ylabel("fusion accuracy (seed 42)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Test items grouped by unimodal correctness (UA image, UB text)")
    _save(fig, out / "support_conflict.png")
    return sc, group


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="outputs/runs/upmc")
    p.add_argument("--out", default="outputs/comparison")
    p.add_argument("--meta-dir", default="data/meta")
    args = p.parse_args()
    runs, out = resolve_path(args.runs), resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = collect(runs)
    if df.empty:
        sys.exit(f"no evaluated runs under {runs}")
    df = df[df.experiment.isin(ORDER) | ~df.experiment.str.startswith("timing")]
    df.to_csv(out / "runs.csv", index=False)
    cols = [c for c in df.columns if c not in ("experiment", "seed", "run_dir")]
    agg = df.groupby("experiment")[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = df.groupby("experiment").size()
    agg["seeds"] = df.groupby("experiment")["seed"].apply(lambda s: " ".join(map(str, sorted(s))))
    agg = agg.loc[sorted(agg.index, key=order_key)]
    agg.to_csv(out / "summary.csv")
    E = list(agg.index)
    L = lambda e: LABEL.get(e, e)

    md = ["# M2 comparison - UPMC Food-101 (complete test pairs)", "",
          "Mean ± std over seeds (percent). Test = complete test pairs (valid image + valid title). "
          "Dedup = without test items linked to an official-train image and without cross-label clusters.", ""]
    md += [md_table(["model", "seeds", "accuracy", "macro-F1", "top-5", "ECE", "dedup macro-F1", "val macro-F1"],
                    [[L(e), agg.loc[e, "seeds"], pm(agg.loc[e, "accuracy_mean"], agg.loc[e, "accuracy_std"]),
                      pm(agg.loc[e, "macro_f1_mean"], agg.loc[e, "macro_f1_std"]),
                      pm(agg.loc[e, "top5_mean"], agg.loc[e, "top5_std"]),
                      pm(agg.loc[e, "ece_mean"], agg.loc[e, "ece_std"]),
                      pm(agg.loc[e, "dedup_f1_mean"], agg.loc[e, "dedup_f1_std"]),
                      pm(agg.loc[e, "val_f1_mean"], agg.loc[e, "val_f1_std"])] for e in E]), ""]
    first = df.iloc[0]
    md += [f"Test complete pairs: {int(first.n_test):,}; dedup subset: {int(first.n_dedup):,}; "
           f"titles without a class mention: {int(first.n_no_mention):,}; natural missing-text set: "
           f"{int(first.n_missing_text) if np.isfinite(first.n_missing_text) else 0:,}.", ""]
    md += ["## Where the text helps: accuracy by subset", "",
           md_table(["model", "title mentions class", "no class mention", "missing-text set (as is)",
                     "missing-text, text zero-filled / fallback", "image removed", "text removed"],
                    [[L(e), pm(agg.loc[e, "mention_acc_mean"], agg.loc[e, "mention_acc_std"]),
                      pm(agg.loc[e, "no_mention_acc_mean"], agg.loc[e, "no_mention_acc_std"]),
                      pm(agg.loc[e, "missing_text_acc_mean"], agg.loc[e, "missing_text_acc_std"]),
                      pm(agg.loc[e, "missing_text_zero_acc_mean"], agg.loc[e, "missing_text_zero_acc_std"]),
                      pm(agg.loc[e, "img_removed_acc_mean"], agg.loc[e, "img_removed_acc_std"]),
                      pm(agg.loc[e, "txt_removed_acc_mean"], agg.loc[e, "txt_removed_acc_std"])] for e in E]), ""]
    md += ["## Cost", "",
           md_table(["model", "params (M)", "train time (min)", "epochs run / best", "throughput (items/s, bs 128)",
                     "latency (ms, bs 1)"],
                    [[L(e), f"{agg.loc[e, 'params_m_mean']:.1f}", f"{agg.loc[e, 'train_min_mean']:.1f}",
                      (f"{agg.loc[e, 'epochs_run_mean']:.1f} / {agg.loc[e, 'best_epoch_mean']:.1f}"
                       if np.isfinite(agg.loc[e, "epochs_run_mean"]) else "-"),
                      f"{agg.loc[e, 'throughput_mean']:.0f}" if np.isfinite(agg.loc[e, "throughput_mean"]) else "-",
                      f"{agg.loc[e, 'latency_ms_mean']:.2f}" if np.isfinite(agg.loc[e, "latency_ms_mean"]) else "-"]
                     for e in E]), "",
           "F0 train time = UA + UB; F0 weight on the image (tuned on val): "
           + (", ".join(f"seed {int(r.seed)}: {r.weight_image:.2f}" for r in df[df.experiment == "f0_late_fusion"].itertuples())
              or "-"), ""]

    decisions = [d for d in (decide(df, agg, a, b) for a, b in PAIRS) if d]
    save_json(decisions, out / "decisions.json")
    md += ["## Decision rule (mean macro-F1 difference > larger std AND McNemar p < 0.05, seed 42)", "",
           md_table(["A vs B", "Δ macro-F1 (pp)", "larger std (pp)", "McNemar b / c, p", "verdict"],
                    [[f"{L(d['a'])} vs {L(d['b'])}", f"{d['diff_macro_f1_pp']:+.2f}", f"{d['larger_std_pp']:.2f}",
                      (f"{d['mcnemar_seed42']['b']} / {d['mcnemar_seed42']['c']}, {d['mcnemar_seed42']['p_value']:.2g}"
                       if d["mcnemar_seed42"] else "-"), d["verdict"]] for d in decisions]), ""]

    flags = test_flags(resolve_path(args.meta_dir))
    tu = title_unseen(df, flags)
    tu.to_csv(out / "title_seen_unseen.csv", index=False)
    ta = tu.groupby(["experiment", "subset"])[["accuracy", "macro_f1"]].agg(["mean", "std"])
    n_seen = int(tu[tu.subset == "seen"].n.iloc[0])
    md += ["## Title leakage check: titles seen / unseen in train", "",
           f"{n_seen:,} of {int(first.n_test):,} complete test pairs ({100 * n_seen / first.n_test:.1f} %) have a "
           "normalised title that also occurs in train/val (several images of one web page, or a generic title). "
           "The image-duplicate grouping does not see these links.", "",
           md_table(["model", "acc. title seen", "acc. title unseen", "macro-F1 title unseen"],
                    [[L(e), pm(ta.loc[(e, "seen"), ("accuracy", "mean")], ta.loc[(e, "seen"), ("accuracy", "std")]),
                      pm(ta.loc[(e, "unseen"), ("accuracy", "mean")], ta.loc[(e, "unseen"), ("accuracy", "std")]),
                      pm(ta.loc[(e, "unseen"), ("macro_f1", "mean")], ta.loc[(e, "unseen"), ("macro_f1", "std")])]
                     for e in E if (e, "unseen") in ta.index]), ""]
    sc, _ = support_conflict(df, out, flags)
    if sc is not None:
        a = sc[sc.subset == "all"]
        fcols = [c for c in a.columns if c.endswith("_acc")]
        md += ["## Support / conflict (preliminary, seed 42)", "",
               "Groups by unimodal correctness: support = UA and UB right; conflict_I = only the image model right; "
               "conflict_T = only the text model right.", "",
               md_table(["group", "items", "share", *[c.replace("_acc", " acc.") for c in fcols],
                         *[c.replace("_follows_image", " follows image") for c in a.columns if c.endswith("_follows_image")]],
                        [[r.group, f"{r.n:,}", f"{100 * r.share:.1f} %", *[f"{100 * getattr(r, c):.1f}" for c in fcols],
                          *[f"{100 * getattr(r, c):.1f}" for c in a.columns if c.endswith("_follows_image")]]
                         for r in a.itertuples()]), "",
               "The same table split by class-name mention and by title seen / unseen is in support_conflict.csv.", ""]

    # figures
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar([L(e) for e in E], 100 * agg["macro_f1_mean"], yerr=100 * agg["macro_f1_std"].fillna(0), capsize=4,
           color=["#999", "tab:blue", "tab:orange", "tab:olive", "tab:green"][:len(E)])
    for i, e in enumerate(E):
        ax.text(i, 100 * agg.loc[e, "macro_f1_mean"] + 1, f"{100 * agg.loc[e, 'macro_f1_mean']:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("test macro-F1 (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("UPMC Food-101, complete test pairs (mean ± std over seeds)")
    _save(fig, out / "macro_f1.png")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for e in E:
        h = Path(df[(df.experiment == e)].sort_values("seed").run_dir.iloc[0]) / "history.csv"
        if h.exists():
            hh = pd.read_csv(h)
            ax.plot(hh["epoch"], 100 * hh["val_macro_f1"], marker="o", ms=3, label=L(e))
    ax.set_xlabel("epoch")
    ax.set_ylabel("val macro-F1 (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Validation macro-F1 per epoch (first seed)")
    _save(fig, out / "val_curves.png")

    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
