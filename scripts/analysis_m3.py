#!/usr/bin/env python
"""M3 analyses from the evaluated runs (CPU, a few seconds; the committed prediction CSVs are enough, the
temperature-scaling step also needs the predictions_*.npz files of a3_results.zip).

  python scripts/analysis_m3.py            # -> outputs/analysis/

support_conflict_seeds.csv / .png  support / conflict groups (UA vs UB correctness) for every seed, accuracy of each
                                   fusion per group and how often it follows the image model; mean +- std over seeds.
                                   The same with class names masked (UA vs UB-mask; F0/F1/F2-mask) for E2.
removal_curves.csv / .png          a random fraction (0, 25, 50, 75, 100 %) of the complete test pairs loses its title
                                   (or its image); each model uses its own fallback: UA / UB predict the train prior
                                   class without their modality, F0 uses the remaining unimodal model, F1 zero-fills,
                                   F2 uses its [MISSING] token. Same random items for every model (seed 42).
ablations.md                       E1 (auxiliary losses), E2 (class-name masking), E4 (modality dropout) tables
calibration.csv                    test ECE before / after temperature scaling fitted on validation (needs .npz)
attention_groups.csv               F2 [FUS] attention share on image tokens per support / conflict group
                                   (needs eval/fus_attention_test.csv from scripts/qualitative.py)
ANALYSIS.md                        all of the above as tables
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.engine.metrics import classification_metrics, expected_calibration_error
from src.utils.io import load_json, resolve_path
from src.viz.plots import _save, plt

SEEDS = [42, 123, 2026]
LABEL = {"ua_resnet50": "UA image", "ub_distilbert": "UB text", "f0_late_fusion": "F0 late", "f1_concat": "F1 concat",
         "f2_crossmodal": "F2 cross-modal", "f2_noaux": "F2 no aux (E1)", "f2_nomoddrop": "F2 no mod. dropout (E4)",
         "ub_distilbert_mask": "UB-mask", "f0_late_fusion_mask": "F0-mask", "f1_concat_mask": "F1-mask",
         "f2_crossmodal_mask": "F2-mask", "t0_tfidf": "T0 TF-IDF", "t0_tfidf_mask": "T0-mask", "k0_keyword": "K0"}
GROUPS = ["support", "conflict_I", "conflict_T", "both_wrong_agree", "both_wrong_disagree"]


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def pm(values, scale=100.0, digits=1):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return "-"
    return f"{scale * v.mean():.{digits}f}" + (f" ± {scale * v.std(ddof=1):.{digits}f}" if len(v) > 1 else "")


class Runs:
    def __init__(self, root: Path):
        self.root = root
        self.cache = {}

    def dir(self, exp, seed):
        d = self.root / exp / f"seed{seed}"
        return d if (d / "eval" / "test_metrics.json").exists() else None

    def seeds(self, exp):
        return [s for s in SEEDS if self.dir(exp, s) is not None]

    def pred(self, exp, seed, name="test"):
        key = (exp, seed, name)
        if key not in self.cache:
            d = self.dir(exp, seed)
            f = d / "eval" / f"predictions_{name}.csv" if d else None
            if f is None or not f.exists():
                self.cache[key] = None
            else:
                p = pd.read_csv(f, dtype={"id": str}, usecols=["id", "y_true", "y_pred"])
                self.cache[key] = (p["id"].to_numpy(), p["y_true"].to_numpy(), p["y_pred"].to_numpy())
        return self.cache[key]

    def metrics(self, exp, seed):
        d = self.dir(exp, seed)
        return load_json(d / "eval" / "test_metrics.json") if d else None


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return np.nan
        d = d[k]
    return d


# ------------------------------------------------------------------------------------------- support / conflict
def support_conflict(R: Runs, image, text, fusions, tag):
    rows = []
    for s in SEEDS:
        a, b = R.pred(image, s), R.pred(text, s)
        if a is None or b is None:
            continue
        ids, y, pa = a
        assert (b[0] == ids).all()
        pb = b[2]
        grp = np.where(pa == y, np.where(pb == y, "support", "conflict_I"),
                       np.where(pb == y, "conflict_T", np.where(pa == pb, "both_wrong_agree", "both_wrong_disagree")))
        for gname in GROUPS:
            m = grp == gname
            r = {"setting": tag, "seed": s, "group": gname, "n": int(m.sum()), "share": float(m.mean())}
            for e in fusions:
                f = R.pred(e, s)
                if f is None:
                    continue
                assert (f[0] == ids).all()
                pf = f[2]
                r[f"{e}_acc"] = float((pf[m] == y[m]).mean()) if m.any() else np.nan
                r[f"{e}_follows_image"] = float((pf[m] == pa[m]).mean()) if m.any() else np.nan
                r[f"{e}_follows_text"] = float((pf[m] == pb[m]).mean()) if m.any() else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def sc_table(sc, fusions):
    rows = []
    for gname in GROUPS:
        s = sc[sc.group == gname]
        if s.empty:
            continue
        rows.append([gname, pm(s["n"], 1, 0), pm(s["share"]),
                     *[pm(s[f"{e}_acc"]) if f"{e}_acc" in s else "-" for e in fusions],
                     *[pm(s[f"{e}_follows_image"]) if f"{e}_follows_image" in s else "-" for e in fusions]])
    return md_table(["group", "items", "share %", *[f"{LABEL[e]} acc." for e in fusions],
                     *[f"{LABEL[e]} follows image" for e in fusions]], rows)


def plot_sc(scs, out):
    fig, axes = plt.subplots(1, len(scs), figsize=(6.2 * len(scs), 3.9), squeeze=False)
    for ax, (title, sc, fusions) in zip(axes[0], scs):
        names = [gname for gname in GROUPS if gname in set(sc.group)]
        x = np.arange(len(names))
        fus = [e for e in fusions if f"{e}_acc" in sc]
        w = 0.8 / max(1, len(fus))
        for k, e in enumerate(fus):
            agg = sc.groupby("group")[f"{e}_acc"].agg(["mean", "std"]).loc[names]
            ax.bar(x + (k - (len(fus) - 1) / 2) * w, 100 * agg["mean"], w, yerr=100 * agg["std"].fillna(0), capsize=3,
                   label=LABEL[e])
        n = sc.groupby("group")["n"].mean().loc[names]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{gname}\n(n≈{n[gname]:,.0f})" for gname in names], fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_ylabel("fusion accuracy (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_title(title, fontsize=9)
    _save(fig, out / "support_conflict_seeds.png")


# ------------------------------------------------------------------------------------------- removal curves
def removal_curves(R: Runs, prior_class: int, fractions, k: int):
    """Accuracy / macro-F1 when a random fraction of complete test pairs loses the title (or the image)."""
    fallback = {"f0_late_fusion": {"text": "ua_resnet50", "image": "ub_distilbert"}}
    models = ["ua_resnet50", "ub_distilbert", "f0_late_fusion", "f1_concat", "f2_crossmodal", "f2_noaux", "f2_nomoddrop"]
    rows = []
    for e in models:
        for s in R.seeds(e):
            full = R.pred(e, s)
            if full is None:
                continue
            ids, y, p = full
            rng = np.random.default_rng(42)
            u = rng.random(len(y))                       # same items for every model and seed
            for drop in ("text", "image"):
                if e == "ua_resnet50":
                    pd_ = p if drop == "text" else np.full_like(p, prior_class)
                elif e == "ub_distilbert":
                    pd_ = p if drop == "image" else np.full_like(p, prior_class)
                elif e in fallback:
                    other = R.pred(fallback[e][drop], s)
                    pd_ = other[2] if other is not None else None
                else:
                    d = R.pred(e, s, f"test_drop_{drop}")
                    pd_ = d[2] if d is not None else None
                if pd_ is None:
                    continue
                for f in fractions:
                    q = np.where(u < f, pd_, p)
                    m = classification_metrics(y, q, k)
                    rows.append({"model": e, "seed": s, "removed": drop, "fraction": f, "accuracy": m["accuracy"],
                                 "macro_f1": m["macro_f1"]})
    return pd.DataFrame(rows)


def plot_removal(rc, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, drop in zip(axes, ("text", "image")):
        d = rc[rc.removed == drop]
        for e in d.model.unique():
            a = d[d.model == e].groupby("fraction")["macro_f1"].agg(["mean", "std"])
            ax.errorbar(100 * a.index, 100 * a["mean"], yerr=100 * a["std"].fillna(0), marker="o", ms=3, capsize=2,
                        label=LABEL[e], ls="--" if e in ("f2_noaux", "f2_nomoddrop") else "-")
        ax.set_xlabel(f"% of test pairs without the {'title' if drop == 'text' else 'image'}")
        ax.set_ylabel("macro-F1 (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.set_title(f"{'Title' if drop == 'text' else 'Image'} removed at test time (mean ± std over seeds)", fontsize=9)
    axes[0].legend(fontsize=7)
    _save(fig, out / "removal_curves.png")


# ------------------------------------------------------------------------------------------- calibration
def temperature(y_val, p_val):
    logit = np.log(np.clip(p_val.astype(np.float64), 1e-8, 1.0))
    best = None
    for T in np.exp(np.linspace(np.log(0.25), np.log(8.0), 121)):
        z = logit / T
        z -= z.max(1, keepdims=True)
        nll = -(z[np.arange(len(y_val)), y_val] - np.log(np.exp(z).sum(1))).mean()
        if best is None or nll < best[0]:
            best = (nll, T)
    return best[1]


def scale(p, T):
    z = np.log(np.clip(p.astype(np.float64), 1e-8, 1.0)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def calibration(R: Runs, exps):
    rows = []
    for e in exps:
        for s in R.seeds(e):
            d = R.dir(e, s) / "eval"
            if not ((d / "predictions_val.npz").exists() and (d / "predictions_test.npz").exists()):
                continue
            v, t = np.load(d / "predictions_val.npz"), np.load(d / "predictions_test.npz")
            T = temperature(v["y_true"], v["probs"])
            before = expected_calibration_error(t["y_true"], t["probs"].astype(np.float64), 15)["ece"]
            after = expected_calibration_error(t["y_true"], scale(t["probs"], T), 15)["ece"]
            rows.append({"model": e, "seed": s, "temperature": T, "ece_before": before, "ece_after": after})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="outputs/runs/upmc")
    p.add_argument("--out", default="outputs/analysis")
    p.add_argument("--split-dir", default="data/splits")
    p.add_argument("--review", default="data/audit/conflict_review_summary.json")
    args = p.parse_args()
    R = Runs(resolve_path(args.runs))
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# M3 analyses - UPMC Food-101 (complete test pairs)", "",
          "Mean ± std over the seeds available (percent). Produced by `scripts/analysis_m3.py`.", ""]

    # support / conflict
    fus = [e for e in ("f0_late_fusion", "f1_concat", "f2_crossmodal") if R.seeds(e)]
    sc = support_conflict(R, "ua_resnet50", "ub_distilbert", fus + [e for e in ("f2_noaux", "f2_nomoddrop") if R.seeds(e)],
                          "titles as is")
    fus_m = [e for e in ("f0_late_fusion_mask", "f1_concat_mask", "f2_crossmodal_mask") if R.seeds(e)]
    scm = support_conflict(R, "ua_resnet50", "ub_distilbert_mask", fus_m, "class names masked")
    pd.concat([sc, scm]).to_csv(out / "support_conflict_seeds.csv", index=False)
    if not sc.empty:
        md += ["## Support / conflict over seeds", "",
               "Groups from the unimodal models of the same seed: support = UA and UB right; conflict_I = only the "
               "image model right; conflict_T = only the text model right. 'Follows image' = the fused prediction "
               "equals UA's.", "", sc_table(sc, fus), ""]
        abl = [e for e in ("f2_noaux", "f2_nomoddrop") if f"{e}_acc" in sc]
        if abl:
            md += ["F2 ablations on the same groups:", "", sc_table(sc, abl), ""]
    if not scm.empty and fus_m:
        md += ["## Support / conflict with class names masked (E2: UA vs UB-mask)", "", sc_table(scm, fus_m), ""]
    plot_sc([x for x in (("titles as is: UA vs UB", sc, fus), ("class names masked: UA vs UB-mask", scm, fus_m))
             if not x[1].empty and x[2]], out)

    # removal curves
    info = load_json(resolve_path(args.split_dir) / "split.json")
    tr = pd.read_csv(resolve_path(args.split_dir) / "train.csv", dtype={"id": str})
    tr = tr[tr["has_img"].astype(bool) & tr["has_txt"].astype(bool)]
    prior_class = info["classes"].index(tr["label"].value_counts().idxmax())
    rc = removal_curves(R, prior_class, [0.0, 0.25, 0.5, 0.75, 1.0], len(info["classes"]))
    if not rc.empty:
        rc.to_csv(out / "removal_curves.csv", index=False)
        plot_removal(rc, out)
        for drop in ("text", "image"):
            d = rc[rc.removed == drop]
            fr = sorted(d.fraction.unique())
            md += [f"## Macro-F1 with the {'title' if drop == 'text' else 'image'} removed from a fraction of test pairs",
                   "", md_table(["model", *[f"{int(100 * f)} %" for f in fr]],
                                [[LABEL[e], *[pm(d[(d.model == e) & (d.fraction == f)].macro_f1) for f in fr]]
                                 for e in d.model.unique()]), ""]

    # ablation tables from test_metrics.json
    def col(e, *keys):
        return [g(R.metrics(e, s), *keys) for s in R.seeds(e)]

    abl_md = []
    if R.seeds("f2_crossmodal") and R.seeds("f2_noaux"):
        abl_md += ["## E1 - auxiliary unimodal losses (F2, λ = 0.5 vs 0)", "",
                   "With λ = 0 the unimodal heads are trained on detached features (linear probes of the encoders).", "",
                   md_table(["model", "macro-F1", "accuracy", "image-head acc.", "text-head acc.", "acc. image removed",
                             "acc. text removed"],
                            [[LABEL[e], pm(col(e, "macro_f1"), digits=2), pm(col(e, "accuracy"), digits=2),
                              pm(col(e, "aux_heads", "image", "accuracy")), pm(col(e, "aux_heads", "text", "accuracy")),
                              pm(col(e, "modality_removal", "image_removed", "accuracy")),
                              pm(col(e, "modality_removal", "text_removed", "accuracy"))]
                             for e in ("f2_crossmodal", "f2_noaux")]), ""]
    if R.seeds("f2_crossmodal") and R.seeds("f2_nomoddrop"):
        abl_md += ["## E4 - modality dropout (F2, p = 0.15 vs 0)", "",
                   md_table(["model", "macro-F1", "acc. text removed", "acc. image removed",
                             "missing-text set, title as is", "missing-text set, [MISSING] token"],
                            [[LABEL[e], pm(col(e, "macro_f1"), digits=2),
                              pm(col(e, "modality_removal", "text_removed", "accuracy")),
                              pm(col(e, "modality_removal", "image_removed", "accuracy")),
                              pm(col(e, "missing_text", "as_is", "accuracy")),
                              pm(col(e, "missing_text", "text_zero_filled", "accuracy"))]
                             for e in ("f2_crossmodal", "f2_nomoddrop")]), ""]
    pairs = [("t0_tfidf", "t0_tfidf_mask"), ("ub_distilbert", "ub_distilbert_mask"),
             ("f0_late_fusion", "f0_late_fusion_mask"), ("f1_concat", "f1_concat_mask"),
             ("f2_crossmodal", "f2_crossmodal_mask")]
    pairs = [(a, b) for a, b in pairs if R.seeds(a) and R.seeds(b)]
    if pairs:
        rows = []
        for a, b in pairs:
            fa, fb = col(a, "macro_f1"), col(b, "macro_f1")
            rows.append([LABEL[a], pm(fa, digits=2), pm(fb, digits=2),
                         f"{100 * (np.nanmean(fb) - np.nanmean(fa)):+.2f}",
                         pm(col(b, "subsets", "no_mention_true", "accuracy")),
                         pm(col(b, "modality_removal", "text_removed", "accuracy"))])
        abl_md += ["## E2 - class names masked in every title (train, val and test)", "",
                   md_table(["model", "macro-F1 as is", "macro-F1 masked", "Δ (pp)", "masked: acc. no class mention",
                             "masked: acc. text removed"], rows), ""]
        gap = []
        for fz in ("f0_late_fusion", "f1_concat", "f2_crossmodal"):
            if R.seeds(fz) and R.seeds(fz + "_mask"):
                g1 = [g(R.metrics(fz, s), "macro_f1") - g(R.metrics("ub_distilbert", s), "macro_f1") for s in R.seeds(fz)]
                g2 = [g(R.metrics(fz + "_mask", s), "macro_f1") - g(R.metrics("ub_distilbert_mask", s), "macro_f1")
                      for s in R.seeds(fz + "_mask")]
                gap.append([LABEL[fz], pm(g1, digits=2), pm(g2, digits=2)])
        if gap:
            abl_md += ["Fusion gain over the text model of the same seed (macro-F1, pp):", "",
                       md_table(["fusion", "gain over UB (as is)", "gain over UB-mask (masked)"], gap), ""]
    if abl_md:
        (out / "ablations.md").write_text("\n".join(abl_md), encoding="utf-8")
        md += abl_md

    # calibration
    cal = calibration(R, [e for e in LABEL if R.seeds(e) and not e.startswith("k0")])
    if not cal.empty:
        cal.to_csv(out / "calibration.csv", index=False)
        md += ["## Calibration: test ECE before / after temperature scaling (T fitted on validation)", "",
               md_table(["model", "T", "ECE before", "ECE after"],
                        [[LABEL[e], pm(c.temperature, 1, 2), pm(c.ece_before, digits=2), pm(c.ece_after, digits=2)]
                         for e, c in cal.groupby("model", sort=False)]), ""]

    # [FUS] attention per group (seed 42 groups)
    att_rows = []
    a, b = R.pred("ua_resnet50", 42), R.pred("ub_distilbert", 42)
    for e in ("f2_crossmodal", "f2_noaux", "f2_nomoddrop", "f2_crossmodal_mask"):
        f = R.dir(e, 42)
        if f is None or not (f / "eval" / "fus_attention_test.csv").exists() or a is None or b is None:
            continue
        at = pd.read_csv(f / "eval" / "fus_attention_test.csv", dtype={"id": str}).set_index("id").loc[a[0]]
        y, pa, pb = a[1], a[2], b[2]
        grp = np.where(pa == y, np.where(pb == y, "support", "conflict_I"), np.where(pb == y, "conflict_T", "both_wrong"))
        for gname in ("all", "support", "conflict_I", "conflict_T", "both_wrong"):
            m = np.ones(len(y), bool) if gname == "all" else grp == gname
            att_rows.append({"model": e, "group": gname, "n": int(m.sum()), "att_image": float(at.att_image[m].mean()),
                             "att_text": float(at.att_text[m].mean()), "att_fus": float(at.att_fus[m].mean())})
    if att_rows:
        att = pd.DataFrame(att_rows)
        att.to_csv(out / "attention_groups.csv", index=False)
        piv = att.pivot(index="model", columns="group", values="att_image")
        cols = [c for c in ("all", "support", "conflict_I", "conflict_T", "both_wrong") if c in piv]
        md += ["## [FUS] attention on image tokens (last fusion layer, seed 42, mean share)", "",
               md_table(["model", *cols], [[LABEL[e], *[f"{piv.loc[e, c]:.3f}" for c in cols]] for e in piv.index]), ""]

    rv = resolve_path(args.review)
    if rv.exists():
        s = load_json(rv)
        rows = []
        for gname, d in s["groups"].items():
            for v in d["verdicts"]:
                if v["count"]:
                    rows.append([gname, d["n"], v["verdict"], v["count"], f"{100 * v['rate']:.0f}",
                                 f"{100 * v['wilson95_low']:.0f}-{100 * v['wilson95_high']:.0f}"])
        md += ["## Conflict error taxonomy (manual review, seed 42 conflict items)", "",
               md_table(["group", "items", "main cause", "count", "%", "Wilson 95 % CI"], rows), ""]

    (out / "ANALYSIS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
