#!/usr/bin/env python
"""F0 - late fusion (no training): p = w * p_UA + (1 - w) * p_UB, with w chosen on the VALIDATION complete pairs
(grid 0, 0.05, ..., 1, highest val macro-F1). One F0 per seed, built from UA and UB of the same seed.
Missing text: F0 falls back to UA (w = 1) on the natural missing-text set.

  python scripts/late_fusion.py --seeds 42 123 2026     # -> outputs/runs/upmc/f0_late_fusion/seed<s>/
  # E2: UA + the text model trained on masked titles
  python scripts/late_fusion.py --text ub_distilbert_mask --name f0_late_fusion_mask --seeds 42 123 2026
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.upmc import build_datasets
from src.engine.metrics import classification_metrics
from src.engine.report import item_flags, save_predictions, subset_metrics, test_report
from src.utils.config import load_config, save_config
from src.utils.env import git_info
from src.utils.io import load_json, resolve_path, save_json


def load_pred(run_dir: Path, name: str):
    z = np.load(run_dir / "eval" / f"predictions_{name}.npz", allow_pickle=False)
    return z["ids"], z["y_true"], z["probs"].astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="outputs/runs/upmc")
    p.add_argument("--image", default="ua_resnet50")
    p.add_argument("--text", default="ub_distilbert")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--name", default="f0_late_fusion", help="experiment folder of the result")
    args = p.parse_args()
    runs = resolve_path(args.runs)
    grid = np.round(np.arange(0, 1 + 1e-9, args.step), 4)
    ds = meta = None
    for seed in args.seeds:
        ra, rb = runs / args.image / f"seed{seed}", runs / args.text / f"seed{seed}"
        if not ((ra / "eval/test_metrics.json").exists() and (rb / "eval/test_metrics.json").exists()):
            print(f"seed {seed}: UA or UB not evaluated yet, skipping")
            continue
        if ds is None:
            cfg = load_config(rb / "config.yaml")
            ds, meta = build_datasets(cfg, sets=("test",), need_image=False, need_text=False)
        classes, k = meta["classes"], meta["num_classes"]
        P = {}
        for name in ("val", "test", "test_missing_text"):
            ia, ya, pa = load_pred(ra, name)
            ib, yb, pb = load_pred(rb, name)
            assert (ia == ib).all() and (ya == yb).all(), f"{name}: UA and UB predictions are not aligned"
            P[name] = (ia, ya, pa, pb)
        _, yv, pav, pbv = P["val"]
        scores = [classification_metrics(yv, (w * pav + (1 - w) * pbv).argmax(1), k)["macro_f1"] for w in grid]
        w = float(grid[int(np.argmax(scores))])
        run_dir = runs / args.name / f"seed{seed}"
        eid = "F0" if args.name == "f0_late_fusion" else "E2-F0" if args.name.endswith("_mask") else args.name
        out = run_dir / "eval"
        out.mkdir(parents=True, exist_ok=True)
        it, yt, pat, pbt = P["test"]
        pt = w * pat + (1 - w) * pbt
        assert list(it) == ds["test"].ids
        save_predictions(out, "test", it, yt, pt, classes)
        save_predictions(out, "val", P["val"][0], yv, w * pav + (1 - w) * pbv, classes)
        m = test_report(yt, pt, it, classes, item_flags(ds["test"].df, classes), out, seed, cfg["eval"],
                        f"upmc / {args.name} / seed {seed}")
        im_, ym, pam, pbm = P["test_missing_text"]
        save_predictions(out, "test_missing_text", im_, ym, pam, classes)
        ones = lambda y: np.ones(len(y), bool)
        sa, sb = load_json(ra / "train_summary.json"), load_json(rb / "train_summary.json")
        metrics = {"experiment": args.name, "experiment_id": eid, "model": "late_fusion", "seed": seed,
                   "split": "test (complete pairs)", **m, "weight_image": w,
                   "val_grid": {str(g): s for g, s in zip(grid, scores)},
                   "val": subset_metrics(yv, w * pav + (1 - w) * pbv, ones(yv), k),
                   "missing_text": {"as_is": subset_metrics(ym, w * pam + (1 - w) * pbm, ones(ym), k),
                                    "text_zero_filled": subset_metrics(ym, pam, ones(ym), k)} if len(ym) else {},
                   "modality_removal": {"image_removed": subset_metrics(yt, pbt, ones(yt), k),
                                        "text_removed": subset_metrics(yt, pat, ones(yt), k)},
                   "parameters": {"total": sa["parameters"]["total"] + sb["parameters"]["total"], "trainable": 0},
                   "components": {"image": str(ra.relative_to(runs)), "text": str(rb.relative_to(runs))},
                   "split_fingerprints": meta["split_fingerprints"], "git_at_eval": git_info()}
        save_json(metrics, out / "test_metrics.json")
        save_config({"seed": seed, "experiment": {"name": args.name, "id": eid},
                     "model": {"name": "late_fusion", "image_run": str(ra), "text_run": str(rb), "weight_image": w},
                     "data": cfg["data"]}, run_dir / "config.yaml")
        save_json({"experiment": args.name, "model": "late_fusion", "seed": seed, "finished": True,
                   "train_time_s": sa["train_time_s"] + sb["train_time_s"], "parameters": metrics["parameters"],
                   "best_val_macro_f1": float(max(scores)), "weight_image": w}, run_dir / "train_summary.json")
        print(f"seed {seed}: w_image {w:.2f} (val macroF1 {max(scores):.4f}) | test acc {m['accuracy']:.4f} "
              f"macroF1 {m['macro_f1']:.4f} -> {out}")


if __name__ == "__main__":
    main()
