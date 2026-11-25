#!/usr/bin/env python
"""K0 - keyword rule (no learning, CPU, seconds): predict the most-mentioned class name in the title
(lexicon of src/data/text.py; ties -> first mention). Titles without any class name get the train class prior,
so the prediction is the most frequent train class. It is the leakage floor for every text model.

  python scripts/keyword_baseline.py            # -> outputs/runs/upmc/k0_keyword/seed42/eval/

Written in the same format as a trained run (eval/test_metrics.json + predictions), so compare.py treats it
like any other model. Deterministic: one "seed".
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.text import ClassLexicon
from src.data.upmc import build_datasets
from src.engine.report import item_flags, save_predictions, subset_metrics, test_report
from src.utils.config import apply_overrides, load_config, save_config
from src.utils.env import git_info
from src.utils.io import resolve_path, save_json


def rule_probs(lex, titles, classes, prior):
    idx = {c: i for i, c in enumerate(classes)}
    probs = np.tile(prior, (len(titles), 1))
    hit = np.zeros(len(titles), bool)
    for n, t in enumerate(titles):
        c = lex.keyword_rule(t)
        if c is not None:
            probs[n] = 0.001 * prior
            probs[n, idx[c]] += 0.999
            hit[n] = True
    return probs, hit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/upmc/ub_distilbert.yaml", help="only data.* is used")
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args()
    cfg = apply_overrides(load_config(resolve_path(args.config)), args.set)
    cfg["model"] = {"name": "keyword_rule"}
    cfg["experiment"] = {"name": "k0_keyword", "id": "K0", "group": "main", "mandatory": False}
    cfg["seed"] = 42
    ds, meta = build_datasets(cfg, sets=("train", "val", "test", "test_missing_text"), need_image=False, need_text=False)
    classes, k = meta["classes"], meta["num_classes"]
    lex = ClassLexicon(classes)
    prior = np.bincount(ds["train"].labels.numpy(), minlength=k).astype(float)
    prior /= prior.sum()

    run_dir = resolve_path(cfg["output"]["root"]) / "upmc" / "k0_keyword" / "seed42"
    out = run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    res = {}
    for name in ("val", "test", "test_missing_text"):
        d = ds[name]
        probs, hit = rule_probs(lex, d.titles, classes, prior)
        y = d.labels.numpy()
        save_predictions(out, name, d.ids, y, probs, classes)
        res[name] = (y, probs, hit)
    y, probs, hit = res["test"]
    m = test_report(y, probs, ds["test"].ids, classes, item_flags(ds["test"].df, classes), out, 42, cfg["eval"],
                    "upmc / k0_keyword")
    yv, pv, _ = res["val"]
    ym, pm, hm = res["test_missing_text"]
    metrics = {"experiment": "k0_keyword", "experiment_id": "K0", "model": "keyword_rule", "seed": 42,
               "split": "test (complete pairs)", **m,
               "coverage_test": float(hit.mean()), "accuracy_when_mentioned": float((probs[hit].argmax(1) == y[hit]).mean()),
               "val": subset_metrics(yv, pv, np.ones(len(yv), bool), k),
               "missing_text": {"as_is": subset_metrics(ym, pm, np.ones(len(ym), bool), k)} if len(ym) else {},
               "modality_removal": {}, "parameters": {"total": 0, "trainable": 0},
               "split_fingerprints": meta["split_fingerprints"], "set_fingerprints": meta["set_fingerprints"],
               "git_at_eval": git_info()}
    save_json(metrics, out / "test_metrics.json")
    save_json({"experiment": "k0_keyword", "model": "keyword_rule", "seed": 42, "finished": True, "train_time_s": 0.0,
               "parameters": {"total": 0}, "best_val_macro_f1": metrics["val"]["macro_f1"]}, run_dir / "train_summary.json")
    print(f"K0 test acc {m['accuracy']:.4f} macroF1 {m['macro_f1']:.4f} | coverage {hit.mean():.3f} | "
          f"acc when a class is mentioned {metrics['accuracy_when_mentioned']:.4f} -> {out}")


if __name__ == "__main__":
    main()
