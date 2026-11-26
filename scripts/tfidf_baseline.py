#!/usr/bin/env python
"""T0 - TF-IDF + logistic regression on titles [OPTIONAL] (CPU only, a few minutes; proposal section 5).

Features: word 1-2 grams + character 3-5 grams (within word boundaries) of the cleaned, lower-cased title,
sublinear tf, min_df 2, fitted on TRAIN only. Classifier: multinomial logistic regression; C is chosen on the
validation complete pairs from a small grid (highest val macro-F1), then evaluated once on test.

  python scripts/tfidf_baseline.py                 # -> outputs/runs/upmc/t0_tfidf/seed42/
  python scripts/tfidf_baseline.py --mask          # E2 variant on masked titles -> t0_tfidf_mask/

Written in the same format as a trained run (eval/test_metrics.json + predictions), so compare.py treats it
like any other model. Deterministic (lbfgs): one "seed".
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.upmc import build_datasets
from src.engine.metrics import classification_metrics
from src.engine.report import item_flags, save_predictions, subset_metrics, test_report
from src.utils.config import apply_overrides, load_config, save_config
from src.utils.env import git_info
from src.utils.io import resolve_path, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/upmc/ub_distilbert.yaml", help="only data.* / text.* are used")
    p.add_argument("--mask", action="store_true", help="E2: class-name mentions replaced by [MASK]")
    p.add_argument("--grid", nargs="+", type=float, default=[1.0, 4.0, 16.0], help="values of C tried on val")
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args()
    cfg = apply_overrides(load_config(resolve_path(args.config)), args.set)
    name = "t0_tfidf_mask" if args.mask else "t0_tfidf"
    cfg["text"]["mask_class_names"] = bool(args.mask)
    cfg["model"] = {"name": "tfidf_logreg", "word_ngrams": [1, 2], "char_ngrams": [3, 5], "min_df": 2,
                    "C_grid": args.grid, "max_iter": args.max_iter}
    cfg["experiment"] = {"name": name, "id": "E2-T0" if args.mask else "T0", "group": "main", "mandatory": False}
    cfg["seed"] = 42
    ds, meta = build_datasets(cfg, sets=("train", "val", "test", "test_missing_text"), need_image=False,
                              need_text=False)
    classes, k = meta["classes"], meta["num_classes"]
    t0 = time.perf_counter()
    word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    tr = ds["train"].titles
    Xtr = hstack([word.fit_transform(tr), char.fit_transform(tr)]).tocsr()
    feats = lambda t: hstack([word.transform(t), char.transform(t)]).tocsr()
    X = {n: feats(ds[n].titles) for n in ("val", "test", "test_missing_text")}
    y = {n: ds[n].labels.numpy() for n in ds}
    print(f"{name}: {Xtr.shape[0]:,} train titles x {Xtr.shape[1]:,} features", flush=True)

    best = None
    grid = {}
    for C in args.grid:
        clf = LogisticRegression(C=C, max_iter=args.max_iter)
        clf.fit(Xtr, y["train"])
        f1 = classification_metrics(y["val"], clf.predict(X["val"]), k)["macro_f1"]
        grid[str(C)] = f1
        print(f"  C={C:g}: val macro-F1 {f1:.4f}", flush=True)
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    _, C, clf = best
    fit_time = time.perf_counter() - t0

    def probs(n):          # columns of predict_proba follow clf.classes_ (all 101 classes occur in train)
        out = np.zeros((X[n].shape[0], k))
        if X[n].shape[0]:
            out[:, clf.classes_] = clf.predict_proba(X[n])
        return out

    run_dir = resolve_path(cfg["output"]["root"]) / "upmc" / name / "seed42"
    out = run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    P = {n: probs(n) for n in X}
    for n in X:
        save_predictions(out, n, ds[n].ids, y[n], P[n], classes)
    t1 = time.perf_counter()
    _ = clf.predict_proba(X["test"])
    infer_s = time.perf_counter() - t1
    m = test_report(y["test"], P["test"], ds["test"].ids, classes, item_flags(ds["test"].df, classes), out, 42,
                    cfg["eval"], f"upmc / {name}")
    ones = lambda a: np.ones(len(a), bool)
    n_params = int(clf.coef_.size + clf.intercept_.size)
    metrics = {"experiment": name, "experiment_id": cfg["experiment"]["id"], "model": "tfidf_logreg", "seed": 42,
               "split": "test (complete pairs)", **m, "C": C, "val_grid": grid, "n_features": int(Xtr.shape[1]),
               "val": subset_metrics(y["val"], P["val"], ones(y["val"]), k),
               "missing_text": ({"as_is": subset_metrics(y["test_missing_text"], P["test_missing_text"],
                                                         ones(y["test_missing_text"]), k)}
                                if len(y["test_missing_text"]) else {}),
               "modality_removal": {}, "parameters": {"total": n_params, "trainable": n_params},
               "inference": {"device": "cpu", "throughput_items_per_s": len(y["test"]) / infer_s,
                             "note": "features precomputed; predict_proba only"},
               "masking": meta.get("masking"),
               "split_fingerprints": meta["split_fingerprints"], "set_fingerprints": meta["set_fingerprints"],
               "git_at_eval": git_info()}
    save_json(metrics, out / "test_metrics.json")
    save_json({"experiment": name, "model": "tfidf_logreg", "seed": 42, "finished": True, "train_time_s": fit_time,
               "parameters": metrics["parameters"], "best_val_macro_f1": grid[str(C)], "C": C,
               "device": "cpu"}, run_dir / "train_summary.json")
    print(f"{name}: C={C:g} | test acc {m['accuracy']:.4f} macroF1 {m['macro_f1']:.4f} | fit {fit_time / 60:.1f} min "
          f"-> {out}")


if __name__ == "__main__":
    main()
