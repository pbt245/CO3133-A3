"""Metrics from predicted probabilities, shared by trained models (evaluate.py), F0 late fusion and the K0
keyword rule, so every row of the comparison table is computed the same way.

Test subsets reported next to the full complete-pair test set (proposal 3.3, 5.5):
  dedup            items whose image is not linked (md5 / pHash) to an official-train image and not in a
                   cross-label image cluster
  mention_true     the title mentions the true class (lexicon of src/data/text.py)
  no_mention_true  the title does not mention the true class
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.text import ClassLexicon
from src.engine.metrics import (bootstrap_ci, classification_metrics, expected_calibration_error, top_confusions,
                                top_k_accuracy)
from src.viz.plots import plot_confusion_matrix


def item_flags(df: pd.DataFrame, classes) -> pd.DataFrame:
    """Per-item boolean flags used to define test subsets. df needs label, title_clean, test_dup, cross_label."""
    lex = ClassLexicon(classes)
    mentioned = [lex.mentioned_classes(t) for t in df["title_clean"]]
    out = pd.DataFrame({"id": df["id"].to_numpy()})
    out["mention_true"] = [lab in m for lab, m in zip(df["label"], mentioned)]
    out["mention_other"] = [bool(m - {lab}) for lab, m in zip(df["label"], mentioned)]
    out["dedup"] = ~(df["test_dup"].astype(bool).to_numpy() | df["cross_label"].astype(bool).to_numpy())
    return out


def subset_metrics(y_true, probs, mask, k) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"n_samples": 0}
    yt, pr = np.asarray(y_true)[mask], probs[mask]
    m = classification_metrics(yt, pr.argmax(1), k)
    return {"n_samples": int(mask.sum()), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "top5_accuracy": top_k_accuracy(yt, pr, 5)}


def save_predictions(out_dir: Path, name: str, ids, y_true, probs, classes):
    """predictions_<name>.npz (float16 probabilities, for F0 / McNemar / support-conflict) + a readable CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"predictions_{name}.npz", ids=np.asarray(ids), y_true=np.asarray(y_true),
                        probs=probs.astype(np.float16))
    y_pred = probs.argmax(1)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    pd.DataFrame({"id": ids, "y_true": y_true, "y_pred": y_pred,
                  "true_name": [classes[i] for i in y_true], "pred_name": [classes[i] for i in y_pred],
                  "confidence": probs.max(1), "prob_true_class": probs[np.arange(len(y_true)), y_true],
                  "correct": y_pred == np.asarray(y_true),
                  "top5": [" ".join(map(str, r)) for r in top5]}).to_csv(out_dir / f"predictions_{name}.csv", index=False)


def test_report(y_true, probs, ids, classes, flags: pd.DataFrame, out_dir: Path, seed: int, ecfg: dict,
                title: str) -> dict:
    """Full metric set on the complete-pair test set; writes per-class report, confusion matrix, top confusions."""
    k = len(classes)
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(1)
    m = classification_metrics(y_true, y_pred, k)
    ece = expected_calibration_error(y_true, probs, n_bins=int(ecfg.get("ece_bins", 15)))
    ci = bootstrap_ci(y_true, y_pred, k, n_boot=int(ecfg.get("bootstrap_samples", 1000)), seed=seed)
    flags = flags.set_index("id").loc[list(ids)]
    report = pd.DataFrame({"class": classes, "precision": m["per_class"]["precision"],
                           "recall": m["per_class"]["recall"], "f1": m["per_class"]["f1"],
                           "support": m["per_class"]["support"]})
    report.to_csv(out_dir / "classification_report.csv", index=False)
    cm = m["confusion_matrix"]
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / "confusion_matrix.csv")
    plot_confusion_matrix(cm, classes, out_dir / "confusion_matrix_normalized.png", normalize=True,
                          title=title + " (row-normalized)", annotate=False)
    conf = top_confusions(cm, classes, top_k=25)
    conf.to_csv(out_dir / "top_confusions.csv", index=False)
    errors = y_pred != y_true
    c = probs.max(1)
    return {
        "n_samples": int(len(y_true)), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"], "top5_accuracy": top_k_accuracy(y_true, probs, 5), "ece": ece["ece"],
        "calibration_bins": ece["bins"], "bootstrap_ci": ci,
        "subsets": {name: subset_metrics(y_true, probs, flags[col].to_numpy() if col else ~flags["mention_true"].to_numpy(), k)
                    for name, col in (("dedup", "dedup"), ("mention_true", "mention_true"),
                                      ("no_mention_true", None), ("mention_other", "mention_other"))},
        "per_class": report.to_dict("records"),
        "error_summary": {"n_errors": int(errors.sum()), "error_rate": float(errors.mean()),
                          "mean_confidence_correct": float(c[~errors].mean()) if (~errors).any() else None,
                          "mean_confidence_errors": float(c[errors].mean()) if errors.any() else None,
                          "high_confidence_errors_conf_gt_0.9": int((errors & (c > 0.9)).sum()),
                          "top_confusions": conf.head(10).to_dict("records")},
    }
