#!/usr/bin/env python
"""Per-item predictions of an already evaluated fusion run with one modality removed (E-mod), for runs evaluated
before evaluate.py saved them (the M2 F1 runs). Writes eval/predictions_test_drop_{image,text}.npz/.csv only;
test_metrics.json is not touched (its modality_removal block already holds the same accuracies).

  python scripts/removal_predictions.py outputs/runs/upmc/f1_concat/seed42 outputs/runs/upmc/f1_concat/seed123
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from torch.utils.data import DataLoader

from src.data.upmc import build_datasets
from src.engine.evaluate import load_run
from src.engine.loops import predict, resolve_amp_dtype
from src.engine.report import save_predictions
from src.utils.env import resolve_device
from src.utils.io import load_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--device", default="auto")
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args()
    device = resolve_device(args.device)
    for rd in map(Path, args.runs):
        out = rd / "eval"
        if all((out / f"predictions_test_drop_{d}.csv").exists() for d in ("image", "text")):
            print(f"{rd}: already done")
            continue
        cfg, model, _, _ = load_run(rd, device, overrides=args.set)
        ds, meta = build_datasets(cfg, sets=("test",))
        amp = resolve_amp_dtype(True, cfg["train"].get("amp_dtype", "auto"), device)
        loader = DataLoader(ds["test"], batch_size=int(cfg["train"].get("eval_batch_size", 256)), shuffle=False,
                            num_workers=int(cfg["data"].get("num_workers", 0)), pin_memory=device.type == "cuda")
        saved = load_json(out / "test_metrics.json")["modality_removal"]
        for drop in ("image", "text"):
            r = predict(model, loader, device, amp_dtype=amp, drop=drop)
            save_predictions(out, f"test_drop_{drop}", ds["test"].ids, r["y_true"], r["probs"], meta["classes"])
            acc = float((r["y_pred"] == r["y_true"]).mean())
            ref = saved.get(f"{drop}_removed", {}).get("accuracy", float("nan"))
            print(f"{rd}: {drop} removed acc {acc:.4f} (test_metrics.json: {ref:.4f})")
            if np.isfinite(ref) and abs(acc - ref) > 0.002:
                print("  WARNING: differs from the stored value by more than 0.2 pp")


if __name__ == "__main__":
    main()
