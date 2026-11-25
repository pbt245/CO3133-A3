"""Evaluate the selected checkpoint of a run and write every artefact the report needs.

eval/
  test_metrics.json                complete-pair test: accuracy, macro-F1, top-5, ECE, bootstrap CI, subsets
                                   (dedup, class-name mention), natural missing-text set, modality removal
                                   (fusion only), inference cost, checkpoint sha256, git
  predictions_{val,test,test_missing_text}.npz/.csv
  classification_report.csv, confusion_matrix.csv, confusion_matrix_normalized.png, top_confusions.csv
  examples_*.png                   image grids (models that see the image)
Validation predictions are saved for F0 (late-fusion weight tuning on val, scripts/late_fusion.py).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.upmc import build_datasets
from src.engine.inference import measure_inference
from src.engine.loops import predict, resolve_amp_dtype
from src.engine.report import item_flags, save_predictions, subset_metrics, test_report
from src.models import build_model, count_parameters
from src.utils.config import apply_overrides, load_config
from src.utils.env import git_info, synchronize
from src.utils.io import save_json, sha256_file
from src.utils.seed import set_seed
from src.viz.plots import plot_image_grid


def load_run(run_dir, device, checkpoint: str = "best.pt", overrides=None):
    run_dir = Path(run_dir)
    cfg = apply_overrides(load_config(run_dir / "config.yaml"), overrides)
    model_cfg = dict(cfg["model"], pretrained=False)          # weights come from the checkpoint
    ckpt_path = run_dir / "checkpoints" / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(model_cfg, ckpt["meta"]["num_classes"], cfg.get("text"))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return cfg, model, ckpt, ckpt_path


def evaluate_run(run_dir, device, logger=None, overrides=None) -> dict:
    log = logger.info if logger is not None else print
    run_dir = Path(run_dir)
    cfg, model, ckpt, ckpt_path = load_run(run_dir, device, overrides=overrides)
    seed = int(cfg["seed"])
    set_seed(seed)
    ecfg = cfg.get("eval", {})
    datasets, meta = build_datasets(cfg, sets=("val", "test", "test_missing_text"))
    classes, k = meta["classes"], meta["num_classes"]
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    amp_dtype = resolve_amp_dtype(bool(ecfg.get("amp", True)), cfg["train"].get("amp_dtype", "auto"), device)
    nw = int(cfg["data"].get("num_workers", 0))

    def run(name, drop=None):
        ds = datasets[name]
        loader = DataLoader(ds, batch_size=int(cfg["train"].get("eval_batch_size", 256)), shuffle=False,
                            num_workers=nw, pin_memory=device.type == "cuda")
        res = predict(model, loader, device, nn.CrossEntropyLoss(), amp_dtype=amp_dtype, drop=drop)
        assert (res["index"] == np.arange(len(ds))).all()
        return res

    title = f"{meta['dataset']} / {cfg['experiment']['name']} / seed {seed}"
    va = run("val")
    save_predictions(out_dir, "val", datasets["val"].ids, va["y_true"], va["probs"], classes)
    synchronize(device)
    t0 = time.perf_counter()
    te = run("test")
    synchronize(device)
    eval_pass_time = time.perf_counter() - t0
    test_ds = datasets["test"]
    save_predictions(out_dir, "test", test_ds.ids, te["y_true"], te["probs"], classes)
    flags = item_flags(test_ds.df, classes)
    metrics = test_report(te["y_true"], te["probs"], test_ds.ids, classes, flags, out_dir, seed, ecfg, title)

    # natural missing-text set: titles that fail the validity criterion (proposal 2.3, 5.4)
    miss = {}
    if len(datasets["test_missing_text"]):
        mt = run("test_missing_text")
        save_predictions(out_dir, "test_missing_text", datasets["test_missing_text"].ids, mt["y_true"], mt["probs"], classes)
        all_items = np.ones(len(mt["y_true"]), bool)
        miss["as_is"] = subset_metrics(mt["y_true"], mt["probs"], all_items, k)
        if getattr(model, "is_fusion", False):
            mz = run("test_missing_text", drop="text")
            miss["text_zero_filled"] = subset_metrics(mz["y_true"], mz["probs"], all_items, k)
    # test-time modality removal on complete test pairs (E-mod; fusion models only)
    removal = {}
    if getattr(model, "is_fusion", False):
        for drop in ("image", "text"):
            r = run("test", drop=drop)
            removal[f"{drop}_removed"] = subset_metrics(r["y_true"], r["probs"], np.ones(len(r["y_true"]), bool), k)

    # example grids (models that see the image)
    if getattr(model, "uses_image", False):
        y_true, probs = te["y_true"], te["probs"]
        y_pred, conf = probs.argmax(1), probs.max(1)
        correct = y_pred == y_true
        rng = np.random.default_rng(seed)
        n_ex = int(ecfg.get("n_examples", 20))

        def grid(indices, filename, heading):
            if len(indices) == 0:
                return
            titles = [f"T: {classes[y_true[i]]}\nP: {classes[y_pred[i]]} ({conf[i]:.2f})\n"
                      f"\"{test_ds.titles[i][:38]}\"" for i in indices]
            plot_image_grid([test_ds.display_image(int(i)) for i in indices], titles, out_dir / filename, ncols=5,
                            suptitle=f"{title}: {heading}", title_colors=["tab:green" if correct[i] else "tab:red"
                                                                          for i in indices], cell=2.6, fontsize=6)

        wrong = np.flatnonzero(~correct)
        if len(wrong):
            grid(np.sort(rng.choice(wrong, size=min(n_ex, len(wrong)), replace=False)), "examples_incorrect_random.png",
                 "random errors")
            grid(wrong[np.argsort(-conf[wrong])[:n_ex]], "examples_incorrect_high_confidence.png", "most confident errors")
        right = np.flatnonzero(correct)
        if len(right):
            grid(np.sort(rng.choice(right, size=min(n_ex, len(right)), replace=False)), "examples_correct_random.png",
                 "random correct predictions")

    icfg = ecfg.get("inference", {})
    inference = measure_inference(model, test_ds, device, batch_size=int(icfg.get("batch_size", 128)),
                                  warmup_iters=int(icfg.get("warmup_iters", 10)),
                                  timed_iters=int(icfg.get("timed_iters", 30)),
                                  latency_samples=int(icfg.get("latency_samples", 200)), amp_dtype=amp_dtype)
    metrics = {
        "run_id": cfg.get("run", {}).get("run_id"), "dataset": meta["dataset"],
        "experiment": cfg["experiment"]["name"], "experiment_id": cfg["experiment"].get("id"),
        "model": cfg["model"]["name"], "seed": seed, "split": "test (complete pairs)",
        "loss": te["loss"], **metrics,
        "val": subset_metrics(va["y_true"], va["probs"], np.ones(len(va["y_true"]), bool), k),
        "missing_text": miss, "modality_removal": removal,
        "parameters": count_parameters(model), "inference": inference, "eval_amp_dtype": str(amp_dtype),
        "eval_pass_time_s": eval_pass_time,
        "checkpoint": {"file": str(ckpt_path.relative_to(run_dir)), "epoch": ckpt.get("epoch"),
                       "selection_metric": ckpt.get("checkpoint_metric"), "val_metrics": ckpt.get("val_metrics"),
                       "sha256": sha256_file(ckpt_path)},
        "split_file": meta["split_file"], "split_fingerprints": meta["split_fingerprints"],
        "set_fingerprints": meta["set_fingerprints"], "git_at_eval": git_info(),
    }
    save_json(metrics, out_dir / "test_metrics.json")
    ci = metrics["bootstrap_ci"]
    log(f"[test] acc {metrics['accuracy']:.4f} (95% CI {ci['accuracy'][0]:.4f}-{ci['accuracy'][1]:.4f}) | "
        f"macroF1 {metrics['macro_f1']:.4f} (95% CI {ci['macro_f1'][0]:.4f}-{ci['macro_f1'][1]:.4f}) | "
        f"top5 {metrics['top5_accuracy']:.4f} | ECE {metrics['ece']:.4f} | "
        f"dedup macroF1 {metrics['subsets']['dedup'].get('macro_f1', float('nan')):.4f} | "
        f"{inference['throughput_items_per_s']:.0f} items/s | bs1 {inference['latency_ms_bs1_mean']:.2f} ms")
    if removal:
        log(f"[test] modality removal: " + ", ".join(f"{k2} acc {v['accuracy']:.4f}" for k2, v in removal.items()))
    return metrics
