"""Training procedure shared by UA, UB, F1 and F2: optimizer groups, warm-up + cosine schedule,
mixed precision, validation every epoch, checkpoint selection, early stopping, resume.

Checkpoint selection (identical for all models): the epoch with the best `train.checkpoint_metric` on the
VALIDATION complete pairs (default: highest val macro-F1; first epoch reaching it). Early stopping after
`early_stopping_patience` epochs without improvement. `last.pt` (model + optimizer + scheduler + history) is
written after every epoch, so a run stopped by the Kaggle session limit or `train.max_hours` resumes.
The test split is never touched here.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from src.data.upmc import build_dataloaders
from src.engine.loops import make_grad_scaler, predict, resolve_amp_dtype, to_device, train_one_epoch
from src.engine.metrics import classification_metrics, top_k_accuracy
from src.models import build_model, count_macs, count_parameters, param_group_name
from src.utils.env import synchronize
from src.utils.io import save_json
from src.utils.seed import set_seed
from src.viz.plots import plot_training_curves

METRIC_MODES = {"val_macro_f1": "max", "val_acc": "max", "val_loss": "min"}


def build_optimizer(model: nn.Module, optim_cfg: dict):
    """One group per (image_encoder | text_encoder | head) x (decay | no decay). LR and weight decay per group
    come from optim.groups; biases and normalisation parameters get no weight decay."""
    groups_cfg = optim_cfg["groups"]
    buckets: dict[tuple, list] = {}
    for pname, p in model.named_parameters():
        g = param_group_name(pname)
        decay = not (p.ndim <= 1 or pname.endswith(".bias"))
        buckets.setdefault((g, decay), []).append(p)
    param_groups = []
    for (g, decay), ps in sorted(buckets.items()):
        gc = groups_cfg[g]
        param_groups.append({"params": ps, "lr": float(gc["lr"]),
                             "weight_decay": float(gc.get("weight_decay", 0.0)) if decay else 0.0,
                             "name": g + ("" if decay else "_no_decay")})
    name = str(optim_cfg.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"Unknown optimizer '{name}'")
    return torch.optim.AdamW(param_groups, betas=tuple(optim_cfg.get("betas", (0.9, 0.999))))


def build_scheduler(optimizer, sched_cfg: dict, total_steps: int):
    """Per-iteration LambdaLR: linear warm-up over warmup_ratio of all steps, then cosine to min_lr_ratio."""
    name = str(sched_cfg.get("name", "cosine")).lower()
    total = max(1, total_steps)
    warm = int(round(float(sched_cfg.get("warmup_ratio", 0.05)) * total))
    min_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))
    if name not in ("cosine", "constant"):
        raise ValueError(f"Unknown scheduler '{name}'")

    def lr_lambda(step: int) -> float:
        if warm > 0 and step < warm:
            return (step + 1) / warm
        if name == "constant":
            return 1.0
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_encoders_frozen(model: nn.Module, frozen: bool) -> list:
    """Stage 1 of the proposal's fine-tuning strategy: encoders frozen, new layers trained."""
    mods = [getattr(model, n) for n in ("image_encoder", "text_encoder") if hasattr(model, n)]
    for m in mods:
        for p in m.parameters():
            p.requires_grad = not frozen
    return mods if frozen else []


def sample_batch(dataset, device, n: int = 1) -> dict:
    from torch.utils.data import default_collate
    return to_device(default_collate([dataset[i] for i in range(n)]), device)


def run_training(cfg: dict, run_dir, device: torch.device, logger, resume: bool = False) -> dict:
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["seed"])
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    tcfg = cfg["train"]
    show_progress = bool(tcfg.get("show_progress", True))

    loaders, datasets, meta = build_dataloaders(cfg, seed, device, logger)
    logger.info("Dataset %s | split %s | train/val = %d/%d complete pairs | modalities image=%s text=%s",
                meta["dataset"], meta["split_file"], meta["sizes"]["train"], meta["sizes"]["val"],
                meta["use_image"], meta["use_text"])

    model = build_model(cfg["model"], meta["num_classes"], cfg.get("text")).to(device)
    params = count_parameters(model)
    macs = count_macs(model, sample_batch(datasets["val"], device))
    logger.info("Model %s | parameters %s | %s GMACs/item", cfg["model"]["name"],
                {k: f"{v:,}" for k, v in params.items()}, f"{macs / 1e9:.2f}" if macs else "n/a")
    (run_dir / "model_architecture.txt").write_text(
        f"{model}\n\nparameters: {params}\nGMACs per item: {macs / 1e9 if macs else None}\n", encoding="utf-8")

    epochs = int(tcfg["epochs"])
    max_steps = int(tcfg.get("max_steps_per_epoch") or 0)
    steps_per_epoch = min(len(loaders["train"]), max_steps) if max_steps else len(loaders["train"])
    amp_dtype = resolve_amp_dtype(bool(tcfg.get("amp", True)), tcfg.get("amp_dtype", "auto"), device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(tcfg.get("label_smoothing", 0.0)))
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg["optim"])
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), epochs * steps_per_epoch)
    scaler = make_grad_scaler(amp_dtype)
    grad_clip = float(tcfg.get("grad_clip") or 0.0)
    freeze_epochs = int(tcfg.get("freeze_encoders_epochs") or 0)
    metric_name = tcfg.get("checkpoint_metric", "val_macro_f1")
    mode = METRIC_MODES[metric_name]
    patience = int(tcfg.get("early_stopping_patience") or 0)
    max_hours = float(tcfg.get("max_hours") or 0.0)
    run_id = cfg.get("run", {}).get("run_id")
    logger.info("AMP dtype %s | optimizer groups %s", amp_dtype,
                [(g["name"], g["lr"], g["weight_decay"], sum(p.numel() for p in g["params"]))
                 for g in optimizer.param_groups])

    history = []
    best_value, best_epoch, no_improve, start_epoch = None, None, 0, 1
    train_time_total = 0.0
    if resume and (ckpt_dir / "last.pt").exists():
        last = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optimizer_state"])
        scheduler.load_state_dict(last["scheduler_state"])
        scaler.load_state_dict(last["scaler_state"])
        torch.set_rng_state(last["rng_state"])
        history = last["history"]
        best_value, best_epoch, no_improve = last["best_value"], last["best_epoch"], last["no_improve"]
        train_time_total = sum(r["epoch_train_time_s"] for r in history)
        start_epoch = last["epoch"] + 1
        logger.info("Resumed from last.pt at epoch %d (best epoch %s)", last["epoch"], best_epoch)
        if patience and no_improve >= patience:
            start_epoch = epochs + 1
    wall_start = time.perf_counter()
    stopped_by_time = False

    for epoch in range(start_epoch, epochs + 1):
        frozen = set_encoders_frozen(model, epoch <= freeze_epochs) if freeze_epochs else []
        lr_start = {g["name"]: g["lr"] for g in optimizer.param_groups}
        synchronize(device)
        t0 = time.perf_counter()
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler, scaler, device, amp_dtype,
                             grad_clip, desc=f"epoch {epoch}/{epochs}", show_progress=show_progress,
                             max_steps=max_steps, frozen_modules=frozen)
        synchronize(device)
        t_train = time.perf_counter() - t0
        train_time_total += t_train

        t1 = time.perf_counter()
        va = predict(model, loaders["val"], device, eval_criterion, amp_dtype=amp_dtype)
        t_val = time.perf_counter() - t1
        va_m = classification_metrics(va["y_true"], va["y_pred"], meta["num_classes"])
        row = {"epoch": epoch, **{f"lr_{k}": v for k, v in lr_start.items() if not k.endswith("_no_decay")},
               "encoders_frozen": bool(frozen),
               "train_loss": tr["loss"], "train_acc": tr["acc"],
               "val_loss": va["loss"], "val_acc": va_m["accuracy"], "val_macro_f1": va_m["macro_f1"],
               "val_top5": top_k_accuracy(va["y_true"], va["probs"], 5),
               **{f"val_aux_{k}_acc": float((pa.argmax(1) == va["y_true"]).mean()) for k, pa in va["aux"].items()},
               "epoch_train_time_s": t_train, "epoch_val_time_s": t_val,
               "train_items_per_s": tr["n"] / t_train if t_train > 0 else float("nan")}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        value = row[metric_name]
        improved = best_value is None or (value > best_value + 1e-6 if mode == "max" else value < best_value - 1e-6)
        if improved:
            best_value, best_epoch, no_improve = value, epoch, 0
        else:
            no_improve += 1
        state = {"model_state": model.state_dict(), "epoch": epoch,
                 "val_metrics": {k: row[k] for k in ("val_loss", "val_acc", "val_macro_f1", "val_top5")},
                 "checkpoint_metric": metric_name, "run_id": run_id, "model_cfg": cfg["model"],
                 "meta": {k: meta[k] for k in ("dataset", "num_classes", "image_size", "classes")}}
        if improved:
            torch.save(state, ckpt_dir / "best.pt")
        torch.save({**state, "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(), "rng_state": torch.get_rng_state(), "history": history,
                    "best_value": best_value, "best_epoch": best_epoch, "no_improve": no_improve},
                   ckpt_dir / "last.pt")
        logger.info("epoch %2d | train loss %.4f acc %.4f | val loss %.4f acc %.4f macroF1 %.4f top5 %.4f | "
                    "%.0fs (%.0f items/s)%s%s", epoch, tr["loss"], tr["acc"], va["loss"], va_m["accuracy"],
                    va_m["macro_f1"], row["val_top5"], t_train, row["train_items_per_s"],
                    " [encoders frozen]" if frozen else "", "  *best*" if improved else "")

        if patience and no_improve >= patience:
            logger.info("Early stopping: no improvement in %s for %d epochs.", metric_name, patience)
            break
        if max_hours and (time.perf_counter() - wall_start) / 3600 > max_hours and epoch < epochs:
            logger.warning("train.max_hours=%.2f reached after epoch %d: stopping, resume with --resume.",
                           max_hours, epoch)
            stopped_by_time = True
            break

    best_row = history[best_epoch - 1]
    finished = not stopped_by_time
    summary = {
        "run_id": run_id, "dataset": meta["dataset"], "experiment": cfg["experiment"]["name"],
        "experiment_id": cfg["experiment"].get("id"), "model": cfg["model"]["name"], "seed": seed,
        "device": str(device), "amp_dtype": str(amp_dtype), "finished": finished,
        "epochs_max": epochs, "epochs_run": len(history), "early_stopped": finished and len(history) < epochs,
        "max_steps_per_epoch": max_steps, "checkpoint_metric": metric_name, "best_epoch": best_epoch,
        "best_val_loss": best_row["val_loss"], "best_val_acc": best_row["val_acc"],
        "best_val_macro_f1": best_row["val_macro_f1"], "best_val_top5": best_row["val_top5"],
        "train_time_s": train_time_total, "train_time_per_epoch_s": train_time_total / len(history),
        "train_items_per_s_mean": float(pd.DataFrame(history)["train_items_per_s"].mean()),
        "parameters": params, "gmacs_per_item": macs / 1e9 if macs else None,
        "train_size": meta["sizes"]["train"], "val_size": meta["sizes"]["val"],
        "split_file": meta["split_file"], "split_fingerprints": meta["split_fingerprints"],
        "set_fingerprints": meta["set_fingerprints"],
    }
    save_json(summary, run_dir / "train_summary.json")
    plot_training_curves(pd.DataFrame(history), run_dir / "curves.png",
                         title=f"{meta['dataset']} / {cfg['experiment']['name']} / seed {seed}", best_epoch=best_epoch)
    logger.info("Best epoch %d | val macroF1 %.4f | val acc %.4f | train time %.1f min%s", best_epoch,
                best_row["val_macro_f1"], best_row["val_acc"], train_time_total / 60,
                "" if finished else " | NOT FINISHED (time limit)")
    return summary
