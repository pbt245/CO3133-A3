"""Train / predict loops on the batch dicts of src/data/upmc.py (shared by every model)."""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.data.upmc import trim_tokens


def resolve_amp_dtype(enabled: bool, name: str, device: torch.device):
    """bf16 where supported (RTX PRO 6000 / Ampere+), else fp16 with gradient scaling (T4). None = fp32."""
    if not enabled or device.type != "cuda":
        return None
    name = str(name or "auto").lower()
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}[name]


def autocast_ctx(device: torch.device, dtype):
    if dtype is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def make_grad_scaler(dtype):
    return torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)


def split_output(out):
    """Model output -> (fused logits, {aux name: logits}, aux loss weight). UA / UB / F1 return a tensor;
    F2 returns {"logits", "aux_image", "aux_text", "aux_weight"}."""
    if isinstance(out, dict):
        aux = {k[4:]: v for k, v in out.items() if k.startswith("aux_") and torch.is_tensor(v)}
        return out["logits"], aux, float(out.get("aux_weight", 0.0))
    return out, {}, 0.0


def to_device(batch: dict, device, drop: str | None = None) -> dict:
    """Move a batch to the device. drop='image' / 'text' marks that modality as missing for every item
    (test-time modality removal; fusion models zero-fill it). drop='keep_text' marks the text as present even
    where the title failed the validity criterion (feed the uninformative title as it is)."""
    batch = trim_tokens(dict(batch))
    out = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if drop == "image":
        out["has_img"] = torch.zeros_like(out["has_img"])
    elif drop == "text":
        out["has_txt"] = torch.zeros_like(out["has_txt"])
    elif drop == "keep_text":
        out["has_txt"] = torch.ones_like(out["has_txt"])
    return out


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, amp_dtype=None,
                    grad_clip: float = 0.0, desc: str = "train", show_progress: bool = True, max_steps: int = 0,
                    frozen_modules=()) -> dict:
    """One pass over the training loader; scheduler stepped per iteration; loss averaged over samples.
    frozen_modules stay in eval mode (BatchNorm statistics and dropout of frozen encoders unchanged)."""
    model.train()
    for m in frozen_modules:
        m.eval()
    loss_sum, correct, n = 0.0, 0, 0
    total = min(len(loader), max_steps) if max_steps else len(loader)
    for step, batch in enumerate(tqdm(loader, desc=desc, leave=False, disable=not show_progress, total=total)):
        if max_steps and step >= max_steps:
            break
        batch = to_device(batch, device)
        y = batch["label"]
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp_dtype):
            logits, aux, aux_w = split_output(model(batch))
            loss = criterion(logits.float(), y)
            if aux and aux_w > 0:                    # F2: L = CE(fused) + w * sum of the unimodal CE terms
                loss = loss + aux_w * sum(criterion(a.float(), y) for a in aux.values())
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite training loss ({loss_value}); lower the learning rate.")
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        bs = y.size(0)
        loss_sum += loss_value * bs
        correct += (logits.detach().argmax(dim=1) == y).sum().item()
        n += bs
    return {"loss": loss_sum / max(n, 1), "acc": correct / max(n, 1), "n": n}


@torch.no_grad()
def predict(model, loader, device, criterion=None, amp_dtype=None, drop: str | None = None,
            desc: str = "eval", show_progress: bool = False) -> dict:
    """Inference over a non-shuffled loader. Logits are cast to fp32 before softmax. F2 also returns the
    probabilities of its auxiliary unimodal heads (`aux`: {"image": probs, "text": probs})."""
    model.eval()
    all_logits, all_y, all_idx, all_aux = [], [], [], {}
    loss_sum, n = 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
        batch = to_device(batch, device, drop=drop)
        with autocast_ctx(device, amp_dtype):
            logits, aux, _ = split_output(model(batch))
            logits = logits.float()
        for k, v in aux.items():
            all_aux.setdefault(k, []).append(torch.softmax(v.float(), dim=1).cpu())
        if criterion is not None:
            loss_sum += criterion(logits, batch["label"]).item() * logits.size(0)
        all_logits.append(logits.cpu())
        all_y.append(batch["label"].cpu())
        all_idx.append(batch["index"].cpu())
        n += logits.size(0)
    probs = torch.softmax(torch.cat(all_logits), dim=1).numpy()
    return {"loss": (loss_sum / n) if criterion is not None else None,
            "y_true": torch.cat(all_y).numpy(), "index": torch.cat(all_idx).numpy(),
            "y_pred": probs.argmax(axis=1).astype(np.int64), "probs": probs,
            "aux": {k: torch.cat(v).numpy() for k, v in all_aux.items()}}
