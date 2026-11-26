"""Model registry built from the `model:` config section (+ `text:` for the DistilBERT folder)."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.models.classifiers import ConcatFusion, CrossModalFusion, ImageOnly, TextOnly

MODELS = {"image_only": ImageOnly, "text_only": TextOnly, "concat_fusion": ConcatFusion,
          "crossmodal_fusion": CrossModalFusion}
GROUP_PREFIXES = ("image_encoder", "text_encoder")       # everything else -> "head"


def build_model(model_cfg: dict, num_classes: int, text_cfg: dict | None = None) -> nn.Module:
    cfg = copy.deepcopy(model_cfg)
    name = str(cfg.pop("name")).lower()
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    cfg.setdefault("text_model_dir", (text_cfg or {}).get("model_dir", "auto"))
    return MODELS[name](num_classes=num_classes, **cfg)


def param_group_name(pname: str) -> str:
    for p in GROUP_PREFIXES:
        if pname.startswith(p + "."):
            return p
    return "head"


def count_parameters(model: nn.Module) -> dict:
    out = {"total": int(sum(p.numel() for p in model.parameters())),
           "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}
    for g in (*GROUP_PREFIXES, "head"):
        out[g] = int(sum(p.numel() for n, p in model.named_parameters() if param_group_name(n) == g))
    return out


def count_macs(model: nn.Module, batch: dict) -> float | None:
    """Multiply-accumulates for one item (batch of size 1, given on the model's device)."""
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(batch)
        return fc.get_total_flops() / 2
    except Exception:
        return None
    finally:
        model.train(was_training)
