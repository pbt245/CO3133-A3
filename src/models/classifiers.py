"""M2 models (proposal section 5).

UA  image_only     ResNet-50 pooled feature (2048) -> dropout -> linear 101
UB  text_only      DistilBERT [CLS] (768) -> dropout -> linear 101
F1  concat_fusion  image 2048 -> linear 512 + LayerNorm ; text 768 -> linear 512 + LayerNorm
                   -> concat (1024) -> linear 512 -> GELU -> dropout -> linear 101  (intermediate fusion)
                   Missing modality: its projected vector is multiplied by has_img / has_txt, i.e. zero-filled.

Every model takes the batch dict of src/data/upmc.py and returns logits. Parameter-name prefixes define the
optimizer groups: image_encoder.*, text_encoder.*, everything else = "head" (new layers).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders import ImageEncoder, TextEncoder


class ImageOnly(nn.Module):
    uses_image, uses_text, is_fusion = True, False, False

    def __init__(self, num_classes: int, pretrained: bool = True, head_dropout: float = 0.0,
                 image_weights="auto", **_):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained, image_weights)
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(self.image_encoder.out_dim, num_classes))

    def forward(self, batch):
        return self.head(self.image_encoder(batch["image"]))


class TextOnly(nn.Module):
    uses_image, uses_text, is_fusion = False, True, False

    def __init__(self, num_classes: int, pretrained: bool = True, head_dropout: float = 0.1,
                 text_model_dir="auto", text_config_overrides=None, **_):
        super().__init__()
        self.text_encoder = TextEncoder(pretrained, text_model_dir, text_config_overrides)
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(self.text_encoder.out_dim, num_classes))

    def forward(self, batch):
        return self.head(self.text_encoder(batch["input_ids"], batch["attention_mask"]))


class ConcatFusion(nn.Module):
    uses_image, uses_text, is_fusion = True, True, True

    def __init__(self, num_classes: int, pretrained: bool = True, proj_dim: int = 512, hidden_dim: int = 512,
                 dropout: float = 0.1, image_weights="auto", text_model_dir="auto", text_config_overrides=None, **_):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained, image_weights)
        self.text_encoder = TextEncoder(pretrained, text_model_dir, text_config_overrides)
        self.img_proj = nn.Sequential(nn.Linear(self.image_encoder.out_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.txt_proj = nn.Sequential(nn.Linear(self.text_encoder.out_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.head = nn.Sequential(nn.Linear(2 * proj_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden_dim, num_classes))

    def forward(self, batch):
        zi = self.img_proj(self.image_encoder(batch["image"]))
        zt = self.txt_proj(self.text_encoder(batch["input_ids"], batch["attention_mask"]))
        zi = zi * batch["has_img"].to(zi.dtype)[:, None]
        zt = zt * batch["has_txt"].to(zt.dtype)[:, None]
        return self.head(torch.cat([zi, zt], dim=1))
