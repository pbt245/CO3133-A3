"""Models (proposal section 5).

UA  image_only     ResNet-50 pooled feature (2048) -> dropout -> linear 101
UB  text_only      DistilBERT [CLS] (768) -> dropout -> linear 101
F1  concat_fusion  image 2048 -> linear 512 + LayerNorm ; text 768 -> linear 512 + LayerNorm
                   -> concat (1024) -> linear 512 -> GELU -> dropout -> linear 101  (intermediate fusion)
                   Missing modality: its projected vector is multiplied by has_img / has_txt, i.e. zero-filled.
F2  crossmodal_fusion  [FUS] + 49 image tokens + title tokens -> 2 transformer layers -> 101, auxiliary unimodal
                   heads, modality dropout and learned [MISSING] tokens (M3, see CrossModalFusion)

Every model takes the batch dict of src/data/upmc.py and returns logits (F2: a dict with the logits, see
src/engine/loops.py:split_output). Parameter-name prefixes define the
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


class FusionBlock(nn.Module):
    """Pre-LN transformer encoder layer (self-attention + GELU FFN). Written out instead of
    nn.TransformerEncoderLayer so the attention weights of the [FUS] token can be read (capture=True)."""

    def __init__(self, d: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.capture, self.last_attn = False, None

    def forward(self, x, pad_mask):
        h = self.ln1(x)
        a, w = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=self.capture, average_attn_weights=True)
        self.last_attn = w[:, 0].detach().float() if self.capture else None      # [FUS] row: (B, L)
        x = x + self.drop1(a)
        return x + self.drop2(self.ffn(self.ln2(x)))


class CrossModalFusion(nn.Module):
    """F2 - improved fusion (proposal 5.1).

    tokens   image: ResNet-50 layer4 map (7 x 7 x 2048) -> 49 tokens -> linear 512 + learned 2-D position + type
             text:  DistilBERT last hidden states (<= 64 tokens, padding masked) -> linear 512 + type
             a learned [FUS] token is prepended: [FUS ; image tokens ; text tokens]
    fusion   `layers` pre-LN transformer layers (d 512, 8 heads, FFN 2048, dropout 0.1); [FUS] -> LayerNorm -> 101
    aux      linear heads on the pooled image feature (2048) and on the text [CLS] (768).
             aux_weight > 0: L = CE(fused) + aux_weight * (CE(image head) + CE(text head)), gradients reach the encoders.
             aux_weight = 0 (ablation E1): the heads are trained on DETACHED features (weight 1), i.e. as linear
             probes that measure how discriminative each encoder is, without changing the encoders.
    missing  an item without a valid image (has_img = 0) / title (has_txt = 0) gets ONE learned [MISSING_IMG] /
             [MISSING_TXT] token instead of that modality's tokens (the other positions are masked).
             Training: modality dropout - with probability p one modality (image or text, 50/50) of an item is
             replaced by its missing token, never both.
    The forward pass returns a dict: logits (fused), aux_image, aux_text (logits) and aux_weight (for the loss).
    """
    uses_image, uses_text, is_fusion = True, True, True

    def __init__(self, num_classes: int, pretrained: bool = True, d_model: int = 512, layers: int = 2, heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1, image_grid: int = 7, aux_weight: float = 0.5,
                 modality_dropout: float = 0.15, image_weights="auto", text_model_dir="auto",
                 text_config_overrides=None, **_):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained, image_weights)
        self.text_encoder = TextEncoder(pretrained, text_model_dir, text_config_overrides)
        self.aux_weight, self.modality_dropout, self.grid = float(aux_weight), float(modality_dropout), int(image_grid)
        self.img_proj = nn.Linear(self.image_encoder.out_dim, d_model)
        self.txt_proj = nn.Linear(self.text_encoder.out_dim, d_model)
        self.img_pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, d_model))
        self.type_emb = nn.Parameter(torch.zeros(2, d_model))                  # 0 image, 1 text
        self.fus_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.missing = nn.Parameter(torch.zeros(2, d_model))                   # [MISSING_IMG], [MISSING_TXT]
        for p in (self.img_pos, self.type_emb, self.fus_token, self.missing):
            nn.init.trunc_normal_(p, std=0.02)
        self.blocks = nn.ModuleList(FusionBlock(d_model, heads, ffn_dim, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.aux_drop = nn.Dropout(dropout)
        self.aux_image = nn.Linear(self.image_encoder.out_dim, num_classes)
        self.aux_text = nn.Linear(self.text_encoder.out_dim, num_classes)
        self.n_img_tokens = None

    def set_capture(self, on: bool = True):
        for b in self.blocks:
            b.capture = on

    def fus_attention(self):
        """[FUS] attention of the last layer (B, 1 + n_img + n_txt), heads averaged; needs set_capture(True)."""
        return self.blocks[-1].last_attn

    def _positions(self, n):
        if n == self.img_pos.shape[1]:
            return self.img_pos
        g = int(round(n ** 0.5))                     # other input sizes (smoke test): interpolate the 7 x 7 grid
        pos = self.img_pos.reshape(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        pos = nn.functional.interpolate(pos, size=(g, g), mode="bilinear", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, g * g, -1)

    def forward(self, batch):
        fmap = self.image_encoder.feature_map(batch["image"])                  # (B, 2048, h, w)
        pooled = fmap.mean(dim=(2, 3))                                         # = ResNet-50 average pool
        hid = self.text_encoder.token_states(batch["input_ids"], batch["attention_mask"])   # (B, T, 768)
        cls = hid[:, 0]
        B, dev = fmap.shape[0], fmap.device

        has_img, has_txt = batch["has_img"] > 0.5, batch["has_txt"] > 0.5
        if self.training and self.modality_dropout > 0:
            both = has_img & has_txt
            drop = (torch.rand(B, device=dev) < self.modality_dropout) & both
            which = torch.rand(B, device=dev) < 0.5                            # True: drop the image
            has_img = has_img & ~(drop & which)
            has_txt = has_txt & ~(drop & ~which)

        it = self.img_proj(fmap.flatten(2).transpose(1, 2))                    # (B, N, d)
        n_img = it.shape[1]
        self.n_img_tokens = n_img
        it = it + self._positions(n_img).to(it.dtype) + self.type_emb[0].to(it.dtype)
        tt = self.txt_proj(hid) + self.type_emb[1].to(hid.dtype)              # (B, T, d)
        tt = tt.to(it.dtype)
        n_txt = tt.shape[1]
        img_pad = torch.zeros(B, n_img, dtype=torch.bool, device=dev)
        txt_pad = batch["attention_mask"] == 0
        first_i = torch.arange(n_img, device=dev)[None] > 0
        first_t = torch.arange(n_txt, device=dev)[None] > 0
        mi, mt = ~has_img, ~has_txt
        it = torch.where(mi[:, None, None], (self.missing[0] + self.type_emb[0]).to(it.dtype).expand_as(it), it)
        tt = torch.where(mt[:, None, None], (self.missing[1] + self.type_emb[1]).to(tt.dtype).expand_as(tt), tt)
        img_pad = torch.where(mi[:, None], first_i.expand(B, -1), img_pad)
        txt_pad = torch.where(mt[:, None], first_t.expand(B, -1), txt_pad)

        x = torch.cat([self.fus_token.to(it.dtype).expand(B, -1, -1), it, tt], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), img_pad, txt_pad], dim=1)
        for blk in self.blocks:
            x = blk(x, pad)
        logits = self.head(self.norm(x[:, 0]))

        probe = self.aux_weight <= 0
        fi, ft = (pooled.detach(), cls.detach()) if probe else (pooled, cls)
        return {"logits": logits, "aux_image": self.aux_image(self.aux_drop(fi)),
                "aux_text": self.aux_text(self.aux_drop(ft)), "aux_weight": 1.0 if probe else self.aux_weight}
