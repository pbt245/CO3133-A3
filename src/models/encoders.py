"""Modality-specific encoders.

ImageEncoder: torchvision ResNet-50, ImageNet-1k weights IMAGENET1K_V2 (the A2 P1 backbone), fc removed.
              forward -> pooled 2048-d feature; feature_map -> layer4 map (B, 2048, 7, 7), used by F2 in M3.
TextEncoder:  Hugging Face DistilBERT-base-uncased (6 layers, d = 768); forward -> hidden state of [CLS].
Weights are loaded from local files (Kaggle runs with Internet OFF, see src/utils/paths.py).
`pretrained: false` builds the same architecture with random weights; evaluation uses it and then loads the
fine-tuned checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm

from src.utils import paths


class ImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, weights_file="auto"):
        super().__init__()
        net = tvm.resnet50(weights=None)
        if pretrained:
            p = paths.weights_path("resnet50", weights_file)
            if p is None:                                        # online machine: torchvision download
                state = tvm.ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True)
            else:
                state = torch.load(p, map_location="cpu", weights_only=True)
            net.load_state_dict(state)
        net.fc = nn.Identity()
        self.net = net
        self.out_dim = 2048

    def feature_map(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        return n.layer4(n.layer3(n.layer2(n.layer1(x))))

    def forward(self, x):
        return self.net(x)


class TextEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, model_dir="auto", config_overrides: dict | None = None):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        p = paths.weights_path("distilbert", model_dir)
        src, local = (str(p), True) if p is not None else ("distilbert-base-uncased", False)
        if pretrained and not config_overrides:
            self.net = AutoModel.from_pretrained(src, local_files_only=local)
        else:
            config = AutoConfig.from_pretrained(src, local_files_only=local, **(config_overrides or {}))
            self.net = AutoModel.from_config(config)
        self.out_dim = int(self.net.config.dim if hasattr(self.net.config, "dim") else self.net.config.hidden_size)

    def token_states(self, input_ids, attention_mask):
        return self.net(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward(self, input_ids, attention_mask):
        return self.token_states(input_ids, attention_mask)[:, 0]
