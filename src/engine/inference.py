"""Inference-time cost (forward pass only, inputs already on the device)."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import default_collate

from src.engine.loops import autocast_ctx, to_device
from src.utils.env import synchronize


@torch.no_grad()
def measure_inference(model, dataset, device, batch_size: int = 128, warmup_iters: int = 10, timed_iters: int = 30,
                      latency_samples: int = 200, amp_dtype=None) -> dict:
    """throughput: items/s for repeated forward passes of one fixed batch; latency: ms per item at batch 1."""
    model.eval()
    n = min(max(int(batch_size), int(latency_samples)), len(dataset))
    items = [dataset[i] for i in range(n)]
    bs = min(int(batch_size), n)
    batch = to_device(default_collate(items[:bs]), device)
    singles = [to_device(default_collate([it]), device) for it in items[:int(latency_samples)]]

    with autocast_ctx(device, amp_dtype):
        for _ in range(int(warmup_iters)):
            model(batch)
        synchronize(device)
        t0 = time.perf_counter()
        iters = max(int(timed_iters), 1)
        for _ in range(iters):
            model(batch)
        synchronize(device)
        elapsed = time.perf_counter() - t0
        for s in singles[:20]:
            model(s)
        lat = []
        for s in singles:
            synchronize(device)
            t = time.perf_counter()
            model(s)
            synchronize(device)
            lat.append((time.perf_counter() - t) * 1000.0)
    lat = np.asarray(lat) if lat else np.asarray([np.nan])
    return {"device": str(device), "amp_dtype": str(amp_dtype), "batch_size": bs, "timed_iters": iters,
            "throughput_items_per_s": float(bs * iters / elapsed),
            "ms_per_item_batched": float(1000.0 * elapsed / (bs * iters)),
            "latency_samples": len(singles), "latency_ms_bs1_mean": float(np.mean(lat)),
            "latency_ms_bs1_median": float(np.median(lat)), "latency_ms_bs1_p95": float(np.percentile(lat, 95)),
            "note": "titles are trimmed to the longest title of the batch; images are 224 x 224"}
