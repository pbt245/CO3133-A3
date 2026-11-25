"""Device selection plus hardware / software / git information for traceability."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

import numpy as np
import torch

from src.utils.io import ROOT


def resolve_device(pref: str = "auto") -> torch.device:
    pref = (pref or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU kernels so wall-clock timings are correct."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def git_info() -> dict:
    """Commit of the running code. On Kaggle there is no .git: scripts/make_kaggle_bundle.py writes
    BUNDLE_INFO (commit, dirty flag, sha256 of the bundled files) into the code bundle instead."""
    commit = _git("rev-parse", "HEAD")
    if commit is None and (ROOT / "BUNDLE_INFO").exists():
        info = json.loads((ROOT / "BUNDLE_INFO").read_text())
        return {"commit": info.get("commit"), "dirty": info.get("dirty"), "tag": None,
                "bundle_sha256": info.get("sha256"), "source": "BUNDLE_INFO"}
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit,
            "dirty": (bool(status) if commit is not None and status is not None else None),
            "tag": _git("describe", "--tags", "--exact-match"), "source": "git"}


def _version(module: str):
    try:
        return __import__(module).__version__
    except Exception:
        return None


def environment_info(device: torch.device | None = None) -> dict:
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": props.name, "total_memory_gb": round(props.total_memory / 1024**3, 2)})
    return {
        "device_used": str(device) if device is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "versions": {m: _version(m) for m in ("torch", "torchvision", "transformers", "tokenizers", "numpy",
                                               "scipy", "sklearn", "pandas", "matplotlib", "yaml", "PIL")},
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "gpus": gpus,
    }
