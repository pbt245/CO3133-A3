"""Locate inputs that live outside the repo: the UPMC Food-101 mirror, the pre-resized image copy and the
pretrained weights. Kaggle notebooks run with Internet OFF, so everything comes from /kaggle/input.

Search order for every item: explicit path (config / CLI) -> environment variable -> repo folder -> /kaggle/input.
The /kaggle/input walk skips image folders, so it stays fast although the mirror has 90k files.
"""
from __future__ import annotations

import os
import zipfile
from functools import lru_cache
from pathlib import Path

from src.utils.io import ROOT, resolve_path

KAGGLE_INPUT = Path("/kaggle/input")
RESNET50_FILE = "resnet50-11ad3fa6.pth"            # torchvision ResNet50_Weights.IMAGENET1K_V2
DISTILBERT_DIR = "distilbert-base-uncased"
RESIZED_NAME = "upmc256"


@lru_cache(maxsize=None)
def _walk_input(max_depth: int = 7) -> tuple:
    """(dirpath, dirnames, filenames) under /kaggle/input, not descending into images/ folders."""
    out = []
    if not KAGGLE_INPUT.is_dir():
        return tuple()
    base = len(str(KAGGLE_INPUT).rstrip(os.sep).split(os.sep))
    for dirpath, dirnames, filenames in os.walk(KAGGLE_INPUT, followlinks=True):
        depth = len(dirpath.rstrip(os.sep).split(os.sep)) - base
        dirnames[:] = [d for d in dirnames if d != "images" and not d.startswith(".")] if depth < max_depth else []
        out.append((dirpath, tuple(dirnames), tuple(filenames)))
    return tuple(out)


def find_in_input(name: str, want_dir: bool = False):
    for dirpath, dirnames, filenames in _walk_input():
        if (name in dirnames) if want_dir else (name in filenames):
            return Path(dirpath) / name
    return None


def upmc_root(explicit=None) -> Path:
    """Folder of the original mirror (has texts/ and images/{train,test}/<label>/)."""
    if explicit and explicit != "auto":
        return resolve_path(explicit)
    if os.environ.get("UPMC_ROOT"):
        return Path(os.environ["UPMC_ROOT"])
    for dirpath, dirnames, filenames in _walk_input():
        texts = Path(dirpath) / "texts"
        if "texts" in dirnames and (texts / "train_titles.csv").exists():
            return Path(dirpath)
    local = ROOT / "data" / "raw" / "upmc_food101"
    if (local / "texts").exists():
        return local
    raise FileNotFoundError("UPMC Food-101 mirror not found: add gianmarco96/upmcfood101 as notebook input, "
                            "set UPMC_ROOT, or pass data.upmc_root=<folder with texts/ and images/>.")


def image_root(explicit=None, fallback_raw=True, extract_to="/tmp") -> Path:
    """Folder with images/{train,test}/<label>/<id>.jpg. Prefers the pre-resized copy (scripts/prepare_data.py):
    an extracted upmc256/ folder, or upmc256.zip (extracted once to /tmp/upmc256)."""
    if explicit and explicit != "auto":
        return resolve_path(explicit)
    if os.environ.get("UPMC_IMAGE_ROOT"):
        return Path(os.environ["UPMC_IMAGE_ROOT"])
    for cand in (ROOT / "data" / "resized" / RESIZED_NAME, Path(extract_to) / RESIZED_NAME):
        if (cand / ".complete").exists() or (cand / "images" / "test").is_dir():
            return cand
    found = find_in_input(RESIZED_NAME, want_dir=True)
    if found is not None and (found / "images").is_dir():
        return found
    z = find_in_input(RESIZED_NAME + ".zip")
    if z is not None:
        dst = Path(extract_to)
        print(f"extracting {z} -> {dst / RESIZED_NAME} (once)")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dst)
        (dst / RESIZED_NAME / ".complete").write_text(str(z))
        return dst / RESIZED_NAME
    if fallback_raw:
        return upmc_root()
    raise FileNotFoundError("pre-resized images (upmc256) not found")


def weights_path(kind: str, explicit=None) -> Path | None:
    """kind = 'resnet50' (a .pth file) or 'distilbert' (a Hugging Face folder). None -> download allowed."""
    name, is_dir = {"resnet50": (RESNET50_FILE, False), "distilbert": (DISTILBERT_DIR, True)}[kind]
    if explicit and explicit != "auto":
        return resolve_path(explicit)
    env = os.environ.get("A3_WEIGHTS_DIR")
    for cand in ([Path(env) / name] if env else []) + [ROOT / "weights" / name]:
        if cand.exists():
            return cand
    return find_in_input(name, want_dir=is_dir)
