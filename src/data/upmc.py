"""UPMC Food-101 image-title pairs: titles, committed split files, multimodal Dataset, transforms, DataLoaders.

Mirror layout (Kaggle gianmarco96/upmcfood101, verified by scripts/inventory_upmc.py):
  images/{train,test}/<label>/<id>.jpg          id = file stem, unique over both splits (e.g. apple_pie_17)
  texts/{train,test}_titles.csv                 no header: file name, page title, label
The pre-resized copy written by scripts/prepare_data.py (upmc256) keeps the same images/ layout.

Split files (scripts/prepare_data.py, committed): data/splits/{train,val,test}.csv with one row per item
  id, label, official_split, has_img, has_txt, cluster_id, test_dup, cross_label
and data/splits/split.json (classes, counts, fingerprints, parameters). Titles are never committed; they are
read from the mirror by id.

Evaluation sets (proposal 2.3): models are trained and selected on COMPLETE pairs (valid image and valid
title). `test` = complete test pairs; `test_missing_text` = test items whose image is valid but whose title
fails the validity criterion (the natural missing-text set).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.data.text import ClassLexicon, clean_title
from src.utils import paths
from src.utils.io import load_json, resolve_path
from src.utils.seed import seed_worker

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- raw mirror
def read_titles(root) -> pd.DataFrame:
    """All title rows of the mirror: id, official_split, label, title (raw text, as crawled)."""
    frames = []
    for split in ("train", "test"):
        df = pd.read_csv(Path(root) / "texts" / f"{split}_titles.csv", header=None, names=["file", "title", "label"],
                         dtype=str, keep_default_na=False)
        df["id"] = df["file"].str.strip().str.rsplit(".", n=1).str[0]
        df["official_split"] = split
        frames.append(df[["id", "official_split", "label", "title"]])
    out = pd.concat(frames, ignore_index=True)
    if out["id"].duplicated().any():
        raise ValueError("duplicate ids in the title CSVs")
    return out


def image_file(root, official_split: str, label: str, item_id: str) -> Path:
    return Path(root) / "images" / official_split / label / f"{item_id}.jpg"


def fingerprint(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- split files
def load_split(split_dir) -> tuple[dict, dict]:
    d = resolve_path(split_dir)
    if not (d / "split.json").exists():
        raise FileNotFoundError(f"{d}/split.json not found: run scripts/prepare_data.py and commit data/splits")
    info = load_json(d / "split.json")
    frames = {}
    for s in SPLITS:
        df = pd.read_csv(d / f"{s}.csv", dtype={"id": str, "label": str, "official_split": str})
        if fingerprint(df["id"]) != info["fingerprints"][s]:
            raise ValueError(f"{d / (s + '.csv')} does not match the fingerprint in split.json")
        frames[s] = df
    return info, frames


# --------------------------------------------------------------------------- transforms
def build_transforms(dcfg: dict, train: bool):
    size = int(dcfg.get("image_size", 224))
    resize = int(dcfg.get("resize", 256))
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose([T.Resize(resize), T.CenterCrop(size), *norm])
    a = dcfg.get("augment") or {}
    ops = [T.RandomResizedCrop(size, scale=tuple(a.get("crop_scale", (0.35, 1.0))))]
    if a.get("hflip", True):
        ops.append(T.RandomHorizontalFlip())
    cj = float(a.get("color_jitter", 0.0) or 0.0)
    if cj > 0:
        ops.append(T.ColorJitter(brightness=cj, contrast=cj, saturation=cj))
    return T.Compose([*ops, *norm])


# --------------------------------------------------------------------------- dataset
class UPMCDataset(Dataset):
    """One item = one web page: image + title + label. Returns a dict; only the modalities the model uses
    are loaded (the text-only model never decodes a JPEG)."""

    def __init__(self, df: pd.DataFrame, class_index: dict, image_root, transform, tokens: dict | None,
                 use_image: bool, use_text: bool, display_size: int = 224, resize: int = 256):
        self.df = df.reset_index(drop=True)
        self.ids = self.df["id"].tolist()
        col = "title_model" if "title_model" in self.df else "title_clean"     # what the text encoder sees
        self.titles = self.df[col].tolist() if col in self.df else [""] * len(self.df)
        self.labels = torch.tensor(self.df["label"].map(class_index).to_numpy(dtype=np.int64))
        self.has_img = torch.tensor(self.df["has_img"].astype(bool).to_numpy(dtype=np.float32))
        self.has_txt = torch.tensor(self.df["has_txt"].astype(bool).to_numpy(dtype=np.float32))
        self.image_root = Path(image_root) if image_root is not None else None
        self.paths = ([image_file(self.image_root, s, l, i) for s, l, i in
                       zip(self.df["official_split"], self.df["label"], self.df["id"])]
                      if self.image_root is not None else None)
        self.transform = transform
        self.tokens = tokens
        self.use_image, self.use_text = bool(use_image), bool(use_text)
        self.display = T.Compose([T.Resize(resize), T.CenterCrop(display_size)])

    def __len__(self) -> int:
        return len(self.ids)

    def load(self, idx) -> Image.Image:
        with Image.open(self.paths[idx]) as im:
            return im.convert("RGB")

    def __getitem__(self, idx):
        item = {"label": self.labels[idx], "has_img": self.has_img[idx], "has_txt": self.has_txt[idx],
                "index": torch.tensor(idx)}
        if self.use_image:
            item["image"] = self.transform(self.load(idx))
        if self.use_text:
            item["input_ids"] = torch.from_numpy(self.tokens["input_ids"][idx])
            item["attention_mask"] = torch.from_numpy(self.tokens["attention_mask"][idx])
        return item

    def display_image(self, idx) -> np.ndarray:
        """Un-normalized centre crop (H, W, 3) uint8, as the model sees it at eval time."""
        return np.asarray(self.display(self.load(idx)))


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    path = paths.weights_path("distilbert", cfg.get("text", {}).get("model_dir"))
    if path is None:                                     # online machine only
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def tokenize(tokenizer, titles, max_length: int) -> dict:
    enc = tokenizer(list(titles), padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
    return {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}


def model_modalities(model_cfg: dict) -> tuple[bool, bool]:
    name = str(model_cfg["name"]).lower()
    return {"image_only": (True, False), "text_only": (False, True)}.get(name, (True, True))


def build_datasets(cfg: dict, logger=None, sets=("train", "val", "test", "test_missing_text"),
                   need_image=None, need_text=None):
    """Datasets of the configured model. need_image / need_text override the model's own modalities."""
    dcfg = cfg["data"]
    info, frames = load_split(dcfg["split_dir"])
    classes = info["classes"]
    class_index = {c: i for i, c in enumerate(classes)}
    use_image, use_text = model_modalities(cfg["model"])
    use_image = use_image if need_image is None else need_image
    use_text = use_text if need_text is None else need_text

    titles = read_titles(paths.upmc_root(dcfg.get("upmc_root"))).set_index("id")["title"]
    mask_names = bool(cfg.get("text", {}).get("mask_class_names", False))
    lexicon = ClassLexicon(classes) if mask_names else None
    img_root = paths.image_root(dcfg.get("image_root")) if use_image else None
    tokenizer = load_tokenizer(cfg) if use_text else None
    max_len = int(cfg.get("text", {}).get("max_length", 64))

    def subset(split, which):
        df = frames[split]
        if which == "complete":
            df = df[df["has_img"].astype(bool) & df["has_txt"].astype(bool)]
        elif which == "missing_text":
            df = df[df["has_img"].astype(bool) & ~df["has_txt"].astype(bool)]
        df = df.copy()
        df["title_clean"] = [clean_title(titles.get(i, "")) for i in df["id"]]
        # E2: every class-name mention replaced by [MASK] in train, val and test alike; title_clean keeps the
        # original title for the test subsets (mention / no mention), which are defined on the real title
        df["title_model"] = [lexicon.mask(t) for t in df["title_clean"]] if mask_names else df["title_clean"]
        return df

    spec = {"train": ("train", dcfg.get("train_on", "complete")), "val": ("val", "complete"),
            "test": ("test", "complete"), "test_missing_text": ("test", "missing_text")}
    datasets = {}
    for name in sets:
        split, which = spec[name]
        df = subset(split, which)
        tokens = tokenize(tokenizer, df["title_model"], max_len) if use_text else None
        tf = build_transforms(dcfg, train=(name == "train"))
        datasets[name] = UPMCDataset(df, class_index, img_root, tf, tokens, use_image, use_text,
                                     int(dcfg.get("image_size", 224)), int(dcfg.get("resize", 256)))
    meta = {
        "dataset": "upmc",
        "classes": classes,
        "num_classes": len(classes),
        "image_size": int(dcfg.get("image_size", 224)),
        "image_root": str(img_root) if img_root is not None else None,
        "use_image": use_image, "use_text": use_text, "max_length": max_len, "mask_class_names": mask_names,
        "split_file": str(resolve_path(dcfg["split_dir"]) / "split.json"),
        "split_fingerprints": info["fingerprints"],
        "set_fingerprints": {k: fingerprint(v.ids) for k, v in datasets.items()},
        "sizes": {k: len(v) for k, v in datasets.items()},
    }
    if mask_names:
        ds0 = next(iter(datasets.values()))
        meta["masking"] = {"titles_changed": int(sum(a != b for a, b in zip(ds0.df["title_clean"], ds0.titles))),
                           "titles": len(ds0), "residual_mentions": int(sum(bool(lexicon.mentions(t)) for t in ds0.titles)),
                           "set": next(iter(datasets))}
    if logger is not None:
        logger.info("Data: images %s | sizes %s%s", meta["image_root"], meta["sizes"],
                    f" | class names masked: {meta['masking']}" if mask_names else "")
    return datasets, meta


def build_dataloaders(cfg: dict, seed: int, device: torch.device, logger=None, sets=("train", "val")):
    datasets, meta = build_datasets(cfg, logger=logger, sets=sets)
    nw = int(cfg["data"].get("num_workers", 0))
    common = {"num_workers": nw, "pin_memory": device.type == "cuda",
              "worker_init_fn": seed_worker if nw > 0 else None, "persistent_workers": nw > 0,
              "prefetch_factor": 4 if nw > 0 else None}
    g = torch.Generator()
    g.manual_seed(seed)
    bs = int(cfg["train"]["batch_size"])
    ebs = int(cfg["train"].get("eval_batch_size", bs))
    loaders = {}
    for name, ds in datasets.items():
        if name == "train":
            loaders[name] = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, generator=g, **common)
        else:
            loaders[name] = DataLoader(ds, batch_size=ebs, shuffle=False, **common)
    return loaders, datasets, meta


def trim_tokens(batch: dict) -> dict:
    """Cut the padded 64-token rows to the longest title of the batch (titles have median 7 words)."""
    if "attention_mask" in batch:
        n = int(batch["attention_mask"].sum(dim=1).max().item())
        n = max(n, 1)
        batch["input_ids"] = batch["input_ids"][:, :n]
        batch["attention_mask"] = batch["attention_mask"][:, :n]
    return batch
