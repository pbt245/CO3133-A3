#!/usr/bin/env python
"""Pre-flight checks for the lab machine (run by setup.sh; also safe to run alone). Exit code 1 if anything is wrong.
Checks: package versions, CUDA + bf16, GPUs, pre-resized images (count per split), title CSVs, committed split
fingerprints, every split id has an image file, weights files (+ SHA256SUMS), tokenizer, a forward pass of F1 on GPU."""
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name}{': ' + str(detail) if detail else ''}", flush=True)


import torch  # noqa: E402

from src.utils.env import environment_info  # noqa: E402

env = environment_info()
print("versions:", env["versions"])
check("CUDA available", torch.cuda.is_available(), f"torch {torch.__version__}, CUDA {torch.version.cuda}")
check("bf16 supported", env["bf16_supported"])
for g in env["gpus"]:
    free, total = torch.cuda.mem_get_info(g["index"])
    print(f"    GPU {g['index']} {g['name']}: {free / 1024**3:.1f} / {total / 1024**3:.1f} GB free "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})")

from src.data.upmc import image_file, load_split, read_titles  # noqa: E402
from src.utils import paths  # noqa: E402

img_root = paths.image_root("auto", fallback_raw=False)
n = {s: sum(1 for _ in (img_root / "images" / s).rglob("*.jpg")) for s in ("train", "test")}
check("pre-resized images", n == {"train": 67988, "test": 22716}, f"{img_root} {n}")
titles = read_titles(paths.upmc_root())
check("title rows", len(titles) == 90688, len(titles))
info, frames = load_split(ROOT / "data/splits")
check("committed split fingerprints", True, info["fingerprints"])
missing = [r.id for s in frames.values() for r in s.itertuples()
           if not image_file(img_root, r.official_split, r.label, r.id).exists()]
check("every split item has an image", not missing, f"{len(missing)} missing")

w = paths.weights_path("resnet50")
d = paths.weights_path("distilbert")
check("ResNet-50 weights", w is not None and w.exists(), w)
check("DistilBERT folder", d is not None and (d / "model.safetensors").exists(), d)
sums = Path(os.environ.get("A3_WEIGHTS_DIR", ROOT / "weights")) / "SHA256SUMS"
if sums.exists():
    for line in sums.read_text().split("\n"):
        if line.strip():
            h, rel = line.split(None, 1)
            f = sums.parent / Path(rel).relative_to("weights")
            check(f"sha256 {f.name}", f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == h)

try:
    from src.data.upmc import load_tokenizer
    from src.models import build_model
    from src.utils.config import load_config
    tok = load_tokenizer({"text": {"model_dir": "auto"}})
    check("tokenizer", True, tok.convert_ids_to_tokens(tok("Easy Apple Pie | Taste of Home")["input_ids"]))
    cfg = load_config(ROOT / "configs/upmc/f1_concat.yaml")
    m = build_model(cfg["model"], 101, cfg.get("text")).cuda().eval()
    enc = tok(["Easy Apple Pie"], return_tensors="pt")
    b = {"image": torch.randn(2, 3, 224, 224, device="cuda"), "input_ids": enc["input_ids"].repeat(2, 1).cuda(),
         "attention_mask": enc["attention_mask"].repeat(2, 1).cuda(), "has_img": torch.ones(2, device="cuda"),
         "has_txt": torch.ones(2, device="cuda")}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(b)
    check("F1 forward pass on GPU (bf16)", tuple(out.shape) == (2, 101), tuple(out.shape))
except Exception as e:  # noqa: BLE001
    check("model / tokenizer", False, f"{type(e).__name__}: {e}")

print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
