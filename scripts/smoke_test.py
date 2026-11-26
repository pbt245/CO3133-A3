#!/usr/bin/env python
"""End-to-end smoke test on a tiny synthetic mirror (CPU, ~1-2 min, no downloads, no real data):
prepare_data -> audit make/score -> eda -> train UA / UB / F1 (random-init ResNet-50 at 64 px, 1-layer DistilBERT
with a toy vocabulary, 2 epochs x 3 steps) -> keyword baseline -> late fusion -> M3: F2 + E1 / E2 / E4 variants,
masked F0, T0, removal predictions, qualitative panels, conflict review -> compare -> analysis_m3.
It checks that the code runs and that files line up; the numbers mean nothing.

  python scripts/smoke_test.py --workdir /tmp/a3_smoke
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["apple_pie", "pizza", "sushi", "waffles"]
TITLES = {"apple_pie": ["Easy Apple Pie | Taste of Home", "Grandma's apple pies", "Dutch Apple Pie Recipe",
                        "Apple Pie Egg Rolls &amp; more"],
          "pizza": ["Margherita Pizza - Food Network", "Homemade pizza dough", "Pizzas on Pinterest", "Detroit style"],
          "sushi": ["Salmon Sushi Rolls", "How to make sushi rice", "Sushi bake | Just One Cookbook", "Maki"],
          "waffles": ["Belgian Waffles", "Crispy waffle recipe", "Chicken and Waffles", "Fluffy buttermilk waffles"]}


def make_mirror(root: Path, rng):
    rows = {"train": [], "test": []}
    for split, n in (("train", 14), ("test", 6)):
        for ci, c in enumerate(CLASSES):
            d = root / "images" / split / c
            d.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                i = f"{c}_{split[:2]}{k}"
                w, h = int(rng.integers(90, 320)), int(rng.integers(90, 320))
                base = np.zeros((h, w, 3), np.uint8)
                base[..., ci % 3] = 160
                img = np.clip(base + rng.integers(0, 90, (h, w, 3)), 0, 255).astype(np.uint8)
                Image.fromarray(img).save(d / f"{i}.jpg", quality=90)
                title = TITLES[c][k % 4]
                if split == "train" and k == 0:
                    continue                                   # image without a title row
                if k == 1:
                    title = "Recipes"                          # fails the content-word criterion
                rows[split].append((f"{i}.jpg", title, c))
    # near-constant image (image-missing), exact duplicate of a train image in test (test_dup), cross-label copy
    Image.fromarray(np.full((120, 120, 3), 200, np.uint8)).save(root / "images/train/pizza/pizza_tr5.jpg")
    shutil.copy(root / "images/train/sushi/sushi_tr3.jpg", root / "images/test/sushi/sushi_te2.jpg")
    shutil.copy(root / "images/train/sushi/sushi_tr4.jpg", root / "images/train/waffles/waffles_tr4.jpg")
    (root / "texts").mkdir(parents=True, exist_ok=True)
    for split, r in rows.items():
        pd.DataFrame(r).to_csv(root / "texts" / f"{split}_titles.csv", header=False, index=False)


def make_tiny_distilbert(dst: Path):
    from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast
    words = sorted({w.strip(".,'|&;-").lower() for ts in TITLES.values() for t in ts for w in t.split()} - {""})
    vocab = ["[PAD]"] + [f"[unused{i}]" for i in range(5)] + ["[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words + list("abcdefghijklmnopqrstuvwxyz")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "vocab.txt").write_text("\n".join(vocab) + "\n")
    DistilBertTokenizerFast(vocab_file=str(dst / "vocab.txt"), do_lower_case=True).save_pretrained(dst)
    cfg = DistilBertConfig(vocab_size=len(vocab), dim=32, n_layers=1, n_heads=2, hidden_dim=64, max_position_embeddings=64)
    DistilBertModel(cfg).save_pretrained(dst)


def run(cmd, env):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, env=env, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    args = p.parse_args()
    w = Path(args.workdir).resolve()
    if w.exists():
        shutil.rmtree(w)
    rng = np.random.default_rng(0)
    make_mirror(w / "mirror", rng)
    make_tiny_distilbert(w / "weights" / "distilbert-base-uncased")
    env = dict(os.environ, UPMC_ROOT=str(w / "mirror"), A3_WEIGHTS_DIR=str(w / "weights"), PYTHONHASHSEED="0")
    prep = w / "prepare"
    run(["scripts/prepare_data.py", "--out", prep, "--resized", w / "upmc256.zip", "--workers", 2, "--val-size", 15], env)
    with zipfile.ZipFile(w / "upmc256.zip") as zf:
        zf.extractall(w)
    env["UPMC_IMAGE_ROOT"] = str(w / "upmc256")
    split_dir, meta_dir = prep / "data/splits", prep / "data/meta"
    run(["scripts/audit_pairs.py", "make", "--split-dir", split_dir, "--out", w / "audit", "--n-classes", 3,
         "--per-class", 2], env)
    t = pd.read_csv(w / "audit/pair_audit_template.csv")
    t["verdict"] = ["valid", "valid", "text_unrelated", "valid", "image_unrelated", "wrong_label"]
    t.to_csv(w / "audit/pair_audit.csv", index=False)
    run(["scripts/audit_pairs.py", "score", "--csv", w / "audit/pair_audit.csv", "--out", w / "audit/summary.json"], env)
    run(["scripts/eda.py", "--split-dir", split_dir, "--meta-dir", meta_dir, "--out", w / "eda"], env)
    common = [f"data.split_dir={split_dir}", f"output.root={w / 'runs'}", "model.pretrained=false",
              "data.num_workers=0", "data.image_size=64", "data.resize=72", "train.epochs=2", "train.batch_size=8",
              "train.eval_batch_size=16", "train.max_steps_per_epoch=3", "train.show_progress=false",
              "eval.bootstrap_samples=20", "eval.n_examples=4", "eval.inference.batch_size=4",
              "eval.inference.warmup_iters=1", "eval.inference.timed_iters=2", "eval.inference.latency_samples=3"]
    for cfg, seeds in (("ua_resnet50", (42, 123)), ("ub_distilbert", (42, 123)), ("f1_concat", (42,))):
        for s in seeds:
            run(["scripts/train.py", "--config", f"configs/upmc/{cfg}.yaml", "--seed", s, "--overwrite", "--device", "cpu",
                 "--set", *common, *(["train.freeze_encoders_epochs=1"] if cfg == "f1_concat" else [])], env)
    run(["scripts/train.py", "--config", "configs/upmc/f1_concat.yaml", "--seed", 42, "--resume", "--no-eval", "--device", "cpu",
         "--set", *common, "train.epochs=3"], env)                                    # resume path
    run(["scripts/keyword_baseline.py", "--set", f"data.split_dir={split_dir}", f"output.root={w / 'runs'}"], env)
    run(["scripts/late_fusion.py", "--runs", w / "runs/upmc", "--seeds", 42, 123], env)
    # M3: F2 (frozen first epoch), E1 / E4 / E2 variants, then the M3 scripts
    m3 = [("configs/upmc/f2_crossmodal.yaml", (42, 123)), ("configs/upmc/ablations/f2_noaux.yaml", (42,)),
          ("configs/upmc/ablations/f2_nomoddrop.yaml", (42,)), ("configs/upmc/ablations/ub_distilbert_mask.yaml", (42, 123)),
          ("configs/upmc/ablations/f2_crossmodal_mask.yaml", (42,))]
    for cfg, seeds in m3:
        for s in seeds:
            run(["scripts/train.py", "--config", cfg, "--seed", s, "--overwrite", "--device", "cpu", "--set", *common,
                 "train.freeze_encoders_epochs=1"], env)
    runs = w / "runs/upmc"
    run(["scripts/late_fusion.py", "--runs", runs, "--text", "ub_distilbert_mask", "--name", "f0_late_fusion_mask",
         "--seeds", 42, 123], env)
    for extra in ([], ["--mask"]):
        run(["scripts/tfidf_baseline.py", *extra, "--grid", 1, "--set", f"data.split_dir={split_dir}",
             f"output.root={w / 'runs'}"], env)
    run(["scripts/removal_predictions.py", runs / "f1_concat/seed42", "--device", "cpu", "--set",
         f"data.split_dir={split_dir}"], env)
    run(["scripts/qualitative.py", "--runs", runs / "f2_crossmodal/seed42", runs / "f2_noaux/seed42",
         "--image-run", runs / "ua_resnet50/seed42", "--text-run", runs / "ub_distilbert/seed42", "--out", w / "analysis",
         "--per-group", 2, "--device", "cpu", "--set", f"data.split_dir={split_dir}"], env)
    run(["scripts/conflict_review.py", "make", "--split-dir", split_dir, "--image-run", runs / "ua_resnet50/seed42",
         "--text-run", runs / "ub_distilbert/seed42", "--fusion-run", runs / "f2_crossmodal/seed42", "--out",
         w / "review", "--per-group", 2], env)
    t = pd.read_csv(w / "review/conflict_review_template.csv")
    t["verdict"] = (["title_generic", "image_unrelated", "wrong_label", "other"] * len(t))[:len(t)]
    t.to_csv(w / "review/conflict_review.csv", index=False)
    run(["scripts/conflict_review.py", "score", "--csv", w / "review/conflict_review.csv", "--out",
         w / "review/summary.json"], env)
    run(["scripts/compare.py", "--runs", runs, "--out", w / "comparison", "--meta-dir", meta_dir], env)
    run(["scripts/analysis_m3.py", "--runs", runs, "--out", w / "analysis", "--split-dir", split_dir,
         "--review", w / "review/summary.json"], env)
    print(f"\nSMOKE TEST PASSED -> {w}")


if __name__ == "__main__":
    main()
