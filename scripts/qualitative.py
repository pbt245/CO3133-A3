#!/usr/bin/env python
"""Qualitative evidence for the support / conflict analysis (proposal 5.5 item 4). Needs the images and the
F2 checkpoints, so it runs on the training machine after the M3 runs.

1. [FUS] attention over the whole complete-pair test set, for every F2-type run given (last fusion layer, heads
   averaged): share of attention on image tokens, on title tokens and on [FUS] itself, per item
   -> <run>/eval/fus_attention_test.csv
2. Example panels for the first run (seed 42): items grouped by the unimodal models of the same seed
   (support / conflict_I / conflict_T / both wrong), `--per-group` random items per group, each with
   - the image and a Grad-CAM map of F2's image encoder (layer4) for F2's predicted class
   - the title with the words that receive the most [FUS] attention
   - the UA, UB and F2 predictions
   -> outputs/analysis/qualitative_<group>.png

  python scripts/qualitative.py --runs outputs/runs/upmc/f2_crossmodal/seed42 outputs/runs/upmc/f2_noaux/seed42
"""
import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

from src.data.upmc import build_datasets, load_tokenizer
from src.engine.evaluate import load_run
from src.engine.loops import autocast_ctx, resolve_amp_dtype, split_output, to_device
from src.utils.env import resolve_device
from src.utils.io import resolve_path, save_json
from src.viz.plots import plt

GROUPS = ["support", "conflict_I", "conflict_T", "both_wrong"]
GROUP_TITLE = {"support": "Support: image model and text model both right",
               "conflict_I": "Conflict-I: only the image model right (title misleading)",
               "conflict_T": "Conflict-T: only the text model right (image misleading)",
               "both_wrong": "Both unimodal models wrong"}


def read_preds(run_dir, name="test"):
    d = pd.read_csv(Path(run_dir) / "eval" / f"predictions_{name}.csv", dtype={"id": str})
    return d.set_index("id")


@torch.no_grad()
def attention_pass(model, ds, device, amp, bs, nw):
    model.set_capture(True)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda")
    rows = []
    for batch in loader:
        b = to_device(batch, device)
        with autocast_ctx(device, amp):
            logits, _, _ = split_output(model(b))
        att = model.fus_attention()
        n = model.n_img_tokens
        rows.append(np.stack([att[:, 0].cpu().numpy(), att[:, 1:1 + n].sum(1).cpu().numpy(),
                              att[:, 1 + n:].sum(1).cpu().numpy(), logits.float().argmax(1).cpu().numpy()], 1))
    model.set_capture(False)
    a = np.concatenate(rows)
    return pd.DataFrame({"id": ds.ids, "att_fus": a[:, 0], "att_image": a[:, 1], "att_text": a[:, 2],
                         "f2_pred": a[:, 3].astype(int)})


def gradcam_and_words(model, ds, idx, device, tokenizer):
    """Grad-CAM (layer4 of F2's image encoder, predicted class) + [FUS] attention per title word, one item."""
    batch = to_device(default_collate([ds[idx]]), device)
    acts = {}
    layer = model.image_encoder.net.layer4

    def hook(_, __, out):
        acts["a"] = out
        out.register_hook(lambda g: acts.__setitem__("g", g))

    h = layer.register_forward_hook(hook)
    model.set_capture(True)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits, _, _ = split_output(model(batch))
            cls = int(logits.argmax(1))
            logits[0, cls].backward()
    finally:
        h.remove()
    att = model.fus_attention()[0].cpu().numpy()
    model.set_capture(False)
    a, g = acts["a"][0].detach(), acts["g"][0]
    cam = F.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
    cam = F.interpolate(cam[None, None], size=batch["image"].shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
    n = model.n_img_tokens
    ids = batch["input_ids"][0].cpu().numpy()
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())
    tatt = att[1 + n:1 + n + len(toks)]
    words, w_att = [], []
    for t, v in zip(toks, tatt):
        if t in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if t.startswith("##") and words:
            words[-1] += t[2:]
            w_att[-1] += v
        else:
            words.append(t)
            w_att.append(float(v))
    return cam, cls, words, np.asarray(w_att), float(att[1:1 + n].sum()), float(tatt.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["outputs/runs/upmc/f2_crossmodal/seed42"],
                   help="F2-type runs; the first one is used for the example panels")
    p.add_argument("--image-run", default="outputs/runs/upmc/ua_resnet50/seed42")
    p.add_argument("--text-run", default="outputs/runs/upmc/ub_distilbert/seed42")
    p.add_argument("--out", default="outputs/analysis")
    p.add_argument("--per-group", type=int, default=6)
    p.add_argument("--device", default="auto")
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args()
    device = resolve_device(args.device)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    first = None
    for rd in map(resolve_path, args.runs):
        cfg, model, _, _ = load_run(rd, device, overrides=args.set)
        ds, meta = build_datasets(cfg, sets=("test",))
        amp = resolve_amp_dtype(True, cfg["train"].get("amp_dtype", "auto"), device)
        att = attention_pass(model, ds["test"], device, amp, int(cfg["train"].get("eval_batch_size", 256)),
                             int(cfg["data"].get("num_workers", 0)))
        att.to_csv(rd / "eval" / "fus_attention_test.csv", index=False, float_format="%.5f")
        ref = read_preds(rd).loc[att["id"], "y_pred"].to_numpy()
        print(f"{rd}: [FUS] attention image {att.att_image.mean():.3f} | text {att.att_text.mean():.3f} | "
              f"self {att.att_fus.mean():.3f} | predictions equal to eval: {(ref == att.f2_pred).mean():.4f}")
        if first is None:
            first = (rd, cfg, model, ds["test"], meta)
        else:
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

    rd, cfg, model, test_ds, meta = first
    classes = meta["classes"]
    tok = load_tokenizer(cfg)
    ids = test_ds.ids
    ua, ub, f2 = (read_preds(r).loc[ids] for r in (args.image_run, args.text_run, rd))
    y = f2["y_true"].to_numpy()
    a_ok, b_ok = ua["y_pred"].to_numpy() == y, ub["y_pred"].to_numpy() == y
    group = np.where(a_ok & b_ok, "support", np.where(a_ok, "conflict_I", np.where(b_ok, "conflict_T", "both_wrong")))
    rng = np.random.default_rng(42)
    chosen = {}
    for gname in GROUPS:
        pool = np.flatnonzero(group == gname)
        sel = np.sort(rng.choice(pool, size=min(args.per_group, len(pool)), replace=False))
        chosen[gname] = [ids[i] for i in sel]
        if not len(sel):
            continue
        ncols = 3
        nrows = int(np.ceil(len(sel) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 5.6), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, sel):
            cam, cls, words, w_att, s_img, s_txt = gradcam_and_words(model, test_ds, int(i), device, tok)
            ax.imshow(test_ds.display_image(int(i)))
            ax.imshow(cam, cmap="jet", alpha=0.4)
            top = [words[j] for j in np.argsort(-w_att)[:3]] if len(words) else []
            mark = lambda ok: "right" if ok else "wrong"
            lines = [f"true: {classes[y[i]]}",
                     f"UA (image): {classes[ua['y_pred'].iloc[i]]} [{mark(a_ok[i])}]",
                     f"UB (text): {classes[ub['y_pred'].iloc[i]]} [{mark(b_ok[i])}]",
                     f"F2: {classes[cls]} [{mark(cls == y[i])}]",
                     f"[FUS] attention: image {s_img:.2f} / title {s_txt:.2f}",
                     "top title words: " + ", ".join(top)]
            title = textwrap.fill('"' + test_ds.titles[int(i)][:110] + '"', 46)
            ax.set_title(title + "\n" + "\n".join(lines), fontsize=7.5, loc="left",
                         color="tab:green" if cls == y[i] else "tab:red")
        fig.suptitle(f"{GROUP_TITLE[gname]} - F2 seed 42 (Grad-CAM: F2 image encoder, predicted class)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out / f"qualitative_{gname}.png", dpi=150)
        plt.close(fig)
    save_json({"run": str(rd), "examples": chosen, "seed": 42}, out / "qualitative_examples.json")
    print(f"panels -> {out}/qualitative_*.png")


if __name__ == "__main__":
    main()
