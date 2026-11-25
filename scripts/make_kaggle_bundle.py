#!/usr/bin/env python
"""Build the private Kaggle datasets used by the Internet-OFF notebooks.

  python scripts/make_kaggle_bundle.py code       # -> outputs/kaggle_bundle/code/     (every code change; commit first)
  python scripts/make_kaggle_bundle.py weights    # -> outputs/kaggle_bundle/weights/  (once)
  kaggle datasets create  -p outputs/kaggle_bundle/<code|weights>                      # first time (private)
  kaggle datasets version -p outputs/kaggle_bundle/code -m "what changed"              # later code updates

code    <user>/co3133-a3-code: co3133-a3-code.zip = co3133-a3/{scripts,src,configs,data/splits,data/meta,...}
        + BUNDLE_INFO (git commit, dirty flag, build time, sha256 over all bundled files). Every run records it
        in env.json, so results trace back to a commit.
weights <user>/co3133-a3-weights: co3133-a3-weights.zip = weights/resnet50-11ad3fa6.pth (torchvision
        ResNet50_Weights.IMAGENET1K_V2) + weights/distilbert-base-uncased/ (config, safetensors, tokenizer files from
        huggingface.co/distilbert/distilbert-base-uncased). Downloaded once on this (online) machine and cached in
        ./weights/ (git-ignored). src/utils/paths.py finds both under /kaggle/input.
Kaggle extracts uploaded zips; the notebook code accepts both the extracted folder and the zip.
"""
import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["scripts", "src", "configs", "data/splits", "data/meta", "data/audit", "requirements.txt", "README.md",
           "PROPOSAL.md"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}
PREFIX = "co3133-a3"
RESNET50 = ("resnet50-11ad3fa6.pth", "https://download.pytorch.org/models/resnet50-11ad3fa6.pth")
HF_REPO = "https://huggingface.co/distilbert/distilbert-base-uncased/resolve/main/"
HF_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()


def files_to_bundle():
    out = []
    for p in INCLUDE:
        q = ROOT / p
        if q.is_file():
            out.append(q)
        elif q.is_dir():
            out += [f for f in sorted(q.rglob("*")) if f.is_file() and not SKIP_PARTS & set(f.parts)]
    return out


def git_state():
    """(commit or None, dirty) for the bundled paths; (None, None) outside a git repo."""
    try:
        git("rev-parse", "--git-dir")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    try:
        commit = git("rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        return None, True
    tracked_or_new = git("status", "--porcelain", "--", *[p for p in INCLUDE if (ROOT / p).exists()])
    return commit, bool(tracked_or_new)


def metadata(out: Path, user: str, slug: str):
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}, indent=1))


def build_code(args):
    files = files_to_bundle()
    sha = hashlib.sha256()
    for f in sorted(files):
        sha.update(str(f.relative_to(ROOT)).encode())
        sha.update(f.read_bytes())
    commit, dirty = git_state()
    if commit and dirty and not args.allow_dirty:
        sys.exit("bundled files differ from HEAD (or are untracked): commit first (or pass --allow-dirty)")
    info = {"commit": commit, "dirty": dirty, "built": dt.datetime.now().isoformat(timespec="seconds"),
            "files": len(files), "sha256": sha.hexdigest()}
    out = ROOT / args.out / "code"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.zip"):
        old.unlink()
    slug = "co3133-a3-code"
    with zipfile.ZipFile(out / f"{slug}.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f"{PREFIX}/{f.relative_to(ROOT).as_posix()}")
        zf.writestr(f"{PREFIX}/BUNDLE_INFO", json.dumps(info, indent=1) + "\n")
    metadata(out, args.user, slug)
    print(json.dumps(info, indent=1))
    print(f"{len(files)} files -> {out / (slug + '.zip')} ({(out / (slug + '.zip')).stat().st_size / 1024:.0f} KB)")


def download(url: str, dst: Path):
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dst)


def build_weights(args):
    cache = ROOT / "weights"
    download(RESNET50[1], cache / RESNET50[0])
    for f in HF_FILES:
        download(HF_REPO + f, cache / "distilbert-base-uncased" / f)
    out = ROOT / args.out / "weights"
    out.mkdir(parents=True, exist_ok=True)
    slug = "co3133-a3-weights"
    manifest = []
    with zipfile.ZipFile(out / f"{slug}.zip", "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(cache.rglob("*")):
            if f.is_file() and not f.name.endswith(".part"):
                rel = f.relative_to(ROOT).as_posix()
                manifest.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel}")
                zf.write(f, rel)
        zf.writestr("weights/SHA256SUMS", "\n".join(manifest) + "\n")
    metadata(out, args.user, slug)
    print("\n".join(manifest))
    print(f"-> {out / (slug + '.zip')} ({(out / (slug + '.zip')).stat().st_size / 1024**2:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="code", choices=["code", "weights"])
    ap.add_argument("--user", default="bathanhphan", help="Kaggle username (dataset owner)")
    ap.add_argument("--out", default="outputs/kaggle_bundle")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    build_code(args) if args.what == "code" else build_weights(args)


if __name__ == "__main__":
    main()
