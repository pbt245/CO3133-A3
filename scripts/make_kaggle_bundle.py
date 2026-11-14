#!/usr/bin/env python
"""Build the private Kaggle dataset `<user>/co3133-a3-code` (the repo as one zip) for Internet-OFF notebooks.

  python scripts/make_kaggle_bundle.py                       # -> outputs/kaggle_bundle/code/
  kaggle datasets create  -p outputs/kaggle_bundle/code      # first time (private by default)
  kaggle datasets version -p outputs/kaggle_bundle/code -m "what changed"   # every later update

The zip holds co3133-a3/{scripts,src,configs,data/splits,...} plus BUNDLE_INFO (git commit + dirty flag, a timestamp and a
sha256 over all bundled files), so every Kaggle run can record exactly
which code it ran. Kaggle may or may not extract the zip on upload; the notebook setup cell accepts both.
Pretrained weights (ResNet-50, DistilBERT) will be added to this bundle in M2, when training code exists.
"""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["scripts", "src", "configs", "data/splits", "data/meta", "requirements.txt", "README.md", "PROPOSAL.md"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}
PREFIX = "co3133-a3"


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
        return None, True  # repo without any commit yet
    dirty = bool(git("status", "--porcelain", "--", *[p for p in INCLUDE if (ROOT / p).exists()]))
    return commit, dirty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="bathanhphan", help="Kaggle username (dataset owner)")
    ap.add_argument("--slug", default="co3133-a3-code")
    ap.add_argument("--out", default="outputs/kaggle_bundle/code")
    ap.add_argument("--allow-dirty", action="store_true", help="git repo with uncommitted changes")
    args = ap.parse_args()

    files = files_to_bundle()
    if not files:
        sys.exit("nothing to bundle")
    sha = hashlib.sha256()
    for f in sorted(files):
        sha.update(str(f.relative_to(ROOT)).encode())
        sha.update(f.read_bytes())
    info = {"built": dt.datetime.now().isoformat(timespec="seconds"), "files": len(files),
            "content_sha256": sha.hexdigest()}
    commit, dirty = git_state()
    if commit and dirty and not args.allow_dirty:
        sys.exit("bundled files differ from HEAD: commit first (or pass --allow-dirty)")
    if commit is None:
        print("WARNING: no git commit yet -> BUNDLE_INFO records only the content sha256", file=sys.stderr)
    info["git_commit"], info["dirty"] = commit, dirty

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.zip"):
        old.unlink()
    with zipfile.ZipFile(out / f"{args.slug}.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f"{PREFIX}/{f.relative_to(ROOT).as_posix()}")
        zf.writestr(f"{PREFIX}/BUNDLE_INFO", json.dumps(info, indent=1) + "\n")
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": args.slug, "id": f"{args.user}/{args.slug}", "licenses": [{"name": "other"}]}, indent=1))
    print(json.dumps(info, indent=1))
    print(f"-> {out / (args.slug + '.zip')}  ({(out / (args.slug + '.zip')).stat().st_size / 1024:.0f} KB)")
    for f in files:
        print("  ", f.relative_to(ROOT))


if __name__ == "__main__":
    main()
