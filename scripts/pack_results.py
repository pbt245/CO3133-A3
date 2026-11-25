#!/usr/bin/env python
"""Zip the comparison and run folders to move results from Kaggle to the repo.

Usage:  python scripts/pack_results.py --out-dir /kaggle/working            # -> a3_results.zip
        python scripts/pack_results.py --checkpoints-only --out-dir /kaggle/working
Output: <out-dir>/a3_results.zip  (paths relative to the repo root: unzip at the repo root)
        <out-dir>/a3_checkpoints_upmc.zip + CHECKPOINTS_SHA256.txt (best.pt of every finished run, timing runs excluded)
Checkpoints are excluded from a3_results.zip (best.pt: UA ~95 MB, UB ~255 MB, F1 ~365 MB).
"""
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def pack_checkpoints(out_dir: Path):
    """best.pt of every finished run, stored as outputs/runs/upmc/<experiment>/seed<seed>/checkpoints/best.pt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in (OUT / "runs").glob("*/*/seed*/checkpoints/best.pt")
                   if not f.parts[-4].startswith("timing_") and (f.parents[1] / "eval" / "test_metrics.json").exists())
    bundle = out_dir / "a3_checkpoints_upmc.zip"
    lines = []
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as zf:      # weights do not compress
        for f in files:
            rel = f.relative_to(ROOT).as_posix()
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            lines.append(f"{h}  {rel}")
            zf.write(f, rel)
        manifest = "\n".join(lines) + "\n"
        zf.writestr("CHECKPOINTS_SHA256.txt", manifest)
    (out_dir / "CHECKPOINTS_SHA256.txt").write_text(manifest)
    print(f"{len(files)} checkpoints -> {bundle} ({bundle.stat().st_size / 1024**2:.0f} MB)\n"
          f"Unzip at the repo root; check with: sha256sum -c CHECKPOINTS_SHA256.txt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--with-checkpoints", choices=["none", "best", "all"], default="none")
    p.add_argument("--checkpoints-only", action="store_true")
    p.add_argument("--out-dir", default=str(OUT))
    args = p.parse_args()
    if args.checkpoints_only:
        return pack_checkpoints(Path(args.out_dir))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "a3_results.zip"
    n_files = 0
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for root in [OUT / "comparison", OUT / "runs"]:
            if not root.exists():
                print(f"(missing) {root.relative_to(ROOT)}")
                continue
            for f in root.rglob("*"):
                if f.is_dir() or f.suffix == ".zip":
                    continue
                if "checkpoints" in f.parts and not (args.with_checkpoints == "all" or
                                                     (args.with_checkpoints == "best" and f.name == "best.pt")):
                    continue
                zf.write(f, f.relative_to(ROOT))
                n_files += 1
    print(f"{n_files} files -> {bundle} ({bundle.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
