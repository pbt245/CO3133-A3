#!/usr/bin/env bash
# Only needed if the M2 run folders are no longer on this machine: downloads the M2 results (with prediction .npz)
# and the M2 checkpoints from the private Kaggle datasets and unzips them at the repo root.
#   bash scripts/remote/fetch_m2.sh
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"
activate_venv
kaggle_login
d="$TMPDIR/m2"; rm -rf "$d"; mkdir -p "$d"
for slug in co3133-a3-m2-results co3133-a3-m2-checkpoints; do
  say "downloading $slug"
  kaggle datasets download "$KAGGLE_USER/$slug" -p "$d/$slug" --unzip
done
unset KAGGLE_API_TOKEN KAGGLE_KEY KAGGLE_USERNAME
unzip -qo "$d/co3133-a3-m2-results/a3_results.zip" -d "$REPO"
unzip -qo "$d/co3133-a3-m2-checkpoints/a3_checkpoints_upmc.zip" -d "$REPO"
rm -rf "$d"
say "M2 runs restored under outputs/runs/upmc. Next: bash scripts/remote/launch.sh m3"
