#!/usr/bin/env bash
# Upload the M2 results to private Kaggle datasets (this machine has no file transfer to the laptop):
#   bathanhphan/co3133-a3-m2-results      a3_results.zip + a3_logs.zip   (metrics, predictions, figures, run logs)
#   bathanhphan/co3133-a3-m2-checkpoints  a3_checkpoints_upmc.zip + CHECKPOINTS_SHA256.txt (best.pt of every run)
# The Kaggle token is asked (hidden) and kept in memory only.   bash scripts/remote/upload_results.sh
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv
[ -f "$RESULTS/a3_results.zip" ] || { echo "no $RESULTS/a3_results.zip yet (run_m2.sh not finished?)"; exit 1; }
kaggle_login
upload() {  # slug, files...
  local slug="$1"; shift
  local d="$RESULTS/upload_$slug"; rm -rf "$d"; mkdir -p "$d"
  for f in "$@"; do ln "$f" "$d/" 2>/dev/null || cp "$f" "$d/"; done
  printf '{"title": "%s", "id": "%s/%s", "licenses": [{"name": "other"}]}\n' "$slug" "$KAGGLE_USER" "$slug" > "$d/dataset-metadata.json"
  say "uploading $slug"
  kaggle datasets version -p "$d" -m "M2 results from lab A6000 $(date +%F)" 2>/dev/null || kaggle datasets create -p "$d"
  rm -rf "$d"
}
upload co3133-a3-m2-results "$RESULTS/a3_results.zip" "$RESULTS/a3_logs.zip"
upload co3133-a3-m2-checkpoints "$RESULTS/a3_checkpoints_upmc.zip" "$RESULTS/CHECKPOINTS_SHA256.txt"
unset KAGGLE_API_TOKEN KAGGLE_KEY KAGGLE_USERNAME
say "done. On the laptop Claude downloads: kaggle datasets download $KAGGLE_USER/co3133-a3-m2-results"
