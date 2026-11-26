#!/usr/bin/env bash
# Upload the results to private Kaggle datasets (this machine has no file transfer to the laptop):
#   M2 (default):  bathanhphan/co3133-a3-m2-results      a3_results.zip + a3_logs.zip
#                  bathanhphan/co3133-a3-m2-checkpoints  a3_checkpoints_upmc.zip + CHECKPOINTS_SHA256.txt
#   M3:            bathanhphan/co3133-a3-m3-results      a3_results.zip + a3_logs.zip + a3_review.zip
#                  bathanhphan/co3133-a3-m3-checkpoints  a3_checkpoints_m3.zip + a3_checkpoints_m3_SHA256.txt
# The Kaggle token is asked (hidden) and kept in memory only.
#   A3_MILESTONE=m3 bash scripts/remote/upload_results.sh
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv
MS="${A3_MILESTONE:-m2}"
[ -f "$RESULTS/a3_results.zip" ] || { echo "no $RESULTS/a3_results.zip yet (run_$MS.sh not finished?)"; exit 1; }
if [ "$MS" = m3 ] && [ ! -f "$RESULTS/a3_checkpoints_m3.zip" ]; then echo "no a3_checkpoints_m3.zip yet"; exit 1; fi
kaggle_login
upload() {  # slug, files...
  local slug="$1"; shift
  local d="$RESULTS/upload_$slug"; rm -rf "$d"; mkdir -p "$d"
  for f in "$@"; do ln "$f" "$d/" 2>/dev/null || cp "$f" "$d/"; done
  printf '{"title": "%s", "id": "%s/%s", "licenses": [{"name": "other"}]}\n' "$slug" "$KAGGLE_USER" "$slug" > "$d/dataset-metadata.json"
  say "uploading $slug"
  kaggle datasets version -p "$d" -m "${MS^^} results from lab A6000 $(date +%F)" 2>/dev/null || kaggle datasets create -p "$d"
  rm -rf "$d"
}
if [ "$MS" = m3 ]; then
  upload co3133-a3-m3-results "$RESULTS/a3_results.zip" "$RESULTS/a3_logs.zip" "$RESULTS/a3_review.zip"
  upload co3133-a3-m3-checkpoints "$RESULTS/a3_checkpoints_m3.zip" "$RESULTS/a3_checkpoints_m3_SHA256.txt"
else
  upload co3133-a3-m2-results "$RESULTS/a3_results.zip" "$RESULTS/a3_logs.zip"
  upload co3133-a3-m2-checkpoints "$RESULTS/a3_checkpoints_upmc.zip" "$RESULTS/CHECKPOINTS_SHA256.txt"
fi
unset KAGGLE_API_TOKEN KAGGLE_KEY KAGGLE_USERNAME
say "done. On the laptop Claude downloads: kaggle datasets download $KAGGLE_USER/co3133-a3-$MS-results"
