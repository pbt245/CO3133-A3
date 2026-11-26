#!/usr/bin/env bash
# The whole M3 pipeline, meant to run inside tmux (started by: bash scripts/remote/launch.sh m3). Safe to re-run at
# any time: finished runs are skipped, interrupted runs resume from checkpoints/last.pt, finished steps are skipped.
#   0. checks that the M2 runs are still here (UA / UB / F1 predictions, F1 checkpoints)
#   1. timing check: 200 steps of F2 (catches code / data / GPU problems in a few minutes)
#   2. F2, then the ablations (E2: UB / F1 / F2 on masked titles; E1: F2 without aux losses; E4: F2 without modality
#      dropout) x seeds 42 123 2026, with retries (a crash waits 5 min, re-picks the GPU and resumes)
#   3. F0 on masked titles, T0 TF-IDF (+ masked), per-item removal predictions of the M2 F1 runs, [FUS] attention and
#      Grad-CAM panels, the conflict-review page, comparison and analyses
#   4. results -> $A3_WORK/results/a3_results.zip (+ a3_review.zip, a3_logs.zip),
#      M3 checkpoints -> a3_checkpoints_m3.zip (+ SHA256)
# Progress: bash scripts/remote/status.sh   |   master log: $A3_WORK/logs/m3_latest.log
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"
activate_venv
LOG="$LOGS/m3_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$LOG" "$LOGS/m3_latest.log"
exec > >(tee -a "$LOG") 2>&1
STATUS="$LOGS/STATUS"
status() { echo "$(ts) $*" > "$STATUS"; say "STATUS: $*"; }

NW="${A3_WORKERS:-16}"                     # shared 64-core machine: do not take every core
SEEDS="${A3_SEEDS:-42 123 2026}"
MODELS="${A3_MODELS:-f2_crossmodal ub_distilbert_mask f1_concat_mask f2_crossmodal_mask f2_noaux f2_nomoddrop}"
RETRIES="${A3_RETRIES:-4}"
COMMON=("data.num_workers=$NW" "train.show_progress=false")
RUNS=outputs/runs/upmc
say "repo $REPO @ $(git rev-parse --short HEAD) (dirty: $(git status --porcelain --untracked-files=no | wc -l) files)"
say "models: $MODELS | seeds: $SEEDS | workers: $NW"

wait_for_gpu() {   # wait (up to A3_MAX_WAIT_H hours) until a GPU has enough free memory
  local waited=0 max=$(( ${A3_MAX_WAIT_H:-12} * 60 ))
  until pick_gpu; do
    [ "$waited" -ge "$max" ] && { status "FAILED: no free GPU for ${A3_MAX_WAIT_H:-12} h"; exit 1; }
    status "WAITING for a free GPU (checked every 10 min)"; sleep 600; waited=$((waited + 10))
  done
}
step() {   # step <name> <command...>: run once, remember success in $LOGS/.m3_<name>
  local name="$1"; shift
  if [ -f "$LOGS/.m3_$name" ]; then say "step $name already done"; return 0; fi
  status "RUNNING $name"
  if "$@"; then touch "$LOGS/.m3_$name"; else status "FAILED in $name: see $LOG"; exit 1; fi
}

# ---- 0. M2 runs needed by M3 (late fusion on masked titles, removal curves, support / conflict, qualitative)
miss=()
for s in $SEEDS; do
  for f in "ua_resnet50/seed$s/eval/predictions_val.npz" "ua_resnet50/seed$s/eval/predictions_test.npz" \
           "ua_resnet50/seed$s/eval/predictions_test_missing_text.npz" "ub_distilbert/seed$s/eval/predictions_test.csv" \
           "f1_concat/seed$s/checkpoints/best.pt"; do
    [ -f "$RUNS/$f" ] || miss+=("$f")
  done
done
if [ ${#miss[@]} -gt 0 ]; then
  say "missing M2 files: ${miss[*]}"
  status "FAILED: M2 runs missing here; run: bash scripts/remote/fetch_m2.sh, then launch again"; exit 1
fi

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv -l 300 \
  >> "$LOGS/gpu_usage.csv" 2>/dev/null &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

# ---- 1. timing check (once)
if [ ! -f "$LOGS/.m3_timing" ]; then
  wait_for_gpu
  status "RUNNING timing check on GPU $CUDA_VISIBLE_DEVICES"
  python scripts/train.py --config configs/upmc/f2_crossmodal.yaml --no-eval --overwrite \
    --set train.epochs=2 train.freeze_encoders_epochs=1 train.max_steps_per_epoch=200 experiment.name=timing_f2_crossmodal \
          experiment.group=debug "${COMMON[@]}" \
    || { status "FAILED in timing check (F2): see $LOG"; exit 1; }
  grep -h "items/s" "$RUNS/timing_f2_crossmodal/seed42/train.log" | tail -2
  touch "$LOGS/.m3_timing"
fi

# ---- 2. main runs with retries
ok=0
for attempt in $(seq 1 "$RETRIES"); do
  wait_for_gpu
  status "RUNNING M3 runs, attempt $attempt/$RETRIES on GPU $CUDA_VISIBLE_DEVICES"
  # shellcheck disable=SC2086
  if python scripts/run_experiments.py --models $MODELS --seeds $SEEDS --set "${COMMON[@]}"; then ok=1; break; fi
  say "run_experiments failed (attempt $attempt); waiting 5 min, then resuming"
  sleep 300
done
[ "$ok" = 1 ] || { status "FAILED after $RETRIES attempts: see $LOG"; exit 1; }

# ---- 3. references, extra predictions, qualitative evidence, review page, comparison
# shellcheck disable=SC2086
step f0_mask python scripts/late_fusion.py --text ub_distilbert_mask --name f0_late_fusion_mask --seeds $SEEDS
step t0 python scripts/tfidf_baseline.py
step t0_mask python scripts/tfidf_baseline.py --mask
wait_for_gpu
f1runs=(); for s in $SEEDS; do f1runs+=("$RUNS/f1_concat/seed$s"); done
step removal_f1 python scripts/removal_predictions.py "${f1runs[@]}" --set "data.num_workers=$NW"
q=(); for e in f2_crossmodal f2_noaux f2_nomoddrop f2_crossmodal_mask; do
  [ -f "$RUNS/$e/seed42/eval/test_metrics.json" ] && q+=("$RUNS/$e/seed42"); done
step qualitative python scripts/qualitative.py --runs "${q[@]}" --set "data.num_workers=$NW"
step review python scripts/conflict_review.py make --out outputs/review
status "RUNNING compare / analyses"
python scripts/compare.py > /dev/null && python scripts/analysis_m3.py > /dev/null \
  || { status "FAILED in compare / analysis: see $LOG"; exit 1; }
tail -n 40 outputs/comparison/summary.md

# ---- 4. pack
status "RUNNING packing"
for f in "$RUNS"/*/seed*/checkpoints/last.pt; do
  [ -f "$f" ] && [ -f "$(dirname "$(dirname "$f")")/eval/test_metrics.json" ] && rm -f "$f"
done
python scripts/pack_results.py --out-dir "$RESULTS"
# shellcheck disable=SC2086
python scripts/pack_results.py --checkpoints-only --experiments $MODELS --name a3_checkpoints_m3 --out-dir "$RESULTS"
(rm -f "$RESULTS/a3_review.zip" && cd outputs && zip -qr "$RESULTS/a3_review.zip" review)
mkdir -p "$RESULTS/logs" && cp "$LOG" "$LOGS/gpu_usage.csv" "$LOGS/setup_check.txt" "$RESULTS/logs/" 2>/dev/null
(cd "$RESULTS" && rm -f a3_logs.zip && zip -qr a3_logs.zip logs)
ls -lh "$RESULTS"
status "DONE: results in $RESULTS (next: A3_MILESTONE=m3 bash scripts/remote/upload_results.sh)"
