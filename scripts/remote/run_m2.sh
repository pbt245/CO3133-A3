#!/usr/bin/env bash
# The whole M2 pipeline, meant to run inside tmux (started by launch.sh). Safe to re-run at any time:
# finished runs are skipped, interrupted runs resume from checkpoints/last.pt.
#   1. timing check: 200 steps of UA, UB, F1 (catches code / data / GPU problems in ~5 min)
#   2. UA, UB, F1 x seeds (42 123 2026), with retries: a crash (e.g. out of memory because another user grabbed
#      the GPU) waits 5 min, re-picks the GPU and resumes
#   3. K0 keyword rule, F0 late fusion, comparison
#   4. results -> $A3_WORK/results/a3_results.zip, checkpoints -> a3_checkpoints_upmc.zip (+ SHA256), logs
# Progress: bash scripts/remote/status.sh   |   master log: $A3_WORK/logs/m2_latest.log
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"
activate_venv
LOG="$LOGS/m2_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$LOG" "$LOGS/m2_latest.log"
exec > >(tee -a "$LOG") 2>&1
STATUS="$LOGS/STATUS"
status() { echo "$(ts) $*" > "$STATUS"; say "STATUS: $*"; }

NW="${A3_WORKERS:-16}"                     # shared 64-core machine: do not take every core
SEEDS="${A3_SEEDS:-42 123 2026}"
MODELS="${A3_MODELS:-ua_resnet50 ub_distilbert f1_concat}"
RETRIES="${A3_RETRIES:-4}"
COMMON=("data.num_workers=$NW" "train.show_progress=false")
say "repo $REPO @ $(git rev-parse --short HEAD) (dirty: $(git status --porcelain --untracked-files=no | wc -l) files)"
say "models: $MODELS | seeds: $SEEDS | workers: $NW"

wait_for_gpu() {   # wait (up to A3_MAX_WAIT_H hours) until a GPU has enough free memory
  local waited=0 max=$(( ${A3_MAX_WAIT_H:-12} * 60 ))
  until pick_gpu; do
    [ "$waited" -ge "$max" ] && { status "FAILED: no free GPU for ${A3_MAX_WAIT_H:-12} h"; exit 1; }
    status "WAITING for a free GPU (checked every 10 min)"; sleep 600; waited=$((waited + 10))
  done
}

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv -l 300 \
  >> "$LOGS/gpu_usage.csv" 2>/dev/null &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

# ---- 1. timing check (once)
if [ ! -f "$LOGS/.timing_done" ]; then
  wait_for_gpu
  status "RUNNING timing check on GPU $CUDA_VISIBLE_DEVICES"
  for m in $MODELS; do
    python scripts/train.py --config "configs/upmc/$m.yaml" --no-eval --overwrite \
      --set train.epochs=1 train.max_steps_per_epoch=200 "experiment.name=timing_$m" experiment.group=debug "${COMMON[@]}" \
      || { status "FAILED in timing check ($m): see $LOG"; exit 1; }
    grep -h "items/s" "outputs/runs/upmc/timing_$m/seed42/train.log" | tail -1
  done
  touch "$LOGS/.timing_done"
fi

# ---- 2. main runs with retries
ok=0
for attempt in $(seq 1 "$RETRIES"); do
  wait_for_gpu
  status "RUNNING main runs, attempt $attempt/$RETRIES on GPU $CUDA_VISIBLE_DEVICES"
  # shellcheck disable=SC2086
  if python scripts/run_experiments.py --models $MODELS --seeds $SEEDS --set "${COMMON[@]}"; then ok=1; break; fi
  say "run_experiments failed (attempt $attempt); waiting 5 min, then resuming"
  sleep 300
done
[ "$ok" = 1 ] || { status "FAILED after $RETRIES attempts: see $LOG"; exit 1; }

# ---- 3. K0, F0, comparison (CPU, a few minutes)
status "RUNNING K0 / F0 / compare"
# shellcheck disable=SC2086
python scripts/keyword_baseline.py && python scripts/late_fusion.py --seeds $SEEDS && python scripts/compare.py \
  || { status "FAILED in K0/F0/compare: see $LOG"; exit 1; }

# ---- 4. pack
status "RUNNING packing"
for f in outputs/runs/upmc/*/seed*/checkpoints/last.pt; do
  [ -f "$f" ] && [ -f "$(dirname "$(dirname "$f")")/eval/test_metrics.json" ] && rm -f "$f"
done
python scripts/pack_results.py --out-dir "$RESULTS"
python scripts/pack_results.py --checkpoints-only --out-dir "$RESULTS"
mkdir -p "$RESULTS/logs" && cp "$LOG" "$LOGS/gpu_usage.csv" "$LOGS/setup_check.txt" "$RESULTS/logs/" 2>/dev/null
(cd "$RESULTS" && rm -f a3_logs.zip && zip -qr a3_logs.zip logs)
ls -lh "$RESULTS"
status "DONE: results in $RESULTS (next: bash scripts/remote/upload_results.sh)"
