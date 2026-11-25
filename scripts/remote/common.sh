# Shared settings for the scripts in scripts/remote/ (sourced, not run).
# Everything lives under $A3_WORK (default ~/co3133_thanh): repo, venv, data, weights, logs, results.
# Nothing is written to /tmp (the root disk of the lab machine is full) or to the shared conda base env.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
A3_WORK="${A3_WORK:-$(dirname "$REPO")}"
VENV="$A3_WORK/venv"
DATA="$A3_WORK/data"
LOGS="$A3_WORK/logs"
RESULTS="$A3_WORK/results"
SESSION="${A3_SESSION:-thanh_a3}"          # tmux session name (the account is shared: keep it distinctive)
KAGGLE_USER="${KAGGLE_USER:-bathanhphan}"

export TMPDIR="$A3_WORK/tmp"
export UPMC_ROOT="$DATA/upmc_texts"          # contains texts/{train,test}_titles.csv
export UPMC_IMAGE_ROOT="$DATA/upmc256"       # contains images/{train,test}/<label>/<id>.jpg (pre-resized, 256 px)
export A3_WEIGHTS_DIR="$A3_WORK/weights"     # resnet50-11ad3fa6.pth + distilbert-base-uncased/
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # CUDA GPU indices = nvidia-smi indices
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$TMPDIR" "$DATA" "$LOGS" "$RESULTS"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*"; }

activate_venv() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || { echo "venv missing: run bash scripts/remote/setup.sh first"; exit 1; }
}

# Ask for the Kaggle API token without echoing it and without writing it to disk (exported to this process only).
kaggle_login() {
  if [ -n "${KAGGLE_API_TOKEN:-}${KAGGLE_KEY:-}" ]; then return; fi
  read -r -s -p "Paste your Kaggle API token (input hidden), then Enter: " tok; echo
  if [[ "$tok" == KGAT_* ]]; then export KAGGLE_API_TOKEN="$tok"; else export KAGGLE_USERNAME="$KAGGLE_USER" KAGGLE_KEY="$tok"; fi
  unset tok
}

# Choose the GPU with the most free memory; refuse if it has less than A3_MIN_FREE_GB. A3_GPU forces an index.
pick_gpu() {
  local min_mb=$(( ${A3_MIN_FREE_GB:-16} * 1024 ))
  nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits \
    | sed 's/ //g' > "$TMPDIR/gpus.csv"
  say "GPUs (index,free MiB,used MiB,util %,temp C):"; sed 's/^/    /' "$TMPDIR/gpus.csv"
  local best
  if [ -n "${A3_GPU:-}" ]; then best="$A3_GPU"
  else best=$(sort -t, -k2,2nr -k4,4n "$TMPDIR/gpus.csv" | head -1 | cut -d, -f1); fi
  local free; free=$(awk -F, -v g="$best" '$1==g{print $2}' "$TMPDIR/gpus.csv")
  if [ -z "$free" ] || [ "$free" -lt "$min_mb" ]; then
    say "GPU $best has only ${free:-?} MiB free (< $min_mb). Not starting. Set A3_MIN_FREE_GB or A3_GPU to override."
    return 1
  fi
  export CUDA_VISIBLE_DEVICES="$best"
  say "using GPU $best (${free} MiB free)"
}
