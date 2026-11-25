#!/usr/bin/env bash
# One-time setup on the lab machine (Ubuntu, shared account, shared GPUs). Safe to re-run: finished steps are skipped.
#
#   cd ~/co3133_thanh/co3133-a3 && bash scripts/remote/setup.sh
#
# 1. venv $A3_WORK/venv on top of the existing Python (reuses its CUDA torch; never modifies conda base)
# 2. installs only the missing packages (transformers, kaggle, ...) into that venv
# 3. downloads from Kaggle (token asked once, hidden, kept in memory only):
#      weights   bathanhphan/co3133-a3-weights        (ResNet-50 + DistilBERT, 354 MB)
#      titles    gianmarco96/upmcfood101 texts/*.csv   (2 small files)
#      images    output upmc256.zip of the notebook bathanhphan/co3133-a3-prepare (3.2 GB) -> extracted, zip deleted
# 4. checks everything (python scripts/remote/check_env.py) and writes $A3_WORK/logs/setup_check.txt
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"
say "repo $REPO | work dir $A3_WORK"
had_kaggle_dir=0; [ -d "$HOME/.kaggle" ] && had_kaggle_dir=1

# ---- 1-2. environment
if [ ! -x "$VENV/bin/python" ]; then
  BASE_PY="${A3_BASE_PYTHON:-$(command -v python3)}"
  say "creating venv with $BASE_PY (--system-site-packages: reuses its torch)"
  "$BASE_PY" -m venv --system-site-packages "$VENV"
fi
activate_venv
python - <<'PY' || { echo "torch with CUDA is required in the base Python"; exit 1; }
import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "CUDA", torch.version.cuda)
PY
missing=()
for spec in "torchvision:torchvision" "transformers:transformers" "tokenizers:tokenizers" "sklearn:scikit-learn" \
            "scipy:scipy" "pandas:pandas" "matplotlib:matplotlib" "PIL:pillow" "yaml:pyyaml" "tqdm:tqdm" "kaggle:kaggle"; do
  mod="${spec%%:*}"; pkg="${spec##*:}"
  python -c "import $mod" 2>/dev/null || missing+=("$pkg")
done
if [ ${#missing[@]} -gt 0 ]; then
  if printf '%s\n' "${missing[@]}" | grep -qx torchvision; then
    echo "torchvision is missing in the base Python: install the build that matches its torch by hand"; exit 1
  fi
  say "installing into the venv only: ${missing[*]}"
  pip install --upgrade-strategy only-if-needed "${missing[@]}"
fi
python -c "import torch; assert torch.cuda.is_available(), 'pip replaced torch?'"

# ---- 3. data and weights
need_dl=0
[ -f "$A3_WEIGHTS_DIR/resnet50-11ad3fa6.pth" ] && [ -f "$A3_WEIGHTS_DIR/distilbert-base-uncased/model.safetensors" ] || need_dl=1
[ -f "$UPMC_ROOT/texts/train_titles.csv" ] && [ -f "$UPMC_ROOT/texts/test_titles.csv" ] || need_dl=1
[ -f "$UPMC_IMAGE_ROOT/.complete" ] || need_dl=1
if [ "$need_dl" = 1 ]; then
  kaggle_login
  unzip_all() { find "$1" -name "*.zip" -print0 | while IFS= read -r -d '' z; do unzip -qo "$z" -d "$(dirname "$z")" && rm -f "$z"; done; }

  if [ ! -f "$A3_WEIGHTS_DIR/resnet50-11ad3fa6.pth" ] || [ ! -f "$A3_WEIGHTS_DIR/distilbert-base-uncased/model.safetensors" ]; then
    say "downloading weights (bathanhphan/co3133-a3-weights)"
    rm -rf "$TMPDIR/w" && mkdir -p "$TMPDIR/w"
    kaggle datasets download "$KAGGLE_USER/co3133-a3-weights" -p "$TMPDIR/w"
    unzip_all "$TMPDIR/w"; unzip_all "$TMPDIR/w"            # Kaggle may wrap our zip in its own zip
    src=$(dirname "$(find "$TMPDIR/w" -name resnet50-11ad3fa6.pth | head -1)")
    mkdir -p "$A3_WEIGHTS_DIR" && cp -r "$src"/. "$A3_WEIGHTS_DIR"/ && rm -rf "$TMPDIR/w"
  fi

  if [ ! -f "$UPMC_ROOT/texts/train_titles.csv" ] || [ ! -f "$UPMC_ROOT/texts/test_titles.csv" ]; then
    say "downloading titles (gianmarco96/upmcfood101 texts/)"
    rm -rf "$TMPDIR/t" && mkdir -p "$TMPDIR/t" "$UPMC_ROOT/texts"
    for f in train_titles.csv test_titles.csv; do
      kaggle datasets download gianmarco96/upmcfood101 -f "texts/$f" -p "$TMPDIR/t"
    done
    unzip_all "$TMPDIR/t"
    for f in train_titles.csv test_titles.csv; do cp "$(find "$TMPDIR/t" -name "$f" | head -1)" "$UPMC_ROOT/texts/$f"; done
    rm -rf "$TMPDIR/t"
  fi

  if [ ! -f "$UPMC_IMAGE_ROOT/.complete" ]; then
    say "downloading upmc256.zip (3.2 GB, output of bathanhphan/co3133-a3-prepare)"
    mkdir -p "$TMPDIR/k"
    [ -f "$TMPDIR/k/upmc256.zip" ] || kaggle kernels output "$KAGGLE_USER/co3133-a3-prepare" -p "$TMPDIR/k" --file-pattern 'upmc256\.zip'
    say "extracting to $DATA (90,704 files)"
    rm -rf "$UPMC_IMAGE_ROOT"
    unzip -q "$TMPDIR/k/upmc256.zip" -d "$DATA"
    date > "$UPMC_IMAGE_ROOT/.complete"
    rm -rf "$TMPDIR/k"
  fi
  unset KAGGLE_API_TOKEN KAGGLE_KEY KAGGLE_USERNAME
fi
if [ "$had_kaggle_dir" = 0 ] && [ -d "$HOME/.kaggle" ]; then
  say "NOTE: the kaggle CLI created $HOME/.kaggle on this shared account; contents:"; ls -la "$HOME/.kaggle"
  say "remove it with: rm -rf ~/.kaggle   (the token itself was never written by this script)"
fi

# ---- 4. checks (on the GPU with the most free memory; the other one is busy)
pick_gpu || true
python scripts/remote/check_env.py | tee "$LOGS/setup_check.txt"
say "setup finished. Next: bash scripts/remote/launch.sh"
