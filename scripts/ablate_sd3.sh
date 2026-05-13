#!/bin/bash
# Ablation: CFG x Caption for SD3 GFM (lambda fixed at 10).
# Outputs go to ./output/sd3_interp/ablation/<tag>/

set -e

# --- Fixed paths / config ---
IMG_A="/cns/USERS/zzhixuan/data/MorphBench/Metamorphosis/castle_0.png"
IMG_B="/cns/USERS/zzhixuan/data/MorphBench/Metamorphosis/castle_1.png"
MODEL_ID="stabilityai/stable-diffusion-3-medium-diffusers"
CACHE_DIR="/cns/USERS/zzhixuan/weights"
DEVICE="cuda"

# --- Fixed hyperparams (from best run so far) ---
LAM=10.0
LR=0.001
MAX_ITERS=800
NUM_STEPS=10
RESOLUTION=512
NOISE_LEVEL=0.6

# --- Manual prompts (only used when CAPTION_MODE=manual) ---
PROMPT_A_MANUAL="a photo of Mont Saint-Michel castle on a sunny day with blue sky"
PROMPT_B_MANUAL="a photo of Mont Saint-Michel castle at twilight with orange sky"

# --- Ablation grid ---
CFG_VALUES=(0.0 3.0 7.5)
CAPTION_MODES=("manual" "empty")

# --- Output layout ---
BASE_OUT="./output/sd3_interp/ablation"
mkdir -p "$BASE_OUT"

export CUDA_VISIBLE_DEVICES=6

for cap in "${CAPTION_MODES[@]}"; do
  for cfg in "${CFG_VALUES[@]}"; do
    TAG="cap-${cap}__cfg-${cfg}"
    OUT="${BASE_OUT}/${TAG}"

    if [ -f "${OUT}/strip.png" ]; then
      echo "[SKIP] ${TAG} (strip.png already exists)"
      continue
    fi

    mkdir -p "$OUT"
    echo ""
    echo "=================================================="
    echo "RUN: ${TAG}"
    echo "=================================================="

    ARGS=(
      --imgA "$IMG_A" --imgB "$IMG_B"
      --noise_level "$NOISE_LEVEL"
      --cfg_scale "$cfg"
      --lam "$LAM" --lr "$LR" --max_iters "$MAX_ITERS"
      --num_steps "$NUM_STEPS" --resolution "$RESOLUTION"
      --output_dir "$OUT"
      --model_id "$MODEL_ID" --cache_dir "$CACHE_DIR"
      --device "$DEVICE" --keep_t5 --no_blip
    )

    case "$cap" in
      manual)
        ARGS+=(--promptA "$PROMPT_A_MANUAL" --promptB "$PROMPT_B_MANUAL")
        ;;
      empty)
        ARGS+=(--promptA "" --promptB "")
        ;;
    esac

    python gfm/path/sd3_wrapper.py "${ARGS[@]}" 2>&1 | tee "${OUT}/log.txt"
  done
done

echo ""
echo "All runs finished. Building comparison grid..."
python scripts/build_ablation_grid.py "$BASE_OUT"
echo "Done. Grid saved to ${BASE_OUT}/grid.png"
