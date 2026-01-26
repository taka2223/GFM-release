#!/bin/bash
# Batch evaluation script for all methods and categories

# ============ Configuration ============
GPU_ID=0
REF_DIR="./data/shapenet"
BASE_DIR="./output/shapenet"
BATCH_SIZE=256
STEP_FILTER="step[3-6]"
THRESHOLDS="0.05 0.1 0.15 0.2"

# Methods to evaluate
METHODS=("score" "el" "stein" "land" "rbf")

# Categories to evaluate (add more as needed)
CATEGORIES=("47" "5")

# ============ Run Evaluation ============
for method in "${METHODS[@]}"; do
    for cat in "${CATEGORIES[@]}"; do
        GEN_DIR="${BASE_DIR}/${method}/test_denoise_sigma0.3_category${cat}_seed99995_pairs200"

        echo "========================================"
        echo "Evaluating: ${method} / category ${cat}"
        echo "Gen dir: ${GEN_DIR}"
        echo "========================================"

        # Check if directory exists
        if [ ! -d "$GEN_DIR" ]; then
            echo "WARNING: Directory not found, skipping..."
            echo ""
            continue
        fi

        CUDA_VISIBLE_DEVICES=$GPU_ID python dist4eval.py \
            --ref_dir "$REF_DIR" \
            --gen_dir "$GEN_DIR" \
            --batch_size $BATCH_SIZE \
            --step_filter "$STEP_FILTER" \
            --thresholds $THRESHOLDS

        echo ""
    done
done

echo "========================================"
echo "All evaluations complete!"
echo "========================================"
