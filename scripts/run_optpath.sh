#!/bin/bash

# Categories to run
CATEGORIES=(47 5)

# Fixed noise level
NOISE_LEVEL=0.3

# Common arguments
COMMON_ARGS="--dataset shapenet --seed 99995 --lr 0.01 --num_pairs 200 --max_iters 500 --noise_level $NOISE_LEVEL --output_dir ./output/shapenet"

for category in "${CATEGORIES[@]}"; do
    echo "=========================================="
    echo "Running experiments for category=$category"
    echo "=========================================="

    # ========== Noise-unrelated methods ==========
    echo "Running RBF (category=$category)..."
    python optim_path.py $COMMON_ARGS --category $category --approximator rbf --batch_size 256 --h_path output/kernel/shapenet/rbf/h.pth

    echo "Running LAND (category=$category)..."
    python optim_path.py $COMMON_ARGS --category $category --approximator land --batch_size 32 --h_path output/kernel/shapenet/land/h.pth

    # ========== Noise-related methods ==========
    echo "Running Score (category=$category)..."
    python optim_path.py $COMMON_ARGS --category $category --approximator score --batch_size 2

    echo "Running EL (category=$category)..."
    python optim_path.py $COMMON_ARGS --category $category --approximator el --batch_size 16

    echo "Running Stein (category=$category)..."
    python optim_path.py $COMMON_ARGS --category $category --approximator stein --batch_size 16
done

echo "All experiments completed!"
