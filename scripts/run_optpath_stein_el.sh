#!/bin/bash

# Categories to run
CATEGORIES=(47)

# Fixed noise level
NOISE_LEVEL=0.3

# Common arguments
COMMON_ARGS="--dataset shapenet --seed 99995 --lr 0.005 --num_pairs 200 --max_iters 500 --noise_level $NOISE_LEVEL --output_dir ./output/shapenet"

for category in "${CATEGORIES[@]}"; do
    echo "=========================================="
    echo "Running experiments for category=$category"
    echo "=========================================="

    echo "Running EL (category=$category)..."
    python optim_path.py $COMMON_ARGS --category $category --approximator el --batch_size 2

    # echo "Running Stein (category=$category)..."
    # python optim_path.py $COMMON_ARGS --category $category --approximator stein --batch_size 2
done

echo "All experiments completed!"