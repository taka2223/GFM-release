#!/bin/bash

# Noise levels for ablation
NOISE_LEVELS=(0.5 0.4 0.3 0.2 0.1 0.05)

# Common arguments
COMMON_ARGS="--dataset shapenet --category 5 --seed 9999 --lr 0.01"

# ========== Noise-unrelated methods (run once) ==========
echo "Running RBF (noise-unrelated)..."
python pair4vis.py $COMMON_ARGS --approximator rbf --noise_level 0.5 --h_path output/kernel/shapenet/rbf/h.pth

echo "Running LAND (noise-unrelated)..."
python pair4vis.py $COMMON_ARGS --approximator land --noise_level 0.5 --h_path output/kernel/shapenet/land/h.pth


# ========== Noise-related methods (ablation over noise levels) ==========
for sigma in "${NOISE_LEVELS[@]}"; do
    echo "Running Score with noise_level=$sigma..."
    python pair4vis.py $COMMON_ARGS --approximator score --noise_level $sigma

    echo "Running EL with noise_level=$sigma..."
    python pair4vis.py $COMMON_ARGS --approximator el --noise_level $sigma

    echo "Running Stein with noise_level=$sigma..."
    python pair4vis.py $COMMON_ARGS --approximator stein --noise_level $sigma
done

echo "All experiments completed!"