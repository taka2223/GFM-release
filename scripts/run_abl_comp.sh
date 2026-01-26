#!/bin/bash

# Noise levels for ablation
NOISE_LEVELS=(0.15)

# Common arguments
COMMON_ARGS="--dataset shapenet --category 0 --seed 89999 --lr 0.01"


# ========== Noise-related methods (ablation over noise levels) ==========
for sigma in "${NOISE_LEVELS[@]}"; do
    echo "Running Linear with noise_level=$sigma..."
    python pair4vis.py $COMMON_ARGS --approximator linear --noise_level $sigma

    echo "Running EL2 with noise_level=$sigma..."
    python pair4vis.py $COMMON_ARGS --approximator el2 --noise_level $sigma

    echo "Running EL with noise_level=$sigma..."
    python pair4vis.py $COMMON_ARGS --approximator el --noise_level $sigma

    echo "Running Spherical with noise_level=$sigma..."
    python pair4vis.py $COMMON_ARGS --approximator spherical --noise_level $sigma
done

echo "All experiments completed!"