#!/usr/bin/env bash
set -euo pipefail

full_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
full_cache=${MORPHBENCH_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}
full_output="$full_repo/output/morphbench_c2f_sigma0p6_lambda20"
full_log_dir="$full_output/logs"

cd "$full_repo"
mkdir -p "$full_log_dir"

env \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}" \
    HF_HOME="$full_cache" \
    .venv/bin/python scripts/run_morphbench.py \
        --data_root data/MorphBench \
        --subsets Metamorphosis Animation \
        --out_root "$full_output" \
        --cache_dir "$full_cache" \
        --caption_model_id ./BLIP_large \
        --device cuda:0 \
        --methods gfm_c2f slerp \
        --num_shards 2 \
        --shard_id 0 \
        --num_steps 17 \
        --n_inference_steps 50 \
        --resolution 512 \
        --cfg_scale 0 \
        --noise_level 0.6 \
        --lam 20 \
        --c2f_steps 3 5 9 17 \
        --c2f_iters 400 300 200 100 \
        --c2f_lrs 0.002 0.001 0.0005 0.00025 \
        > "$full_log_dir/shard0.log" 2>&1 &
full_pid0=$!

env \
    CUDA_VISIBLE_DEVICES=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}" \
    HF_HOME="$full_cache" \
    .venv/bin/python scripts/run_morphbench.py \
        --data_root data/MorphBench \
        --subsets Metamorphosis Animation \
        --out_root "$full_output" \
        --cache_dir "$full_cache" \
        --caption_model_id ./BLIP_large \
        --device cuda:0 \
        --methods gfm_c2f slerp \
        --num_shards 2 \
        --shard_id 1 \
        --num_steps 17 \
        --n_inference_steps 50 \
        --resolution 512 \
        --cfg_scale 0 \
        --noise_level 0.6 \
        --lam 20 \
        --c2f_steps 3 5 9 17 \
        --c2f_iters 400 300 200 100 \
        --c2f_lrs 0.002 0.001 0.0005 0.00025 \
        > "$full_log_dir/shard1.log" 2>&1 &
full_pid1=$!

full_status=0
wait "$full_pid0" || full_status=$?
wait "$full_pid1" || full_status=$?
exit "$full_status"
