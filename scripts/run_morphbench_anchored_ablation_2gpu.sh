#!/usr/bin/env bash
set -euo pipefail

ablation_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ablation_cache=${MORPHBENCH_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}
ablation_caption_root="$ablation_repo/output/morphbench_c2f_sigma0p6_lambda20/captions"
ablation_pythonpath=${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}
ablation_mplconfig="$ablation_repo/.rebuttal-runtime/matplotlib"

cd "$ablation_repo"
mkdir -p "$ablation_mplconfig"

run_ablation() {
    local gpu_id=$1
    local sigma=$2
    local lambda=$3
    local output_name=$4
    local output_root="$ablation_repo/output/$output_name"
    local log_dir="$output_root/logs"

    mkdir -p "$log_dir"
    env \
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="$ablation_pythonpath" \
        MPLCONFIGDIR="$ablation_mplconfig" \
        HF_HOME="$ablation_cache" \
        .venv/bin/python scripts/run_morphbench.py \
            --data_root data/MorphBench \
            --subsets Metamorphosis Animation \
            --out_root "$output_root" \
            --caption_root "$ablation_caption_root" \
            --cache_dir "$ablation_cache" \
            --caption_model_id ./BLIP_large \
            --device cuda:0 \
            --methods gfm_c2f slerp \
            --num_shards 1 \
            --shard_id 0 \
            --num_steps 17 \
            --n_inference_steps 50 \
            --resolution 512 \
            --cfg_scale 0 \
            --noise_level "$sigma" \
            --lam "$lambda" \
            --c2f_steps 3 5 9 17 \
            --c2f_iters 800 800 80 40 \
            --c2f_lrs 0.002 0.001 0.0005 0.00025 \
            --c2f_freeze_anchors \
            > "$log_dir/run.log" 2>&1
}

run_ablation \
    0 \
    0.5 \
    20 \
    morphbench_anchored_c2f_sigma0p5_lambda20 &
ablation_pid0=$!

run_ablation \
    1 \
    0.6 \
    10 \
    morphbench_anchored_c2f_sigma0p6_lambda10 &
ablation_pid1=$!

ablation_status=0
wait "$ablation_pid0" || ablation_status=$?
wait "$ablation_pid1" || ablation_status=$?
exit "$ablation_status"
