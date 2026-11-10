#!/usr/bin/env bash
set -u

rebuttal_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rebuttal_output=${1:-"$rebuttal_repo/output/morphbench_sd3"}
rebuttal_cache=${MORPHBENCH_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}
rebuttal_lambda=${MORPHBENCH_LAMBDA:-10}
rebuttal_log_dir="$rebuttal_output/logs"

cd "$rebuttal_repo" || exit 1
mkdir -p "$rebuttal_log_dir"

run_shard() {
    local gpu_id=$1
    local shard_id=$2
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    PYTHONPATH="${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}" \
    HF_HOME="$rebuttal_cache" \
    .venv/bin/python scripts/run_morphbench.py \
        --data_root data/MorphBench \
        --out_root "$rebuttal_output" \
        --cache_dir "$rebuttal_cache" \
        --caption_model_id ./BLIP_large \
        --device cuda:0 \
        --methods gfm slerp \
        --num_shards 2 \
        --shard_id "$shard_id" \
        --num_steps 17 \
        --n_inference_steps 50 \
        --resolution 512 \
        --noise_level 0.6 \
        --cfg_scale 0 \
        --lam "$rebuttal_lambda" \
        --lr 0.001 \
        --max_iters 400 \
        >"$rebuttal_log_dir/shard_${shard_id}.log" 2>&1
}

run_shard 0 0 &
rebuttal_pid_0=$!
run_shard 1 1 &
rebuttal_pid_1=$!

rebuttal_status=0
wait "$rebuttal_pid_0" || rebuttal_status=$?
wait "$rebuttal_pid_1" || rebuttal_status=$?
exit "$rebuttal_status"
