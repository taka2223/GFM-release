#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
DINO_MODEL="${DINO_MODEL:-facebook/dinov3-vitl16-pretrain-lvd1689m}"
EVAL_PACKAGES="${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}"
export PYTHONPATH="${EVAL_PACKAGES}"

run_suite() {
    local gpu="$1"
    local root="$2"
    local log_dir="${root}/eval_logs"

    mkdir -p "${log_dir}"

    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_iqa.py \
        --out_root "${root}" \
        --methods gfm_c2f slerp \
        --batch_size 16 \
        > "${log_dir}/musiq.log" 2>&1

    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_pairwise_dinov3.py \
        --out_root "${root}" \
        --methods gfm_c2f slerp \
        --model_path "${DINO_MODEL}" \
        --batch_size 64 \
        --dtype bfloat16 \
        > "${log_dir}/dinov3.log" 2>&1

    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_metrics.py \
        --out_root "${root}" \
        --methods gfm_c2f slerp \
        --device cuda \
        > "${log_dir}/lpips.log" 2>&1

    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_fid.py \
        --out_root "${root}" \
        --methods gfm_c2f slerp \
        --device cuda \
        --batch_size 64 \
        > "${log_dir}/fid.log" 2>&1

    printf 'Evaluation complete: %s\n' "${root}" > "${log_dir}/complete"
}

run_suite 0 output/morphbench_anchored_c2f_sigma0p5_lambda20 &
pid0=$!
run_suite 1 output/morphbench_anchored_c2f_sigma0p6_lambda10 &
pid1=$!

wait "${pid0}"
wait "${pid1}"

echo "Both anchored C2F evaluation suites are complete."
