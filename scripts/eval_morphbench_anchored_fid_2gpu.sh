#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
EVAL_PACKAGES="${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}"
export PYTHONPATH="${EVAL_PACKAGES}"

run_fid() {
    local gpu="$1"
    local root="$2"
    mkdir -p "${root}/eval_logs"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_fid.py \
        --out_root "${root}" \
        --methods gfm_c2f slerp \
        --device cuda \
        --batch_size 64 \
        > "${root}/eval_logs/fid.log" 2>&1
}

run_fid 0 output/morphbench_anchored_c2f_sigma0p5_lambda20 &
pid0=$!
run_fid 1 output/morphbench_anchored_c2f_sigma0p6_lambda10 &
pid1=$!

wait "${pid0}"
wait "${pid1}"

echo "Both anchored C2F FID evaluations are complete."
