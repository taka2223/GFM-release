#!/usr/bin/env bash
set -euo pipefail

eval_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval_root="$eval_repo/output/morphbench_c2f_sigma0p6_lambda20"
eval_log_dir="$eval_root/eval_logs"

cd "$eval_repo"
mkdir -p "$eval_log_dir"

env \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}" \
    .venv/bin/python scripts/eval_metrics.py \
        --out_root "$eval_root" \
        --methods gfm_c2f slerp \
        --subsets Metamorphosis Animation \
        --device cuda:0 \
        --lpips_backbone alex \
        --expected_frames 17 \
        --bootstrap_repeats 10000 \
        --seed 20260727 \
        > "$eval_log_dir/lpips.log" 2>&1 &
eval_lpips_pid=$!

env \
    CUDA_VISIBLE_DEVICES=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}" \
    .venv/bin/python scripts/eval_fid.py \
        --out_root "$eval_root" \
        --methods gfm_c2f slerp \
        --subsets Metamorphosis Animation \
        --device cuda:0 \
        --batch_size 64 \
        --num_workers 4 \
        --dims 2048 \
        --fid_repeats 10 \
        --seed 20260727 \
        > "$eval_log_dir/fid.log" 2>&1 &
eval_fid_pid=$!

eval_status=0
wait "$eval_lpips_pid" || eval_status=$?
wait "$eval_fid_pid" || eval_status=$?
exit "$eval_status"
