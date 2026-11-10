#!/usr/bin/env bash
set -u

rebuttal_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rebuttal_output=${1:-"$rebuttal_repo/output/morphbench_sd3"}
rebuttal_log_dir="$rebuttal_output/logs"

cd "$rebuttal_repo" || exit 1
mkdir -p "$rebuttal_log_dir"

while tmux has-session -t morphbench90 2>/dev/null; do
    sleep 30
done

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}" \
.venv/bin/python scripts/eval_metrics.py \
    --out_root "$rebuttal_output" \
    --methods gfm slerp \
    --subsets Metamorphosis Animation \
    --device cuda:0 \
    --expected_frames 17 \
    --bootstrap_repeats 10000 \
    >"$rebuttal_log_dir/lpips.log" 2>&1
rebuttal_lpips_status=$?

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}" \
.venv/bin/python scripts/eval_fid.py \
    --out_root "$rebuttal_output" \
    --methods gfm slerp \
    --subsets Metamorphosis Animation \
    --device cuda:0 \
    --expected_frames 17 \
    --batch_size 64 \
    --num_workers 4 \
    --dims 2048 \
    --fid_repeats 10 \
    >"$rebuttal_log_dir/fid.log" 2>&1
rebuttal_fid_status=$?

if [[ "$rebuttal_lpips_status" -ne 0 || "$rebuttal_fid_status" -ne 0 ]]; then
    exit 1
fi
