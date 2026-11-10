#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$repo_root/output/morphbench_ddim_fullnoise100_benchmark"
fullnoise_root="$repo_root/output/morphbench_ddim_slerp_t1_steps100/slerp"
partial_root="$repo_root/output/morphbench_anchored_c2f_sigma0p6_lambda10/slerp"
runtime_packages=${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}
dino_model=${DINO_MODEL:-facebook/dinov3-vitl16-pretrain-lvd1689m}

cd "$repo_root"
mkdir -p "$benchmark_root/eval_logs"
ln -sfn "$fullnoise_root" "$benchmark_root/fullnoise_slerp"
ln -sfn "$partial_root" "$benchmark_root/partial_slerp"
export PYTHONPATH="$runtime_packages${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/eval_iqa.py \
    --out_root "$benchmark_root" \
    --methods fullnoise_slerp partial_slerp \
    --batch_size 16 \
    > "$benchmark_root/eval_logs/musiq.log" 2>&1

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/eval_pairwise_dinov3.py \
    --out_root "$benchmark_root" \
    --methods fullnoise_slerp partial_slerp \
    --model_path "$dino_model" \
    --batch_size 64 \
    --dtype bfloat16 \
    > "$benchmark_root/eval_logs/dinov3.log" 2>&1

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/eval_metrics.py \
    --out_root "$benchmark_root" \
    --methods fullnoise_slerp partial_slerp \
    --device cuda \
    > "$benchmark_root/eval_logs/lpips.log" 2>&1

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/eval_fid.py \
    --out_root "$benchmark_root" \
    --methods fullnoise_slerp partial_slerp \
    --device cuda \
    --batch_size 64 \
    > "$benchmark_root/eval_logs/fid.log" 2>&1

echo "Full-noise benchmark evaluation complete."
