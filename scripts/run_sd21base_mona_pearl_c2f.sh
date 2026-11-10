#!/usr/bin/env bash
set -euo pipefail

out_root="${SD21_MONA_OUT_ROOT:-output/morphbench_sd21base_smoke_current17/mona_pearl_pretest}"
caption_root="${SD21_CAPTION_ROOT:-output/morphbench_sd3/captions}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
revision="4e63672c03103b6c636b8fb4119ba982469b2955"
noise_level="${SD21_NOISE_LEVEL:-0.5}"
read -r -a c2f_lrs <<< "${SD21_C2F_LRS:-0.002 0.001 0.0005 0.00025}"
if [[ "${#c2f_lrs[@]}" -ne 4 ]]; then
  echo "SD21_C2F_LRS must contain exactly four learning rates" >&2
  exit 2
fi

mkdir -p "${out_root}/logs"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${python_bin}" \
  scripts/run_morphbench.py \
  --backbone sd21 \
  --model_id sd2-community/stable-diffusion-2-1-base \
  --revision "${revision}" \
  --data_root data/MorphBench \
  --subsets Metamorphosis \
  --pair_names mona_pearl \
  --out_root "${out_root}" \
  --caption_root "${caption_root}" \
  --methods gfm_c2f \
  --resolution 512 \
  --n_inference_steps 50 \
  --num_steps 17 \
  --noise_level "${noise_level}" \
  --cfg_scale 0 \
  --lam 20 \
  --lr 0.001 \
  --max_iters 400 \
  --c2f_steps 3 5 9 17 \
  --c2f_iters 800 800 80 40 \
  --c2f_lrs "${c2f_lrs[@]}" \
  --c2f_freeze_anchors \
  2>&1 | tee "${out_root}/logs/generation.log"

"${python_bin}" scripts/validate_sd21base_smoke.py \
  --out_root "${out_root}" \
  --methods gfm_c2f \
  --pairs mona_pearl \
  --subset Metamorphosis \
  --input_root data/MorphBench \
  --log "${out_root}/logs/generation.log" \
  --require_model_revision "${revision}" \
  --require_c2f
