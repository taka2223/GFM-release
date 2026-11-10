#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 3 ]]; then
    echo "Usage: $0 [gpu_sigma05=0] [gpu_lambda20=1] [gpu_sphere=2]" >&2
    exit 2
fi

gpu_sigma05=${1:-0}
gpu_lambda20=${2:-1}
gpu_sphere=${3:-2}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_cache=${MORPHBENCH_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}
output_base="$repo_dir/output/morphbench_blur_ablation"
log_dir="$output_base/logs"

cd "$repo_dir"
mkdir -p "$log_dir"

common_args=(
    --data_root data/MorphBench
    --subsets Metamorphosis
    --pair_names boy_girl_2
    --cache_dir "$model_cache"
    --caption_model_id ./BLIP_large
    --device cuda:0
    --num_steps 17
    --n_inference_steps 50
    --resolution 512
    --lr 0.001
    --max_iters 800
    --cfg_scale 0
)

run_one() {
    local gpu_id=$1
    local run_name=$2
    local noise_level=$3
    local lam=$4
    local gfm_method=$5

    echo "Launching $run_name on physical GPU $gpu_id"
    env \
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        PYTHONPATH="${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}" \
        HF_HOME="$model_cache" \
        .venv/bin/python scripts/run_morphbench.py \
            "${common_args[@]}" \
            --out_root "$output_base/$run_name" \
            --methods "$gfm_method" slerp \
            --noise_level "$noise_level" \
            --lam "$lam" \
        > "$log_dir/$run_name.log" 2>&1 &
    launched_pid=$!
}

child_pids=()

run_one "$gpu_sigma05" sigma0p5_iter800 0.5 10 gfm
child_pids+=("$launched_pid")

run_one "$gpu_lambda20" lambda20_iter800 0.6 20 gfm
child_pids+=("$launched_pid")

run_one "$gpu_sphere" sphere_sigma0p6_iter800 0.6 10 gfm_sphere
child_pids+=("$launched_pid")

echo "Started PIDs: ${child_pids[*]}"
echo "Per-run logs: $log_dir/{sigma0p5_iter800,lambda20_iter800,sphere_sigma0p6_iter800}.log"

exit_code=0
for child_pid in "${child_pids[@]}"; do
    if ! wait "$child_pid"; then
        exit_code=1
    fi
done

if [[ $exit_code -eq 0 ]]; then
    echo "All three MorphBench blur ablations completed."
else
    echo "At least one MorphBench blur ablation failed; inspect the per-run logs." >&2
fi
exit "$exit_code"
