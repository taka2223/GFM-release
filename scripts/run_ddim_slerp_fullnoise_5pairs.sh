#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_cache=${MORPHBENCH_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}
caption_root="$repo_root/output/morphbench_c2f_sigma0p6_lambda20/captions"
output_root="$repo_root/output/morphbench_ddim_slerp_t1_pairs5"
runtime_packages=${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}
mpl_config="$repo_root/.rebuttal-runtime/matplotlib"

cd "$repo_root"
mkdir -p "$output_root/logs" "$mpl_config"

run_shard() {
    local gpu_id="$1"
    local shard_id="$2"
    env \
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="$runtime_packages" \
        MPLCONFIGDIR="$mpl_config" \
        HF_HOME="$model_cache" \
        .venv/bin/python scripts/run_morphbench.py \
            --data_root data/MorphBench \
            --subsets Metamorphosis Animation \
            --out_root "$output_root" \
            --caption_root "$caption_root" \
            --cache_dir "$model_cache" \
            --device cuda:0 \
            --methods slerp \
            --pair_names boy_girl_2 mona_pearl david cake woman \
            --num_shards 2 \
            --shard_id "$shard_id" \
            --num_steps 17 \
            --n_inference_steps 50 \
            --resolution 512 \
            --cfg_scale 0 \
            --noise_level 1.0 \
            > "$output_root/logs/shard${shard_id}.log" 2>&1
}

run_shard 0 0 &
pid0=$!
run_shard 1 1 &
pid1=$!

status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
exit "$status"
