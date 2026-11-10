#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 <gpu_id> <iter800|lr2e3|cfg1p5|sigma0p5|sigma0p3_lambda20_iter800|sigma0p5_lambda50_iter800|sigma0p5_lambda50_lr2e3_earlystop|midpoint_sigma0p6_lambda20_iter800|control5_pairs5_sigma0p6_lambda20_iter800|control7_mona_sigma0p6_lambda20_iter800|control9_mona_sigma0p6_lambda20_iter800|c2f_mona_sigma0p6_lambda20|c2f_mona_sigma0p3_lambda20|c2f_mona_sigma0p3_lambda500|c2f5full_mona_sigma0p6_lambda20|c2f5full_refine_mona_sigma0p6_lambda20|cond_interp> [num_shards] [shard_id]" >&2
    exit 2
fi

blur_gpu_id=$1
blur_variant=$2
blur_num_shards=${3:-1}
blur_shard_id=${4:-0}
blur_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
blur_cache=${MORPHBENCH_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}
blur_output="$blur_repo/output/morphbench_blur_ablation/$blur_variant"
blur_methods=(gfm)
blur_extra_args=()
blur_pair_args=(--pair_names boy_girl_2)

case "$blur_variant" in
    iter800)
        blur_extra_args=(--max_iters 800 --lr 0.001 --cfg_scale 0 --noise_level 0.6)
        ;;
    lr2e3)
        blur_extra_args=(--max_iters 400 --lr 0.002 --cfg_scale 0 --noise_level 0.6)
        ;;
    cfg1p5)
        blur_methods=(gfm slerp)
        blur_extra_args=(--max_iters 400 --lr 0.001 --cfg_scale 1.5 --noise_level 0.6)
        ;;
    sigma0p5)
        blur_methods=(gfm slerp)
        blur_extra_args=(--max_iters 400 --lr 0.001 --cfg_scale 0 --noise_level 0.5)
        ;;
    sigma0p3_lambda20_iter800)
        blur_methods=(gfm slerp)
        blur_extra_args=(
            --max_iters 800
            --lr 0.001
            --cfg_scale 0
            --noise_level 0.3
            --lam 20
        )
        ;;
    sigma0p5_lambda50_iter800)
        blur_methods=(gfm slerp)
        blur_extra_args=(
            --max_iters 800
            --lr 0.001
            --cfg_scale 0
            --noise_level 0.5
            --lam 50
        )
        ;;
    sigma0p5_lambda50_lr2e3_earlystop)
        blur_methods=(gfm slerp)
        blur_extra_args=(
            --max_iters 2000
            --lr 0.002
            --cfg_scale 0
            --noise_level 0.5
            --lam 50
            --early_stop_patience 200
            --early_stop_rel_delta 0.0001
            --early_stop_warmup 500
        )
        ;;
    midpoint_sigma0p6_lambda20_iter800)
        blur_methods=(gfm slerp)
        blur_extra_args=(
            --opt_num_steps 3
            --max_iters 800
            --lr 0.001
            --cfg_scale 0
            --noise_level 0.6
            --lam 20
        )
        ;;
    control5_pairs5_sigma0p6_lambda20_iter800)
        blur_methods=(gfm slerp)
        blur_pair_args=(
            --pair_names
            boy_girl_2
            dog_wolf
            cake_burger
            red_car
            mona_pearl
        )
        blur_extra_args=(
            --opt_num_steps 5
            --max_iters 800
            --lr 0.001
            --cfg_scale 0
            --noise_level 0.6
            --lam 20
        )
        ;;
    control7_mona_sigma0p6_lambda20_iter800)
        blur_pair_args=(--pair_names mona_pearl)
        blur_extra_args=(
            --opt_num_steps 7
            --max_iters 800
            --lr 0.001
            --cfg_scale 0
            --noise_level 0.6
            --lam 20
        )
        ;;
    control9_mona_sigma0p6_lambda20_iter800)
        blur_pair_args=(--pair_names mona_pearl)
        blur_extra_args=(
            --opt_num_steps 9
            --max_iters 800
            --lr 0.001
            --cfg_scale 0
            --noise_level 0.6
            --lam 20
        )
        ;;
    c2f_mona_sigma0p6_lambda20)
        blur_methods=(gfm_c2f slerp)
        blur_pair_args=(--pair_names mona_pearl)
        blur_extra_args=(
            --cfg_scale 0
            --noise_level 0.6
            --lam 20
            --c2f_steps 3 5 9 17
            --c2f_iters 400 300 200 100
            --c2f_lrs 0.002 0.001 0.0005 0.00025
        )
        ;;
    c2f_mona_sigma0p3_lambda20)
        blur_methods=(gfm_c2f slerp)
        blur_pair_args=(--pair_names mona_pearl)
        blur_extra_args=(
            --cfg_scale 0
            --noise_level 0.3
            --lam 20
            --c2f_steps 3 5 9 17
            --c2f_iters 400 300 200 100
            --c2f_lrs 0.002 0.001 0.0005 0.00025
        )
        ;;
    c2f_mona_sigma0p3_lambda500)
        blur_methods=(gfm_c2f)
        blur_pair_args=(--pair_names mona_pearl)
        blur_extra_args=(
            --cfg_scale 0
            --noise_level 0.3
            --lam 500
            --c2f_steps 3 5 9 17
            --c2f_iters 400 300 200 100
            --c2f_lrs 0.002 0.001 0.0005 0.00025
        )
        ;;
    c2f5full_mona_sigma0p6_lambda20)
        blur_methods=(gfm_c2f)
        blur_pair_args=(--pair_names mona_pearl)
        blur_extra_args=(
            --cfg_scale 0
            --noise_level 0.6
            --lam 20
            --c2f_steps 5 9 17
            --c2f_iters 800 100 50
            --c2f_lrs 0.001 0.0002 0.0001
        )
        ;;
    c2f5full_refine_mona_sigma0p6_lambda20)
        blur_methods=(gfm_c2f)
        blur_pair_args=(--pair_names mona_pearl)
        blur_extra_args=(
            --cfg_scale 0
            --noise_level 0.6
            --lam 20
            --c2f_steps 5 9 17
            --c2f_iters 800 200 100
            --c2f_lrs 0.001 0.0005 0.00025
        )
        ;;
    cond_interp)
        blur_methods=(cond_interp)
        blur_extra_args=(
            --cfg_scale 7
            --cond_seed 20260727
        )
        ;;
    *)
        echo "Unknown variant: $blur_variant" >&2
        exit 2
        ;;
esac

cd "$blur_repo"
mkdir -p "$blur_output"

exec env \
    CUDA_VISIBLE_DEVICES="$blur_gpu_id" \
    PYTHONPATH="${GFM_EXTRA_PYTHONPATH:-${PYTHONPATH:-}}" \
    HF_HOME="$blur_cache" \
    .venv/bin/python scripts/run_morphbench.py \
        --data_root data/MorphBench \
        --subsets Metamorphosis \
        "${blur_pair_args[@]}" \
        --out_root "$blur_output" \
        --cache_dir "$blur_cache" \
        --caption_model_id ./BLIP_large \
        --device cuda:0 \
        --methods "${blur_methods[@]}" \
        --num_shards "$blur_num_shards" \
        --shard_id "$blur_shard_id" \
        --num_steps 17 \
        --n_inference_steps 50 \
        --resolution 512 \
        --lam 10 \
        "${blur_extra_args[@]}"
