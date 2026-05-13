#!/bin/bash

# --- 基础路径配置 ---
IMG_A="/cns/USERS/zzhixuan/data/MorphBench/Metamorphosis/castle_0.png"
IMG_B="/cns/USERS/zzhixuan/data/MorphBench/Metamorphosis/castle_1.png"
OUTPUT_DIR="./output/sd3_interp_blip"

# --- 核心超参数 ---
NOISE_LEVEL=0.6
CFG_SCALE=0.0
LAM=10.0
LR=0.001
MAX_ITERS=400
NUM_STEPS=10
RESOLUTION=512
SNAPSHOT_ITERS=""

# --- 提示词 (留空则使用默认值) ---
PROMPT_A=""
PROMPT_B=""

# --- 模型与设备配置 ---
MODEL_ID="stabilityai/stable-diffusion-3-medium-diffusers"
CACHE_DIR="/cns/USERS/zzhixuan/weights"
DEVICE="cuda"

# --- 开关参数 (1表示启用, 0表示禁用) ---
USE_TEXT_INV=0
NO_BLIP=0
KEEP_T5=1

# ---------------------------------------------------------
# 逻辑处理：根据开关构建动态参数列表
# ---------------------------------------------------------
CMD_ARGS=(
    --imgA "$IMG_A"
    --imgB "$IMG_B"
    --noise_level "$NOISE_LEVEL"
    --cfg_scale "$CFG_SCALE"
    --lam "$LAM"
    --lr "$LR"
    --max_iters "$MAX_ITERS"
    --num_steps "$NUM_STEPS"
    --resolution "$RESOLUTION"
    --output_dir "$OUTPUT_DIR"
    --model_id "$MODEL_ID"
    --cache_dir "$CACHE_DIR"
    --device "$DEVICE"
    --snapshot_iters "$SNAPSHOT_ITERS"
)

# 处理可选的 Prompt
if [ -n "$PROMPT_A" ]; then CMD_ARGS+=(--promptA "$PROMPT_A"); fi
if [ -n "$PROMPT_B" ]; then CMD_ARGS+=(--promptB "$PROMPT_B"); fi

# 处理布尔开关 (Action store_true)
if [ "$USE_TEXT_INV" -eq 1 ]; then CMD_ARGS+=(--use_text_inv); fi
if [ "$NO_BLIP" -eq 1 ]; then CMD_ARGS+=(--no_blip); fi
if [ "$KEEP_T5" -eq 1 ]; then CMD_ARGS+=(--keep_t5); fi

export CUDA_VISIBLE_DEVICES=6
# 执行程序 (假设你的 Python 文件名为 main.py)
python gfm/path/sd3_wrapper.py "${CMD_ARGS[@]}"