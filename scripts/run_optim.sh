#!/bin/bash

# ==========================================
# Configuration
# ==========================================
REQUIRED_MEM=30720
# Wait time (seconds) to ensure the python process occupies GPU memory
WAIT_TIME=360

mkdir -p logs

CATEGORIES=(47 5)
NOISE_LEVEL=0.3
COMMON_ARGS="--dataset shapenet --seed 99995 --lr 0.01 --num_pairs 200 --max_iters 500 --noise_level $NOISE_LEVEL --output_dir ./output/shapenet"

# ==========================================
# Function to find available GPU
# ==========================================
get_free_gpu() {
    while true; do
        # Find a GPU with free memory > 30GB
        local GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
                       awk -v mem=$REQUIRED_MEM -F', ' '$2 >= mem {print $1; exit}')

        if [ -n "$GPU_ID" ]; then
            echo "$GPU_ID"
            return 0
        fi

        echo "[$(date +'%T')] Insufficient GPU memory, waiting..." >&2
        sleep 30
    done
}

# ==========================================
# Main loop
# ==========================================

# Helper function to launch tasks
launch_task() {
    local LOG_NAME=$1
    shift

    # 1. Find available GPU (will wait if none available)
    TARGET_GPU=$(get_free_gpu)

    echo ">>> [Launch] $LOG_NAME using GPU: $TARGET_GPU"

    # 2. Run in background
    CUDA_VISIBLE_DEVICES=$TARGET_GPU python -u optim_path.py $@ > "logs/${LOG_NAME}.log" 2>&1 &

    # 3. Wait to ensure GPU memory is occupied
    echo ">>> [Wait] Sleeping ${WAIT_TIME}s to ensure GPU memory allocation..."
    sleep $WAIT_TIME
}

for category in "${CATEGORIES[@]}"; do
    echo "Processing category=$category"

    launch_task "cat${category}_03_rbf"   $COMMON_ARGS --category $category --approximator rbf --batch_size 256 --h_path output/kernel/shapenet/rbf/h.pth

    launch_task "cat${category}_03_land"  $COMMON_ARGS --category $category --approximator land --batch_size 32 --h_path output/kernel/shapenet/land/h.pth

    launch_task "cat${category}_03_score" $COMMON_ARGS --category $category --approximator score --batch_size 2

    launch_task "cat${category}_03_el"    $COMMON_ARGS --category $category --approximator el --batch_size 16

    launch_task "cat${category}_03_stein" $COMMON_ARGS --category $category --approximator stein --batch_size 16
done

echo "All tasks submitted. Main script exiting."
# Uncomment the following line if you want to wait for all tasks to complete
# wait
