#!/usr/bin/env bash
set -euo pipefail

control_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
control_variant=control5_pairs5_sigma0p6_lambda20_iter800
control_log_dir="$control_repo/output/morphbench_blur_ablation/logs"

cd "$control_repo"
mkdir -p "$control_log_dir"

bash scripts/run_blur_ablation.sh \
    0 "$control_variant" 2 0 \
    > "$control_log_dir/${control_variant}_shard0.log" 2>&1 &
control_pid0=$!

bash scripts/run_blur_ablation.sh \
    1 "$control_variant" 2 1 \
    > "$control_log_dir/${control_variant}_shard1.log" 2>&1 &
control_pid1=$!

control_status=0
wait "$control_pid0" || control_status=$?
wait "$control_pid1" || control_status=$?
exit "$control_status"
