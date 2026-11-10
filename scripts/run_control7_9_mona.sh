#!/usr/bin/env bash
set -euo pipefail

control_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
control_log_dir="$control_repo/output/morphbench_blur_ablation/logs"
control_variant7=control7_mona_sigma0p6_lambda20_iter800
control_variant9=control9_mona_sigma0p6_lambda20_iter800

cd "$control_repo"
mkdir -p "$control_log_dir"

bash scripts/run_blur_ablation.sh \
    0 "$control_variant7" \
    > "$control_log_dir/${control_variant7}.log" 2>&1 &
control_pid7=$!

bash scripts/run_blur_ablation.sh \
    1 "$control_variant9" \
    > "$control_log_dir/${control_variant9}.log" 2>&1 &
control_pid9=$!

control_status=0
wait "$control_pid7" || control_status=$?
wait "$control_pid9" || control_status=$?
exit "$control_status"
