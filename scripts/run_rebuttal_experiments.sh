#!/usr/bin/env bash
#
# Single approved entry point for GPU-backed rebuttal experiments.
#
# This script intentionally does not accept arbitrary commands. It can only:
#   - check GPU visibility;
#   - run scripts/rebuttal_experiments.py;
#   - inspect the runner's PID and log.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${GFM_REBUTTAL_RUNTIME_DIR:-${REPO_ROOT}/.rebuttal-runtime}"
RUNNER="${REPO_ROOT}/scripts/rebuttal_experiments.py"
METRICS_RUNNER="${REPO_ROOT}/scripts/rebuttal_metrics.py"
FINALIZER="${REPO_ROOT}/scripts/rebuttal_finalize.py"
PYTHON_BIN="${GFM_REBUTTAL_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
PID_FILE="${RUNTIME_DIR}/runner.pid"
LOG_FILE="${RUNTIME_DIR}/runner.log"
TMUX_SESSION="${GFM_REBUTTAL_TMUX_SESSION:-gfm-rebuttal}"
METRICS_PID_FILE="${RUNTIME_DIR}/metrics.pid"
METRICS_LOG_FILE="${RUNTIME_DIR}/metrics.log"
METRICS_TMUX_SESSION="${GFM_REBUTTAL_METRICS_TMUX_SESSION:-gfm-rebuttal-metrics}"
WORKER2_PID_FILE="${RUNTIME_DIR}/worker2.pid"
WORKER2_LOG_FILE="${RUNTIME_DIR}/worker2.log"
WORKER2_TMUX_SESSION="${GFM_REBUTTAL_WORKER2_TMUX_SESSION:-gfm-rebuttal-worker2}"
FINALIZER_PID_FILE="${RUNTIME_DIR}/finalizer.pid"
FINALIZER_LOG_FILE="${RUNTIME_DIR}/finalizer.log"
FINALIZER_TMUX_SESSION="${GFM_REBUTTAL_FINALIZER_TMUX_SESSION:-gfm-rebuttal-finalizer}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_rebuttal_experiments.sh gpu-check
  bash scripts/run_rebuttal_experiments.sh foreground [runner arguments...]
  bash scripts/run_rebuttal_experiments.sh start [runner arguments...]
  bash scripts/run_rebuttal_experiments.sh resume [runner arguments...]
  bash scripts/run_rebuttal_experiments.sh status
  bash scripts/run_rebuttal_experiments.sh tail [line count]
  bash scripts/run_rebuttal_experiments.sh worker2-start [runner arguments...]
  bash scripts/run_rebuttal_experiments.sh worker2-status
  bash scripts/run_rebuttal_experiments.sh worker2-tail [line count]
  bash scripts/run_rebuttal_experiments.sh finalize-start
  bash scripts/run_rebuttal_experiments.sh finalize-status
  bash scripts/run_rebuttal_experiments.sh finalize-tail [line count]
  bash scripts/run_rebuttal_experiments.sh metrics-foreground [metric arguments...]
  bash scripts/run_rebuttal_experiments.sh metrics-start [metric arguments...]
  bash scripts/run_rebuttal_experiments.sh metrics-status
  bash scripts/run_rebuttal_experiments.sh metrics-tail [line count]

Environment:
  CUDA_VISIBLE_DEVICES          GPU selection (default: 0)
  GFM_REBUTTAL_PYTHON           Python executable (default: .venv/bin/python)
  GFM_REBUTTAL_RUNTIME_DIR      PID/log directory (default: .rebuttal-runtime)
EOF
}

ensure_runtime_dir() {
    mkdir -p -- "${RUNTIME_DIR}"
}

ensure_runner() {
    if [[ ! -f "${RUNNER}" ]]; then
        echo "Missing runner: ${RUNNER}" >&2
        echo "Implement scripts/rebuttal_experiments.py before starting E1/E2." >&2
        exit 2
    fi
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
        exit 2
    fi
}

read_live_pid() {
    read_live_pid_from "${PID_FILE}"
}

read_live_pid_from() {
    local source=$1
    local pid
    [[ -f "${source}" ]] || return 1
    read -r pid < "${source}"
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    printf '%s\n' "${pid}"
}

run_foreground() {
    ensure_runner
    ensure_runtime_dir
    cd -- "${REPO_ROOT}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export TMPDIR="${RUNTIME_DIR}/tmp"
    export CUDA_CACHE_PATH="${RUNTIME_DIR}/cuda-cache"
    export TORCH_EXTENSIONS_DIR="${RUNTIME_DIR}/torch-extensions"
    export MPLCONFIGDIR="${RUNTIME_DIR}/matplotlib"
    mkdir -p -- \
        "${TMPDIR}" "${CUDA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}" "${MPLCONFIGDIR}"
    exec "${PYTHON_BIN}" "${RUNNER}" "$@"
}

run_metrics_foreground() {
    ensure_runtime_dir
    if [[ ! -f "${METRICS_RUNNER}" ]]; then
        echo "Missing metrics runner: ${METRICS_RUNNER}" >&2
        exit 2
    fi
    cd -- "${REPO_ROOT}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    export TMPDIR="${RUNTIME_DIR}/tmp"
    export CUDA_CACHE_PATH="${RUNTIME_DIR}/cuda-cache"
    export TORCH_EXTENSIONS_DIR="${RUNTIME_DIR}/torch-extensions"
    export MPLCONFIGDIR="${RUNTIME_DIR}/matplotlib"
    mkdir -p -- \
        "${TMPDIR}" "${CUDA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}" "${MPLCONFIGDIR}"
    exec "${PYTHON_BIN}" "${METRICS_RUNNER}" "$@"
}

start_tmux_job() {
    local session=$1
    local pid_file=$2
    local log_file=$3
    local program=$4
    shift 4
    local existing_pid
    ensure_runtime_dir
    if existing_pid="$(read_live_pid_from "${pid_file}")"; then
        echo "Runner is already active (PID ${existing_pid})." >&2
        exit 3
    fi

    cd -- "${REPO_ROOT}"
    export TMPDIR="${RUNTIME_DIR}/tmp"
    export CUDA_CACHE_PATH="${RUNTIME_DIR}/cuda-cache"
    export TORCH_EXTENSIONS_DIR="${RUNTIME_DIR}/torch-extensions"
    export MPLCONFIGDIR="${RUNTIME_DIR}/matplotlib"
    mkdir -p -- "${TMPDIR}" "${CUDA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}"
    mkdir -p -- "${MPLCONFIGDIR}"

    if tmux has-session -t "=${session}" 2>/dev/null; then
        echo "tmux session already exists: ${session}" >&2
        exit 3
    fi
    tmux new-session -d -s "${session}" -c "${REPO_ROOT}" -- \
        bash -c 'log_file=$1; shift; exec "$@" >> "${log_file}" 2>&1' \
        bash "${log_file}" env \
        "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
        "TMPDIR=${TMPDIR}" \
        "CUDA_CACHE_PATH=${CUDA_CACHE_PATH}" \
        "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}" \
        "MPLCONFIGDIR=${MPLCONFIGDIR}" \
        "${PYTHON_BIN}" "${program}" "$@"
    local pid
    pid="$(tmux display-message -p -t "${session}:" '#{pane_pid}')"
    printf '%s\n' "${pid}" > "${pid_file}"
    echo "Started runner (tmux=${session}, PID ${pid})."
    echo "Log: ${log_file}"
}

start_background() {
    ensure_runner
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    start_tmux_job \
        "${TMUX_SESSION}" "${PID_FILE}" "${LOG_FILE}" "${RUNNER}" "$@"
}

start_metrics_background() {
    if [[ ! -f "${METRICS_RUNNER}" ]]; then
        echo "Missing metrics runner: ${METRICS_RUNNER}" >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    start_tmux_job \
        "${METRICS_TMUX_SESSION}" "${METRICS_PID_FILE}" \
        "${METRICS_LOG_FILE}" "${METRICS_RUNNER}" "$@"
}

start_worker2_background() {
    ensure_runner
    export CUDA_VISIBLE_DEVICES=1
    start_tmux_job \
        "${WORKER2_TMUX_SESSION}" "${WORKER2_PID_FILE}" \
        "${WORKER2_LOG_FILE}" "${RUNNER}" "$@"
}

start_finalizer_background() {
    if [[ ! -f "${FINALIZER}" ]]; then
        echo "Missing finalizer: ${FINALIZER}" >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES=1
    start_tmux_job \
        "${FINALIZER_TMUX_SESSION}" "${FINALIZER_PID_FILE}" \
        "${FINALIZER_LOG_FILE}" "${FINALIZER}"
}

show_job_status() {
    local pid_file=$1
    local session=$2
    local pid
    if pid="$(read_live_pid_from "${pid_file}")"; then
        echo "running tmux=${session} PID=${pid}"
        ps -o pid=,ppid=,stat=,etime=,cmd= -p "${pid}"
        return 0
    fi
    echo "not running"
    [[ ! -f "${pid_file}" ]] || echo "stale PID file: ${pid_file}"
}

show_status() {
    show_job_status "${PID_FILE}" "${TMUX_SESSION}"
}

show_metrics_status() {
    show_job_status "${METRICS_PID_FILE}" "${METRICS_TMUX_SESSION}"
}

show_worker2_status() {
    show_job_status "${WORKER2_PID_FILE}" "${WORKER2_TMUX_SESSION}"
}

show_finalizer_status() {
    show_job_status "${FINALIZER_PID_FILE}" "${FINALIZER_TMUX_SESSION}"
}

show_tail_file() {
    local log_file=$1
    local lines=$2
    if [[ ! -f "${log_file}" ]]; then
        echo "No log exists yet: ${log_file}" >&2
        exit 4
    fi
    tail -n "${lines}" -- "${log_file}"
}

validate_line_count() {
    local lines="${1:-80}"
    if [[ ! "${lines}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Line count must be a positive integer." >&2
        exit 2
    fi
    printf '%s\n' "${lines}"
}

command_name="${1:-}"
if [[ -z "${command_name}" ]]; then
    usage
    exit 2
fi
shift

case "${command_name}" in
    gpu-check)
        if (($# != 0)); then
            usage
            exit 2
        fi
        exec nvidia-smi \
            --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader
        ;;
    foreground)
        run_foreground "$@"
        ;;
    start)
        start_background "$@"
        ;;
    resume)
        start_background --resume "$@"
        ;;
    status)
        if (($# != 0)); then
            usage
            exit 2
        fi
        show_status
        ;;
    tail)
        if (($# > 1)); then
            usage
            exit 2
        fi
        lines="$(validate_line_count "${1:-80}")"
        show_tail_file "${LOG_FILE}" "${lines}"
        ;;
    worker2-start)
        start_worker2_background "$@"
        ;;
    worker2-status)
        if (($# != 0)); then
            usage
            exit 2
        fi
        show_worker2_status
        ;;
    worker2-tail)
        if (($# > 1)); then
            usage
            exit 2
        fi
        lines="$(validate_line_count "${1:-80}")"
        show_tail_file "${WORKER2_LOG_FILE}" "${lines}"
        ;;
    finalize-start)
        if (($# != 0)); then
            usage
            exit 2
        fi
        start_finalizer_background
        ;;
    finalize-status)
        if (($# != 0)); then
            usage
            exit 2
        fi
        show_finalizer_status
        ;;
    finalize-tail)
        if (($# > 1)); then
            usage
            exit 2
        fi
        lines="$(validate_line_count "${1:-80}")"
        show_tail_file "${FINALIZER_LOG_FILE}" "${lines}"
        ;;
    metrics-foreground)
        run_metrics_foreground "$@"
        ;;
    metrics-start)
        start_metrics_background "$@"
        ;;
    metrics-status)
        if (($# != 0)); then
            usage
            exit 2
        fi
        show_metrics_status
        ;;
    metrics-tail)
        if (($# > 1)); then
            usage
            exit 2
        fi
        lines="$(validate_line_count "${1:-80}")"
        show_tail_file "${METRICS_LOG_FILE}" "${lines}"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unsupported command: ${command_name}" >&2
        usage >&2
        exit 2
        ;;
esac
