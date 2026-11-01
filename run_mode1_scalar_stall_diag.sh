#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_mode1_scalar_stall_diag.sh [floor]
  ./run_mode1_scalar_stall_diag.sh --floor 2
  ./run_mode1_scalar_stall_diag.sh --floor 2 --no-batch

Purpose:
  Reproduce the pre-fix mode=1 shrink path that reads expert_map slots from
  NPU scalar .item(), while collecting Python-level timings plus Ascend/HCCL logs.

Important defaults:
  floor=2
  batch direct NPU import remains enabled, matching the old chunks=4/8 shrink path.
  CPU export slot map is disabled only for this diagnostic run.

Useful overrides:
  MODE1_DIAG_RUN_ROOT=/path/to/runs
  ASCEND_GLOBAL_LOG_LEVEL=0|1|2|3
  VLLM_ASCEND_MODE1_DIAG_SCALAR_SLOW_LOG_MS=1
  VLLM_ASCEND_MODE1_DIAG_SLOW_EXPORT_LOG_MS=10
  VLLM_ASCEND_MODE1_DIAG_SLOW_P2P_LOG_MS=10
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
floor="${MODE1_FLOOR:-2}"
batch_direct="${VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--floor)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; usage >&2; exit 2; }
      floor="$2"
      shift 2
      ;;
    --no-batch)
      batch_direct=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        floor="$1"
        shift
      else
        echo "unknown argument: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

case "$floor" in
  1|2|4|8|16) ;;
  *) echo "unsupported mode=1 floor: $floor; expected one of 1,2,4,8,16" >&2; exit 2 ;;
esac

stamp=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="${MODE1_DIAG_RUN_ROOT:-${SCRIPT_DIR}/mode1_scalar_stall_diag_runs}"
RUN_DIR="${OUT_DIR:-${RUN_ROOT}/${stamp}_floor${floor}_npu_scalar}"
ASCEND_LOG_DIR="${RUN_DIR}/ascend_log"
mkdir -p "$RUN_DIR" "$ASCEND_LOG_DIR"

export OUT_DIR="$RUN_DIR"
export LOGFILE="${LOGFILE:-${RUN_DIR}/launcher.log}"
export RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${RUN_DIR}}"
export RECORD_DIR="${RECORD_DIR:-${RUN_DIR}/record}"
export MODE1_FLOOR="$floor"
if [[ "$floor" != "2" ]]; then
  export ALLOW_NON_FLOOR2="${ALLOW_NON_FLOOR2:-1}"
fi

# Force the old suspect path for this run only: direct NPU export reads slot ids
# from NPU expert_map tensors via .item(). Stable/default launchers keep CPU maps.
export MODE1_USE_CPU_EXPORT_SLOT_MAP=0
export VLLM_ASCEND_MODE1_USE_CPU_EXPORT_SLOT_MAP=0
export VLLM_ASCEND_MODE1_ALLOW_NPU_EXPORT_SLOT_FALLBACK=1
export VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT="$batch_direct"
export VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT=1
export VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT=0
export VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS="${VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS:-8}"

# Python-level diagnostics around the exact old stall candidates.
export MODE1_DIAG_EXPORT_INTERNAL_TIMING=1
export VLLM_ASCEND_MODE1_DIAG_EXPORT_INTERNAL_TIMING=1
export VLLM_ASCEND_MODE1_DIAG_SCALAR_TRACE="${VLLM_ASCEND_MODE1_DIAG_SCALAR_TRACE:-1}"
export VLLM_ASCEND_MODE1_DIAG_SCALAR_TRACE_SYNC_BEFORE="${VLLM_ASCEND_MODE1_DIAG_SCALAR_TRACE_SYNC_BEFORE:-0}"
export VLLM_ASCEND_MODE1_DIAG_SCALAR_SLOW_LOG_MS="${VLLM_ASCEND_MODE1_DIAG_SCALAR_SLOW_LOG_MS:-1}"
export MODE1_DIAG_SLOW_EXPORT_LOG_MS="${MODE1_DIAG_SLOW_EXPORT_LOG_MS:-10}"
export VLLM_ASCEND_MODE1_DIAG_SLOW_EXPORT_LOG_MS="${VLLM_ASCEND_MODE1_DIAG_SLOW_EXPORT_LOG_MS:-${MODE1_DIAG_SLOW_EXPORT_LOG_MS}}"
export VLLM_ASCEND_MODE1_DIAG_SLOW_PRELOAD_LAYER_LOG_MS="${VLLM_ASCEND_MODE1_DIAG_SLOW_PRELOAD_LAYER_LOG_MS:-10}"
export VLLM_ASCEND_MODE1_DIAG_SLOW_P2P_LOG_MS="${VLLM_ASCEND_MODE1_DIAG_SLOW_P2P_LOG_MS:-10}"
export VLLM_ASCEND_MODE1_STREAM_IMPORT_TRACE="${VLLM_ASCEND_MODE1_STREAM_IMPORT_TRACE:-0}"

# Ascend/HCCL logs. The launcher now respects these envs instead of overwriting.
export ASCEND_PROCESS_LOG_PATH="${ASCEND_PROCESS_LOG_PATH:-${ASCEND_LOG_DIR}}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"
export ASCEND_GLOBAL_EVENT_ENABLE="${ASCEND_GLOBAL_EVENT_ENABLE:-1}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-360}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-360}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-1}"
export HCCL_DEBUG="${HCCL_DEBUG:-INFO}"

{
  echo "[mode1 scalar stall diag] repo=${SCRIPT_DIR}"
  echo "[mode1 scalar stall diag] run_dir=${RUN_DIR}"
  echo "[mode1 scalar stall diag] wrapper_log=${LOGFILE}"
  echo "[mode1 scalar stall diag] train_log_dir=${RUN_OUTPUT_DIR}"
  echo "[mode1 scalar stall diag] ascend_log_dir=${ASCEND_LOG_DIR}"
  echo "[mode1 scalar stall diag] floor=${floor} batch_direct=${VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT} batch_experts=${VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS}"
  echo "[mode1 scalar stall diag] use_cpu_export_slot_map=${VLLM_ASCEND_MODE1_USE_CPU_EXPORT_SLOT_MAP} allow_npu_export_slot_fallback=${VLLM_ASCEND_MODE1_ALLOW_NPU_EXPORT_SLOT_FALLBACK}"
  echo "[mode1 scalar stall diag] scalar_trace=${VLLM_ASCEND_MODE1_DIAG_SCALAR_TRACE} scalar_slow_ms=${VLLM_ASCEND_MODE1_DIAG_SCALAR_SLOW_LOG_MS} slow_export_ms=${VLLM_ASCEND_MODE1_DIAG_SLOW_EXPORT_LOG_MS} slow_p2p_ms=${VLLM_ASCEND_MODE1_DIAG_SLOW_P2P_LOG_MS}"
  echo "[mode1 scalar stall diag] ASCEND_PROCESS_LOG_PATH=${ASCEND_PROCESS_LOG_PATH} ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL} ASCEND_GLOBAL_EVENT_ENABLE=${ASCEND_GLOBAL_EVENT_ENABLE}"
  echo "[mode1 scalar stall diag] HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT} HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT} HCCL_DEBUG=${HCCL_DEBUG}"
} | tee "${RUN_DIR}/diag_header.log"

set +e
bash "${SCRIPT_DIR}/run_mode1_floor2_local.sh"
rc=$?
set -e

{
  echo "[mode1 scalar stall diag] exit_code=${rc}"
  train_log="$(ls -1t "${RUN_OUTPUT_DIR}"/wjeagerqwen30b-a3b-with_draft_*_elastic.txt 2>/dev/null | head -1 || true)"
  if [[ -n "${train_log}" ]]; then
    echo "[mode1 scalar stall diag] train_log=${train_log}"
  else
    echo "[mode1 scalar stall diag] train_log=<not-found>"
  fi
  echo "[mode1 scalar stall diag] quick checks:"
  echo "  python3 ${SCRIPT_DIR}/analyze_mode1_scalar_stall_diag.py ${RUN_DIR}"
  echo "  rg -n 'Mode1 NPU scalar slot read|slow export|P2P .*timing|stream import breakdown' ${train_log:-${RUN_OUTPUT_DIR}}"
  echo "  rg -n 'HCCL|hccl|timeout|retry|error|failed|wait' ${ASCEND_LOG_DIR}"
} | tee -a "${RUN_DIR}/diag_header.log"

exit "$rc"
