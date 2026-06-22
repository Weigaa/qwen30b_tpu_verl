#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./run_mode5_cpu_fraction_dispatch_mem_test.sh [remote_fraction] [floor]

Purpose:
  Run the existing DispatchV2/HCCL memory diagnostic through mode=5 while
  varying the remote/CPU expert split. This isolates whether OOM is caused by
  CPU-row count itself or by mode=3's CPU-source runtime organization.

Interpretation at floor=4 with 32 active experts per rank:
  remote_fraction=0.75 -> about 18 remote-NPU + 6 CPU rows
  remote_fraction=0.50 -> about 12 remote-NPU + 12 CPU rows
  remote_fraction=0.25 -> about 6 remote-NPU + 18 CPU rows
  remote_fraction=0.00 -> about 0 remote-NPU + 24 CPU rows

Defaults:
  remote_fraction=0.75
  floor=4

Examples:
  ./run_mode5_cpu_fraction_dispatch_mem_test.sh 0.75 4
  ./run_mode5_cpu_fraction_dispatch_mem_test.sh 0.50 4
  ./run_mode5_cpu_fraction_dispatch_mem_test.sh 0.00 4
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
remote_fraction=${1:-0.75}
floor=${2:-4}

case "${remote_fraction}" in
    -h|--help) usage; exit 0 ;;
esac

if ! [[ "${remote_fraction}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
    echo "unsupported remote_fraction: ${remote_fraction}; expected [0, 1]" >&2
    usage >&2
    exit 2
fi

case "${floor}" in
    1|2|4|8|16) ;;
    *) echo "unsupported floor: ${floor}" >&2; usage >&2; exit 2 ;;
esac

stamp=$(date -u +%Y%m%dT%H%M%SZ)
export DIAG_RUN_ROOT="${DIAG_RUN_ROOT:-${SCRIPT_DIR}/mode5_cpu_fraction_dispatch_mem_runs/${stamp}_fraction${remote_fraction}_floor${floor}}"
export VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION="${remote_fraction}"
export VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY="${VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY:-fixed}"
export VLLM_ASCEND_MODE5_BALANCE_REMOTE_SOURCE_FANOUT="${VLLM_ASCEND_MODE5_BALANCE_REMOTE_SOURCE_FANOUT:-0}"

cat <<EOF_RUN
[mode5-cpu-fraction-diag] remote_fraction=${remote_fraction}
[mode5-cpu-fraction-diag] floor=${floor}
[mode5-cpu-fraction-diag] DIAG_RUN_ROOT=${DIAG_RUN_ROOT}
[mode5-cpu-fraction-diag] expected floor=4 CPU rows ~= round((1 - fraction) * 24)
EOF_RUN

exec "${SCRIPT_DIR}/run_mode3_mode4_floor4_dispatch_mem_diag.sh" mode5 "${floor}"
