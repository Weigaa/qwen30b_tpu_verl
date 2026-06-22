#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_mode3_perf_clean_test.sh [floor]
  ./run_mode3_perf_clean_test.sh --floor 1

Environment overrides:
  MODE3_FLOOR=1
  REPO_ROOT=/path/to/log/output/root
  PATCH_TREE=/path/to/code/tree
  HCCL_IF_BASE_PORT=47041
  MASTER_PORT=26040
  VERL_HCCL_IF_BASE_PORT_START=47041

Examples:
  ./run_mode3_perf_clean_test.sh
  ./run_mode3_perf_clean_test.sh 1
  ./run_mode3_perf_clean_test.sh --floor 4
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
PATCH_TREE="${PATCH_TREE:-$REPO_ROOT}"
LAUNCHER="$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"

floor="${MODE3_FLOOR:-1}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--floor)
            [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; usage >&2; exit 2; }
            floor="$2"
            shift 2
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
    *)
        echo "unsupported mode=3 floor: $floor; expected one of 1,2,4,8,16" >&2
        exit 2
        ;;
esac

cd "$PATCH_TREE"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
tee_log="$REPO_ROOT/mode3_floor${floor}_perf_clean_${stamp}.log"

# Performance run: disable profiler, stage-decode MSTX markers, and per-layer
# timing sync/logging.
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=3
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="$floor"
export VLLM_ASCEND_MODE3_TRANSFER_LOG=0
export VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG=0
export VLLM_ASCEND_MODE3_TIMING_LOG=0
export VLLM_ASCEND_MODE3_TIMING_SYNC=0
export VLLM_ASCEND_MODE3_TIMING_EVERY=1000000
export VLLM_ASCEND_MODE3_TIMING_FIRST_N=0
export VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS=0
export VLLM_ASCEND_BUCKET_OP_PROFILE=0
export VLLM_ASCEND_BUCKET_OP_PROFILE_BY_STAGE=0
export VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS=""
export VLLM_ASCEND_DUMMY_WASTE_TIMING=0
export VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=0
export VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE=0
export VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS=0

export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-47041}"
export MASTER_PORT="${MASTER_PORT:-26040}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-47041}"

export HOME="$REPO_ROOT"
export CONFIG_DIR="$PATCH_TREE/verl/trainer/config"
export PYTHONPATH="$PATCH_TREE${PYTHONPATH:+:$PYTHONPATH}"

printf '[mode3 perf clean] runtime_cwd=%s
' "$PATCH_TREE"
printf '[mode3 perf clean] tee_log=%s
' "$tee_log"
printf '[mode3 perf clean] floor=%s\n' "$floor"
printf '[mode3 perf clean] profile=%s timing_log=%s timing_sync=%s markers=%s
' "$VLLM_ASCEND_BUCKET_OP_PROFILE" "$VLLM_ASCEND_MODE3_TIMING_LOG" "$VLLM_ASCEND_MODE3_TIMING_SYNC" "$VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS"
printf '[mode3 perf clean] ports HCCL_IF_BASE_PORT=%s MASTER_PORT=%s VERL_HCCL_IF_BASE_PORT_START=%s
' "$HCCL_IF_BASE_PORT" "$MASTER_PORT" "$VERL_HCCL_IF_BASE_PORT_START"

bash "$LAUNCHER" 2>&1 | tee "$tee_log"
