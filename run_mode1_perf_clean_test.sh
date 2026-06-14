#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_mode1_perf_clean_test.sh [floor]
  ./run_mode1_perf_clean_test.sh --floor 2

Environment overrides:
  MODE1_FLOOR=2
  REPO_ROOT=/path/to/log/output/root
  PATCH_TREE=/path/to/code/tree
  HCCL_IF_BASE_PORT=47241
  MASTER_PORT=26240
  VERL_HCCL_IF_BASE_PORT_START=47241

Examples:
  ./run_mode1_perf_clean_test.sh
  ./run_mode1_perf_clean_test.sh 2
  ./run_mode1_perf_clean_test.sh --floor 4
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
PATCH_TREE="${PATCH_TREE:-$REPO_ROOT}"
LAUNCHER="$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"

floor="${MODE1_FLOOR:-2}"
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
        echo "unsupported mode=1 floor: $floor; expected one of 1,2,4,8,16" >&2
        exit 2
        ;;
esac

cd "$PATCH_TREE"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
tee_log="$REPO_ROOT/mode1_floor${floor}_perf_clean_${stamp}.log"

# Performance run: force the stable mode=1 path while disabling profiler,
# per-layer timing sync/logging, and extra diagnostic noise. The actual
# training log is still produced by the launcher under $HOME.
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="$floor"
export VLLM_ASCEND_CUSTOM_MODE1_DEBUG=0
export VLLM_ASCEND_CUSTOM_MODE1_TIMING_EVENTS=0
export VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG=0

# Keep the host-stable mode=1 floor=2 path: direct NPU->NPU payload transfer,
# but no scalar NPU slot-map reads. These defaults can still be overridden by
# the caller for diagnostics.
export VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT="${VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT:-1}"
export VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT="${VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT:-0}"
export VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT="${VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT:-0}"
export VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS="${VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS:-8}"
export VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC="${VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC:-1}"

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

export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-47241}"
export MASTER_PORT="${MASTER_PORT:-26240}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-47241}"

export HOME="$REPO_ROOT"
export CONFIG_DIR="$PATCH_TREE/verl/trainer/config"
export PYTHONPATH="$PATCH_TREE${PYTHONPATH:+:$PYTHONPATH}"

printf '[mode1 perf clean] runtime_cwd=%s\n' "$PATCH_TREE"
printf '[mode1 perf clean] tee_log=%s\n' "$tee_log"
printf '[mode1 perf clean] floor=%s batch_direct_npu=%s allow_scalar=%s batch_experts=%s cpu_dp_metadata_sync=%s\n' \
    "$floor" \
    "$VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT" \
    "$VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT" \
    "$VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS" \
    "$VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC"
printf '[mode1 perf clean] profile=%s timing_log=%s timing_sync=%s markers=%s\n' \
    "$VLLM_ASCEND_BUCKET_OP_PROFILE" \
    "$VLLM_ASCEND_MODE3_TIMING_LOG" \
    "$VLLM_ASCEND_MODE3_TIMING_SYNC" \
    "$VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS"
printf '[mode1 perf clean] ports HCCL_IF_BASE_PORT=%s MASTER_PORT=%s VERL_HCCL_IF_BASE_PORT_START=%s\n' \
    "$HCCL_IF_BASE_PORT" "$MASTER_PORT" "$VERL_HCCL_IF_BASE_PORT_START"

bash "$LAUNCHER" 2>&1 | tee "$tee_log"
