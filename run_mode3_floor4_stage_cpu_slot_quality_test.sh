#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./run_mode3_floor4_stage_cpu_slot_quality_test.sh [floor]
  ./run_mode3_floor4_stage_cpu_slot_quality_test.sh --floor 4

Purpose:
  Run mode=3 with the old CPU staging path, i.e. disable direct CPU -> final
  runtime NPU slot copies. This is intended as the A/B pair for the archived
  direct CPU slot run under mode3_cpu_slot_quality_compare_runs/.

Key overrides forced by this wrapper:
  VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT=0
  VLLM_ASCEND_MODE3_BULK_CPU_DIRECT=0
  VLLM_ASCEND_MODE3_ASYNC_CPU_STAGE=1
  VLLM_ASCEND_MODE3_BULK_CPU_STAGE=1

Environment overrides:
  MODE3_FLOOR=4
  REPO_ROOT=/path/to/output/root
  PATCH_TREE=/path/to/code/tree
  HCCL_IF_BASE_PORT=47041
  MASTER_PORT=26040
  VERL_HCCL_IF_BASE_PORT_START=47041
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
PATCH_TREE="${PATCH_TREE:-$SCRIPT_DIR}"
LAUNCHER="$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"

floor="${MODE3_FLOOR:-4}"
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

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$REPO_ROOT/mode3_cpu_slot_quality_compare_runs/${stamp}_floor${floor}_stage_cpu_slot"
record_dir="$run_dir/record"
mkdir -p "$record_dir"

cd "$PATCH_TREE"

# Match the successful direct-slot run's main mode/floor, but force the old
# staging path for CPU-only experts.
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=3
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="$floor"
export VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT=0
export VLLM_ASCEND_MODE3_BULK_CPU_DIRECT=0
export VLLM_ASCEND_MODE3_ASYNC_CPU_STAGE=1
export VLLM_ASCEND_MODE3_ASYNC_CPU_PACK=1
export VLLM_ASCEND_MODE3_BULK_CPU_STAGE=1
export VLLM_ASCEND_MODE3_BULK_NPU_COPY="${VLLM_ASCEND_MODE3_BULK_NPU_COPY:-1}"
export VLLM_ASCEND_MODE3_DEVICE_READY_WAIT="${VLLM_ASCEND_MODE3_DEVICE_READY_WAIT:-1}"
export VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH="${VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH:-1}"
export VLLM_ASCEND_MODE3_FULL_DISPATCH_DOMAIN="${VLLM_ASCEND_MODE3_FULL_DISPATCH_DOMAIN:-0}"
export VLLM_ASCEND_MODE3_MODE5_LIKE_CPU_SHADOW="${VLLM_ASCEND_MODE3_MODE5_LIKE_CPU_SHADOW:-1}"

# Keep the perf-clean logging profile unless explicitly overridden.
export VLLM_ASCEND_MODE3_TRANSFER_LOG="${VLLM_ASCEND_MODE3_TRANSFER_LOG:-0}"
export VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG="${VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG:-0}"
export VLLM_ASCEND_MODE3_TIMING_LOG="${VLLM_ASCEND_MODE3_TIMING_LOG:-0}"
export VLLM_ASCEND_MODE3_TIMING_SYNC="${VLLM_ASCEND_MODE3_TIMING_SYNC:-0}"
export VLLM_ASCEND_MODE3_TIMING_EVERY="${VLLM_ASCEND_MODE3_TIMING_EVERY:-1000000}"
export VLLM_ASCEND_MODE3_TIMING_FIRST_N="${VLLM_ASCEND_MODE3_TIMING_FIRST_N:-0}"
export VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS="${VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS:-0}"
export VLLM_ASCEND_BUCKET_OP_PROFILE="${VLLM_ASCEND_BUCKET_OP_PROFILE:-0}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_BY_STAGE="${VLLM_ASCEND_BUCKET_OP_PROFILE_BY_STAGE:-0}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS="${VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS:-}"
export VLLM_ASCEND_DUMMY_WASTE_TIMING="${VLLM_ASCEND_DUMMY_WASTE_TIMING:-0}"
export VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC="${VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC:-0}"
export VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE="${VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE:-0}"
export VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS="${VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS:-0}"

export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-47041}"
export MASTER_PORT="${MASTER_PORT:-26040}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-47041}"

# Put launcher log and generation dumps under the A/B run directory.
export HOME="$run_dir"
export RECORD_DIR="$record_dir"
export CONFIG_DIR="$PATCH_TREE/verl/trainer/config"
export PYTHONPATH="$PATCH_TREE${PYTHONPATH:+:$PYTHONPATH}"

{
    echo "[mode3 stage-cpu-slot quality] run_dir=$run_dir"
    echo "[mode3 stage-cpu-slot quality] patch_tree=$PATCH_TREE"
    echo "[mode3 stage-cpu-slot quality] floor=$floor"
    echo "[mode3 stage-cpu-slot quality] forced direct_cpu_slot=$VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT bulk_cpu_direct=$VLLM_ASCEND_MODE3_BULK_CPU_DIRECT async_cpu_stage=$VLLM_ASCEND_MODE3_ASYNC_CPU_STAGE bulk_cpu_stage=$VLLM_ASCEND_MODE3_BULK_CPU_STAGE"
    echo "[mode3 stage-cpu-slot quality] headroom=${VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES:-launcher-default}"
    echo "[mode3 stage-cpu-slot quality] ports HCCL_IF_BASE_PORT=$HCCL_IF_BASE_PORT MASTER_PORT=$MASTER_PORT VERL_HCCL_IF_BASE_PORT_START=$VERL_HCCL_IF_BASE_PORT_START"
    git -C "$PATCH_TREE" rev-parse HEAD || true
    git -C "$PATCH_TREE" status --short || true
} | tee "$run_dir/run_header.log"

bash "$LAUNCHER" 2>&1 | tee "$run_dir/launcher.log"

latest_log=$(ls -t "$run_dir"/wjeagerqwen30b-a3b-with_draft_*_elastic.txt 2>/dev/null | head -n 1 || true)
if [[ -n "$latest_log" ]]; then
    echo "[mode3 stage-cpu-slot quality] elastic_log=$latest_log" | tee -a "$run_dir/run_header.log"
else
    echo "[mode3 stage-cpu-slot quality] WARNING: no elastic log found under $run_dir" | tee -a "$run_dir/run_header.log"
fi
