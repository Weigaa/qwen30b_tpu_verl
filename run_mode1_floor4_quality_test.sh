#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./run_mode1_floor4_quality_test.sh [floor]
  ./run_mode1_floor4_quality_test.sh --floor 4

Purpose:
  Run mode=1 with an isolated RECORD_DIR so the generated 1.jsonl can be
  compared against the mode=3 direct/stage quality runs.

Environment overrides:
  MODE1_FLOOR=4
  REPO_ROOT=/path/to/output/root
  PATCH_TREE=/path/to/code/tree
  HCCL_IF_BASE_PORT=47241
  MASTER_PORT=26240
  VERL_HCCL_IF_BASE_PORT_START=47241
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
PATCH_TREE="${PATCH_TREE:-$SCRIPT_DIR}"
LAUNCHER="$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"

floor="${MODE1_FLOOR:-4}"
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

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$REPO_ROOT/mode_quality_compare_runs/${stamp}_floor${floor}_mode1"
record_dir="$run_dir/record"
mkdir -p "$record_dir"

cd "$PATCH_TREE"

before_list=$(mktemp)
find "$PATCH_TREE" -maxdepth 1 -name 'wjeagerqwen30b-a3b-with_draft_*_elastic.txt' -printf '%f\n' | sort > "$before_list"

export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="$floor"
export VLLM_ASCEND_CUSTOM_MODE1_DEBUG=0
export VLLM_ASCEND_CUSTOM_MODE1_TIMING_EVENTS=0
export VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG=0

# Match the stable mode=1 path used by run_mode1_perf_clean_test.sh.
export VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT="${VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT:-1}"
export VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT="${VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT:-0}"
export VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT="${VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT:-0}"
export VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS="${VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS:-8}"
export VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC="${VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC:-1}"

# Keep perf-clean diagnostics unless explicitly overridden.
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

export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-47241}"
export MASTER_PORT="${MASTER_PORT:-26240}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-47241}"

export RECORD_DIR="$record_dir"
export CONFIG_DIR="$PATCH_TREE/verl/trainer/config"
export PYTHONPATH="$PATCH_TREE${PYTHONPATH:+:$PYTHONPATH}"

{
    echo "[mode1 quality] run_dir=$run_dir"
    echo "[mode1 quality] patch_tree=$PATCH_TREE"
    echo "[mode1 quality] floor=$floor"
    echo "[mode1 quality] mode1_batch_direct_npu=$VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT allow_scalar=$VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT batch_experts=$VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS cpu_dp_metadata_sync=$VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC"
    echo "[mode1 quality] headroom=${VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES:-launcher-default}"
    echo "[mode1 quality] ports HCCL_IF_BASE_PORT=$HCCL_IF_BASE_PORT MASTER_PORT=$MASTER_PORT VERL_HCCL_IF_BASE_PORT_START=$VERL_HCCL_IF_BASE_PORT_START"
    git -C "$PATCH_TREE" rev-parse HEAD || true
    git -C "$PATCH_TREE" status --short || true
} | tee "$run_dir/run_header.log"

bash "$LAUNCHER" 2>&1 | tee "$run_dir/launcher.log"

latest_log=$(comm -13 "$before_list" <(find "$PATCH_TREE" -maxdepth 1 -name 'wjeagerqwen30b-a3b-with_draft_*_elastic.txt' -printf '%f\n' | sort) | tail -n 1 || true)
rm -f "$before_list"
if [[ -n "$latest_log" && -f "$PATCH_TREE/$latest_log" ]]; then
    cp -a "$PATCH_TREE/$latest_log" "$run_dir/"
    echo "[mode1 quality] elastic_log=$run_dir/$latest_log" | tee -a "$run_dir/run_header.log"
else
    echo "[mode1 quality] WARNING: could not identify new elastic log; check $PATCH_TREE" | tee -a "$run_dir/run_header.log"
fi
