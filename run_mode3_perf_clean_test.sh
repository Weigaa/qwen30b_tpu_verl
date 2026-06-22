#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_mode3_perf_clean_test.sh [floor]
  ./run_mode3_perf_clean_test.sh --floor 1

Environment overrides:
  MODE3_FLOOR=2
  REPO_ROOT=/path/to/log/output/root
  PATCH_TREE=/path/to/code/tree
  HCCL_IF_BASE_PORT=47041
  MASTER_PORT=26040
  VERL_HCCL_IF_BASE_PORT_START=47041

Examples:
  ./run_mode3_perf_clean_test.sh        # defaults to floor=2
  ./run_mode3_perf_clean_test.sh 1
  ./run_mode3_perf_clean_test.sh --floor 4
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
PATCH_TREE="${PATCH_TREE:-$REPO_ROOT}"
LAUNCHER="$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"

floor="${MODE3_FLOOR:-2}"
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
# Quality/KV validation baseline: bind mode=3 to the direct CPU -> final
# runtime slot path. The old staging path remains available via the dedicated
# run_mode3_floor4_stage_cpu_slot_quality_test.sh wrapper.
export VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT=${VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT:-1}
export VLLM_ASCEND_MODE3_RELEASE_CPU_STAGE_ON_DIRECT=${VLLM_ASCEND_MODE3_RELEASE_CPU_STAGE_ON_DIRECT:-1}
export VLLM_ASCEND_MODE3_RELEASE_RUNTIME_ON_STAGE_CHANGE=${VLLM_ASCEND_MODE3_RELEASE_RUNTIME_ON_STAGE_CHANGE:-1}
export VLLM_ASCEND_MODE3_DYNAMIC_RUNTIME_CAPACITY=${VLLM_ASCEND_MODE3_DYNAMIC_RUNTIME_CAPACITY:-1}
export VLLM_ASCEND_MODE3_REBUILD_MC2_ON_RESTORE=${VLLM_ASCEND_MODE3_REBUILD_MC2_ON_RESTORE:-0}
export VLLM_ASCEND_MODE3_BULK_CPU_DIRECT=${VLLM_ASCEND_MODE3_BULK_CPU_DIRECT:-1}
export VLLM_ASCEND_MODE3_CPU_DIRECT_SYNC_COPY=${VLLM_ASCEND_MODE3_CPU_DIRECT_SYNC_COPY:-0}
export VLLM_ASCEND_MODE3_CPU_STAGE_HEADROOM_BYTES=${VLLM_ASCEND_MODE3_CPU_STAGE_HEADROOM_BYTES:-0}
export VLLM_ASCEND_MODE3_DENSE_RUNTIME_EXPERT_MAP=${VLLM_ASCEND_MODE3_DENSE_RUNTIME_EXPERT_MAP:-1}
export VLLM_ASCEND_MODE3_PERSIST_DISPATCHER_DOMAIN=${VLLM_ASCEND_MODE3_PERSIST_DISPATCHER_DOMAIN:-0}
# Keep mode=3 communicator lifecycle aligned with mode=4/5 by default:
# stale shrink groups are cleared on restore, not destroyed in shrink hot path.
export VLLM_ASCEND_MODE3_DROP_STALE_GROUP_CACHE_AFTER_SHRINK=${VLLM_ASCEND_MODE3_DROP_STALE_GROUP_CACHE_AFTER_SHRINK:-0}
export VLLM_ASCEND_MODE3_DROP_STALE_MC2_GROUP_CACHE_AFTER_SHRINK=${VLLM_ASCEND_MODE3_DROP_STALE_MC2_GROUP_CACHE_AFTER_SHRINK:-0}
export VLLM_ASCEND_MODE3_KEEP_STALE_MC2_ON_ACTIVE_RANKS=${VLLM_ASCEND_MODE3_KEEP_STALE_MC2_ON_ACTIVE_RANKS:-0}
export VLLM_ASCEND_MODE3_DEFER_GROUP_DESTROY=${VLLM_ASCEND_MODE3_DEFER_GROUP_DESTROY:-0}
export VLLM_ASCEND_MODE3_DEFER_DESTROY_FLOOR_GROUP_SIZES=${VLLM_ASCEND_MODE3_DEFER_DESTROY_FLOOR_GROUP_SIZES:-1,2,4,8}
export VLLM_ASCEND_MODE3_DESTROY_DEVICE_PG_ON_RETIRE=${VLLM_ASCEND_MODE3_DESTROY_DEVICE_PG_ON_RETIRE:-1}
# Diagnostic only. The synthetic MC2 dispatcher warmup can expose a 300s HCCL
# timeout on floor=4; keep it off unless explicitly bisecting that path.
export VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP=${VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP:-0}
export VLLM_ASCEND_MODE3_FORCE_DISPATCHER_WARMUP=${VLLM_ASCEND_MODE3_FORCE_DISPATCHER_WARMUP:-0}
export VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP_TOKENS=${VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP_TOKENS:-32}
export VLLM_ASCEND_MODE3_ENABLE_POST_SHRINK_DP_WARMUP=${VLLM_ASCEND_MODE3_ENABLE_POST_SHRINK_DP_WARMUP:-0}
export VLLM_ASCEND_MODE3_TRANSFER_LOG=0
export VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG=0
export VLLM_ASCEND_MODE3_TIMING_LOG=0
export VLLM_ASCEND_MODE3_TIMING_SYNC=0
export VLLM_ASCEND_MODE3_TIMING_EVERY=1000000
export VLLM_ASCEND_MODE3_TIMING_FIRST_N=0
export VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING=${VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING:-1}
export VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_SYNC=${VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_SYNC:-0}
export VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_FIRST_N=${VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_FIRST_N:-4}
export VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_THRESHOLD_MS=${VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_THRESHOLD_MS:-1000}
export VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_LAYERS=${VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_LAYERS:-all}
export VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_MAX_STAGE=${VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_MAX_STAGE:-4}
export VLLM_ASCEND_MC2_HOST_TIMING_LOG=${VLLM_ASCEND_MC2_HOST_TIMING_LOG:-1}
export VLLM_ASCEND_MC2_HOST_TIMING_FIRST_N=${VLLM_ASCEND_MC2_HOST_TIMING_FIRST_N:-4}
export VLLM_ASCEND_MC2_HOST_TIMING_THRESHOLD_MS=${VLLM_ASCEND_MC2_HOST_TIMING_THRESHOLD_MS:-1000}
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
printf '[mode3 perf clean] dense_runtime_expert_map=%s persist_dispatcher_domain=%s dynamic_runtime_capacity=%s\n' \
    "$VLLM_ASCEND_MODE3_DENSE_RUNTIME_EXPERT_MAP" \
    "$VLLM_ASCEND_MODE3_PERSIST_DISPATCHER_DOMAIN" \
    "$VLLM_ASCEND_MODE3_DYNAMIC_RUNTIME_CAPACITY"
printf '[mode3 perf clean] restore_rebuild_mc2=%s\n' \
    "$VLLM_ASCEND_MODE3_REBUILD_MC2_ON_RESTORE"
printf '[mode3 perf clean] mc2_host_timing=%s first_n=%s threshold_ms=%s\n' \
    "$VLLM_ASCEND_MC2_HOST_TIMING_LOG" \
    "$VLLM_ASCEND_MC2_HOST_TIMING_FIRST_N" \
    "$VLLM_ASCEND_MC2_HOST_TIMING_THRESHOLD_MS"
printf '[mode3 perf clean] stale_drop=%s stale_mc2_drop=%s keep_active_mc2=%s defer_destroy=%s defer_sizes=%s destroy_device_pg_on_retire=%s\n' \
    "$VLLM_ASCEND_MODE3_DROP_STALE_GROUP_CACHE_AFTER_SHRINK" \
    "$VLLM_ASCEND_MODE3_DROP_STALE_MC2_GROUP_CACHE_AFTER_SHRINK" \
    "$VLLM_ASCEND_MODE3_KEEP_STALE_MC2_ON_ACTIVE_RANKS" \
    "$VLLM_ASCEND_MODE3_DEFER_GROUP_DESTROY" \
    "$VLLM_ASCEND_MODE3_DEFER_DESTROY_FLOOR_GROUP_SIZES" \
    "$VLLM_ASCEND_MODE3_DESTROY_DEVICE_PG_ON_RETIRE"
printf '[mode3 perf clean] profile=%s timing_log=%s timing_sync=%s markers=%s
' "$VLLM_ASCEND_BUCKET_OP_PROFILE" "$VLLM_ASCEND_MODE3_TIMING_LOG" "$VLLM_ASCEND_MODE3_TIMING_SYNC" "$VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS"
printf '[mode3 perf clean] post_shrink_warmup=%s force_dispatcher_warmup=%s warmup_tokens=%s moe_forward_timing=%s/%s threshold_ms=%s layers=%s max_stage=%s\n' \
    "$VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP" \
    "$VLLM_ASCEND_MODE3_FORCE_DISPATCHER_WARMUP" \
    "$VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP_TOKENS" \
    "$VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING" \
    "$VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_SYNC" \
    "$VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_THRESHOLD_MS" \
    "$VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_LAYERS" \
    "$VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING_MAX_STAGE"
printf '[mode3 perf clean] post_shrink_dp_warmup=%s\n' \
    "$VLLM_ASCEND_MODE3_ENABLE_POST_SHRINK_DP_WARMUP"
printf '[mode3 perf clean] ports HCCL_IF_BASE_PORT=%s MASTER_PORT=%s VERL_HCCL_IF_BASE_PORT_START=%s
' "$HCCL_IF_BASE_PORT" "$MASTER_PORT" "$VERL_HCCL_IF_BASE_PORT_START"

bash "$LAUNCHER" 2>&1 | tee "$tee_log"
