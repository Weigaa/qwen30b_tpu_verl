#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Usage:
  CASE=current KV_TOKENS=50000  bash run_mode1_floor2_diag_kv_lifecycle_ab.sh [hydra args...]
  CASE=current KV_TOKENS=100000 bash run_mode1_floor2_diag_kv_lifecycle_ab.sh [hydra args...]
  CASE=no_tail KV_TOKENS=100000 bash run_mode1_floor2_diag_kv_lifecycle_ab.sh [hydra args...]

Purpose:
  Isolate the floor=2 step2+ slowdown by changing exactly two variables:
    1. KV cache budget.
    2. shrink/restore lifecycle cleanup/sync policy.

Cases:
  current
    Use the current floor2 script behavior.

  no_tail
    Keep MoE runtime release/cleanup, but remove shrink/restore tail NPU sync
    and CPU barrier. If this removes the ~50s step2+ gap, the regression is
    mostly tail synchronization/waiting after lifecycle transitions.

  no_moe_release
    Keep tail sync/barrier, but do not release MoE comm runtime transient state
    during shrink/restore. If this improves speed but increases memory, the
    release path is the latency source and needs a lighter cleanup strategy.

  no_release_no_tail
    Disable both MoE runtime release and tail guards. This is unsafe for long
    runs but useful to prove whether lifecycle maintenance is the slowdown
    source.

  release_no_empty_cache
    Still drop transient MoE comm runtime Python references, but avoid explicit
    synchronize/empty_cache in that release path. This tests allocator pressure
    vs explicit cleanup overhead.

Suggested first pass:
  CASE=current KV_TOKENS=50000  bash run_mode1_floor2_diag_kv_lifecycle_ab.sh
  CASE=current KV_TOKENS=100000 bash run_mode1_floor2_diag_kv_lifecycle_ab.sh
  CASE=no_tail KV_TOKENS=100000 bash run_mode1_floor2_diag_kv_lifecycle_ab.sh
  CASE=release_no_empty_cache KV_TOKENS=100000 bash run_mode1_floor2_diag_kv_lifecycle_ab.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

CASE="${CASE:-current}"
KV_TOKENS="${KV_TOKENS:-100000}"
STEPS_TAG="${STEPS_TAG:-5steps}"

export KV_TOKENS
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export DATASET_FRACTION="${DATASET_FRACTION:-0.005}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-640}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-128,256,384,640,640}"

# Keep the backend fixed while testing lifecycle/KV effects.
export VLLM_ASCEND_FORCE_MC2_MOE="${VLLM_ASCEND_FORCE_MC2_MOE:-1}"
export VLLM_ASCEND_FORCE_ALLTOALL_MOE="${VLLM_ASCEND_FORCE_ALLTOALL_MOE:-0}"
export VLLM_ASCEND_AVOID_ALLTOALL_ON_SHRINK_STAGE="${VLLM_ASCEND_AVOID_ALLTOALL_ON_SHRINK_STAGE:-1}"
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP="${VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP:-0}"
export VLLM_ASCEND_MODE1_PARITY_ALLTOALL_HOST_METADATA="${VLLM_ASCEND_MODE1_PARITY_ALLTOALL_HOST_METADATA:-0}"
export VLLM_ASCEND_POST_SHRINK_PREFILL_ALLTOALL_HEADROOM_BYTES="${VLLM_ASCEND_POST_SHRINK_PREFILL_ALLTOALL_HEADROOM_BYTES:-0}"
export VLLM_ASCEND_RESERVED_MC2_MASK_TOKENS="${VLLM_ASCEND_RESERVED_MC2_MASK_TOKENS:-4096}"

case "$CASE" in
    current)
        ;;
    no_tail)
        export VLLM_ASCEND_MODE1_RESTORE_TAIL_NPU_SYNC=0
        export VLLM_ASCEND_MODE1_RESTORE_TAIL_CPU_BARRIER=0
        export VLLM_ASCEND_MODE1_SHRINK_TAIL_NPU_SYNC=0
        export VLLM_ASCEND_MODE1_SHRINK_TAIL_CPU_BARRIER=0
        ;;
    no_moe_release)
        export VLLM_ASCEND_MODE1_RELEASE_MOE_COMM_RUNTIME_ON_SHRINK=0
        export VLLM_ASCEND_MODE1_RELEASE_MOE_COMM_RUNTIME_ON_RESTORE=0
        ;;
    no_release_no_tail)
        export VLLM_ASCEND_MODE1_RELEASE_MOE_COMM_RUNTIME_ON_SHRINK=0
        export VLLM_ASCEND_MODE1_RELEASE_MOE_COMM_RUNTIME_ON_RESTORE=0
        export VLLM_ASCEND_MODE1_RESTORE_TAIL_NPU_SYNC=0
        export VLLM_ASCEND_MODE1_RESTORE_TAIL_CPU_BARRIER=0
        export VLLM_ASCEND_MODE1_SHRINK_TAIL_NPU_SYNC=0
        export VLLM_ASCEND_MODE1_SHRINK_TAIL_CPU_BARRIER=0
        ;;
    release_no_empty_cache)
        export VLLM_ASCEND_MODE1_RELEASE_MOE_COMM_RUNTIME_ON_SHRINK=1
        export VLLM_ASCEND_MODE1_RELEASE_MOE_COMM_RUNTIME_ON_RESTORE=1
        export VLLM_ASCEND_MODE1_RELEASE_MOE_COMM_RUNTIME_EMPTY_CACHE=0
        export VLLM_ASCEND_MODE1_SYNC_BEFORE_MOE_COMM_RELEASE=0
        export VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE=0
        export VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC=0
        export VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE=0
        export VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE=0
        ;;
    *)
        echo "unknown CASE=$CASE" >&2
        usage >&2
        exit 2
        ;;
esac

PORT_BASE="${PORT_BASE:-$((30000 + ($$ % 20000)))}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-$((PORT_BASE))-$((PORT_BASE + 511))}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-$((PORT_BASE + 512))-$((PORT_BASE + 1023))}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-$((PORT_BASE + 1024))}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-$HCCL_IF_BASE_PORT}"
export MASTER_PORT="${MASTER_PORT:-$((PORT_BASE + 1536))}"

export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_baseline_random_batch_floor2_diag_${CASE}_kv${KV_TOKENS}_${STEPS_TAG}}"

printf '[floor2 kv lifecycle ab] case=%s kv=%s output=%s\n' "$CASE" "$KV_TOKENS" "$OUTPUT_SUBDIR"
printf '[floor2 kv lifecycle ab] release_shrink=%s release_restore=%s release_empty_cache=%s sync_before_release=%s\n' \
    "${VLLM_ASCEND_MODE1_RELEASE_MOE_COMM_RUNTIME_ON_SHRINK:-<base>}" \
    "${VLLM_ASCEND_MODE1_RELEASE_MOE_COMM_RUNTIME_ON_RESTORE:-<base>}" \
    "${VLLM_ASCEND_MODE1_RELEASE_MOE_COMM_RUNTIME_EMPTY_CACHE:-<base>}" \
    "${VLLM_ASCEND_MODE1_SYNC_BEFORE_MOE_COMM_RELEASE:-<base>}"
printf '[floor2 kv lifecycle ab] restore_tail_sync=%s restore_tail_barrier=%s shrink_tail_sync=%s shrink_tail_barrier=%s ports=%s/%s master=%s\n' \
    "${VLLM_ASCEND_MODE1_RESTORE_TAIL_NPU_SYNC:-<base>}" \
    "${VLLM_ASCEND_MODE1_RESTORE_TAIL_CPU_BARRIER:-<base>}" \
    "${VLLM_ASCEND_MODE1_SHRINK_TAIL_NPU_SYNC:-<base>}" \
    "${VLLM_ASCEND_MODE1_SHRINK_TAIL_CPU_BARRIER:-<base>}" \
    "$HCCL_NPU_SOCKET_PORT_RANGE" "$HCCL_HOST_SOCKET_PORT_RANGE" "$MASTER_PORT"

exec "$SCRIPT_DIR/run_mode1_local_baseline_random_batch_floor2_threshold_multistep.sh" "$@"
