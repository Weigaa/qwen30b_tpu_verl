#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_RUNNER="$SCRIPT_DIR/run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh"
ACTION="${1:-dry-run}"

case "$ACTION" in
    run|dry-run|summarize) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1.sh dry-run
  ./run_qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1.sh run
  ./run_qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1.sh summarize

Runs the same Qwen3-30B-A3B Vanilla Full16 epoch0 protocol as the validated
0.11 FULL/FIA script, with the compatible runtime optimizations ported from
the fastest vLLM/vLLM-Ascend 0.14 graph configuration. This script does not
change the seed, five-step workload, n=16, max length, KV contract, or model.
EOF
        exit 0
        ;;
    *)
        echo "unknown action: $ACTION" >&2
        exit 2
        ;;
esac

[[ -x "$BASE_RUNNER" ]] || {
    echo "missing base FULL graph runner: $BASE_RUNNER" >&2
    exit 2
}

FILTERED_OPP="${VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH:-$SCRIPT_DIR/../qwen3/vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend}"
[[ -d "$FILTERED_OPP" ]] || {
    echo "missing filtered 0.14 custom OPP package: $FILTERED_OPP" >&2
    exit 2
}
FILTERED_OPP=$(realpath "$FILTERED_OPP")
FILTERED_OPP_BUNDLE_SHA256=$(
    find "$FILTERED_OPP" -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        | sha256sum \
        | awk '{print $1}'
)

# Preserve the validated 0.11 FULL_DECODE_ONLY/FIA graph implementation.
export FULL_DECODE_OPTIMIZATION_PROFILE=v014_runtime_port
export FULL_DECODE_EXTRA_CODE_PATH="$SCRIPT_DIR/run_qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1.sh"
export FULL_DECODE_EPOCH0_ROOT="${FULL_DECODE_EPOCH0_ROOT:-/workspace/adafloor_graph_results/qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1_seed0}"
export FULL_DECODE_EPOCH0_RUN_NAME="${FULL_DECODE_EPOCH0_RUN_NAME:-common_epoch0_full_decode_fia_v014opt_tq1_seed0}"
export FULL_DECODE_CAPTURE_SIZES="${FULL_DECODE_CAPTURE_SIZES:-[32]}"

# 0.14 runtime lifecycle port. CaMem restores the same parameter/KV virtual
# addresses. Runtime pointer guards fall back to invalidate+recapture if the
# older 0.11 loader changes an address.
export VLLM_ROLLOUT_NATIVE_SLEEP_MODE=1
export VLLM_ROLLOUT_SLEEP_LEVEL=1
export VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=0
export VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1
export MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-garbage_collection_threshold:0.6,max_split_size_mb:24}"

# Scheduler and prompt-work optimizations used by the fastest 0.14 graph run.
export VLLM_ROLLOUT_ASYNC_SCHEDULING=true
export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true
export VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true

# Keep graph-only custom ops while using the system CANN MoE dispatch/combine
# kernels. This is a runtime artifact dependency, so its content hash is sealed
# into protocol.env by the base runner.
export VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1
export VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH="$FILTERED_OPP"
export VLLM_ASCEND_FILTERED_CUSTOM_OPP_BUNDLE_SHA256="$FILTERED_OPP_BUNDLE_SHA256"

# Fast 0.11 backends that have real consumers in this source tree.
export VLLM_ASCEND_FORCE_ALLTOALL_MOE=0
export VLLM_ASCEND_MC2_MIN_EP_SIZE=2
export VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1
export VLLM_ENABLE_MC2=1
export TASK_QUEUE_ENABLE=1

exec "$BASE_RUNNER" "$ACTION"
