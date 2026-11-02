#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_RUNNER="$SCRIPT_DIR/run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh"
COMPARE_TOOL="$SCRIPT_DIR/tools/compare_qwen3_full_decode_buckets.py"
ACTION="${1:-dry-run}"

BASELINE_ROOT="${FULL_DECODE_BUCKET_BASELINE_ROOT:-/workspace/adafloor_graph_results/qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1_allocatorfix_seed0}"
CANDIDATE_ROOT="${FULL_DECODE_EPOCH0_ROOT:-/workspace/adafloor_graph_results/qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1_buckets16_32_seed0}"
CANDIDATE_RUN_NAME="${FULL_DECODE_EPOCH0_RUN_NAME:-common_epoch0_full_decode_fia_v014opt_tq1_buckets16_32_seed0}"

case "$ACTION" in
    run|dry-run|summarize|compare) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1_buckets16_32.sh dry-run
  ./run_qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1_buckets16_32.sh run
  ./run_qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1_buckets16_32.sh summarize
  ./run_qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1_buckets16_32.sh compare

Runs the validated Qwen3-30B-A3B vLLM-Ascend 0.11 FULL_DECODE_ONLY/FIA
Vanilla epoch0 protocol with exactly two decode capture sizes, [16,32]. All
other workload and runtime settings are inherited unchanged from the validated
v014-runtime-port [32] runner.

The run and summarize actions also compare the completed candidate against the
existing [32] result. The script never starts NPU work for dry-run, summarize,
or compare.
EOF
        exit 0
        ;;
    *)
        echo "unknown action: $ACTION" >&2
        exit 2
        ;;
esac

[[ -x "$BASE_RUNNER" ]] || {
    echo "missing optimized FULL graph runner: $BASE_RUNNER" >&2
    exit 2
}
[[ -f "$COMPARE_TOOL" ]] || {
    echo "missing bucket comparison tool: $COMPARE_TOOL" >&2
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

compare_buckets() {
    python3 "$COMPARE_TOOL" \
        --baseline-root "$BASELINE_ROOT" \
        --candidate-root "$CANDIDATE_ROOT"
}

if [[ "$ACTION" == compare ]]; then
    compare_buckets
    exit 0
fi

# This is a deliberately fixed experiment contract. Do not allow an inherited
# environment to silently turn the candidate back into the [32] baseline.
export FULL_DECODE_CAPTURE_SIZES='[16,32]'
export FULL_DECODE_EPOCH0_ROOT="$CANDIDATE_ROOT"
export FULL_DECODE_EPOCH0_RUN_NAME="$CANDIDATE_RUN_NAME"
export FULL_DECODE_EXTRA_CODE_PATH="$SCRIPT_DIR/run_qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1_buckets16_32.sh"

# Match the validated v014-runtime-port [32] arm exactly, except for the graph
# bucket list above.
export FULL_DECODE_OPTIMIZATION_PROFILE=v014_runtime_port
export VLLM_ROLLOUT_NATIVE_SLEEP_MODE=0
unset VLLM_ROLLOUT_SLEEP_LEVEL
export VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=0
export VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1
unset MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VERL_FULL_DECODE_MAX_UPDATE_ACTOR_SECONDS="${VERL_FULL_DECODE_MAX_UPDATE_ACTOR_SECONDS:-180}"
export VLLM_ROLLOUT_ASYNC_SCHEDULING=false
export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true
export VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true
export VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1
export VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH="$FILTERED_OPP"
export VLLM_ASCEND_FILTERED_CUSTOM_OPP_BUNDLE_SHA256="$FILTERED_OPP_BUNDLE_SHA256"
export VLLM_ASCEND_FORCE_ALLTOALL_MOE=0
export VLLM_ASCEND_MC2_MIN_EP_SIZE=2
export VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1
export VLLM_ENABLE_MC2=1
export TASK_QUEUE_ENABLE=1

"$BASE_RUNNER" "$ACTION"

if [[ "$ACTION" == run || "$ACTION" == summarize ]]; then
    compare_buckets
fi
