#!/usr/bin/env bash
# Qwen3-30B-A3B GRPO on vLLM/vLLM-Ascend 0.14 with native
# FULL_DECODE_ONLY ACLGraph.
#
# Prefill and mixed batches stay eager. Uniform decode captures the complete
# model, including KV write, Attention, MoE communication, and dense layers.
# This wrapper keeps the workload and validated runtime optimizations aligned
# with wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_graph_fast.sh while
# changing only the native ACLGraph policy and its explicit capture sizes.
#
# Usage:
#   bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_full_decode_only.sh dry-run
#   bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_full_decode_only.sh run
#
# For a strict one-graph comparison with the 0.11 experiment, retain the
# default FULL_DECODE_CAPTURE_SIZES=[32]. A deliberate multi-graph tail study
# can override it, for example FULL_DECODE_CAPTURE_SIZES='[1,8,16,32]'.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GRAPH_BASE="${SCRIPT_DIR}/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_graph_fast.sh"

ACTION="${1:-dry-run}"
case "$ACTION" in
    run|dry-run)
        shift
        ;;
    -h|--help)
        sed -n '2,20p' "$0"
        exit 0
        ;;
    *)
        echo "usage: $0 {dry-run|run} [additional Hydra overrides ...]" >&2
        exit 2
        ;;
esac

[[ -f "$GRAPH_BASE" ]] || {
    echo "missing graph base launcher: $GRAPH_BASE" >&2
    exit 2
}

FULL_DECODE_CAPTURE_SIZES=${FULL_DECODE_CAPTURE_SIZES:-[32]}
python3 - "$FULL_DECODE_CAPTURE_SIZES" <<'PY'
import ast
import sys

try:
    sizes = ast.literal_eval(sys.argv[1])
except (SyntaxError, ValueError) as exc:
    raise SystemExit(f"invalid FULL_DECODE_CAPTURE_SIZES: {exc}")
if not isinstance(sizes, list) or not sizes:
    raise SystemExit("FULL_DECODE_CAPTURE_SIZES must be a non-empty list")
if any(type(size) is not int or size <= 0 for size in sizes):
    raise SystemExit("FULL_DECODE_CAPTURE_SIZES must contain positive integers")
if sizes != sorted(set(sizes)):
    raise SystemExit("FULL_DECODE_CAPTURE_SIZES must be sorted and unique")
if sizes[-1] != 32:
    raise SystemExit("FULL_DECODE_CAPTURE_SIZES must end at max_num_seqs=32")
PY

# Frozen workload for the cross-version Vanilla graph comparison.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-32}
export ROLLOUT_N=${ROLLOUT_N:-16}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
export DATASET_FRACTION=${DATASET_FRACTION:-0.005}
export TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-1}
export TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS:-5}
export SAVE_CKPT_ENABLE=${SAVE_CKPT_ENABLE:-0}
export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_full_decode_only_v014}

# Native full-decode graph contract. The VLLM_ENABLE_GRAPH_MODE toggle is kept
# for the historical outer launcher gate; the actual mode is selected by the
# explicit Hydra cudagraph_mode override below.
export VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER=1
export VLLM_ROLLOUT_UNSAFE_TRUE_GRAPH_WITH_RESAMPLER=1
export VLLM_ENABLE_GRAPH_MODE=1
export ROLLOUT_ENFORCE_EAGER=False
export VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0

# Scheduler and memory settings from the fastest validated 0.14 graph path.
export VLLM_ROLLOUT_ASYNC_SCHEDULING=${VLLM_ROLLOUT_ASYNC_SCHEDULING:-true}
export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=${VLLM_ROLLOUT_ENABLE_PREFIX_CACHING:-true}
export VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=${VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL:-true}
export VLLM_ROLLOUT_MAX_NUM_BATCHED_TOKENS=${VLLM_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-17408}
export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=${VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}
export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
export VLLM_ROLLOUT_FREE_CACHE_ENGINE=True
export VLLM_ROLLOUT_SLEEP_LEVEL=1
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=0
export VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1
export MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1

# Keep the validated high-performance kernels and filtered graph OPP bundle.
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=${VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP:-1}
export VLLM_ASCEND_USE_LEGACY_FUSED_MOE=0
export VLLM_ASCEND_USE_LEGACY_ATTENTION=0
export VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE=0
export VLLM_ASCEND_FORCE_ALLTOALL_MOE=0
export VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE=0
export VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM=${VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM:-1}
export VLLM_ASCEND_MC2_TOKENS_CAPACITY=512
export VLLM_ASCEND_MC2_GLOBAL_BS=0
export VLLM_ASCEND_MC2_MIN_EP_SIZE=2

# The full-decode path is native ACLGraph, not TorchAir. Inductor/GraphEX
# fusion is not required for capture or replay in this 0.14 implementation.
export VLLM_ASCEND_FORCE_CUDAGRAPH_NONE=0
export VLLM_ASCEND_FORCE_COMPILE_WITHOUT_ACLGRAPH=0
export VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE=0
export NPUGRAPH_EX_ENABLE_STATIC_KERNEL=False

# Match the validated 0.14 RL weight/sleep lifecycle. Graph entries keep stable
# parameter addresses and are not rebuilt after every policy synchronization.
export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
export VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
export VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD=0
export VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD=0

# Full graph must not inherit PIECEWISE-only input/Attention update experiments.
export VLLM_ASCEND_PIECEWISE_COPY_INPUTS=0
export VLLM_ASCEND_UPDATE_PIECEWISE_GRAPH_ATTN=0
export VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC=0

HYDRA_ARGS=(
    actor_rollout_ref.rollout.seed=0
    actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY
    "actor_rollout_ref.rollout.cudagraph_capture_sizes=${FULL_DECODE_CAPTURE_SIZES}"
)
HYDRA_ARGS+=("$@")

if [[ "$ACTION" == dry-run ]]; then
    cat <<EOF
launcher=$GRAPH_BASE
output_subdir=$OUTPUT_SUBDIR
vllm_stack=0.14
cudagraph_mode=FULL_DECODE_ONLY
capture_sizes=$FULL_DECODE_CAPTURE_SIZES
prefill=eager
decode=full_aclgraph
task_queue_enable=$TASK_QUEUE_ENABLE
async_scheduling=$VLLM_ROLLOUT_ASYNC_SCHEDULING
prefix_caching=$VLLM_ROLLOUT_ENABLE_PREFIX_CACHING
chunked_prefill=$VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL
gpu_memory_utilization=$VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION
steps=$TRAINER_TOTAL_TRAINING_STEPS
EOF
    printf 'command=bash %q' "$GRAPH_BASE"
    printf ' %q' "${HYDRA_ARGS[@]}"
    printf '\n'
    exit 0
fi

exec bash "$GRAPH_BASE" "${HYDRA_ARGS[@]}"
