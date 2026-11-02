#!/usr/bin/env bash
# Async rollout variant built on top of the regroup script.
# This entrypoint is the qwen3 eager fast path: keep the efficient scheduler
# shape observed in old-stack runs, but use the newer vLLM-Ascend eager
# attention / split-MoE implementation by default instead of forcing legacy
# kernels back in.
# Scheduler semantics:
#   - async_scheduling = true
#   - enable_prefix_caching = false in the initialized engine
#   - max_num_batched_tokens = 17408
#   - enable_chunked_prefill = true
# Legacy attention / legacy fused MoE remain opt-in diagnostic paths only.
# Fast qwen3 rollout entrypoint.
#
# Keep only two performance switches exposed:
#   - VLLM_ROLLOUT_ASYNC_SCHEDULING: true/false
#   - VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER: 0/1
#
# Everything else is pinned to the best configuration validated during the
# qwen3-vs-old-stack rollout perf investigation. This avoids stale shell
# exports accidentally moving the run back onto slower legacy/debug paths.
set -ex

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="${SCRIPT_DIR}/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"

export VLLM_ROLLOUT_ASYNC_SCHEDULING=${VLLM_ROLLOUT_ASYNC_SCHEDULING:-true}
export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=${VLLM_ROLLOUT_ENABLE_PREFIX_CACHING:-false}
export VLLM_ROLLOUT_MAX_NUM_BATCHED_TOKENS=${VLLM_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-17408}
export VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=${VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL:-true}
export VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER=${VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER:-0}

# Scheduler semantics matched to the best eager run.
export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=false
export VLLM_ROLLOUT_MAX_NUM_BATCHED_TOKENS=17408
# Default to chunked prefill for the validated eager path, but allow explicit
# no-chunk experiments while keeping the full 17408 scheduling window.
export VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=${VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL:-true}

if [ "${VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER}" = "1" ]; then
    export VLLM_ENABLE_GRAPH_MODE=1
    export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
    export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=${VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}
    export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=${VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE:-0}
    export ROLLOUT_ENFORCE_EAGER=False
    export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=0.65
    export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
    export VLLM_ASCEND_EAGER_COMPILE=0
else
    # Stable eager fast path: keep eager semantics and disable full graph
    # replay, but preserve vLLM's compile wrapper.  In threshold runs this
    # enforce_eager=True + VLLM_ASCEND_EAGER_COMPILE=1 baseline had the best
    # end-to-end rollout/gen time; the no-enforce/no-ACLGraph variant was more
    # variable and did not beat it.
    export VLLM_ENABLE_GRAPH_MODE=0
    export ROLLOUT_ENFORCE_EAGER=True
    unset VLLM_ASCEND_FORCE_CUDAGRAPH_NONE
    unset VLLM_ASCEND_FORCE_COMPILE_WITHOUT_ACLGRAPH
    # Keep this aligned with the best threshold eager run.  0.83 leaves a bit
    # more workspace headroom for compiled-eager split-MoE/attention than 0.85
    # and was marginally faster in the comparable 441-token threshold sweep.
    export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=0.83
    export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=1
    export VLLM_ASCEND_EAGER_COMPILE=1
fi

# RL rollout fully reloads weights after wake. Level 2 avoids backing up the
# model buffers inside vllm-ascend's native sleep mode.
export VLLM_ROLLOUT_SLEEP_LEVEL=${VLLM_ROLLOUT_SLEEP_LEVEL:-2}
export VLLM_ROLLOUT_SLEEP_LEVEL=2
# Keep reload-time MoE post-processing allocations inside vLLM-Ascend's
# sleep-mode weight pool so phase switching can release them.
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=${VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD:-1}
# Optional old RL phase-switch experiment: disable native vLLM sleep and
# manually free KV cache / drop parameter storage like the old repo.
#
# Prefer the old verl phase switch for this perf probe. The qwen3 rollout code
# restores split-MoE parameters to loader layout before each load_weights call,
# then process_weights_after_loading converts them back to the new execution
# layout. This keeps the new split-MoE runtime while avoiding native CaMem
# sleep's multi-second per-step release cost.
export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=${VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE:-1}
export VLLM_EAGER_BASELINE_NO_RESAMPLE=${VLLM_EAGER_BASELINE_NO_RESAMPLE:-0}
# Pin the default perf probe to the qwen3 fast path. Use the rollout-scoped
# variables below for explicit legacy experiments so stale shell exports do not
# accidentally switch this script onto a slower/debug path.
export VLLM_ASCEND_USE_LEGACY_FUSED_MOE=${VLLM_ROLLOUT_USE_LEGACY_FUSED_MOE:-0}
export VLLM_ASCEND_USE_LEGACY_ATTENTION=${VLLM_ROLLOUT_USE_LEGACY_ATTENTION:-0}
export VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE=${VLLM_ROLLOUT_LEGACY_ATTENTION_SPLITFUSE:-0}
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
export VLLM_EAGER_BASELINE_NO_RESAMPLE=0

# Pin backend kernels to the validated qwen3 fast path by default, while still
# letting explicit rollout-scoped diagnostic knobs override it.
export VLLM_ASCEND_USE_LEGACY_FUSED_MOE=${VLLM_ROLLOUT_USE_LEGACY_FUSED_MOE:-0}
export VLLM_ASCEND_USE_LEGACY_ATTENTION=${VLLM_ROLLOUT_USE_LEGACY_ATTENTION:-0}
export VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE=${VLLM_ROLLOUT_LEGACY_ATTENTION_SPLITFUSE:-0}

# Keep the MC2 active-mask pool at the old eager rollout size. The qwen3 auto
# split-MoE eager path uses MC2_GLOBAL_BS=0.
export VLLM_ASCEND_MC2_TOKENS_CAPACITY=${VLLM_ASCEND_MC2_TOKENS_CAPACITY:-512}
export VLLM_ASCEND_MC2_GLOBAL_BS=${VLLM_ASCEND_MC2_GLOBAL_BS:-0}
export VLLM_ASCEND_MC2_MIN_EP_SIZE=${VLLM_ASCEND_MC2_MIN_EP_SIZE:-2}
export VLLM_ASCEND_ENABLE_FUSED_MC2=${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}
export VLLM_ASCEND_FORCE_ALLTOALL_MOE=${VLLM_ASCEND_FORCE_ALLTOALL_MOE:-0}
export VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE=${VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE:-0}
export VLLM_ASCEND_EAGER_COMPILE=${VLLM_ASCEND_EAGER_COMPILE:-1}
export VLLM_ROLLOUT_FORCE_ELASTIC_MOE_POLICY=${VLLM_ROLLOUT_FORCE_ELASTIC_MOE_POLICY:-1}
export VLLM_ASCEND_ATTENTION_BLOCK_SIZE=${VLLM_ASCEND_ATTENTION_BLOCK_SIZE:-64}
export VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=${VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE:-1}
export VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=${VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2:-1}
export VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM=${VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM:-0}
export VLLM_ROLLOUT_USE_TQDM=${VLLM_ROLLOUT_USE_TQDM:-0}
# Keep repeated-prompt locality by default. Rebalancing remains available for
# diagnostics, but it spreads one copy of every prompt to every DP worker and
# substantially hurts n=16 prefill/locality in this workload.
export VLLM_ROLLOUT_DATA_REBALANCE=${VLLM_ROLLOUT_DATA_REBALANCE:-0}
# Preserve repeated-prompt locality by default. Length balancing is useful as a
# diagnostic knob, but the full-step run on this workload was slower than the
# best new-stack baseline, so keep it opt-in for now.
export VLLM_ROLLOUT_LENGTH_BALANCE=${VLLM_ROLLOUT_LENGTH_BALANCE:-0}
# Avoid inheriting stale TASK_QUEUE_ENABLE=1 from previous debug shells. The
# old regroup baseline uses 2; use VLLM_ROLLOUT_TASK_QUEUE_ENABLE for explicit
# experiments.
export TASK_QUEUE_ENABLE=${VLLM_ROLLOUT_TASK_QUEUE_ENABLE:-2}
# Keep the vLLM Ascend legacy MoE custom op path enabled above, but do not
# default the Megatron actor logprob path to the Triton fused CE kernel on NPU.
# That kernel currently hits BiShengIR UB-overflow for this long-sequence shape;
# callers can still opt in explicitly with ACTOR_USE_FUSED_KERNELS=true.
export ACTOR_USE_FUSED_KERNELS=${ACTOR_USE_FUSED_KERNELS:-false}
export VLLM_ASCEND_MC2_TOKENS_CAPACITY=512
export VLLM_ASCEND_MC2_GLOBAL_BS=0
export VLLM_ASCEND_MC2_MIN_EP_SIZE=2
# Default to the validated MC2/AllToAll split, but preserve explicit
# experiments such as VLLM_ASCEND_ENABLE_FUSED_MC2=2 (decode-only fused MC2).
export VLLM_ASCEND_ENABLE_FUSED_MC2=${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}
export VLLM_ASCEND_MC2_USE_ACTIVE_MASK=${VLLM_ROLLOUT_MC2_USE_ACTIVE_MASK:-${VLLM_ASCEND_MC2_USE_ACTIVE_MASK:-0}}
export VLLM_ASCEND_FORCE_ALLTOALL_MOE=0
export VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE=${VLLM_ROLLOUT_FORCE_PAGED_ATTENTION_DECODE:-${VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE:-0}}
export VLLM_ASCEND_EAGER_OLD_FORWARD_CONTEXT=${VLLM_ROLLOUT_OLD_FORWARD_CONTEXT:-${VLLM_ASCEND_EAGER_OLD_FORWARD_CONTEXT:-0}}
export VLLM_ROLLOUT_FORCE_ELASTIC_MOE_POLICY=1
export VLLM_ASCEND_ATTENTION_BLOCK_SIZE=64
export VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=1
export VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=1
export VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM=0
unset VLLM_ASCEND_ELASTIC_EXECUTION_MODE
unset VLLM_ASCEND_ELASTIC_MOE_MODE
export VLLM_ROLLOUT_USE_TQDM=0
unset VLLM_ROLLOUT_FAST_WEIGHT_LOAD

# Preserve GRPO n=16 prompt locality and the validated training path.
export VLLM_ROLLOUT_DATA_REBALANCE=0
export VLLM_ROLLOUT_LENGTH_BALANCE=0
export TASK_QUEUE_ENABLE=2
export ACTOR_USE_FUSED_KERNELS=false
export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_async_oldcfg}

exec bash "${BASE_SCRIPT}" "$@"
