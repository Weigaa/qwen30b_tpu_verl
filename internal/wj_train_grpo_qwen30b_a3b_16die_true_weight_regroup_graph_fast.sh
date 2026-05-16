#!/usr/bin/env bash
# qwen3 30B-A3B GRPO, fastest validated true-graph rollout path.
#
# This is the graph entrypoint we keep separate from eager because graph needs
# pointer-stable native sleep semantics:
#   - true vllm-ascend ACL graph mode
#   - TP=1 + EP over 16 rollout workers
#   - native sleep level 1, not manual free
#   - prefix caching and chunked prefill enabled
#
# Optional quick probe:
#   VLLM_ROLLOUT_FAST_DEBUG=1 bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_graph_fast.sh
set -ex

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="${SCRIPT_DIR}/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"

# Workload defaults.  These may be overridden for cheap probes.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
export ROLLOUT_N=${ROLLOUT_N:-16}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
export ROLLOUT_TOP_K=${ROLLOUT_TOP_K:-50}
export DATASET_FRACTION=${DATASET_FRACTION:-0.005}
if [ "${VLLM_ROLLOUT_FAST_DEBUG:-0}" = "1" ]; then
    export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
    export TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-1}
    export TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS:-1}
    export VLLM_ROLLOUT_DEBUG_GENERATION=${VLLM_ROLLOUT_DEBUG_GENERATION:-1}
    export VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE=${VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE:-1}
    export VLLM_ROLLOUT_DEBUG_GENERATION_SAMPLES=${VLLM_ROLLOUT_DEBUG_GENERATION_SAMPLES:-8}
    export VLLM_ROLLOUT_DEBUG_GROUPS=${VLLM_ROLLOUT_DEBUG_GROUPS:-8}
else
    export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
    export TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-3}
    export TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS:-null}
fi

# Parallelism: pure rollout EP, no rollout TP.
export VLLM_ROLLOUT_PARALLEL_MODE=ep
export ROLLOUT_TP_SIZE=1
export VLLM_ENABLE_EXPERT_PARALLEL=1
export ALL_TO_ALL_RESHARD=1
export USE_ALLTOALL_OVERLAP=1
export VLLM_ASCEND_NO_EP_KEEP_TP_ACROSS_DP=0
export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=0
unset VLLM_ASCEND_ELASTIC_MOE_MODE

# True graph mode.  The base script still names this gate "UNSAFE" because it
# was introduced as a diagnostic; with sleep level 1 and the current qwen3
# graph fixes this is the graph path we intentionally preserve.
export VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER=1
export VLLM_ROLLOUT_UNSAFE_TRUE_GRAPH_WITH_RESAMPLER=1
export VLLM_ROLLOUT_ZIYI_ALIGN=0
export VLLM_ROLLOUT_EAGER_OLDREF_ALIGN=0
export VLLM_ENABLE_GRAPH_MODE=1
export ROLLOUT_ENFORCE_EAGER=False
export VLLM_ASCEND_EAGER_COMPILE=0
# Graph still needs bundled graph-only OPPs such as AddRmsNormBias, but the
# bundled MoE dispatch/combine kernels are slower than the system CANN versions.
# The base script defaults to the filtered local OPP package when available.
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=${VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP:-1}
export VLLM_ASCEND_FORCE_CUDAGRAPH_NONE=0
export VLLM_ASCEND_FORCE_COMPILE_WITHOUT_ACLGRAPH=0
export VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE=0
export VLLM_ASCEND_PIECEWISE_COPY_INPUTS=0
export VLLM_ASCEND_UPDATE_PIECEWISE_GRAPH_ATTN=0
export VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC=0
export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
export VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
export VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD=0
export VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD=0

# Scheduler and memory policy validated for qwen3 graph rollout.
export VLLM_ROLLOUT_ASYNC_SCHEDULING=${VLLM_ROLLOUT_ASYNC_SCHEDULING:-true}
export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true
export VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true
export VLLM_ROLLOUT_MAX_NUM_BATCHED_TOKENS=17408
# Graph native sleep must remap the full KV pool on every rollout phase.  The
# actor/ref phase leaves ~11 GiB resident after sleep, so the previous 0.83
# default could OOM in CaMem create_and_map() on the second kv_cache wake.
export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=${VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}
export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
export VLLM_ROLLOUT_FREE_CACHE_ENGINE=True
export VLLM_ROLLOUT_SLEEP_LEVEL=1
# Keep native graph sleep responsible for the real model weight storage, but do
# not put per-step load_weights/process_weights_after_loading allocations into
# the persistent CaMem "weights" pool.  In graph mode those extra handles can
# survive the first sleep and then crash create_and_map() on the next wake.
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=0
export VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1
export VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1
export TASK_QUEUE_ENABLE=1
# True graph/native sleep remaps weights and KV cache to their original device
# addresses on every rollout phase.  Letting the Megatron training phase turn
# expandable segments back on can occupy those address ranges and crash inside
# CaMem create_and_map() on the next wake_up.  Keep the allocator mode stable
# for the whole graph run.
export MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1

# Keep the high-performance qwen3 backend path, not legacy fallback kernels.
export VLLM_ASCEND_USE_LEGACY_FUSED_MOE=0
export VLLM_ASCEND_USE_LEGACY_ATTENTION=0
export VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE=0
export VLLM_ASCEND_DISABLE_GRAPH_FUSION=0
export VLLM_ASCEND_DISABLE_NORM_QUANT_FUSION=0
export VLLM_ASCEND_DISABLE_QKNORM_ROPE_FUSION=0
export VLLM_ASCEND_DISABLE_ALLREDUCE_RMS_FUSION=0
export VLLM_ASCEND_FORCE_ALLTOALL_MOE=0
export VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE=0
export VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM=${VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM:-1}
export VLLM_ASCEND_MC2_TOKENS_CAPACITY=512
export VLLM_ASCEND_MC2_GLOBAL_BS=0
export VLLM_ASCEND_MC2_MIN_EP_SIZE=2

# Preserve GRPO semantics: prompt-local n samples, no cross-rank data rebalance.
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=1
export VLLM_EPOCH_LENGTH_REGROUP_DEFAULT_LENGTH=${VLLM_EPOCH_LENGTH_REGROUP_DEFAULT_LENGTH:-8192}
export VLLM_ROLLOUT_DATA_REBALANCE=0
export VLLM_ROLLOUT_LENGTH_BALANCE=0
export VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED=0
export VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS=0
export VLLM_ROLLOUT_USE_TQDM=1
export ACTOR_USE_FUSED_KERNELS=false
export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_graph_fast}

exec bash "${BASE_SCRIPT}" "$@"
