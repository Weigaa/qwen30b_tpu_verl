#!/usr/bin/env bash
# qwen3 30B-A3B GRPO, fastest validated eager rollout path.
#
# This is the production eager entrypoint:
#   - true graph capture disabled
#   - VLLM_ASCEND_EAGER_COMPILE enabled
#   - new split-MoE / new attention kernels
#   - system CANN MoE custom ops, not the bundled local OPP override
#   - validated opaque vLLM attention wrapper by default
#   - MC2 for <=512 tokens, AllToAll above 512
#   - explicit manual free between rollout and actor phases
#   - TP=1 + EP over 16 rollout workers
#
# Optional quick probe:
#   VLLM_ROLLOUT_FAST_DEBUG=1 bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_eager_fast.sh
set -ex

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
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
    export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-256,512,640,768,896}
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

# Eager mode: keep vLLM compile, but disable ACL/cudagraph replay.
export VLLM_ENABLE_GRAPH_MODE=0
export VLLM_ASCEND_EAGER_COMPILE=${VLLM_ASCEND_EAGER_COMPILE:-1}
export ROLLOUT_ENFORCE_EAGER=${VLLM_ROLLOUT_ENFORCE_EAGER:-True}
# The fastest validated eager probes used the system CANN OPP path, not the
# bundled qwen3 local OPP.  The local package helps graph by providing ops such
# as AddRmsNormBias, but in eager it can either intercept MoE dispatch/combine
# or register partial fusions that compiled eager cannot tile correctly.
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=${VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP:-0}
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=${VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB:-0}
export VLLM_ASCEND_LOCAL_CUSTOM_OPP_PATH=${VLLM_ASCEND_LOCAL_CUSTOM_OPP_PATH:-"${PROJECT_ROOT}/vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend"}
# When local OPP is disabled, the system CANN package has no AddRmsNormBias
# registration.  Keep residual RMSNorm on torch_npu.npu_add_rms_norm so
# compiled eager does not emit the local-only _C_ascend.npu_add_rms_norm_bias.
export VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=${VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM:-1}
# Pass fusion did not improve the threshold workload; keep the current-best
# compiled-eager shape unless explicitly overridden for diagnostics.
export VLLM_ASCEND_EAGER_COMPILE_PASS_FUSION=${VLLM_ASCEND_EAGER_COMPILE_PASS_FUSION:-0}
export VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER=0
export VLLM_ROLLOUT_UNSAFE_TRUE_GRAPH_WITH_RESAMPLER=0
export VLLM_ROLLOUT_ZIYI_ALIGN=0
export VLLM_ROLLOUT_EAGER_OLDREF_ALIGN=0
export VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE=0
export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
export VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0

# Scheduler and memory policy validated for qwen3 eager rollout.
export VLLM_ROLLOUT_ASYNC_SCHEDULING=${VLLM_ROLLOUT_ASYNC_SCHEDULING:-true}
export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=${VLLM_ROLLOUT_ENABLE_PREFIX_CACHING:-false}
export VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=${VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL:-true}
export VLLM_ROLLOUT_MAX_NUM_BATCHED_TOKENS=${VLLM_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-17408}
# Current best wrapper/run used 0.83; 0.85 is reserved for explicit probes.
export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=${VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.83}
export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=${VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE:-1}
export VLLM_ROLLOUT_FREE_CACHE_ENGINE=${VLLM_ROLLOUT_FREE_CACHE_ENGINE:-True}
# Historical eager-fast wrapper pinned level 2 even though manual-free owns the
# phase switch. Keep that surface aligned with the previous validated script.
export VLLM_ROLLOUT_SLEEP_LEVEL=${VLLM_ROLLOUT_SLEEP_LEVEL:-2}
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=${VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD:-1}
export VLLM_ROLLOUT_TASK_QUEUE_ENABLE=${VLLM_ROLLOUT_TASK_QUEUE_ENABLE:-2}
# Avoid inheriting stale TASK_QUEUE_ENABLE=1 from graph/debug shells.
export TASK_QUEUE_ENABLE=${VLLM_ROLLOUT_TASK_QUEUE_ENABLE}

# Keep the high-performance qwen3 backend path, not legacy fallback kernels.
export VLLM_ASCEND_USE_LEGACY_FUSED_MOE=${VLLM_ASCEND_USE_LEGACY_FUSED_MOE:-0}
export VLLM_ASCEND_USE_LEGACY_ATTENTION=${VLLM_ASCEND_USE_LEGACY_ATTENTION:-0}
export VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE=${VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE:-0}
export VLLM_ASCEND_DISABLE_GRAPH_FUSION=0
export VLLM_ASCEND_DISABLE_NORM_QUANT_FUSION=0
export VLLM_ASCEND_DISABLE_QKNORM_ROPE_FUSION=0
export VLLM_ASCEND_DISABLE_ALLREDUCE_RMS_FUSION=0
export VLLM_ASCEND_FORCE_ALLTOALL_MOE=${VLLM_ASCEND_FORCE_ALLTOALL_MOE:-0}
export VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE=${VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE:-0}
# Diagnostic only: bypass the generic opaque attention custom-op wrapper and
# call the Ascend backend impl directly. This was fast on threshold-control
# probes, but its full-16k run produced a much shorter response distribution,
# so keep production eager on the behavior-preserving wrapper unless explicitly
# requested.
export VLLM_ASCEND_FORCE_DIRECT_ATTENTION_IMPL=${VLLM_ASCEND_FORCE_DIRECT_ATTENTION_IMPL:-0}
export VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM=${VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM:-0}
export VLLM_ASCEND_MC2_TOKENS_CAPACITY=${VLLM_ASCEND_MC2_TOKENS_CAPACITY:-512}
export VLLM_ASCEND_MC2_GLOBAL_BS=${VLLM_ASCEND_MC2_GLOBAL_BS:-0}
export VLLM_ASCEND_MC2_MIN_EP_SIZE=${VLLM_ASCEND_MC2_MIN_EP_SIZE:-2}
export VLLM_ASCEND_ENABLE_FUSED_MC2=${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}
export VLLM_ROLLOUT_FORCE_ELASTIC_MOE_POLICY=${VLLM_ROLLOUT_FORCE_ELASTIC_MOE_POLICY:-1}
export VLLM_ASCEND_ATTENTION_BLOCK_SIZE=${VLLM_ASCEND_ATTENTION_BLOCK_SIZE:-64}
export VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=${VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE:-1}
export VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=${VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2:-1}
# Keep the upstream Qwen3Moe reduction semantics by default.  The
# reduce_results=0 probe was slower on the threshold workload, so leave it as
# an explicit diagnostic override only.
export VLLM_QWEN3_MOE_REDUCE_RESULTS=${VLLM_QWEN3_MOE_REDUCE_RESULTS:-1}

# Preserve GRPO semantics: prompt-local n samples, no cross-rank data rebalance.
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=1
export VLLM_EPOCH_LENGTH_REGROUP_DEFAULT_LENGTH=${VLLM_EPOCH_LENGTH_REGROUP_DEFAULT_LENGTH:-8192}
export VLLM_ROLLOUT_DATA_REBALANCE=0
export VLLM_ROLLOUT_LENGTH_BALANCE=0
export VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED=0
export VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS=0
export VLLM_ROLLOUT_USE_TQDM=${VLLM_ROLLOUT_USE_TQDM:-1}
export ACTOR_USE_FUSED_KERNELS=false
# Keep compiled-eager artifacts for this production path separate from older
# OPP experiments. The compile cache key now includes OPP provider envs, but a
# dedicated cache root also makes stale artifacts easier to inspect and purge.
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-"${PWD}/.cache/vllm_eager_fast_nolocalopp"}
export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_eager_fast}

exec bash "${BASE_SCRIPT}" "$@"
