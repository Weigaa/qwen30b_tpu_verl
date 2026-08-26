# Variant: regroup + rollout eager workaround for MoeDistributeDispatchV2 torchair compile issue
set -ex

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
FILTERED_CUSTOM_OPP_PATH="${PROJECT_ROOT}/vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend"
FULL_CUSTOM_OPP_PATH="${PROJECT_ROOT}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
DEFAULT_CUSTOM_OPP_PATH="${FILTERED_CUSTOM_OPP_PATH}"
if [ ! -d "${DEFAULT_CUSTOM_OPP_PATH}" ]; then
    DEFAULT_CUSTOM_OPP_PATH="${FULL_CUSTOM_OPP_PATH}"
fi
CUSTOM_OPP_PATH="${VLLM_ASCEND_LOCAL_CUSTOM_OPP_PATH:-${DEFAULT_CUSTOM_OPP_PATH}}"
CUSTOM_OP_API_LIB="${CUSTOM_OPP_PATH}/op_api/lib"
FILTERED_CUSTOM_OP_API_LIB="${FILTERED_CUSTOM_OPP_PATH}/op_api/lib"
FULL_CUSTOM_OP_API_LIB="${FULL_CUSTOM_OPP_PATH}/op_api/lib"
USE_LOCAL_CUSTOM_OP_API_LIB="${VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB:-0}"

remove_colon_path() {
    local var_name="$1"
    local path_to_remove="$2"
    local current_value="${!var_name:-}"
    local new_value=""
    local entry
    IFS=':' read -r -a entries <<< "${current_value}"
    for entry in "${entries[@]}"; do
        if [ -z "${entry}" ] || [ "${entry}" = "${path_to_remove}" ]; then
            continue
        fi
        if [ -z "${new_value}" ]; then
            new_value="${entry}"
        else
            new_value="${new_value}:${entry}"
        fi
    done
    export "${var_name}=${new_value}"
}

if [ "${VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP:-1}" = "1" ]; then
    remove_colon_path LD_LIBRARY_PATH "${CUSTOM_OP_API_LIB}"
    if [ "${CUSTOM_OPP_PATH}" != "${FULL_CUSTOM_OPP_PATH}" ]; then
        remove_colon_path ASCEND_CUSTOM_OPP_PATH "${FILTERED_CUSTOM_OPP_PATH}"
        remove_colon_path ASCEND_CUSTOM_OPP_PATH "${FULL_CUSTOM_OPP_PATH}"
        remove_colon_path LD_LIBRARY_PATH "${FILTERED_CUSTOM_OP_API_LIB}"
        remove_colon_path LD_LIBRARY_PATH "${FULL_CUSTOM_OP_API_LIB}"
    fi
    if [ -d "${CUSTOM_OPP_PATH}" ]; then
        export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_PATH}:${ASCEND_CUSTOM_OPP_PATH:-}"
    fi
    if [ "${USE_LOCAL_CUSTOM_OP_API_LIB}" = "1" ] && [ -d "${CUSTOM_OP_API_LIB}" ]; then
        export LD_LIBRARY_PATH="${CUSTOM_OP_API_LIB}:${LD_LIBRARY_PATH:-}"
    fi
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-garbage_collection_threshold:0.6,max_split_size_mb:24}"

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
# ATB's set_env.sh can return non-zero under `set -e` when it auto-detects
# torch's CXX ABI and an internal grep misses. Passing the ABI explicitly
# avoids that false failure and keeps the environment initialization stable.
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1

export HYDRA_FULL_ERROR=1
#export ASCEND_LAUNCH_BLOCKING=1         
export RAY_DEDUP_LOGS=0                   

export ASCEND_GLOBAL_EVENT_ENABLE=0         
export ASCEND_SLOG_PRINT_TO_STDOUT=0       
export ASCEND_GLOBAL_LOG_LEVEL=3             
# Keep enough headroom for actor / rollout / ref phased offsets inside
# Megatron workers. A high base port can still trip HCCL setup after the
# ref-phase offset is applied.
export HCCL_IF_BASE_PORT=50021
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_PORT=23300    # vllm port error
export D2D_DATA_TRANSFER=1

export VLLM_USE_V1=1
export PRINT_MEMORY=1
export USE_ALLTOALL_OVERLAP=${USE_ALLTOALL_OVERLAP:-1}
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ENABLE_MC2=1                     # 910C开启
TRAINER_NNODES=${TRAINER_NNODES:-1}
TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE:-16}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
ROLLOUT_WORLD_SIZE=$((TRAINER_NNODES * TRAINER_N_GPUS_PER_NODE))
if [ $((ROLLOUT_WORLD_SIZE % ROLLOUT_TP_SIZE)) -ne 0 ]; then
    echo "Invalid rollout parallel config: world_size=${ROLLOUT_WORLD_SIZE}, tp=${ROLLOUT_TP_SIZE}" >&2
    exit 1
fi
ROLLOUT_DP_SIZE=${ROLLOUT_DP_SIZE_OVERRIDE:-$((ROLLOUT_WORLD_SIZE / ROLLOUT_TP_SIZE))}
export VLLM_DP_SIZE=${ROLLOUT_DP_SIZE}        # world_size // rollout.tp_size
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_BUFFSIZE=800

# Keep this aligned with the old regroup baseline.  Avoid inheriting stale
# TASK_QUEUE_ENABLE=1 from debug shells; use VLLM_ROLLOUT_TASK_QUEUE_ENABLE for
# explicit experiments.
export TASK_QUEUE_ENABLE=${VLLM_ROLLOUT_TASK_QUEUE_ENABLE:-2}

export VLLM_ENABLE_FIX_ROUTE=0    
export VLLM_MODEL_EXECUTE_TIME_OBSERVE=${VLLM_MODEL_EXECUTE_TIME_OBSERVE:-0}     # decode prefill的耗时打印
export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=${VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE:-${VLLM_MODEL_EXECUTE_TIME_OBSERVE}}

#extra env in qwen3_235b_env.sh
# Recipe features
export VLLM_ENABLE_GRAPH_MODE=${VLLM_ENABLE_GRAPH_MODE:-0}             # 0: eager mode, 1: graph mode
export VLLM_ENABLE_EXPERT_PARALLEL=${VLLM_ENABLE_EXPERT_PARALLEL:-1}        # Enable EP in vLLM rollout.
export VLLM_CHUNK_MOE_SIZE=${VLLM_CHUNK_MOE_SIZE:-512}  # The minimum block size set for prefill computation partition.
export VLLM_ASCEND_USE_LEGACY_FUSED_MOE=${VLLM_ASCEND_USE_LEGACY_FUSED_MOE:-0}
export ALL_TO_ALL_RESHARD=${ALL_TO_ALL_RESHARD:-1}                 # Enable EP to reshard parameters with AllToAllV (without communication redundancy).
export USE_ALLTOALL_OVERLAP=${USE_ALLTOALL_OVERLAP:-1}               # Enable to overlap communication in EP with computation to hide MoE communication latency. Should be consistent with model conversion config.
export VLLM_ENABLE_EPLB=0                   # 0: disable eplb, 1: enable eplb
export USE_HDP=0                            # 0: disable hdp, 1: enable hdp
export ROLLOUT_REBALANCE_ENABLE=0          # 0: disable rollout rebalance, 1: enable rollout rebalance
export VLLM_ASCEND_USE_LEGACY_ATTENTION=${VLLM_ASCEND_USE_LEGACY_ATTENTION:-0}
export VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE=${VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE:-0}
# Keep enough MC2 active-mask capacity for mixed small-token chunks in eager
# rollout. The qwen3 auto heuristic can shrink this to max_num_reqs (32 here);
# 512 matches the validated fast eager path while remaining tiny in memory.
export VLLM_ASCEND_MC2_TOKENS_CAPACITY=${VLLM_ASCEND_MC2_TOKENS_CAPACITY:-512}
export VLLM_ASCEND_MC2_GLOBAL_BS=${VLLM_ASCEND_MC2_GLOBAL_BS:-0}
export VLLM_ASCEND_FORCE_ALLTOALL_MOE=${VLLM_ASCEND_FORCE_ALLTOALL_MOE:-0}
export VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE=${VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE:-0}
# Keep regroup script on the non-elastic baseline by default.  When shrink is
# off, do not leak elastic policy knobs into the official EP graph path.
export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=${VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:-0}
if [ "${VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK}" = "1" ]; then
    export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE:-1}
    export VLLM_ASCEND_ELASTIC_MOE_MODE=${VLLM_ASCEND_ELASTIC_MOE_MODE:-lossless}
    export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE:-4}
else
    unset VLLM_ASCEND_ELASTIC_EXECUTION_MODE
    unset VLLM_ASCEND_ELASTIC_MOE_MODE
    unset VLLM_ASCEND_INIT_REDUNDANCY_EXPERT
    unset VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE
    unset VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS
fi
# Preserve an explicit caller-provided tail-cap schedule for quick validation
# probes; otherwise leave it unset by default.
if [ -z "${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS+x}" ]; then
    unset VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS
fi
# export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=4,8,12,16,20
# export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896


# Disable MoE pattern stats by default here to match the old eager/regroup
# scripts unless the caller explicitly turns them on.
export VLLM_MOE_PATTERN_STATS=${VLLM_MOE_PATTERN_STATS:-0}
export VLLM_MOE_STATS=${VLLM_MOE_PATTERN_STATS}
export VLLM_MOE_STATS_TIMING=${VLLM_MOE_STATS_TIMING:-0}

# Default to the old regroup script's watchdog handling and allow callers to
# override when debugging async failures.
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200

#控制llm.py里的profiler
export VLLM_ASCEND_LLM_PROFILE_ENABLE=0

# Unified output root for checkpoints / rollout dumps / draft dumps / logs.
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/cann-recipes-train/llm_rl/qwen3}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_baseline_ft}
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}"
ROLL_OUT_DIR="${OUTPUT_DIR}/rollout_data"
ROLL_LEN_DIR="${OUTPUT_DIR}/rollout_length"
DRAFT_DUMP_DIR="${OUTPUT_DIR}/draft_hidden"
CKPT_DIR="${OUTPUT_DIR}/checkpoints/qwen3moe_for_eagle3"
TB_DIR="${OUTPUT_DIR}/tensorboard"
LOG_DIR="${OUTPUT_DIR}/logs"

# Toggle switches:
#   SAVE_CKPT_ENABLE=1        -> save checkpoints
#   SAVE_DRAFT_HIDDEN_ENABLE=1 -> dump draft hidden states
SAVE_CKPT_ENABLE=${SAVE_CKPT_ENABLE:-0}
SAVE_DRAFT_HIDDEN_ENABLE=${SAVE_DRAFT_HIDDEN_ENABLE:-0}

mkdir -p "${ROLL_OUT_DIR}" "${ROLL_LEN_DIR}" "${TB_DIR}" "${LOG_DIR}"
if [ "${SAVE_DRAFT_HIDDEN_ENABLE}" = "1" ]; then
    mkdir -p "${DRAFT_DUMP_DIR}"
fi
if [ "${SAVE_CKPT_ENABLE}" = "1" ]; then
    mkdir -p "${CKPT_DIR}"
fi

# Draft data collection for offline Eagle3 training
export VLLM_ASCEND_ENABLE_DRAFT_TRAIN=${VLLM_ASCEND_ENABLE_DRAFT_TRAIN:-0}
export VLLM_ASCEND_DRAFT_DUMP_ENABLE=${VLLM_ASCEND_DRAFT_DUMP_ENABLE:-${SAVE_DRAFT_HIDDEN_ENABLE}}
export VLLM_ASCEND_DRAFT_DUMP_DIR=${VLLM_ASCEND_DRAFT_DUMP_DIR:-${DRAFT_DUMP_DIR}}
export VLLM_ASCEND_DRAFT_DUMP_EVERY=${VLLM_ASCEND_DRAFT_DUMP_EVERY:-1}
export VLLM_ASCEND_DRAFT_DUMP_HIDDEN_DTYPE=${VLLM_ASCEND_DRAFT_DUMP_HIDDEN_DTYPE:-bf16}
# Larger queue reduces drop risk when collecting offline draft dataset.
export VLLM_ASCEND_DRAFT_QUEUE_SIZE=${VLLM_ASCEND_DRAFT_QUEUE_SIZE:-4096}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${TB_DIR}}

# Per-epoch auto regrouping by previous rollout response lengths.
# Keep dataloader workers at 0 when using curriculum sampler.
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=${VLLM_EPOCH_LENGTH_REGROUP_ENABLE:-1}
export VLLM_EPOCH_LENGTH_REGROUP_BUCKET_SIZE=${VLLM_EPOCH_LENGTH_REGROUP_BUCKET_SIZE:-1024}
export VLLM_EPOCH_LENGTH_REGROUP_EMA_DECAY=${VLLM_EPOCH_LENGTH_REGROUP_EMA_DECAY:-0.7}
export VLLM_EPOCH_LENGTH_REGROUP_SHUFFLE_BLOCKS=${VLLM_EPOCH_LENGTH_REGROUP_SHUFFLE_BLOCKS:-1}
export VLLM_EPOCH_LENGTH_REGROUP_DEFAULT_LENGTH=${VLLM_EPOCH_LENGTH_REGROUP_DEFAULT_LENGTH:-8192}
# Rollout long-tail guard (default on):
# Cap per-step generation by expected response length from resampler.
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=${VLLM_ROLLOUT_EARLY_STOP_ENABLE:-1}
export VLLM_ROLLOUT_EARLY_STOP_FACTOR=${VLLM_ROLLOUT_EARLY_STOP_FACTOR:-2.0}
export VLLM_ROLLOUT_EARLY_STOP_MIN_TOKENS=${VLLM_ROLLOUT_EARLY_STOP_MIN_TOKENS:-10000}
# Optional backend load balancing: reorder prompt-repeat blocks across rollout
# ranks, then restore the original order before PPO/logprob. Keep off in the
# sync baseline script unless explicitly requested.
export VLLM_ROLLOUT_LENGTH_BALANCE=${VLLM_ROLLOUT_LENGTH_BALANCE:-0}
# Rollout scheduler mode:
#   false -> match the old baseline's sync-style scheduling
#   true  -> force enable
#   null  -> let vLLM decide automatically
export VLLM_ROLLOUT_ASYNC_SCHEDULING=${VLLM_ROLLOUT_ASYNC_SCHEDULING:-false}
# Rollout engine knobs. The threshold-controlled old baseline and the fastest
# qwen3 positive-control run both initialize with prefix caching off, chunked
# prefill on, and a full 17408 scheduling window.
export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=${VLLM_ROLLOUT_ENABLE_PREFIX_CACHING:-false}
export VLLM_ROLLOUT_MAX_NUM_BATCHED_TOKENS=${VLLM_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-17408}
export VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=${VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL:-true}
export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=${VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
export ACTOR_USE_FUSED_KERNELS=${ACTOR_USE_FUSED_KERNELS:-false}
# Baseline switch:
#   1 -> eager on + no resample (disable length regroup sampler)
#   0 -> follow regroup switch below
export VLLM_EAGER_BASELINE_NO_RESAMPLE=${VLLM_EAGER_BASELINE_NO_RESAMPLE:-0}
# Optional experiment: keep the length-regroup sampler enabled while allowing
# the rollout engine to enter vLLM-Ascend ACL graph mode.  Default stays eager
# because the non-eager qwen3 MoE + resampler path currently changes rollout
# stop behavior and produces many max-length responses.
export VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER=${VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER:-0}
export VLLM_ROLLOUT_UNSAFE_TRUE_GRAPH_WITH_RESAMPLER=${VLLM_ROLLOUT_UNSAFE_TRUE_GRAPH_WITH_RESAMPLER:-0}

if [ "${VLLM_EAGER_BASELINE_NO_RESAMPLE}" = "1" ]; then
    SAMPLER_CLASS_PATH=null
    SAMPLER_CLASS_NAME=null
    SAMPLER_EXTRA_ARGS=""
    # Historical "baseline no resample" runs were eager by default, but graph
    # correctness isolation needs a clean no-resampler + non-eager variant.
    ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-${VLLM_NO_RESAMPLE_ENFORCE_EAGER:-True}}
    if [ "${VLLM_ENABLE_GRAPH_MODE}" = "1" ] && [ "${ROLLOUT_ENFORCE_EAGER}" = "False" ] && [ -z "${VLLM_ROLLOUT_TASK_QUEUE_ENABLE+x}" ]; then
        # NPU graph capture rejects TASK_QUEUE_ENABLE=2. Keep this no-resampler
        # graph isolation path consistent with the resampler graph path.
        export TASK_QUEUE_ENABLE=1
    fi
elif [ "${VLLM_EPOCH_LENGTH_REGROUP_ENABLE}" = "1" ]; then
    SAMPLER_CLASS_PATH=pkg://verl.experimental.dataset.length_bucket_sampler
    SAMPLER_CLASS_NAME=LengthAwareEpochSampler
    SAMPLER_EXTRA_ARGS="+data.sampler.bucket_size=${VLLM_EPOCH_LENGTH_REGROUP_BUCKET_SIZE} +data.sampler.ema_decay=${VLLM_EPOCH_LENGTH_REGROUP_EMA_DECAY} +data.sampler.shuffle_batch_blocks=${VLLM_EPOCH_LENGTH_REGROUP_SHUFFLE_BLOCKS} +data.sampler.default_length=${VLLM_EPOCH_LENGTH_REGROUP_DEFAULT_LENGTH}"
    if [ "${VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER}" = "1" ]; then
        # Safe public graph toggle: for now it resolves to eager-compile
        # semantics, because the true non-eager path has a correctness bug in
        # qwen3 MoE + resampler.  To keep investigating true graph mode, opt in
        # with VLLM_ROLLOUT_UNSAFE_TRUE_GRAPH_WITH_RESAMPLER=1.
        if [ "${VLLM_ROLLOUT_UNSAFE_TRUE_GRAPH_WITH_RESAMPLER}" = "1" ]; then
            ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
            export VLLM_ENABLE_GRAPH_MODE=1
            if [ -z "${VLLM_ROLLOUT_TASK_QUEUE_ENABLE+x}" ]; then
                # NPU graph capture rejects TASK_QUEUE_ENABLE=2.
                export TASK_QUEUE_ENABLE=1
            fi
            export VLLM_ASCEND_FORCE_CUDAGRAPH_NONE=${VLLM_ASCEND_FORCE_CUDAGRAPH_NONE:-0}
            export VLLM_ASCEND_FORCE_COMPILE_WITHOUT_ACLGRAPH=${VLLM_ASCEND_FORCE_COMPILE_WITHOUT_ACLGRAPH:-0}
            export VLLM_ASCEND_PIECEWISE_COPY_INPUTS=${VLLM_ASCEND_PIECEWISE_COPY_INPUTS:-0}
            export VLLM_ASCEND_UPDATE_PIECEWISE_GRAPH_ATTN=${VLLM_ASCEND_UPDATE_PIECEWISE_GRAPH_ATTN:-0}
            # True graph mode captures during vLLM init, but RL rollout
            # replaces dummy init weights with real Megatron weights before
            # generation. Clear and recapture those graphs after the reload so
            # replay cannot bind stale MoE layouts or tensor addresses.
            export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=${VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE:-1}
            export VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=${VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE:-1}
            export VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD=${VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD:-0}
            export VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD=${VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD:-0}
            export VLLM_ASCEND_DISABLE_NORM_QUANT_FUSION=${VLLM_ASCEND_DISABLE_NORM_QUANT_FUSION:-0}
        else
            ROLLOUT_ENFORCE_EAGER=True
            export VLLM_ENABLE_GRAPH_MODE=0
            export VLLM_ASCEND_EAGER_COMPILE=1
            export VLLM_ASCEND_FORCE_CUDAGRAPH_NONE=0
            export VLLM_ASCEND_FORCE_COMPILE_WITHOUT_ACLGRAPH=0
            export VLLM_ASCEND_DISABLE_NORM_QUANT_FUSION=0
        fi
        export VLLM_ASCEND_DISABLE_GRAPH_FUSION=${VLLM_ASCEND_DISABLE_GRAPH_FUSION:-0}
        export VLLM_ASCEND_DISABLE_QKNORM_ROPE_FUSION=${VLLM_ASCEND_DISABLE_QKNORM_ROPE_FUSION:-0}
        export VLLM_ASCEND_DISABLE_ALLREDUCE_RMS_FUSION=${VLLM_ASCEND_DISABLE_ALLREDUCE_RMS_FUSION:-0}
        export VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED=${VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED:-0}
        export VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS=${VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS:-0}
    else
        # Work around torchair graph compile instability for MoE dispatch under
        # dynamically regrouped rollout batches.
        ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
        export VLLM_ASCEND_FORCE_CUDAGRAPH_NONE=${VLLM_ASCEND_FORCE_CUDAGRAPH_NONE:-0}
        export VLLM_ASCEND_FORCE_COMPILE_WITHOUT_ACLGRAPH=${VLLM_ASCEND_FORCE_COMPILE_WITHOUT_ACLGRAPH:-0}
        export VLLM_ASCEND_DISABLE_GRAPH_FUSION=${VLLM_ASCEND_DISABLE_GRAPH_FUSION:-0}
        export VLLM_ASCEND_DISABLE_NORM_QUANT_FUSION=${VLLM_ASCEND_DISABLE_NORM_QUANT_FUSION:-0}
        export VLLM_ASCEND_DISABLE_QKNORM_ROPE_FUSION=${VLLM_ASCEND_DISABLE_QKNORM_ROPE_FUSION:-0}
        export VLLM_ASCEND_DISABLE_ALLREDUCE_RMS_FUSION=${VLLM_ASCEND_DISABLE_ALLREDUCE_RMS_FUSION:-0}
        export VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED=${VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED:-0}
        export VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS=${VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS:-0}
    fi
else
    SAMPLER_CLASS_PATH=null
    SAMPLER_CLASS_NAME=null
    SAMPLER_EXTRA_ARGS=""
    ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
    export VLLM_ASCEND_FORCE_CUDAGRAPH_NONE=${VLLM_ASCEND_FORCE_CUDAGRAPH_NONE:-0}
    export VLLM_ASCEND_FORCE_COMPILE_WITHOUT_ACLGRAPH=${VLLM_ASCEND_FORCE_COMPILE_WITHOUT_ACLGRAPH:-0}
    export VLLM_ASCEND_DISABLE_GRAPH_FUSION=${VLLM_ASCEND_DISABLE_GRAPH_FUSION:-0}
    export VLLM_ASCEND_DISABLE_NORM_QUANT_FUSION=${VLLM_ASCEND_DISABLE_NORM_QUANT_FUSION:-0}
    export VLLM_ASCEND_DISABLE_QKNORM_ROPE_FUSION=${VLLM_ASCEND_DISABLE_QKNORM_ROPE_FUSION:-0}
    export VLLM_ASCEND_DISABLE_ALLREDUCE_RMS_FUSION=${VLLM_ASCEND_DISABLE_ALLREDUCE_RMS_FUSION:-0}
    export VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED=${VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED:-0}
    export VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS=${VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS:-0}
fi

# Keep disk usage bounded for per-step save.
# 1) only save model weights in checkpoints (no optimizer/extra),
# 2) keep only the latest checkpoint(s).
ACTOR_CKPT_SAVE_CONTENTS=${ACTOR_CKPT_SAVE_CONTENTS:-[model]}
ACTOR_CKPT_LOAD_CONTENTS=${ACTOR_CKPT_LOAD_CONTENTS:-[model]}
CRITIC_CKPT_SAVE_CONTENTS=${CRITIC_CKPT_SAVE_CONTENTS:-[model]}
CRITIC_CKPT_LOAD_CONTENTS=${CRITIC_CKPT_LOAD_CONTENTS:-[model]}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-1}
# Keep ref loading aligned with the old Megatron dist-checkpoint workflow by default.
REF_USE_DIST_CHECKPOINTING=${REF_USE_DIST_CHECKPOINTING:-True}
ACTOR_USE_KL_LOSS=${ACTOR_USE_KL_LOSS:-True}

if [ "${SAVE_CKPT_ENABLE}" = "1" ]; then
    TRAINER_SAVE_FREQ=1
    TRAINER_DEFAULT_LOCAL_DIR="${CKPT_DIR}"
else
    TRAINER_SAVE_FREQ=-1
    TRAINER_DEFAULT_LOCAL_DIR="${OUTPUT_DIR}"
fi

HOME=$(pwd)
MODEL_PATH=${MODEL_PATH:-"/data/Qwen3-30B-A3B"}
CONFIG_DIR=${CONFIG_DIR:-"${HOME}/verl/trainer/config"}
DISTCP_PATH="/data/Qwen3-30B-A3B_megatron"
TRAIN_FILE=${TRAIN_FILE:-"/data/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"/data/deepscaler/test.parquet"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
DATASET_FRACTION=${DATASET_FRACTION:-0.005}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
ROLLOUT_N=${ROLLOUT_N:-16}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:-50}
TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-3}
TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS:-null}
    

time=$(date +%Y%m%d%H%M%S)
logfile="${LOG_DIR}/wjqwen30b-a3b-record_graph_save4eagle3_${time}.txt"
GIT_TRACK_RUN_LOGS=${GIT_TRACK_RUN_LOGS:-1}

track_run_log_with_git() {
    if [ "${GIT_TRACK_RUN_LOGS}" != "1" ]; then
        return 0
    fi
    if ! command -v git >/dev/null 2>&1; then
        return 0
    fi
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        return 0
    fi
    if ! git add -f "${logfile}" >/dev/null 2>&1; then
        echo "[run][warn] failed to git add logfile=${logfile}" >> "${logfile}"
    fi
    return 0
}

set -x

{
    echo "[run] start_time=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "[run] cwd=$(pwd)"
    echo "[run] logfile=${logfile}"
    echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
    echo "[run] MODEL_PATH=${MODEL_PATH}"
    echo "[run] TRAIN_FILE=${TRAIN_FILE}"
    echo "[run] TEST_FILE=${TEST_FILE}"
    echo "[run] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
    echo "[run] ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE}"
    echo "[run] DATASET_FRACTION=${DATASET_FRACTION}"
    echo "[run] MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
    echo "[run] ROLLOUT_N=${ROLLOUT_N}"
    echo "[run] VLLM_ROLLOUT_PARALLEL_MODE=${VLLM_ROLLOUT_PARALLEL_MODE:-ep}"
    echo "[run] ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE}"
    echo "[run] ROLLOUT_DP_SIZE=${ROLLOUT_DP_SIZE}"
    echo "[run] ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS}"
    echo "[run] ROLLOUT_TOP_K=${ROLLOUT_TOP_K}"
    echo "[run] TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS}"
    echo "[run] TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS}"
    echo "[run] VLLM_ROLLOUT_DEBUG_GENERATION=${VLLM_ROLLOUT_DEBUG_GENERATION:-0}"
    echo "[run] VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE=${VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE:-0}"
    echo "[run] TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE}"
    echo "[run] VLLM_ASCEND_USE_LEGACY_ATTENTION=${VLLM_ASCEND_USE_LEGACY_ATTENTION}"
    echo "[run] VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE=${VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE}"
    echo "[run] VLLM_ASCEND_USE_LEGACY_FUSED_MOE=${VLLM_ASCEND_USE_LEGACY_FUSED_MOE}"
    echo "[run] VLLM_ENABLE_EXPERT_PARALLEL=${VLLM_ENABLE_EXPERT_PARALLEL}"
    echo "[run] VLLM_ASCEND_NO_EP_KEEP_TP_ACROSS_DP=${VLLM_ASCEND_NO_EP_KEEP_TP_ACROSS_DP:-0}"
    echo "[run] ALL_TO_ALL_RESHARD=${ALL_TO_ALL_RESHARD}"
    echo "[run] USE_ALLTOALL_OVERLAP=${USE_ALLTOALL_OVERLAP}"
    echo "[run] HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-}"
    echo "[run] HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-}"
    echo "[run] VLLM_ASCEND_FORCE_ALLTOALL_MOE=${VLLM_ASCEND_FORCE_ALLTOALL_MOE}"
    echo "[run] VLLM_ASCEND_MC2_TOKENS_CAPACITY=${VLLM_ASCEND_MC2_TOKENS_CAPACITY}"
    echo "[run] VLLM_ASCEND_MC2_GLOBAL_BS=${VLLM_ASCEND_MC2_GLOBAL_BS}"
    echo "[run] VLLM_ASCEND_MC2_DISABLE_DISPATCH_EXPERT_SCALES=${VLLM_ASCEND_MC2_DISABLE_DISPATCH_EXPERT_SCALES:-0}"
    echo "[run] VLLM_ASCEND_ENABLE_FUSED_MC2=${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}"
    echo "[run] VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=${VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2:-0}"
    echo "[run] VLLM_ROLLOUT_FORCE_ELASTIC_MOE_POLICY=${VLLM_ROLLOUT_FORCE_ELASTIC_MOE_POLICY:-0}"
    echo "[run] VLLM_ASCEND_ATTENTION_BLOCK_SIZE=${VLLM_ASCEND_ATTENTION_BLOCK_SIZE:-}"
    echo "[run] VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=${VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE:-0}"
    echo "[run] VLLM_QWEN3_MOE_REDUCE_RESULTS=${VLLM_QWEN3_MOE_REDUCE_RESULTS:-1}"
    echo "[run] VLLM_QWEN3_MOE_ASCEND_LEGACY_INIT=${VLLM_QWEN3_MOE_ASCEND_LEGACY_INIT:-0}"
    echo "[run] VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE=${VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE}"
    echo "[run] VLLM_ENABLE_GRAPH_MODE=${VLLM_ENABLE_GRAPH_MODE}"
    echo "[run] VLLM_CHUNK_MOE_SIZE=${VLLM_CHUNK_MOE_SIZE}"
    echo "[run] VLLM_EPOCH_LENGTH_REGROUP_DEFAULT_LENGTH=${VLLM_EPOCH_LENGTH_REGROUP_DEFAULT_LENGTH}"
    echo "[run] VLLM_ASCEND_EAGER_COMPILE=${VLLM_ASCEND_EAGER_COMPILE:-}"
    echo "[run] VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=${VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP:-1}"
    echo "[run] VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=${VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB:-0}"
    echo "[run] VLLM_ASCEND_LOCAL_CUSTOM_OPP_PATH=${CUSTOM_OPP_PATH}"
    echo "[run] VLLM_ASCEND_EAGER_COMPILE_PASS_FUSION=${VLLM_ASCEND_EAGER_COMPILE_PASS_FUSION:-0}"
    echo "[run] VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM=${VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM:-0}"
    echo "[run] VLLM_ROLLOUT_STAGE_DEBUG=${VLLM_ROLLOUT_STAGE_DEBUG:-0}"
    echo "[run] VLLM_ROLLOUT_USE_TQDM=${VLLM_ROLLOUT_USE_TQDM:-}"
    echo "[run] VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER=${VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER}"
    echo "[run] ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER}"
    echo "[run] VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=${VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK}"
    echo "[run] VLLM_ASCEND_ELASTIC_EXECUTION_MODE=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE:-}"
    echo "[run] VLLM_ASCEND_ELASTIC_MOE_MODE=${VLLM_ASCEND_ELASTIC_MOE_MODE:-}"
    echo "[run] VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE:-}"
    echo "[run] VLLM_ASCEND_SHRINK_AWARE_ENABLE=${VLLM_ASCEND_SHRINK_AWARE_ENABLE:-0}"
    echo "[run] VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=${VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY:-}"
    echo "[run] VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS=${VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS:-}"
    echo "[run] VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS}"
    echo "[run] VLLM_ROLLOUT_DATA_REBALANCE=${VLLM_ROLLOUT_DATA_REBALANCE}"
    echo "[run] VLLM_ROLLOUT_ASYNC_SCHEDULING=${VLLM_ROLLOUT_ASYNC_SCHEDULING}"
    echo "[run] VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=${VLLM_ROLLOUT_ENABLE_PREFIX_CACHING}"
    echo "[run] VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=${VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL}"
    echo "[run] VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=${VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION}"
    echo "[run] VLLM_ROLLOUT_SLEEP_LEVEL=${VLLM_ROLLOUT_SLEEP_LEVEL:-}"
    echo "[run] VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=${VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD:-1}"
    echo "[run] VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=${VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE:-}"
    echo "[run] VLLM_ROLLOUT_FREE_CACHE_ENGINE=${VLLM_ROLLOUT_FREE_CACHE_ENGINE:-True}"
    echo "[run] VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=${VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS:-1}"
    echo "[run] VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=${VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE:-}"
    echo "[run] VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=${VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE:-}"
    echo "[run] VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE=${VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE:-}"
    echo "[run] VLLM_ASCEND_FORCE_CUDAGRAPH_NONE=${VLLM_ASCEND_FORCE_CUDAGRAPH_NONE:-0}"
    echo "[run] VLLM_ASCEND_FORCE_COMPILE_WITHOUT_ACLGRAPH=${VLLM_ASCEND_FORCE_COMPILE_WITHOUT_ACLGRAPH:-0}"
    echo "[run] VLLM_ASCEND_PIECEWISE_COPY_INPUTS=${VLLM_ASCEND_PIECEWISE_COPY_INPUTS:-0}"
    echo "[run] VLLM_ASCEND_UPDATE_PIECEWISE_GRAPH_ATTN=${VLLM_ASCEND_UPDATE_PIECEWISE_GRAPH_ATTN:-0}"
    echo "[run] VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC=${VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC:-0}"
    echo "[run] VLLM_ASCEND_GRAPH_META_DEBUG=${VLLM_ASCEND_GRAPH_META_DEBUG:-0}"
    echo "[run] VLLM_ASCEND_DISABLE_GRAPH_FUSION=${VLLM_ASCEND_DISABLE_GRAPH_FUSION:-0}"
    echo "[run] VLLM_ASCEND_DISABLE_NORM_QUANT_FUSION=${VLLM_ASCEND_DISABLE_NORM_QUANT_FUSION:-0}"
    echo "[run] VLLM_ASCEND_DISABLE_QKNORM_ROPE_FUSION=${VLLM_ASCEND_DISABLE_QKNORM_ROPE_FUSION:-0}"
    echo "[run] VLLM_ASCEND_DISABLE_ALLREDUCE_RMS_FUSION=${VLLM_ASCEND_DISABLE_ALLREDUCE_RMS_FUSION:-0}"
    echo "[run] VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED=${VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED:-0}"
    echo "[run] VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS=${VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS:-0}"
    echo "[run] ACTOR_USE_FUSED_KERNELS=${ACTOR_USE_FUSED_KERNELS}"
    echo "[run] GIT_TRACK_RUN_LOGS=${GIT_TRACK_RUN_LOGS}"
} >> "${logfile}"
track_run_log_with_git

set +e
python3 -m verl.trainer.main_ppo --config-path="${CONFIG_DIR}" \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=1024 \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    data.dataloader_num_workers=0 \
    data.sampler.class_path=${SAMPLER_CLASS_PATH} \
    data.sampler.class_name=${SAMPLER_CLASS_NAME} \
    ${SAMPLER_EXTRA_ARGS} \
    +data.dataset_fraction=${DATASET_FRACTION}\
    custom_reward_function.path=deepscaler.py \
    custom_reward_function.name=compute_score  \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_fused_kernels=${ACTOR_USE_FUSED_KERNELS} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.clip_grad=10000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DISTCP_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.free_cache_engine=${VLLM_ROLLOUT_FREE_CACHE_ENGINE:-True} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${VLLM_ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
    actor_rollout_ref.rollout.async_scheduling=${VLLM_ROLLOUT_ASYNC_SCHEDULING} \
    actor_rollout_ref.rollout.enable_prefix_caching=${VLLM_ROLLOUT_ENABLE_PREFIX_CACHING} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K} \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.megatron.context_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.load_weight=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${REF_USE_DIST_CHECKPOINTING} \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DISTCP_PATH}" \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.balance_batch=False \
    trainer.device=npu \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_grpo_example' \
    trainer.experiment_name='qwen3_30_verl_mindspeedllm_vllm' \
    trainer.n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE} \
    trainer.nnodes=${TRAINER_NNODES} \
    trainer.save_freq=${TRAINER_SAVE_FREQ} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP} \
    actor_rollout_ref.actor.checkpoint.save_contents="${ACTOR_CKPT_SAVE_CONTENTS}" \
    actor_rollout_ref.actor.checkpoint.load_contents="${ACTOR_CKPT_LOAD_CONTENTS}" \
    critic.checkpoint.save_contents="${CRITIC_CKPT_SAVE_CONTENTS}" \
    critic.checkpoint.load_contents="${CRITIC_CKPT_LOAD_CONTENTS}" \
    +trainer.save_epoch_freq=0 \
    trainer.default_local_dir="${TRAINER_DEFAULT_LOCAL_DIR}" \
    trainer.test_freq=-1 \
    trainer.total_epochs=${TRAINER_TOTAL_EPOCHS} \
    trainer.total_training_steps=${TRAINER_TOTAL_TRAINING_STEPS} \
    +trainer.rollout_data_dir="${ROLL_OUT_DIR}" \
    +trainer.rollout_length_dir="${ROLL_LEN_DIR}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_num_transformer_layers=[[11],[13],[13],[11]] \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type='alltoall' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_alltoall_overlap_comm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=2048 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True \
    "$@" >> "${logfile}" 2>&1
run_exit_code=$?
set -e

echo "[run] end_time=$(date '+%Y-%m-%dT%H:%M:%S%z') exit_code=${run_exit_code}" | tee -a "${logfile}"
track_run_log_with_git
exit "${run_exit_code}"
