# Variant: regroup + rollout eager workaround for MoeDistributeDispatchV2 torchair compile issue
set -ex

if [[ "${ADAFLOOR_TRAIN_LAUNCHER_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    train_launcher_source=$(realpath "${BASH_SOURCE[0]}")
    train_launcher_snapshot=$(mktemp "${train_launcher_source}.run-snapshot.XXXXXX")
    cp -- "$train_launcher_source" "$train_launcher_snapshot"
    set +e
    ADAFLOOR_TRAIN_LAUNCHER_SNAPSHOT_ACTIVE=1 \
        bash "$train_launcher_snapshot" "$@"
    train_launcher_rc=$?
    set -e
    rm -f -- "$train_launcher_snapshot"
    exit "$train_launcher_rc"
fi

SCRIPT_DIR_INTERNAL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_NOISE_FILTER="${LOG_NOISE_FILTER:-${SCRIPT_DIR_INTERNAL}/filter_known_log_noise.py}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

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
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64021}
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_PORT=${MASTER_PORT:-23300}    # vllm port error
export D2D_DATA_TRANSFER=1

export VLLM_USE_V1=1
export PRINT_MEMORY=1
export USE_ALLTOALL_OVERLAP=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ENABLE_MC2=1                     # 910C开启
export VLLM_DP_SIZE=${VLLM_DP_SIZE:-16}       # vLLM EP group size
export HCCL_BUFFSIZE=800

export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}

export VLLM_ENABLE_FIX_ROUTE=0    
export VLLM_MODEL_EXECUTE_TIME_OBSERVE=0     # decode prefill的耗时打印

#extra env in qwen3_235b_env.sh
# Recipe features
export VLLM_ENABLE_GRAPH_MODE=0             # 0: eager mode, 1: graph mode
export VLLM_ENABLE_EXPERT_PARALLEL=1        # Enable EP in vLLM rollout.
export VLLM_CHUNK_MOE_SIZE=512              # The minimum block size set for prefill computation partition.
export ALL_TO_ALL_RESHARD=1                 # Enable EP to reshard parameters with AllToAllV (without communication redundancy).
export USE_ALLTOALL_OVERLAP=1               # Enable to overlap communication in EP with computation to hide MoE communication latency. Should be consistent with model conversion config.
export VLLM_ENABLE_EPLB=0                   # 0: disable eplb, 1: enable eplb
export USE_HDP=0                            # 0: disable hdp, 1: enable hdp
export ROLLOUT_REBALANCE_ENABLE=0          # 0: disable rollout rebalance, 1: enable rollout rebalance
#关闭看门狗,并控制超时时间
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200

#控制llm.py里的profiler
export VLLM_ASCEND_LLM_PROFILE_ENABLE=0

# Unified output root for checkpoints / rollout dumps / draft dumps / logs.
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/cann-recipes-train/llm_rl/qwen3_shrink_aware}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_baseline_ft}
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}"
ROLL_OUT_DIR="${OUTPUT_DIR}/rollout_data"
ROLL_LEN_DIR="${OUTPUT_DIR}/rollout_length"
DRAFT_DUMP_DIR="${OUTPUT_DIR}/draft_hidden"
CHECKPOINT_MODEL_DIR_NAME=${CHECKPOINT_MODEL_DIR_NAME:-qwen3moe_for_eagle3}
if [[ ! "$CHECKPOINT_MODEL_DIR_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid CHECKPOINT_MODEL_DIR_NAME=$CHECKPOINT_MODEL_DIR_NAME" >&2
    exit 2
fi
CKPT_DIR="${OUTPUT_DIR}/checkpoints/${CHECKPOINT_MODEL_DIR_NAME}"
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
export VLLM_EPOCH_LENGTH_REGROUP_EMA_DECAY=${VLLM_EPOCH_LENGTH_REGROUP_EMA_DECAY:-0.3}
export VLLM_EPOCH_LENGTH_REGROUP_SHUFFLE_BLOCKS=${VLLM_EPOCH_LENGTH_REGROUP_SHUFFLE_BLOCKS:-1}
# Rollout long-tail guard (default on):
# Cap per-step generation by expected response length from resampler.
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=${VLLM_ROLLOUT_EARLY_STOP_ENABLE:-1}
export VLLM_ROLLOUT_EARLY_STOP_FACTOR=${VLLM_ROLLOUT_EARLY_STOP_FACTOR:-2.0}
export VLLM_ROLLOUT_EARLY_STOP_MIN_TOKENS=${VLLM_ROLLOUT_EARLY_STOP_MIN_TOKENS:-10000}
# Baseline switch:
#   1 -> eager on + no resample (disable length regroup sampler)
#   0 -> follow regroup switch below
export VLLM_EAGER_BASELINE_NO_RESAMPLE=${VLLM_EAGER_BASELINE_NO_RESAMPLE:-0}

if [ "${VLLM_EAGER_BASELINE_NO_RESAMPLE}" = "1" ]; then
    SAMPLER_CLASS_PATH=null
    SAMPLER_CLASS_NAME=null
    SAMPLER_EXTRA_ARGS=""
    ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
elif [ "${VLLM_EPOCH_LENGTH_REGROUP_ENABLE}" = "1" ]; then
    SAMPLER_CLASS_PATH=pkg://verl.experimental.dataset.length_bucket_sampler
    SAMPLER_CLASS_NAME=LengthAwareEpochSampler
    SAMPLER_EXTRA_ARGS="+data.sampler.bucket_size=${VLLM_EPOCH_LENGTH_REGROUP_BUCKET_SIZE} +data.sampler.ema_decay=${VLLM_EPOCH_LENGTH_REGROUP_EMA_DECAY} +data.sampler.shuffle_batch_blocks=${VLLM_EPOCH_LENGTH_REGROUP_SHUFFLE_BLOCKS}"
    # Work around torchair graph compile instability for MoE dispatch under
    # dynamically regrouped rollout batches.
    ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
else
    SAMPLER_CLASS_PATH=null
    SAMPLER_CLASS_NAME=null
    SAMPLER_EXTRA_ARGS=""
    ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
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

if [ "${SAVE_CKPT_ENABLE}" = "1" ]; then
    TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-1}
    TRAINER_DEFAULT_LOCAL_DIR="${CKPT_DIR}"
else
    TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:--1}
    TRAINER_DEFAULT_LOCAL_DIR="${OUTPUT_DIR}"
fi

HOME=$(pwd)
MODEL_PATH=${MODEL_PATH:-"/data/Qwen3-30B-A3B"}
CONFIG_DIR=${CONFIG_DIR:-"${HOME}/verl/trainer/config"}
DISTCP_PATH=${DISTCP_PATH:-"/data/Qwen3-30B-A3B_megatron"}
TRAIN_FILE=${TRAIN_FILE:-"/data/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"/data/deepscaler/test.parquet"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
DATASET_FRACTION=${DATASET_FRACTION:-0.005}
MCORE_SEQUENCE_PARALLEL=${MCORE_SEQUENCE_PARALLEL:-True}
MCORE_EXPERT_MODEL_PARALLEL_SIZE=${MCORE_EXPERT_MODEL_PARALLEL_SIZE:-4}
MCORE_TENSOR_MODEL_PARALLEL_SIZE=${MCORE_TENSOR_MODEL_PARALLEL_SIZE:-4}
MCORE_PIPELINE_MODEL_PARALLEL_SIZE=${MCORE_PIPELINE_MODEL_PARALLEL_SIZE:-4}
MCORE_EXPERT_TENSOR_PARALLEL_SIZE=${MCORE_EXPERT_TENSOR_PARALLEL_SIZE:-1}
MCORE_PIPELINE_NUM_TRANSFORMER_LAYERS=${MCORE_PIPELINE_NUM_TRANSFORMER_LAYERS:-'[[11],[13],[13],[11]]'}
MCORE_FIRST_PIPELINE_NUM_LAYERS=${MCORE_FIRST_PIPELINE_NUM_LAYERS:-11}
MCORE_LAST_PIPELINE_NUM_LAYERS=${MCORE_LAST_PIPELINE_NUM_LAYERS:-11}
MCORE_MOE_ALLTOALL_OVERLAP_COMM=${MCORE_MOE_ALLTOALL_OVERLAP_COMM:-True}
MCORE_MOE_SHARED_EXPERT_OVERLAP=${MCORE_MOE_SHARED_EXPERT_OVERLAP:-True}
MCORE_DEALLOCATE_PIPELINE_OUTPUTS=${MCORE_DEALLOCATE_PIPELINE_OUTPUTS:-False}
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
ROLLOUT_N=${ROLLOUT_N:-16}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-20480}
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-20480}
TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-3}
TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-qwen3_30_verl_mindspeedllm_vllm}
    

time=$(date +%Y%m%d%H%M%S)
TRAIN_LOG_PREFIX=${TRAIN_LOG_PREFIX:-wjqwen30b-a3b-record_graph_save4eagle3}
logfile="${LOG_DIR}/${TRAIN_LOG_PREFIX}_${time}.txt"

export VERL_SIDECAR_ENABLE=${VERL_SIDECAR_ENABLE:-0}
export VERL_SIDECAR_MULTI_STAGE=${VERL_SIDECAR_MULTI_STAGE:-0}
export VERL_SIDECAR_TARGET_FLOORS=${VERL_SIDECAR_TARGET_FLOORS:-8,4,2}
export VERL_SIDECAR_PROMPTS_FILE=${VERL_SIDECAR_PROMPTS_FILE:-/data/gsm8k}
export VERL_SIDECAR_DATA_SPLIT=${VERL_SIDECAR_DATA_SPLIT:-train}
if [[ "${VERL_SIDECAR_MULTI_STAGE}" == "1" ]]; then
    export VERL_SIDECAR_MODEL_PATH=${VERL_SIDECAR_MODEL_PATH:-/data/Qwen2.5-1.5B-Instruct}
    export VERL_SIDECAR_PARALLEL_MODE=${VERL_SIDECAR_PARALLEL_MODE:-dp}
    export VERL_SIDECAR_TENSOR_PARALLEL_SIZE=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE:-1}
    export VERL_SIDECAR_DATA_PARALLEL_SIZE=${VERL_SIDECAR_DATA_PARALLEL_SIZE:-1}
    export VERL_SIDECAR_REPLICA_COUNT=${VERL_SIDECAR_REPLICA_COUNT:-1}
    export VERL_SIDECAR_ENABLE_EXPERT_PARALLEL=${VERL_SIDECAR_ENABLE_EXPERT_PARALLEL:-0}
    export VERL_SIDECAR_GPU_MEMORY_UTILIZATION=${VERL_SIDECAR_GPU_MEMORY_UTILIZATION:-0.80}
    export VERL_SIDECAR_MAX_NUM_SEQS=${VERL_SIDECAR_MAX_NUM_SEQS:-128}
    export VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=${VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS:-32768}
    export VERL_SIDECAR_MAX_MODEL_LEN=${VERL_SIDECAR_MAX_MODEL_LEN:-4096}
    export VERL_SIDECAR_MAX_TOKENS=${VERL_SIDECAR_MAX_TOKENS:-1024}
    export VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA=${VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA:-128}
    export VERL_SIDECAR_GENERATE_CHUNK_SIZE=${VERL_SIDECAR_GENERATE_CHUNK_SIZE:-128}
else
    export VERL_SIDECAR_MODEL_PATH=${VERL_SIDECAR_MODEL_PATH:-/data/Qwen3-8B}
    export VERL_SIDECAR_PARALLEL_MODE=${VERL_SIDECAR_PARALLEL_MODE:-hybrid}
    export VERL_SIDECAR_TENSOR_PARALLEL_SIZE=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE:-8}
    export VERL_SIDECAR_DATA_PARALLEL_SIZE=${VERL_SIDECAR_DATA_PARALLEL_SIZE:-1}
    export VERL_SIDECAR_REPLICA_COUNT=${VERL_SIDECAR_REPLICA_COUNT:-1}
    export VERL_SIDECAR_ENABLE_EXPERT_PARALLEL=${VERL_SIDECAR_ENABLE_EXPERT_PARALLEL:-0}
    export VERL_SIDECAR_GPU_MEMORY_UTILIZATION=${VERL_SIDECAR_GPU_MEMORY_UTILIZATION:-0.90}
    export VERL_SIDECAR_MAX_NUM_SEQS=${VERL_SIDECAR_MAX_NUM_SEQS:-278}
    export VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=${VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS:-65536}
    export VERL_SIDECAR_MAX_MODEL_LEN=${VERL_SIDECAR_MAX_MODEL_LEN:-6144}
    export VERL_SIDECAR_MAX_TOKENS=${VERL_SIDECAR_MAX_TOKENS:-4096}
    export VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA=${VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA:-556}
    export VERL_SIDECAR_GENERATE_CHUNK_SIZE=${VERL_SIDECAR_GENERATE_CHUNK_SIZE:-556}
fi
export VERL_SIDECAR_REPEAT_UNTIL_KILLED=${VERL_SIDECAR_REPEAT_UNTIL_KILLED:-1}
export VERL_SIDECAR_STREAM_CHECKPOINT=${VERL_SIDECAR_STREAM_CHECKPOINT:-1}
export VERL_SIDECAR_EXPECTED_ACTIVE_RANKS=${VERL_SIDECAR_EXPECTED_ACTIVE_RANKS:-8}
export VERL_SIDECAR_START_TRIGGER=${VERL_SIDECAR_START_TRIGGER:-shrink_done}
export VERL_SIDECAR_LOG_DIR=${VERL_SIDECAR_LOG_DIR:-"${OUTPUT_DIR}/sidecar"}
export VERL_SIDECAR_STATE_DIR=${VERL_SIDECAR_STATE_DIR:-"${VERL_SIDECAR_LOG_DIR}/state"}
export VERL_SIDECAR_RESTORE_HANDSHAKE_DIR=${VERL_SIDECAR_RESTORE_HANDSHAKE_DIR:-"${VERL_SIDECAR_LOG_DIR}/restore_handshake"}
export VERL_SIDECAR_STOP_ACK_TIMEOUT_SECONDS=${VERL_SIDECAR_STOP_ACK_TIMEOUT_SECONDS:-60}

sidecar_monitor_pid=""
cleanup_sidecar_monitor() {
    if [[ -n "${sidecar_monitor_pid}" ]] && kill -0 "${sidecar_monitor_pid}" 2>/dev/null; then
        kill "${sidecar_monitor_pid}" 2>/dev/null || true
        wait "${sidecar_monitor_pid}" 2>/dev/null || true
    fi
}

if [[ "${VERL_SIDECAR_ENABLE}" == "1" ]]; then
    [[ -f "${VERL_SIDECAR_MODEL_PATH}/config.json" ]] || {
        echo "[elastic sidecar] missing model config: ${VERL_SIDECAR_MODEL_PATH}/config.json" >&2
        exit 2
    }
    [[ -e "${VERL_SIDECAR_PROMPTS_FILE}" ]] || {
        echo "[elastic sidecar] missing prompts: ${VERL_SIDECAR_PROMPTS_FILE}" >&2
        exit 2
    }
    mkdir -p "${VERL_SIDECAR_LOG_DIR}"
    : > "${logfile}"
    export VERL_SIDECAR_TRAIN_LOG="${logfile}"
    export VERL_SIDECAR_LEASE_LOG=${VERL_SIDECAR_LEASE_LOG:-"${VERL_SIDECAR_LOG_DIR}/lease.log"}
    export VERL_SIDECAR_LOG_FILE=${VERL_SIDECAR_LOG_FILE:-"${VERL_SIDECAR_LOG_DIR}/infer.log"}
    export VERL_SIDECAR_OUTPUT_FILE=${VERL_SIDECAR_OUTPUT_FILE:-"${VERL_SIDECAR_LOG_DIR}/outputs.jsonl"}
    sidecar_monitor_log=${VERL_SIDECAR_MONITOR_LOG:-"${VERL_SIDECAR_LOG_DIR}/monitor.log"}
    if [[ "${VERL_SIDECAR_MULTI_STAGE}" == "1" ]]; then
        sidecar_watcher="${SCRIPT_DIR_INTERNAL}/watch_elastic_shrink_and_run_multistage_sidecars.sh"
    else
        sidecar_watcher="${SCRIPT_DIR_INTERNAL}/watch_elastic_shrink_and_run_sidecar.sh"
    fi
    "${sidecar_watcher}" "${logfile}" >> "${sidecar_monitor_log}" 2>&1 &
    sidecar_monitor_pid=$!
    trap cleanup_sidecar_monitor EXIT
    echo "[elastic sidecar] enabled model=${VERL_SIDECAR_MODEL_PATH} prompts=${VERL_SIDECAR_PROMPTS_FILE} multi_stage=${VERL_SIDECAR_MULTI_STAGE} target_floors=${VERL_SIDECAR_TARGET_FLOORS} parallel_mode=${VERL_SIDECAR_PARALLEL_MODE} tp=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE} dp=${VERL_SIDECAR_DATA_PARALLEL_SIZE} replicas=${VERL_SIDECAR_REPLICA_COUNT} watcher=${sidecar_watcher} log_dir=${VERL_SIDECAR_LOG_DIR}"
fi

set -x

run_trainer_with_log_filter() {
    if [[ "${VERL_SUPPRESS_KNOWN_NOISE:-1}" == "1" && -f "${LOG_NOISE_FILTER}" ]]; then
        "$@" > >(python3 "${LOG_NOISE_FILTER}" >> "${logfile}") 2>&1
    else
        "$@" >> "${logfile}" 2>&1
    fi
}

run_trainer_with_log_filter python3 -m verl.trainer.main_ppo --config-path="${CONFIG_DIR}" \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    data.dataloader_num_workers=0 \
    data.sampler.class_path=${SAMPLER_CLASS_PATH} \
    data.sampler.class_name=${SAMPLER_CLASS_NAME} \
    ${SAMPLER_EXTRA_ARGS} \
    +data.dataset_fraction="${DATASET_FRACTION}" \
    custom_reward_function.path=deepscaler.py \
    custom_reward_function.name=compute_score  \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.clip_grad=10000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.megatron.sequence_parallel="${MCORE_SEQUENCE_PARALLEL}" \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${MCORE_EXPERT_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${MCORE_TENSOR_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${MCORE_PIPELINE_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="${MCORE_EXPERT_TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DISTCP_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER} \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.load_weight=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DISTCP_PATH}" \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.balance_batch=False \
    trainer.device=npu \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_grpo_example' \
    trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
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
    trainer.total_epochs="${TRAINER_TOTAL_EPOCHS}" \
    +trainer.rollout_data_dir="${ROLL_OUT_DIR}" \
    +trainer.rollout_length_dir="${ROLL_LEN_DIR}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_num_transformer_layers="${MCORE_PIPELINE_NUM_TRANSFORMER_LAYERS}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type='alltoall' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_alltoall_overlap_comm="${MCORE_MOE_ALLTOALL_OVERLAP_COMM}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap="${MCORE_MOE_SHARED_EXPERT_OVERLAP}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs="${MCORE_DEALLOCATE_PIPELINE_OUTPUTS}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=2048 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage="${MCORE_FIRST_PIPELINE_NUM_LAYERS}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage="${MCORE_LAST_PIPELINE_NUM_LAYERS}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True  "$@"
