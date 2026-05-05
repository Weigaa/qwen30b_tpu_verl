set -euo pipefail
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
# Ascend env scripts probe shell-specific vars like ZSH_VERSION directly,
# so source them with nounset temporarily disabled.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3

export HCCL_CONNECT_TIMEOUT=360
export HCCL_IF_BASE_PORT=64021
export HCCL_EXEC_TIMEOUT=360
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_PORT=23300
export D2D_DATA_TRANSFER=1
export VLLM_USE_V1=1
export PRINT_MEMORY=1
export USE_ALLTOALL_OVERLAP=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ASCEND_FORCE_ALLTOALL_MOE=${VLLM_ASCEND_FORCE_ALLTOALL_MOE:-0}
if [[ "${VLLM_ASCEND_FORCE_ALLTOALL_MOE}" == "1" ]]; then
    export VLLM_ENABLE_MC2=0
else
    export VLLM_ENABLE_MC2=1
fi
export VLLM_DP_SIZE=16
export HCCL_BUFFSIZE=800

export TASK_QUEUE_ENABLE=2

export VLLM_ENABLE_FIX_ROUTE=0
export VLLM_MODEL_EXECUTE_TIME_OBSERVE=0

export VLLM_ENABLE_GRAPH_MODE=0
export VLLM_ENABLE_EXPERT_PARALLEL=1
export VLLM_CHUNK_MOE_SIZE=512
export ALL_TO_ALL_RESHARD=1
export USE_ALLTOALL_OVERLAP=1
export VLLM_ENABLE_EPLB=0
export USE_HDP=0
export ROLLOUT_REBALANCE_ENABLE=0

export HCCL_ASYNC_ERROR_HANDLING=1

export VLLM_ASCEND_ENABLE_DRAFT_TRAIN=0
export VLLM_ASCEND_DRAFT_WARMUP_ON_INIT=1
export VLLM_ASCEND_DRAFT_LR=1e-4
export VLLM_ASCEND_DRAFT_REUSE_TARGET_EMB_LM=1
export VLLM_ASCEND_DRAFT_VOCAB_SIZE=4096
export VLLM_ASCEND_DRAFT_QUEUE_SIZE=4
export VLLM_ASCEND_DRAFT_MAX_SEQ_LEN=16384
export VLLM_ASCEND_DRAFT_TRAIN_DTYPE=bf16
export VLLM_ASCEND_DRAFT_ATTN_IMPL=sdpa
export VLLM_ASCEND_DRAFT_ATTN_CHUNK_SIZE=1024
export VLLM_ASCEND_DRAFT_LORA_ENABLE=0
export VLLM_ASCEND_DRAFT_LORA_BACKEND=custom
export VLLM_ASCEND_DRAFT_LORA_RANK=8
export VLLM_ASCEND_DRAFT_LORA_ALPHA=16
export VLLM_ASCEND_DRAFT_LORA_DROPOUT=0.0
export VLLM_ASCEND_DRAFT_LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,fc
export VLLM_ASCEND_DRAFT_SPARSE_KL_ENABLE=1
export VLLM_ASCEND_DRAFT_SPARSE_KL_TOPK=64
export VLLM_ASCEND_DRAFT_COMPUTE_ACCURACY=0
export VLLM_ASCEND_DRAFT_MICRO_SEQ_LEN=1
export VLLM_ASCEND_DRAFT_GRAD_ACCUM_STEPS=4096

export DRAFT_PROFILE_MODE=${DRAFT_PROFILE_MODE:-breakdown}
export VLLM_ASCEND_DRAFT_PROFILE_ONLY=0
export VLLM_ASCEND_DRAFT_NPU_PROFILE=0
export VLLM_ASCEND_DRAFT_NPU_PROFILE_DIR=./result/profiler/draft_${DRAFT_PROFILE_MODE}
export VLLM_ASCEND_DRAFT_STARTUP_WARMUP_STEPS=5
export VLLM_ASCEND_DRAFT_WARMUP_STEPS=5
export VLLM_ASCEND_DRAFT_PROFILE_BREAKDOWN=1
export VLLM_ASCEND_DRAFT_PROFILE_SYNC=0
export VLLM_ASCEND_DRAFT_ASYNC_TRAIN=0

# mode=0 baseline; no elastic shrink path.
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=0
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=1
export VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS=${VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS:-16}

# Enable MoE stats and include stage=16 so mode=0 runs still produce stage-level CSVs.
export VLLM_MOE_PATTERN_STATS=1
export VLLM_MOE_STATS=${VLLM_MOE_PATTERN_STATS}
export VLLM_MOE_LONG_TAIL_STAGES=${VLLM_MOE_LONG_TAIL_STAGES:-16}

# export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896

if [[ "${DRAFT_PROFILE_MODE}" == "profile_only" ]]; then
    export VLLM_ASCEND_DRAFT_PROFILE_ONLY=1
    export VLLM_ASCEND_DRAFT_PROFILE_ONLY_WARMUP_STEPS=2
    export VLLM_ASCEND_DRAFT_PROFILE_ONLY_STEPS=10
    export VLLM_ASCEND_DRAFT_NPU_PROFILE_STEPS=10
    export VLLM_ASCEND_DRAFT_STARTUP_WARMUP_STEPS=0
    export VLLM_ASCEND_DRAFT_WARMUP_STEPS=0
fi

export ACL_MDL_STREAM_SYNC_TIMEOUT=-1
export ACL_MDL_EVENT_SYNC_TIMEOUT=-1

HOME=$(pwd)
MODEL_PATH=${MODEL_PATH:-"/home/data/Qwen3-30B-A3B"}
CONFIG_DIR=${CONFIG_DIR:-"${HOME}/verl/trainer/config"}
DISTCP_PATH="/home/data/Qwen3-30B-A3B_megatron"
TRAIN_FILE=${TRAIN_FILE:-"/workspace/data/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"/workspace/data/deepscaler/test.parquet"}

TOTAL_BUDGET=${TOTAL_BUDGET:-512}
N_VALUES=(${N_VALUES:-2 4 8 16})
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
FIXED_TOTAL_TRAINING_STEPS=${FIXED_TOTAL_TRAINING_STEPS:-${TOTAL_EPOCHS}}
DATASET_SAMPLE_PAD=${DATASET_SAMPLE_PAD:-32}
TRAIN_DATASET_ROWS=$(python3 - "${TRAIN_FILE}" <<'PY'
import sys

import pyarrow.parquet as pq

parquet_path = sys.argv[1]
pf = pq.ParquetFile(parquet_path)
rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups))
print(rows)
PY
)
RUN_STAMP=$(date +%Y%m%d%H%M%S)
SWEEP_ROOT=${SWEEP_ROOT:-"/workspace/cann-recipes-train/llm_rl/qwen3/mode0_k512_sweep_${RUN_STAMP}"}
mkdir -p "${SWEEP_ROOT}"

MANIFEST="${SWEEP_ROOT}/run_manifest.tsv"
printf "case_id\tn\tbatch_size\tdataset_fraction\tlogfile\trecord_dir\tmoe_stats_dir\n" > "${MANIFEST}"

run_case() {
    local n="$1"
    shift
    local extra_args=("$@")
    if (( TOTAL_BUDGET % n != 0 )); then
        echo "TOTAL_BUDGET=${TOTAL_BUDGET} is not divisible by n=${n}" >&2
        exit 1
    fi

    local batch_size=$((TOTAL_BUDGET / n))
    local effective_sample_pad
    local target_train_rows
    local dataset_fraction
    # Keep all four sweep points comparable:
    # 1. each epoch should contain exactly one aligned train batch
    # 2. total_training_steps should stay fixed across n
    #
    # Trainer-side behavior:
    #   aligned_size = floor(len(train_dataset) / batch_size) * batch_size
    #   total_training_steps = len(train_dataloader) * total_epochs
    #
    # So here we choose dataset_fraction to target a sampled train split in
    # [batch_size, 2 * batch_size). After DataAlign this always becomes exactly
    # one batch. We then also pass trainer.total_training_steps explicitly.
    #
    # With train_rows ~= 36283 and DATASET_SAMPLE_PAD=32:
    #   n=2  -> bs=256 -> target_rows=288 -> align 256 -> 1 batch/epoch
    #   n=4  -> bs=128 -> target_rows=160 -> align 128 -> 1 batch/epoch
    #   n=8  -> bs=64  -> target_rows=96  -> align 64  -> 1 batch/epoch
    #   n=16 -> bs=32  -> target_rows=63  -> align 32  -> 1 batch/epoch
    #
    # The cap below is important for n=16: if we let sample_pad reach 32, then
    # target_rows becomes 64 and DataAlign would produce two batches instead of
    # one.
    effective_sample_pad=$(python3 - "${batch_size}" "${DATASET_SAMPLE_PAD}" <<'PY'
import sys

batch_size = int(sys.argv[1])
sample_pad = int(sys.argv[2])
print(max(0, min(sample_pad, batch_size - 1)))
PY
)
    target_train_rows=$(python3 - "${batch_size}" "${effective_sample_pad}" <<'PY'
import sys

batch_size = int(sys.argv[1])
effective_sample_pad = int(sys.argv[2])
print(batch_size + effective_sample_pad)
PY
)
    dataset_fraction=$(python3 - "${TRAIN_DATASET_ROWS}" "${target_train_rows}" <<'PY'
import sys

train_rows = int(sys.argv[1])
target_train_rows = int(sys.argv[2])
dataset_fraction = target_train_rows / train_rows
print(f"{dataset_fraction:.6f}")
PY
)
    local case_id="mode0_k${TOTAL_BUDGET}_n${n}_bs${batch_size}"
    local case_root="${SWEEP_ROOT}/${case_id}"
    local record_dir="${case_root}/record"
    local moe_stats_dir="${case_root}/moe_stats"
    local logfile="${case_root}/wjeagerqwen30b-a3b-with_draft_${DRAFT_PROFILE_MODE}_${RUN_STAMP}_${case_id}.txt"

    mkdir -p "${case_root}" "${record_dir}" "${moe_stats_dir}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${case_id}" "${n}" "${batch_size}" "${dataset_fraction}" "${logfile}" "${record_dir}" "${moe_stats_dir}" \
        >> "${MANIFEST}"

    export VLLM_MOE_STATS_DIR="${moe_stats_dir}"
    echo "[moe pattern stats] enabled=${VLLM_MOE_PATTERN_STATS} dir=${VLLM_MOE_STATS_DIR} mode=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE} stages=${VLLM_MOE_LONG_TAIL_STAGES} n=${n} batch_size=${batch_size} target_train_rows=${target_train_rows} dataset_fraction=${dataset_fraction} total_training_steps=${FIXED_TOTAL_TRAINING_STEPS}"

    python3 -m verl.trainer.main_ppo --config-path="${CONFIG_DIR}" \
        --config-name='ppo_megatron_trainer.yaml' \
        algorithm.adv_estimator=grpo \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.train_batch_size="${batch_size}" \
        data.max_prompt_length=1024 \
        data.max_response_length=16384 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=False \
        +data.dataset_fraction="${dataset_fraction}" \
        custom_reward_function.path=deepscaler.py \
        custom_reward_function.name=compute_score \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.clip_grad=10000 \
        actor_rollout_ref.actor.ppo_mini_batch_size="${batch_size}" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
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
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.load_weight=True \
        actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
        actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
        actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.max_num_seqs="${batch_size}" \
        actor_rollout_ref.rollout.n="${n}" \
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
        trainer.experiment_name="qwen3_30_verl_mindspeedllm_vllm_${case_id}" \
        trainer.n_gpus_per_node=16 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs="${TOTAL_EPOCHS}" \
        trainer.total_training_steps="${FIXED_TOTAL_TRAINING_STEPS}" \
        +trainer.rollout_data_dir="${record_dir}" \
        +trainer.rollout_length_dir="${record_dir}" \
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
        "${extra_args[@]}" >> "${logfile}" 2>&1
}

for n in "${N_VALUES[@]}"; do
    run_case "${n}" "$@"
done

echo "Sweep completed. Manifest: ${MANIFEST}"
