set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export HYDRA_FULL_ERROR=1
#export ASCEND_LAUNCH_BLOCKING=1         
export RAY_DEDUP_LOGS=0                   

export ASCEND_GLOBAL_EVENT_ENABLE=0         
export ASCEND_SLOG_PRINT_TO_STDOUT=0       
export ASCEND_GLOBAL_LOG_LEVEL=3           

export HCCL_CONNECT_TIMEOUT=360   
export HCCL_IF_BASE_PORT=64021
export HCCL_EXEC_TIMEOUT=360
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_PORT=23300    # vllm port error
export D2D_DATA_TRANSFER=1
export VLLM_USE_V1=1
export PRINT_MEMORY=1
export USE_ALLTOALL_OVERLAP=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ASCEND_FORCE_ALLTOALL_MOE=${VLLM_ASCEND_FORCE_ALLTOALL_MOE:-0}  # 1: force AllToAll for EP MoE, 0: allow MC2
if [[ "${VLLM_ASCEND_FORCE_ALLTOALL_MOE}" == "1" ]]; then
    export VLLM_ENABLE_MC2=0
else
export VLLM_ENABLE_MC2=1
fi
export VLLM_DP_SIZE=16                        # world_size // rollout.tp_size
export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=${VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:-1}  # 1: enable elastic shrink, 0: keep original dummy-run path
export VLLM_ASCEND_ELASTIC_MOE_MODE=${VLLM_ASCEND_ELASTIC_MOE_MODE:-lossless}  # lossy | lossless
export VLLM_ASCEND_INIT_REDUNDANCY_EXPERT=${VLLM_ASCEND_INIT_REDUNDANCY_EXPERT:-0}  # total preloaded redundant expert replicas for lossless shrink
export HCCL_BUFFSIZE=800

export TASK_QUEUE_ENABLE=2

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

#关闭看门狗
export HCCL_ASYNC_ERROR_HANDLING=1

#Train Drafter开关
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
# micro-step training: one dummy window only runs a seq chunk, then accumulate across windows
export VLLM_ASCEND_DRAFT_MICRO_SEQ_LEN=1
export VLLM_ASCEND_DRAFT_GRAD_ACCUM_STEPS=4096

# Draft profile mode:
#   breakdown   -> 正常训练中打印 draft 分段耗时（默认）
#   profile_only -> 只跑 draft profile step 后退出进程
export DRAFT_PROFILE_MODE=${DRAFT_PROFILE_MODE:-breakdown}

# 默认：在正常训练里看 draft 的分段耗时
export VLLM_ASCEND_DRAFT_PROFILE_ONLY=0
export VLLM_ASCEND_DRAFT_NPU_PROFILE=0
# export VLLM_ASCEND_DRAFT_NPU_PROFILE_STEPS=10
# export VLLM_ASCEND_DRAFT_NPU_PROFILE_WAIT=0
# export VLLM_ASCEND_DRAFT_NPU_PROFILE_WARMUP=2
# export VLLM_ASCEND_DRAFT_NPU_PROFILE_ACTIVE=4
# export VLLM_ASCEND_DRAFT_NPU_PROFILE_REPEAT=1
export VLLM_ASCEND_DRAFT_NPU_PROFILE_DIR=./result/profiler/draft_${DRAFT_PROFILE_MODE}
export VLLM_ASCEND_DRAFT_STARTUP_WARMUP_STEPS=5
export VLLM_ASCEND_DRAFT_WARMUP_STEPS=5
export VLLM_ASCEND_DRAFT_PROFILE_BREAKDOWN=1
export VLLM_ASCEND_DRAFT_PROFILE_SYNC=0
export VLLM_ASCEND_DRAFT_ASYNC_TRAIN=0

if [ "${DRAFT_PROFILE_MODE}" = "profile_only" ]; then
    # 只关注 draft train 的耗时拆分，不进入整套 RL 训练
    export VLLM_ASCEND_DRAFT_PROFILE_ONLY=1
    export VLLM_ASCEND_DRAFT_PROFILE_ONLY_WARMUP_STEPS=2
    export VLLM_ASCEND_DRAFT_PROFILE_ONLY_STEPS=10
    export VLLM_ASCEND_DRAFT_NPU_PROFILE_STEPS=10
    export VLLM_ASCEND_DRAFT_STARTUP_WARMUP_STEPS=0
    export VLLM_ASCEND_DRAFT_WARMUP_STEPS=0
fi

#超时配置
export ACL_MDL_STREAM_SYNC_TIMEOUT=-1
export ACL_MDL_EVENT_SYNC_TIMEOUT=-1

HOME=$(pwd)
MODEL_PATH=${MODEL_PATH:-"/home/data/Qwen3-30B-A3B"}
CONFIG_DIR=${CONFIG_DIR:-"${HOME}/verl/trainer/config"}
DISTCP_PATH="/home/data/Qwen3-30B-A3B_megatron"
TRAIN_FILE=${TRAIN_FILE:-"/workspace/data/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"/workspace/data/deepscaler/test.parquet"}
RECORD_DIR="/workspace/cann-recipes-train/llm_rl/qwen3/record"
mkdir -p "${RECORD_DIR}"

time=$(date +%Y%m%d%H%M%S)
elastic_suffix=""
if [ "${VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK}" = "1" ]; then
    elastic_suffix="_elastic"
fi
logfile=wjeagerqwen30b-a3b-with_draft_${DRAFT_PROFILE_MODE}_${time}${elastic_suffix}.txt

set -x

python3 -m verl.trainer.main_ppo --config-path="${CONFIG_DIR}" \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    +data.dataset_fraction=0.001\
    custom_reward_function.path=deepscaler.py \
    custom_reward_function.name=compute_score  \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.clip_grad=10000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.n=16 \
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
    trainer.experiment_name='qwen3_30_verl_mindspeedllm_vllm' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    +trainer.rollout_data_dir="${RECORD_DIR}" \
    +trainer.rollout_length_dir="${RECORD_DIR}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_num_transformer_layers=[[11],[13],[13],[11]] \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type='alltoall' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_alltoall_overlap_comm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=2048 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True  $@ >> "${logfile}" 
