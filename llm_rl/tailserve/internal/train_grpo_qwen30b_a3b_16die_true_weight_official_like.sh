set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

HOME=$(pwd)
CONFIG_DIR=${CONFIG_DIR:-"${HOME}/verl/trainer/config"}
MODEL_PATH=${MODEL_PATH:-"/data/Qwen3-30B-A3B"}
DISTCP_PATH=${DISTCP_PATH:-"/data/Qwen3-30B-A3B_megatron"}
TRAIN_FILE=${TRAIN_FILE:-"/data/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"/data/deepscaler/test.parquet"}

NODES=${NODES:-1}
TRAIN_TP=${TRAIN_TP:-4}
TRAIN_PP=${TRAIN_PP:-4}
TRAIN_CP=${TRAIN_CP:-1}
TRAIN_EP=${TRAIN_EP:-4}
INFER_TP=${INFER_TP:-4}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
CUDAGRAPH_SIZES=${CUDAGRAPH_SIZES:-'[8,16,32]'}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MAX_TOKEN_LEN_PER_GPU=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) / TRAIN_CP))

OUTPUT_DIR=${OUTPUT_DIR:-/workspace/cann-recipes-train/llm_rl/qwen3/official_like_30b_a3b}
mkdir -p "${OUTPUT_DIR}"
time=$(date +%Y%m%d%H%M%S)
logfile="${OUTPUT_DIR}/official_like_30b_a3b_${time}.log"

python3 -m verl.trainer.main_ppo --config-path="${CONFIG_DIR}" \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    custom_reward_function.path=deepscaler.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.clip_grad=10000 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${TRAIN_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DISTCP_PATH}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.recompute_old_log_prob=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.cudagraph_capture_sizes=${CUDAGRAPH_SIZES} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS} \
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
    actor_rollout_ref.ref.megatron.context_parallel_size=${TRAIN_CP} \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.balance_batch=False \
    trainer.device=npu \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_grpo_example_deepscaler' \
    trainer.experiment_name='qwen3_30b_a3b_official_like' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=${NODES} \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=2048 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_num_transformer_layers="[[11],[13],[13],[11]]" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=${TRAIN_CP} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_cp_send_recv_overlap=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo='megatron_cp_algo' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_ring_attention_update=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.cp_window_size=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type='alltoall' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_alltoall_overlap_comm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    $@ 2>&1 | tee "${logfile}"
