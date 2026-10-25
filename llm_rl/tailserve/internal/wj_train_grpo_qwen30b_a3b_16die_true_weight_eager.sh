set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:24"

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1

export HYDRA_FULL_ERROR=1
#export ASCEND_LAUNCH_BLOCKING=1         
export RAY_DEDUP_LOGS=0                   

export ASCEND_GLOBAL_EVENT_ENABLE=0         
export ASCEND_SLOG_PRINT_TO_STDOUT=0       
export ASCEND_GLOBAL_LOG_LEVEL=3           

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64021
export HCCL_EXEC_TIMEOUT=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export TP_SOCKET_IFNAME=enp23s0f3
export HCCL_SOCKET_IFNAME=enp23s0f3
export GLOO_SOCKET_IFNAME=enp23s0f3
unset VERL_REF_INIT_FROM_COLOCATED_ACTOR

export MASTER_PORT=23300    # vllm port error
export D2D_DATA_TRANSFER=1
export VLLM_USE_V1=1
export PRINT_MEMORY=1
export USE_ALLTOALL_OVERLAP=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ENABLE_MC2=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_DP_SIZE=16
export HCCL_BUFFSIZE=800

export TASK_QUEUE_ENABLE=1

export VLLM_ENABLE_FIX_ROUTE=0    
export VLLM_MODEL_EXECUTE_TIME_OBSERVE=0     # decode prefill的耗时打印

#extra env in qwen3_235b_env.sh
# Recipe features
export VLLM_ENABLE_GRAPH_MODE=1             # 0: eager mode, 1: graph mode
export VLLM_ENABLE_EXPERT_PARALLEL=1        # Enable EP in vLLM rollout.
export VLLM_CHUNK_MOE_SIZE=512              # The minimum block size set for prefill computation partition.
export ALL_TO_ALL_RESHARD=1                 # Enable EP to reshard parameters with AllToAllV (without communication redundancy).
export USE_ALLTOALL_OVERLAP=1               # Enable to overlap communication in EP with computation to hide MoE communication latency. Should be consistent with model conversion config.
export VLLM_ENABLE_EPLB=0                   # 0: disable eplb, 1: enable eplb
export USE_HDP=0                            # 0: disable hdp, 1: enable hdp
export ROLLOUT_REBALANCE_ENABLE=0          # 0: disable rollout rebalance, 1: enable rollout rebalance

# Match the official 235B launcher defaults here first; CANN 8.5 is more
# sensitive to HCCL comm establishment on the ref dist-checkpoint path.
export HCCL_ASYNC_ERROR_HANDLING=0
unset VERL_DIST_CKPT_VALIDATE_ACCESS_INTEGRITY
export VERL_DIST_CKPT_LOAD_NO_DIST=1
export VERL_DIST_CKPT_LOAD_PROCESS_GROUP=gloo
unset VERL_DIST_CKPT_CPU_STAGING

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
export VLLM_ASCEND_LLM_PROFILE_ENABLE=0
# 弹性执行模式:
# 0: baseline dummy-run
# 1: 冗余专家模式
# 2: CPU/NPU 混合模式
# 3: 无冗余专家 + 跨层双缓冲 hybrid tail
#    使用建议:
#    - mode=3 主要面向 shrink 到 <=8 rank 之后的 MoE 主路径
#    - 若希望允许 2 -> 1 的 single-rank no-EP tail，保持
#      VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=1
#    - mode=3 不依赖冗余专家；运行时 expert double buffer 固定为 128 expert slots
#    - VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS 在 mode=3 下只保留
#      primary prefix 语义，不再决定运行时 buffer 容量
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE:-1}
# 弹性缩容的最小计算组:
#   1  -> 允许在 2-rank 阶段后进入 single-rank no-EP tail
#   2/4/8/16 -> 最多缩到该 floor 结束，不再进入 1-rank tail
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=8
# mode=2 时每个 rank 固定保留的 NPU resident expert 槽位数
# mode=3 时该值不控制双缓冲大小；当前 runtime double buffer 固定为 128 experts
export VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS=${VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS:-8}
# mode=3 step-1: allow next-layer NPU resident experts to prefetch into the
# alternate runtime buffer. CPU-only experts still fill synchronously at bind.
export VLLM_ASCEND_MODE3_ASYNC_NPU_PREFETCH=${VLLM_ASCEND_MODE3_ASYNC_NPU_PREFETCH:-1}
# mode=3 step-2: prefetch CPU-only experts into a plain NPU staging buffer on
# a separate stream.
export VLLM_ASCEND_MODE3_ASYNC_CPU_STAGE=${VLLM_ASCEND_MODE3_ASYNC_CPU_STAGE:-1}
# mode=3 step-3: after CPU staging, pack CPU-only experts into their final
# runtime slots on the CPU staging stream. The main prefetch stream only waits
# for cpu_pack_event before marking the alternate buffer ready.
export VLLM_ASCEND_MODE3_ASYNC_CPU_PACK=${VLLM_ASCEND_MODE3_ASYNC_CPU_PACK:-1}
# mode=3 step-4: bind current layer by inserting a device-side wait on the
# slot ready event, instead of blocking Python with event.synchronize().
export VLLM_ASCEND_MODE3_DEVICE_READY_WAIT=${VLLM_ASCEND_MODE3_DEVICE_READY_WAIT:-1}
# mode=3 step-5: try direct CPU shadow row copies into the final runtime
# expert slots on the CPU prefetch stream, bypassing the staging buffer.
#
# Default to staging for the next A/B run so it can be compared with the
# previous direct-slot run. Override to 1 to restore direct CPU -> runtime slot.
export VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT=${VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT:-1}
# mode=3 step-6: coalesce contiguous expert slot copies into larger slice
# copies. CPU direct bulk is experimental because the runtime slot may use a
# formatted layout; set it to 0 to fall back to the proven per-expert direct
# copies while keeping NPU/staging bulk copies enabled.
export VLLM_ASCEND_MODE3_BULK_NPU_COPY=${VLLM_ASCEND_MODE3_BULK_NPU_COPY:-1}
export VLLM_ASCEND_MODE3_BULK_CPU_STAGE=${VLLM_ASCEND_MODE3_BULK_CPU_STAGE:-1}
export VLLM_ASCEND_MODE3_BULK_CPU_DIRECT=${VLLM_ASCEND_MODE3_BULK_CPU_DIRECT:-1}
# mode=3 dispatch/group optimization:
#   EXPERT_TOKEN_NUMS_TYPE=1 asks MC2 dispatch to return per-expert counts
#   directly, avoiding a per-layer cumulative->counts conversion.
#   ACTIVE_ROWS_SYNC=1 restores active_rows diagnostics but adds host sync.
export VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH=${VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH:-1}
export VLLM_ASCEND_MODE3_EXPERT_TOKEN_NUMS_TYPE=${VLLM_ASCEND_MODE3_EXPERT_TOKEN_NUMS_TYPE:-0}
export VLLM_ASCEND_MODE3_ACTIVE_ROWS_SYNC=${VLLM_ASCEND_MODE3_ACTIVE_ROWS_SYNC:-0}
# mode=3 profile controls:
#   TRANSFER_LOG=0 closes high-frequency binding/prefetch logs.
#   TIMING_LOG=1 emits sampled compute-vs-prefetch breakdown lines.
#   TIMING_SYNC=1 records NPU timing events for accurate sampled timings.
export VLLM_ASCEND_MODE3_TRANSFER_LOG=${VLLM_ASCEND_MODE3_TRANSFER_LOG:-0}
export VLLM_ASCEND_MODE3_TIMING_LOG=${VLLM_ASCEND_MODE3_TIMING_LOG:-1}
export VLLM_ASCEND_MODE3_TIMING_SYNC=${VLLM_ASCEND_MODE3_TIMING_SYNC:-1}
export VLLM_ASCEND_MODE3_TIMING_EVERY=${VLLM_ASCEND_MODE3_TIMING_EVERY:-1024}
export VLLM_ASCEND_MODE3_TIMING_FIRST_N=${VLLM_ASCEND_MODE3_TIMING_FIRST_N:-1}
export VLLM_ASCEND_MODE3_TIMING_LAYERS=${VLLM_ASCEND_MODE3_TIMING_LAYERS:-all}
#控制moe记录是否开启
export VLLM_MOE_PATTERN_STATS=${VLLM_MOE_PATTERN_STATS:-0}  # 1: enable MoE pattern stats collection, 0: disable
export VLLM_MOE_STATS=${VLLM_MOE_PATTERN_STATS}
export VLLM_MOE_STATS_DIR=${VLLM_MOE_STATS_DIR:-./moe_stats}
echo "[moe pattern stats] enabled=${VLLM_MOE_PATTERN_STATS} dir=${VLLM_MOE_STATS_DIR} mode=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE} floor=${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE} hybrid_slots=${VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS} mode3_async_npu_prefetch=${VLLM_ASCEND_MODE3_ASYNC_NPU_PREFETCH} mode3_async_cpu_stage=${VLLM_ASCEND_MODE3_ASYNC_CPU_STAGE} mode3_async_cpu_pack=${VLLM_ASCEND_MODE3_ASYNC_CPU_PACK} mode3_direct_cpu_slot=${VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT} mode3_bulk_npu_copy=${VLLM_ASCEND_MODE3_BULK_NPU_COPY} mode3_bulk_cpu_stage=${VLLM_ASCEND_MODE3_BULK_CPU_STAGE} mode3_bulk_cpu_direct=${VLLM_ASCEND_MODE3_BULK_CPU_DIRECT} mode3_use_fused_experts_path=${VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH} mode3_expert_token_nums_type=${VLLM_ASCEND_MODE3_EXPERT_TOKEN_NUMS_TYPE} mode3_active_rows_sync=${VLLM_ASCEND_MODE3_ACTIVE_ROWS_SYNC} mode3_device_ready_wait=${VLLM_ASCEND_MODE3_DEVICE_READY_WAIT} mode3_transfer_log=${VLLM_ASCEND_MODE3_TRANSFER_LOG} mode3_timing_log=${VLLM_ASCEND_MODE3_TIMING_LOG} mode3_timing_sync=${VLLM_ASCEND_MODE3_TIMING_SYNC} mode3_timing_every=${VLLM_ASCEND_MODE3_TIMING_EVERY} mode3_timing_first_n=${VLLM_ASCEND_MODE3_TIMING_FIRST_N} mode3_timing_layers=${VLLM_ASCEND_MODE3_TIMING_LAYERS}"
#模拟样本缩短规则
# export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=4,8,12,16,20
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896
echo "[elastic tail validate] VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS}"

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

OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/cann-recipes-train/llm_rl/qwen3}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-save4analyse_ingraphmode}
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}"
ROLL_OUT_DIR="${OUTPUT_DIR}/rollout_data"
ROLL_LEN_DIR="${OUTPUT_DIR}/rollout_length"
TB_DIR="${OUTPUT_DIR}/tensorboard"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${ROLL_OUT_DIR}" "${ROLL_LEN_DIR}" "${TB_DIR}" "${LOG_DIR}"

HOME=$(pwd)
MODEL_PATH=${MODEL_PATH:-"/data/Qwen3-30B-A3B"}
CONFIG_DIR=${CONFIG_DIR:-"${HOME}/verl/trainer/config"}
DISTCP_PATH="/data/Qwen3-30B-A3B_megatron"
TRAIN_FILE=${TRAIN_FILE:-"/data/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"/data/deepscaler/test.parquet"}

time=$(date +%Y%m%d%H%M%S)
elastic_suffix=""
if [ "${VLLM_ASCEND_ELASTIC_EXECUTION_MODE}" != "0" ]; then
    elastic_suffix="_elastic"
fi
logfile="${LOG_DIR}/wjqwen30b-a3b-record_graph_save4eagle3_${time}${elastic_suffix}.txt"

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
    +data.dataset_fraction=0.003\
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
    actor_rollout_ref.rollout.max_num_batched_tokens=17408 \
    actor_rollout_ref.rollout.enforce_eager=False \
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
    trainer.total_epochs=3 \
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
    +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True  $@ >> "${logfile}" 
