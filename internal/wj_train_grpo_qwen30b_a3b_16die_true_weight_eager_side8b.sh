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
export VLLM_DP_SIZE=16                        # world_size // rollout.tp_sizebaseline dummy-run, 1: redundant experts only, 2: redundant experts + CPU/NPU hybrid tail
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

# Train Drafter stays disabled in this sidecar experiment.
export VLLM_ASCEND_ENABLE_DRAFT_TRAIN=0
DRAFT_PROFILE_MODE=${DRAFT_PROFILE_MODE:-breakdown}
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
# Keep DP/EP cached across restore, but do not cache MC2. Keeping old
# MC2 communicators alive makes the next shrink rebuild fall into a very
# slow HCCL path on Ascend in this workload.
export VLLM_ASCEND_ELASTIC_CACHE_MC2_GROUPS=0
# Match the 05-01 fast path: keep only the target-rank DP/EP cache and drop
# stale groups after each transition. Retaining all stale communicators makes
# later MC2 new_group calls fall into an ~80s HCCL slow path.
export VLLM_ASCEND_ELASTIC_KEEP_GROUP_CACHE=0
export VLLM_ASCEND_ELASTIC_GROUP_STAGE_BARRIER=${VLLM_ASCEND_ELASTIC_GROUP_STAGE_BARRIER:-0}
export VERL_RAY_DISABLE_HEAD_DASHBOARD=${VERL_RAY_DISABLE_HEAD_DASHBOARD:-1}
export VERL_RAY_DISABLE_DASHBOARD_METRICS=${VERL_RAY_DISABLE_DASHBOARD_METRICS:-1}
export VERL_RAY_DASHBOARD_AGENT_MINIMAL=${VERL_RAY_DASHBOARD_AGENT_MINIMAL:-0}
export VERL_RAY_DISABLE_API_SERVER=${VERL_RAY_DISABLE_API_SERVER:-1}
export RAY_enable_open_telemetry=${RAY_enable_open_telemetry:-0}
export RAY_USAGE_STATS_ENABLED=${RAY_USAGE_STATS_ENABLED:-0}
export VERL_LOG_FULL_CONFIG=${VERL_LOG_FULL_CONFIG:-0}
export VERL_LOG_FULL_TRANSFORMER_CONFIG=${VERL_LOG_FULL_TRANSFORMER_CONFIG:-0}
export VLLM_ENGINE_CORE_INIT_STACK_LOG=${VLLM_ENGINE_CORE_INIT_STACK_LOG:-0}
export VLLM_ENGINE_CORE_DEBUG_LOG=${VLLM_ENGINE_CORE_DEBUG_LOG:-0}
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
# mode=3 experimental: reuse each layer's resident prefix weight buffer as the
# runtime buffer when resident NPU experts already occupy the required dense
# prefix slots. Keep disabled by default to preserve the strict two-runtime-
# buffer execution model.
export VLLM_ASCEND_MODE3_LAYER_LOCAL_BUFFER=${VLLM_ASCEND_MODE3_LAYER_LOCAL_BUFFER:-0}
# mode=3 dispatch/group optimization:
#   EXPERT_TOKEN_NUMS_TYPE=1 asks MC2 dispatch to return per-expert counts
#   directly, avoiding a per-layer cumulative->counts conversion.
#   ACTIVE_ROWS_SYNC=1 restores active_rows diagnostics but adds host sync.
export VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH=${VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH:-1}
export VLLM_ASCEND_MODE3_EXPERT_TOKEN_NUMS_TYPE=${VLLM_ASCEND_MODE3_EXPERT_TOKEN_NUMS_TYPE:-0}
export VLLM_ASCEND_MODE3_ACTIVE_ROWS_SYNC=${VLLM_ASCEND_MODE3_ACTIVE_ROWS_SYNC:-0}
# mode=3 profile controls:
#   TRANSFER_LOG=0 closes high-frequency binding/prefetch logs.
#   TIMING_LOG/TIMING_SYNC default off for performance runs. Override them to
#   1 only when collecting compute-vs-prefetch diagnostics.
export VLLM_ASCEND_MODE3_TRANSFER_LOG=${VLLM_ASCEND_MODE3_TRANSFER_LOG:-0}
export VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG=${VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG:-0}
export VLLM_ASCEND_MODE3_TRANSFER_PLAN_FIRST_N=${VLLM_ASCEND_MODE3_TRANSFER_PLAN_FIRST_N:-4}
export VLLM_ASCEND_MODE3_TIMING_LOG=${VLLM_ASCEND_MODE3_TIMING_LOG:-1}
export VLLM_ASCEND_MODE3_TIMING_SYNC=${VLLM_ASCEND_MODE3_TIMING_SYNC:-1}
export VLLM_ASCEND_MODE3_TIMING_EVERY=${VLLM_ASCEND_MODE3_TIMING_EVERY:-1024}
export VLLM_ASCEND_MODE3_TIMING_FIRST_N=${VLLM_ASCEND_MODE3_TIMING_FIRST_N:-1}
export VLLM_ASCEND_MODE3_TIMING_LAYERS=${VLLM_ASCEND_MODE3_TIMING_LAYERS:-all}
#控制moe记录是否开启
export VLLM_MOE_PATTERN_STATS=${VLLM_MOE_PATTERN_STATS:-0}  # 1: enable MoE pattern stats collection, 0: disable
export VLLM_MOE_STATS=${VLLM_MOE_PATTERN_STATS}
export VLLM_MOE_STATS_DIR=${VLLM_MOE_STATS_DIR:-./moe_stats}
echo "[moe pattern stats] enabled=${VLLM_MOE_PATTERN_STATS} dir=${VLLM_MOE_STATS_DIR} mode=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE} floor=${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE} hybrid_slots=${VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS} mode3_async_npu_prefetch=${VLLM_ASCEND_MODE3_ASYNC_NPU_PREFETCH} mode3_async_cpu_stage=${VLLM_ASCEND_MODE3_ASYNC_CPU_STAGE} mode3_async_cpu_pack=${VLLM_ASCEND_MODE3_ASYNC_CPU_PACK} mode3_direct_cpu_slot=${VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT} mode3_bulk_npu_copy=${VLLM_ASCEND_MODE3_BULK_NPU_COPY} mode3_bulk_cpu_stage=${VLLM_ASCEND_MODE3_BULK_CPU_STAGE} mode3_bulk_cpu_direct=${VLLM_ASCEND_MODE3_BULK_CPU_DIRECT} mode3_layer_local_buffer=${VLLM_ASCEND_MODE3_LAYER_LOCAL_BUFFER} mode3_use_fused_experts_path=${VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH} mode3_expert_token_nums_type=${VLLM_ASCEND_MODE3_EXPERT_TOKEN_NUMS_TYPE} mode3_active_rows_sync=${VLLM_ASCEND_MODE3_ACTIVE_ROWS_SYNC} mode3_device_ready_wait=${VLLM_ASCEND_MODE3_DEVICE_READY_WAIT} mode3_transfer_log=${VLLM_ASCEND_MODE3_TRANSFER_LOG} mode3_transfer_plan_log=${VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG} mode3_transfer_plan_first_n=${VLLM_ASCEND_MODE3_TRANSFER_PLAN_FIRST_N} mode3_timing_log=${VLLM_ASCEND_MODE3_TIMING_LOG} mode3_timing_sync=${VLLM_ASCEND_MODE3_TIMING_SYNC} mode3_timing_every=${VLLM_ASCEND_MODE3_TIMING_EVERY} mode3_timing_first_n=${VLLM_ASCEND_MODE3_TIMING_FIRST_N} mode3_timing_layers=${VLLM_ASCEND_MODE3_TIMING_LAYERS} elastic_cache_mc2=${VLLM_ASCEND_ELASTIC_CACHE_MC2_GROUPS} elastic_keep_group_cache=${VLLM_ASCEND_ELASTIC_KEEP_GROUP_CACHE} elastic_group_stage_barrier=${VLLM_ASCEND_ELASTIC_GROUP_STAGE_BARRIER}"
#模拟样本缩短规则
# export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=4,8,12,16,20
# export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896
echo "[elastic tail validate] VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS}"


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
if [ "${VLLM_ASCEND_ELASTIC_EXECUTION_MODE}" != "0" ]; then
    elastic_suffix="_elastic"
fi
logfile="${HOME}/wjeagerqwen30b-a3b-with_draft_${DRAFT_PROFILE_MODE}_${time}${elastic_suffix}.txt"

export VERL_SIDECAR_ENABLE=${VERL_SIDECAR_ENABLE:-1}
export VERL_SIDECAR_MODEL_PATH=${VERL_SIDECAR_MODEL_PATH:-"/home/data/Qwen3-8B"}
export VERL_SIDECAR_PROMPTS_FILE=${VERL_SIDECAR_PROMPTS_FILE:-"/home/qiuzy/verl_dev/data/gsm8k"}
# Only keep non-default sidecar knobs here. Defaults live in
# internal/run_elastic_sidecar_infer.sh.
export VERL_SIDECAR_MAX_NUM_SEQS=${VERL_SIDECAR_MAX_NUM_SEQS:-16}
export VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=${VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS:-32768}
export VERL_SIDECAR_MAX_MODEL_LEN=${VERL_SIDECAR_MAX_MODEL_LEN:-9216}
export VERL_SIDECAR_MAX_TOKENS=${VERL_SIDECAR_MAX_TOKENS:-8192}
export VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA=${VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA:-16}
export VERL_SIDECAR_GENERATE_CHUNK_SIZE=${VERL_SIDECAR_GENERATE_CHUNK_SIZE:-16}
export VERL_SIDECAR_STATE_DIR=${VERL_SIDECAR_STATE_DIR:-"sidecar_runs/state/qwen3_8b_gsm8k_train"}
export VERL_SIDECAR_PARALLEL_MODE=${VERL_SIDECAR_PARALLEL_MODE:-hybrid}
export VERL_SIDECAR_TENSOR_PARALLEL_SIZE=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE:-4}
export VERL_SIDECAR_REPLICA_COUNT=${VERL_SIDECAR_REPLICA_COUNT:-2}
export VERL_SIDECAR_ENABLE_EXPERT_PARALLEL=${VERL_SIDECAR_ENABLE_EXPERT_PARALLEL:-0}
export VERL_SIDECAR_LOG_DIR=${VERL_SIDECAR_LOG_DIR:-"sidecar_runs/${time}"}
sidecar_monitor_pid=""
cleanup_sidecar_monitor() {
    if [[ -n "${sidecar_monitor_pid}" ]] && kill -0 "${sidecar_monitor_pid}" 2>/dev/null; then
        kill "${sidecar_monitor_pid}" 2>/dev/null || true
        wait "${sidecar_monitor_pid}" 2>/dev/null || true
    fi
}

if [[ "${VERL_SIDECAR_ENABLE}" == "1" ]]; then
    mkdir -p "${VERL_SIDECAR_LOG_DIR}"
    : > "${logfile}"
    export VERL_SIDECAR_TRAIN_LOG="${logfile}"
    export VERL_SIDECAR_LEASE_LOG=${VERL_SIDECAR_LEASE_LOG:-"${VERL_SIDECAR_LOG_DIR}/lease.log"}
    export VERL_SIDECAR_LOG_FILE=${VERL_SIDECAR_LOG_FILE:-"${VERL_SIDECAR_LOG_DIR}/infer.log"}
    export VERL_SIDECAR_OUTPUT_FILE=${VERL_SIDECAR_OUTPUT_FILE:-"${VERL_SIDECAR_LOG_DIR}/outputs.jsonl"}
    sidecar_monitor_log=${VERL_SIDECAR_MONITOR_LOG:-"${VERL_SIDECAR_LOG_DIR}/monitor.log"}
    echo "[elastic sidecar] enabled=1 train_log=${VERL_SIDECAR_TRAIN_LOG} log_dir=${VERL_SIDECAR_LOG_DIR} lease_log=${VERL_SIDECAR_LEASE_LOG} sidecar_log=${VERL_SIDECAR_LOG_FILE} sidecar_output=${VERL_SIDECAR_OUTPUT_FILE} monitor_log=${sidecar_monitor_log} devices=${VERL_SIDECAR_NPU_DEVICES:-auto_from_inactive_ranks} parallel_mode=${VERL_SIDECAR_PARALLEL_MODE} tensor_parallel_size=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE} replica_count=${VERL_SIDECAR_REPLICA_COUNT} sidecar_ep=${VERL_SIDECAR_ENABLE_EXPERT_PARALLEL} model=${VERL_SIDECAR_MODEL_PATH} prompts=${VERL_SIDECAR_PROMPTS_FILE} state_dir=${VERL_SIDECAR_STATE_DIR} max_prompts_per_replica=${VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA} generate_chunk_size=${VERL_SIDECAR_GENERATE_CHUNK_SIZE} max_num_seqs=${VERL_SIDECAR_MAX_NUM_SEQS} max_num_batched_tokens=${VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS} max_model_len=${VERL_SIDECAR_MAX_MODEL_LEN} max_tokens=${VERL_SIDECAR_MAX_TOKENS}"
    internal/watch_elastic_shrink_and_run_sidecar.sh "${logfile}" >> "${sidecar_monitor_log}" 2>&1 &
    sidecar_monitor_pid=$!
    trap cleanup_sidecar_monitor EXIT
else
    echo "[elastic sidecar] enabled=0"
fi

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
