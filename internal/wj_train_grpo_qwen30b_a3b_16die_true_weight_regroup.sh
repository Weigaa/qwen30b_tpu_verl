# Variant: regroup + rollout eager workaround for MoeDistributeDispatchV2 torchair compile issue
set -ex

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
echo "[pythonpath] staged repo root first: ${REPO_ROOT}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

if [[ "${USE_CANN91_A3_WITH_NNAL91:-0}" == "1" ]]; then
    CANN91_NNAL91_ENV="${CANN91_NNAL91_ENV:-/workspace/cann-recipes-train/llm_rl/qwen3/use_cann91_a3_with_nnal91.sh}"
    if [[ ! -f "${CANN91_NNAL91_ENV}" ]]; then
        echo "[cann91+nnal91] missing env script: ${CANN91_NNAL91_ENV}" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${CANN91_NNAL91_ENV}"
else
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/asdsip/set_env.sh
    # ATB's set_env.sh can return non-zero under `set -e` when it auto-detects
    # torch's CXX ABI and an internal grep misses. Passing the ABI explicitly
    # avoids that false failure and keeps the environment initialization stable.
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1

    # Optional validation path for full CANN 9.1 A3. This changes OPP/runtime too,
    # so it is not suitable for validating an 8.3-compatible HCCL-only fix. Prefer
    # USE_CANN91_A3_WITH_NNAL91=1 for torch_npu 2.9, otherwise ATB remains 8.3.
    if [[ "${USE_HCCL_MASTER_9_1_A3:-0}" == "1" ]]; then
        HCCL_MASTER_A3_ENV="${HCCL_MASTER_A3_ENV:-/workspace/cann-recipes-train/llm_rl/qwen3/use_hccl_master_9_1_a3.sh}"
        if [[ ! -f "${HCCL_MASTER_A3_ENV}" ]]; then
            echo "[hccl master a3] missing env script: ${HCCL_MASTER_A3_ENV}" >&2
            exit 1
        fi
        # shellcheck disable=SC1090
        source "${HCCL_MASTER_A3_ENV}"
    fi
fi

# HCCL-only overlay for vendor-provided fixes that are ABI-compatible with
# CANN 8.3.  Refuse master/9.1 HCCL here because it depends on 9.1 runtime
# libraries such as libhcomm.so and libacl_rt.so, which would no longer keep
# the 8.3 OPP/runtime path intact.
if [[ "${USE_HCCL_8_3_COMPAT_OVERLAY:-0}" == "1" ]]; then
    HCCL_8_3_COMPAT_LIB="${HCCL_8_3_COMPAT_LIB:-}"
    if [[ -z "${HCCL_8_3_COMPAT_LIB}" || ! -f "${HCCL_8_3_COMPAT_LIB}" ]]; then
        echo "[hccl 8.3 overlay] set HCCL_8_3_COMPAT_LIB to an existing libhccl.so" >&2
        exit 1
    fi
    if readelf -d "${HCCL_8_3_COMPAT_LIB}" | grep -Eq 'Shared library: \[(libhcomm|libhccl_compat|libacl_rt)\.so\]'; then
        echo "[hccl 8.3 overlay] rejected non-8.3-compatible lib: ${HCCL_8_3_COMPAT_LIB}" >&2
        echo "[hccl 8.3 overlay] this lib depends on 9.x runtime components; use a CANN 8.3-built HCCL fix instead" >&2
        exit 1
    fi
    HCCL_8_3_COMPAT_DIR="$(cd "$(dirname "${HCCL_8_3_COMPAT_LIB}")" && pwd)"
    export LD_LIBRARY_PATH="${HCCL_8_3_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
    export HCCL_8_3_COMPAT_OVERLAY_ENABLED=1
fi

export HYDRA_FULL_ERROR=1
#export ASCEND_LAUNCH_BLOCKING=1         
export RAY_DEDUP_LOGS=0                   

export ASCEND_GLOBAL_EVENT_ENABLE=0         
export ASCEND_SLOG_PRINT_TO_STDOUT=0       
export ASCEND_GLOBAL_LOG_LEVEL=3             
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-360}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64021}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-360}
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
    default_gloo_ifname=$(
        awk '$2 == "00000000" && $1 != "lo" { print $1; exit }' \
            /proc/net/route 2>/dev/null || true
    )
    if [[ -z "${default_gloo_ifname}" ]]; then
        default_gloo_ifname=$(
            for iface_path in /sys/class/net/*; do
                iface=$(basename "${iface_path}")
                [[ "${iface}" == "lo" ]] && continue
                [[ -r "${iface_path}/operstate" ]] || continue
                [[ "$(cat "${iface_path}/operstate")" == "up" ]] || continue
                echo "${iface}"
                break
            done
        )
    fi
    if [[ -n "${default_gloo_ifname}" ]]; then
        export GLOO_SOCKET_IFNAME="${default_gloo_ifname}"
    fi
fi
echo "[gloo] GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"

export MASTER_PORT=${MASTER_PORT:-23300}    # vllm port error
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
export VLLM_DP_SIZE=${VLLM_DP_SIZE:-16}                        # world_size // rollout.tp_size
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-800}"

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
#关闭看门狗,并控制超时时间
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-1}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-360}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-360}

#控制llm.py里的profiler
export VLLM_ASCEND_LLM_PROFILE_ENABLE=0

# Unified output root for checkpoints / rollout dumps / draft dumps / logs.
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/cann-recipes-train/llm_rl/qwen3_shrink_aware}
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
export VLLM_ASCEND_MODE1_FORCE_NON_MEMBER_PLACEHOLDER_NEW_GROUP=${VLLM_ASCEND_MODE1_FORCE_NON_MEMBER_PLACEHOLDER_NEW_GROUP:-1}
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
    

time=$(date +%Y%m%d%H%M%S)
logfile="${LOG_DIR}/wjqwen30b-a3b-record_graph_save4eagle3_${time}.txt"

: > "${logfile}"
exec > >(tee -a "${logfile}") 2>&1

if [[ "${USE_HCCL_MASTER_9_1_A3:-0}" == "1" || "${USE_HCCL_8_3_COMPAT_OVERLAY:-0}" == "1" ]]; then
    hccl_lib_path=$(python3 - <<'PY'
import ctypes.util
print(ctypes.util.find_library('hccl') or '')
PY
)
    explicit_hccl_lib="${ASCEND_HOME_PATH:-}/lib64/libhccl.so"
    overlay_hccl_lib="${HCCL_8_3_COMPAT_LIB:-}"
    echo "[hccl env] USE_HCCL_MASTER_9_1_A3=${USE_HCCL_MASTER_9_1_A3:-0} USE_HCCL_8_3_COMPAT_OVERLAY=${USE_HCCL_8_3_COMPAT_OVERLAY:-0}"
    echo "[hccl env] ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-} ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-}"
    if [[ -n "${overlay_hccl_lib}" && -f "${overlay_hccl_lib}" ]]; then
        echo "[hccl env] overlay_lib=${overlay_hccl_lib} sha256=$(sha256sum "${overlay_hccl_lib}" | awk '{print $1}')"
    fi
    if [[ -f "${explicit_hccl_lib}" ]]; then
        echo "[hccl env] explicit_lib=${explicit_hccl_lib} sha256=$(sha256sum "${explicit_hccl_lib}" | awk '{print $1}') ctypes_find_library=${hccl_lib_path:-unknown}"
    else
        echo "[hccl env] explicit_lib_missing=${explicit_hccl_lib} ctypes_find_library=${hccl_lib_path:-unknown}"
    fi
    echo "[hccl env] LD_LIBRARY_PATH_HEAD=$(python3 - <<'PY'
import os
print(':'.join(os.environ.get('LD_LIBRARY_PATH', '').split(':')[:10]))
PY
)"
fi

worker_v1_sha256=$(sha256sum "${HOME}/vllm_ascend/worker/worker_v1.py" 2>/dev/null | awk '{print $1}' || true)
llm_engine_sha256=$(sha256sum "${HOME}/vllm/v1/engine/llm_engine.py" 2>/dev/null | awk '{print $1}' || true)
fused_moe_sha256=$(sha256sum "${HOME}/vllm_ascend/ops/fused_moe.py" 2>/dev/null | awk '{print $1}' || true)
ray_trainer_sha256=$(sha256sum "${HOME}/verl/trainer/ppo/ray_trainer.py" 2>/dev/null | awk '{print $1}' || true)
launcher_sha256=$(sha256sum "${BASH_SOURCE[0]}" 2>/dev/null | awk '{print $1}' || true)

echo "[run] start_time=$(date '+%Y-%m-%dT%H:%M:%S%z') logfile=${logfile}"
echo "[mode1 lifecycle] launcher=regroup mode=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE:-} floor=${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE:-} defer_group_destroy=${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY:-} defer_full_restore=${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE:-} defer_pre_rebuild=${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD:-} defer_stash=${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH:-} defer_sizes=${VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES:-} destroy_device_pg=${VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE:-} destroy_cpu_pg=${VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_RETIRE:-} destroy_device_placeholder_pg=${VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_NON_MEMBER_PLACEHOLDER:-} destroy_cpu_placeholder_pg=${VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_NON_MEMBER_PLACEHOLDER:-} force_non_member_placeholder=${VLLM_ASCEND_MODE1_FORCE_NON_MEMBER_PLACEHOLDER_NEW_GROUP:-} allow_rebuild_cache_hit_with_placeholder=${VLLM_ASCEND_MODE1_ALLOW_REBUILD_CACHE_HIT_WITH_PLACEHOLDER:-} group_seq_digest=${VLLM_ASCEND_GROUP_SEQ_DIGEST_LOG:-} hccl_seq_digest=${VLLM_ASCEND_ELASTIC_HCCL_SEQ_DIGEST_LOG:-} hccl_seq_summary=${VLLM_ASCEND_ELASTIC_HCCL_SEQ_SUMMARY_LOG:-} mode1_seq_checkpoint=${VLLM_ASCEND_MODE1_SEQ_CHECKPOINT_LOG:-} moe_dispatch_digest=${VLLM_ASCEND_MOE_DISPATCH_SEQ_DIGEST_LOG:-} moe_dispatch_sample_every=${VLLM_ASCEND_MOE_DISPATCH_SEQ_DIGEST_SAMPLE_EVERY:-} no_resample=${VLLM_EAGER_BASELINE_NO_RESAMPLE:-} length_regroup=${VLLM_EPOCH_LENGTH_REGROUP_ENABLE:-}"
echo "[mode1 lifecycle] post_restore_mc2_warmup=${VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP:-} post_restore_alltoall_warmup=${VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP:-} post_shrink_warmup=${VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP:-} repeat_post_shrink_moe_dispatch_warmup=${VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP:-} post_shrink_release_empty_cache=${VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE:-} post_shrink_release_sync=${VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC:-} release_warmup_cache=${VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE:-} pre_moe_warmup_empty_cache=${VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE:-} force_cleanup_after_floor_group_release=${VLLM_ASCEND_MODE1_PARITY_FORCE_CLEANUP_AFTER_FLOOR_GROUP_RELEASE:-} drop_old_floor_before_rebuild=${VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD:-} mode1_seq_checkpoint=${VLLM_ASCEND_MODE1_SEQ_CHECKPOINT_LOG:-} hccl_seq_digest=${VLLM_ASCEND_ELASTIC_HCCL_SEQ_DIGEST_LOG:-} hccl_seq_summary=${VLLM_ASCEND_ELASTIC_HCCL_SEQ_SUMMARY_LOG:-}"
echo "[mode1 code] worker_v1_sha256=${worker_v1_sha256:-unknown} llm_engine_sha256=${llm_engine_sha256:-unknown} fused_moe_sha256=${fused_moe_sha256:-unknown} ray_trainer_sha256=${ray_trainer_sha256:-unknown} launcher_sha256=${launcher_sha256:-unknown}"

set -x

set +e
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
    data.dataloader_num_workers=0 \
    data.sampler.class_path=${SAMPLER_CLASS_PATH} \
    data.sampler.class_name=${SAMPLER_CLASS_NAME} \
    ${SAMPLER_EXTRA_ARGS} \
    +data.dataset_fraction=0.005\
    custom_reward_function.path=deepscaler.py \
    custom_reward_function.name=compute_score  \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.clip_grad=10000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER} \
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
    +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True  "$@"
run_exit_code=$?
set -e
analysis_file="${logfile%.txt}_analysis.txt"
if [[ "${VLLM_ASCEND_MODE1_AUTO_ANALYZE_LOG:-1}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    set +e
    generated_analysis=$(tools/write_mode1_threshold_analysis.sh "${logfile}" "${analysis_file}" 2>&1)
    analysis_exit_code=$?
    set -e
    if [[ "${analysis_exit_code}" -eq 0 ]]; then
        echo "[mode1 auto-analysis] analysis_file=${generated_analysis}"
        summary_file="${generated_analysis%.txt}_summary.txt"
        if [[ -f "${summary_file}" ]]; then
            echo "[mode1 auto-analysis] summary_file=${summary_file}"
            sed -n '1,80p' "${summary_file}"
        fi
    else
        echo "[mode1 auto-analysis] failed exit_code=${analysis_exit_code} output=${generated_analysis} expected_file=${analysis_file}"
    fi
fi
echo "[run] end_time=$(date '+%Y-%m-%dT%H:%M:%S%z') exit_code=${run_exit_code} logfile=${logfile}"
exit "${run_exit_code}"
