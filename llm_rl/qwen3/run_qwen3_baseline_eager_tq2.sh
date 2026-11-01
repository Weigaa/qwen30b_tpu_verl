#!/usr/bin/env bash
# Qwen3-30B-A3B baseline: eager, TQ2, 16 NPUs, 32 prompts x 16 samples.
set -eo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"

source "${ROOT}/tools/activate_cann85_a3_torch29.sh"

# The root filesystem is intentionally kept out of Ray's spill/session path.
# It is nearly full on this host, while /home/sharedata has ample capacity.
export RAY_TMPDIR=${RAY_TMPDIR:-/home/sharedata/qwen3_ray_tmp}
export TMPDIR=${TMPDIR:-${RAY_TMPDIR}/tmp}
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"

# Do not inherit rollout experiment knobs from the caller's shell. Set options
# that affect plugin/native loading before the runtime audit imports anything.
while IFS= read -r name; do
  case "${name}" in
    VLLM_* | VERL_*) unset "${name}" ;;
  esac
done < <(compgen -e)

# CANN 8.5 HCCL cannot build a multi-NPU communicator when Ray restricts each
# actor to a single logical device through ASCEND_RT_VISIBLE_DEVICES. Keep all
# local NPUs visible and let VERL bind each actor to the physical ID assigned
# by Ray. This is required before ray.init() starts worker processes.
unset ASCEND_RT_VISIBLE_DEVICES NPU_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_OP_EXPANSION_MODE=HOST

# Match the remote 014_baseline execution contract. In particular, keep TQ2
# paired with manual cache-engine release below; changing either setting makes
# the local result no longer comparable with the validated remote run.
export TASK_QUEUE_ENABLE=2
export VLLM_USE_V1=1
export VLLM_ENABLE_GRAPH_MODE=0
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=0
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=0

# A diagnostic override must never make the training launcher bypass its
# driver compatibility gate.
unset QWEN3_ALLOW_UNVALIDATED_DRIVER
python3 "${ROOT}/tools/check_cann85_a3_torch29_runtime.py"

if [ "${1:-}" != "dry-run" ] && [ "${QWEN3_ALLOW_BUSY_NPU:-0}" != 1 ]; then
  if ! NPU_SMI_OUTPUT=$(npu-smi info 2>&1); then
    printf '[qwen3 preflight] npu-smi failed; refusing to start:\n%s\n' \
      "${NPU_SMI_OUTPUT}" >&2
    exit 1
  fi
  NPU_BUSY_LINES=$(printf '%s\n' "${NPU_SMI_OUTPUT}" | awk '
    /\| NPU[[:space:]]+Chip[[:space:]]+\| Process id/ {
      in_process_table = 1
      next
    }
    in_process_table &&
      /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]*[0-9]+/ {
      print
    }
  ')
  if [ -n "${NPU_BUSY_LINES}" ]; then
    printf '[qwen3 preflight] NPU processes already exist; refusing to start:\n%s\n' \
      "${NPU_BUSY_LINES}" >&2
    printf '[qwen3 preflight] set QWEN3_ALLOW_BUSY_NPU=1 only when sharing is intentional\n' >&2
    exit 1
  fi
fi
unset NPU_SMI_OUTPUT NPU_BUSY_LINES

# Imports alone cannot detect a broken HCCL AICPU kernel. Exercise the same
# 16-rank collective families used by checkpoint loading, Megatron and EP, and
# require communicator teardown to finish before starting the expensive job.
if [ "${1:-}" != "dry-run" ] && [ "${QWEN3_SKIP_HCCL_PREFLIGHT:-0}" != 1 ]; then
  bash "${ROOT}/tools/run_hccl_collective_smoke.sh"
fi

# The vendored v0.14 path supports eager compilation with Torch 2.9.
TORCH_VERSION=$(
  python3 -c 'import torch; print("TORCH_VERSION=" + torch.__version__.split("+")[0])' \
    | sed -n 's/^TORCH_VERSION=//p' \
    | tail -n 1
)
if [ -z "${TORCH_VERSION}" ]; then
  printf '[qwen3 runtime] failed to detect torch version\n' >&2
  exit 1
fi
if python3 - "${TORCH_VERSION}" <<'PY'
import sys

major, minor = (int(part) for part in sys.argv[1].split(".")[:2])
raise SystemExit(0 if (major, minor) >= (2, 9) else 1)
PY
then
  export VLLM_ASCEND_EAGER_COMPILE=1
else
  export VLLM_ASCEND_EAGER_COMPILE=0
fi
printf '[qwen3 runtime] torch=%s eager_compile=%s\n' \
  "${TORCH_VERSION}" "${VLLM_ASCEND_EAGER_COMPILE}"
export VLLM_ENABLE_EXPERT_PARALLEL=1
export VLLM_DP_SIZE=16
export VLLM_ENABLE_MC2=1
export ALL_TO_ALL_RESHARD=1
export USE_ALLTOALL_OVERLAP=1

# Fast validated v0.14 eager backend settings.
export VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1
export VLLM_ASCEND_EAGER_COMPILE_PASS_FUSION=0
export VLLM_ASCEND_USE_LEGACY_FUSED_MOE=0
export VLLM_ASCEND_USE_LEGACY_ATTENTION=0
export VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE=0
export VLLM_ASCEND_MC2_TOKENS_CAPACITY=512
export VLLM_ASCEND_MC2_GLOBAL_BS=0
export VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=1
export VLLM_ASCEND_ATTENTION_BLOCK_SIZE=64
export VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=1
export VLLM_QWEN3_MOE_REDUCE_RESULTS=1
export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=1
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
export VLLM_ROLLOUT_SLEEP_LEVEL=2

# Disable the trainer's nonstandard response cap.
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=0

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:24"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-50021}
export HCCL_BUFFSIZE=800
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export MASTER_PORT=${MASTER_PORT:-23300}
export D2D_DATA_TRANSFER=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ASCEND_ENABLE_NZ=0

MODEL_PATH=${MODEL_PATH:-/home/data/Qwen3-30B-A3B}
DISTCP_PATH=${DISTCP_PATH:-/home/data/Qwen3-30B-A3B_megatron}
TRAIN_FILE=${TRAIN_FILE:-/workspace/data/deepscaler/train.parquet}
TEST_FILE=${TEST_FILE:-/workspace/data/deepscaler/test.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/cann-recipes-train/llm_rl/qwen3_014_baseline/llm_rl/qwen3/outputs/qwen3_baseline_results/eager_tq2_bs32_n16_no_async_sche}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
DATASET_FRACTION=${DATASET_FRACTION:-0.005}

ARGS=(
  --config-path="${ROOT}/verl/trainer/config"
  --config-name=ppo_megatron_trainer.yaml
  algorithm.adv_estimator=grpo
  data.train_files="${TRAIN_FILE}"
  data.val_files="${TEST_FILE}"
  data.train_batch_size=32
  data.max_prompt_length=1024
  data.max_response_length=16384
  data.filter_overlong_prompts=True
  data.truncation=error
  data.shuffle=False
  data.dataloader_num_workers=0
  +data.dataset_fraction="${DATASET_FRACTION}"
  custom_reward_function.path="${ROOT}/deepscaler.py"
  custom_reward_function.name=compute_score
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.use_fused_kernels=False
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.clip_grad=10000
  actor_rollout_ref.actor.ppo_mini_batch_size=32
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480
  actor_rollout_ref.actor.megatron.sequence_parallel=True
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4
  actor_rollout_ref.actor.megatron.context_parallel_size=1
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=4
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.actor.megatron.param_offload=True
  actor_rollout_ref.actor.megatron.grad_offload=True
  actor_rollout_ref.actor.megatron.optimizer_offload=False
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
  actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DISTCP_PATH}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.load_weight=True
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  # The shared rollout YAML also serves AdaFloor runners. Remove its extension
  # entirely so this upstream RolloutConfig remains a pure baseline config.
  ~actor_rollout_ref.rollout.shrink_aware
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.gpu_memory_utilization=0.83
  actor_rollout_ref.rollout.max_num_batched_tokens=17408
  actor_rollout_ref.rollout.max_num_seqs=32
  actor_rollout_ref.rollout.async_scheduling=false
  actor_rollout_ref.rollout.enable_prefix_caching=false
  actor_rollout_ref.rollout.enable_chunked_prefill=true
  actor_rollout_ref.rollout.n=16
  actor_rollout_ref.rollout.temperature=0.9
  actor_rollout_ref.rollout.top_k=50
  actor_rollout_ref.rollout.top_p=0.9
  actor_rollout_ref.rollout.ignore_eos=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=4
  actor_rollout_ref.ref.megatron.context_parallel_size=1
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=4
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.ref.megatron.param_offload=True
  actor_rollout_ref.ref.load_weight=True
  actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
  actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DISTCP_PATH}"
  algorithm.kl_ctrl.kl_coef=0.001
  trainer.balance_batch=False
  trainer.device=npu
  trainer.val_before_train=False
  trainer.critic_warmup=0
  trainer.logger="[console,tensorboard]"
  trainer.project_name=qwen3_baseline
  trainer.experiment_name=eager_tq2_bs32_n16
  trainer.n_gpus_per_node=16
  trainer.nnodes=1
  trainer.save_freq=-1
  trainer.test_freq=-1
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.resume_mode=disable
  trainer.default_local_dir="${OUTPUT_DIR}"
  +trainer.rollout_data_dir="${OUTPUT_DIR}/rollout_data"
  +trainer.rollout_length_dir="${OUTPUT_DIR}/rollout_length"
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_num_transformer_layers="[[11],[13],[13],[11]]"
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_alltoall_overlap_comm=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=2048
  +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11
  +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=11
  +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True
)

if [ "${1:-}" = "dry-run" ]; then
  printf 'TASK_QUEUE_ENABLE=%s\nVLLM_ENABLE_GRAPH_MODE=%s\n' \
    "${TASK_QUEUE_ENABLE}" "${VLLM_ENABLE_GRAPH_MODE}"
  printf '%q ' python3 -m verl.trainer.main_ppo "${ARGS[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/rollout_data" "${OUTPUT_DIR}/rollout_length"
export TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
LOG_FILE="${OUTPUT_DIR}/logs/baseline_$(date -u +%Y%m%dT%H%M%SZ).log"
python3 -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
