#!/usr/bin/env bash
set -euo pipefail

# Baseline: full-world mode=0, no elastic shrink, no sidecar. Emits
# lightweight decode-step markers and samples real per-NPU AICore utilization
# with npu-smi.
script_dir=$(cd "$(dirname "$0")" && pwd)
work_dir=$(cd "${script_dir}/.." && pwd)

export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=0
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=0
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE:-16}
export VLLM_ASCEND_SHRINK_AWARE_ENABLE=0
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=${VLLM_ASCEND_REGISTER_CUSTOM_MODELS:-1}
export VLLM_ASCEND_USE_NATIVE_QWEN3_MOE=0
export VLLM_MCORE_DIST_CKPT_RELAXED_LOAD=${VLLM_MCORE_DIST_CKPT_RELAXED_LOAD:-1}
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=0
export VLLM_EAGER_BASELINE_NO_RESAMPLE=1
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=0
export VERL_SIDECAR_ENABLE=0
export VLLM_MOE_PATTERN_STATS=0
export VLLM_MOE_STATS=0
export VLLM_MOE_STATS_TIMING=0
export VLLM_ASCEND_NATIVE_MOE_TOPK_DEBUG=0
export SAVE_CKPT_ENABLE=0
export SAVE_DRAFT_HIDDEN_ENABLE=0
export VLLM_ASCEND_ENABLE_DRAFT_TRAIN=0
export VLLM_ASCEND_DRAFT_DUMP_ENABLE=0
export VLLM_ASCEND_ELASTIC_UTIL_LOG=${VLLM_ASCEND_ELASTIC_UTIL_LOG:-0}
export VLLM_ASCEND_ELASTIC_UTIL_BUCKET_STEPS=${VLLM_ASCEND_ELASTIC_UTIL_BUCKET_STEPS:-500}
export VLLM_ASCEND_DUMMY_WASTE_TIMING=0
export VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS=0
export OUTPUT_ROOT=${OUTPUT_ROOT:-"${work_dir}"}
export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-mode0_no_shrink_baseline}
export RECORD_DIR=${RECORD_DIR:-"${work_dir}/mode0_no_shrink_baseline"}

export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}

run_id=$(date +%Y%m%d%H%M%S)
export WJ_RECORDS_DIR=${WJ_RECORDS_DIR:-/home/sharedata/wj_records}
export VLLM_ASCEND_NPU_UTIL_CSV=${VLLM_ASCEND_NPU_UTIL_CSV:-"${WJ_RECORDS_DIR}/npu_util/${run_id}_baseline.csv"}
export VLLM_ASCEND_NPU_UTIL_SAMPLE_INTERVAL=${VLLM_ASCEND_NPU_UTIL_SAMPLE_INTERVAL:-1.0}
sampler_pid=""
cleanup_sampler() {
    if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
        kill "${sampler_pid}" 2>/dev/null || true
        wait "${sampler_pid}" 2>/dev/null || true
        echo "[npu util sampler] csv=${VLLM_ASCEND_NPU_UTIL_CSV}"
    fi
}
if [[ "${MODE0_BASELINE_SAMPLE_NPU_UTIL:-0}" == "1" ]]; then
    mkdir -p "$(dirname "${VLLM_ASCEND_NPU_UTIL_CSV}")"
    python3 "$(dirname "$0")/sample_npu_util.py" \
        --output "${VLLM_ASCEND_NPU_UTIL_CSV}" \
        --interval "${VLLM_ASCEND_NPU_UTIL_SAMPLE_INTERVAL}" &
    sampler_pid=$!
    trap cleanup_sampler EXIT
else
    echo "[npu util sampler] disabled; set MODE0_BASELINE_SAMPLE_NPU_UTIL=1 to enable"
fi

echo "[mode0 baseline wrapper] mode=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE} floor=${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE} shrink=${VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK} sidecar=${VERL_SIDECAR_ENABLE} custom_models=${VLLM_ASCEND_REGISTER_CUSTOM_MODELS} native_qwen3=${VLLM_ASCEND_USE_NATIVE_QWEN3_MOE} relaxed_ckpt_load=${VLLM_MCORE_DIST_CKPT_RELAXED_LOAD} rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} dataset_fraction=${DATASET_FRACTION:-0.005} data_shuffle=${DATA_SHUFFLE:-True} moe_stats=${VLLM_MOE_PATTERN_STATS} moe_stats_timing=${VLLM_MOE_STATS_TIMING} elastic_util_log=${VLLM_ASCEND_ELASTIC_UTIL_LOG} save_rollout_artifacts=${MODE0_SAVE_ROLLOUT_ARTIFACTS:-0} record_dir=${RECORD_DIR}"

cd "${work_dir}"
baseline_launcher=${BASELINE_LAUNCHER:-"${script_dir}/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"}
extra_overrides=()
if [[ "${MODE0_SAVE_ROLLOUT_ARTIFACTS:-1}" != "1" ]]; then
    extra_overrides+=("trainer.rollout_data_dir=")
    extra_overrides+=("trainer.rollout_length_dir=")
fi
if [[ -n "${TRAINER_LOGGER:-}" ]]; then
    extra_overrides+=("trainer.logger=${TRAINER_LOGGER}")
fi
bash "${baseline_launcher}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-1024}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-16384}" \
    data.shuffle="${DATA_SHUFFLE:-True}" \
    data.dataset_fraction="${DATASET_FRACTION:-0.005}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-17408}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-32}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-16}" \
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER:-True}" \
    trainer.total_epochs="${TRAINER_TOTAL_EPOCHS:-1}" \
    "${extra_overrides[@]}" \
    "$@"
