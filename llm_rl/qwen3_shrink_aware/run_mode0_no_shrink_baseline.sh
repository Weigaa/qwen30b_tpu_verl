#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
cd "${script_dir}"

MODEL_PATH="${MODEL_PATH:-/data/Qwen3-30B-A3B}"
DISTCP_PATH="${DISTCP_PATH:-/data/Qwen3-30B-A3B_megatron}"
TRAIN_FILE="${TRAIN_FILE:-/data/deepscaler/train.parquet}"
TEST_FILE="${TEST_FILE:-/data/deepscaler/test.parquet}"

if [[ "${CHECK_LOCAL_INPUTS:-1}" == "1" ]]; then
    for required_path in "$MODEL_PATH" "$DISTCP_PATH" "$TRAIN_FILE" "$TEST_FILE"; do
        if [[ ! -e "$required_path" ]]; then
            echo "missing local input path: $required_path" >&2
            echo "Set MODEL_PATH/DISTCP_PATH/TRAIN_FILE/TEST_FILE, or CHECK_LOCAL_INPUTS=0 to skip checks." >&2
            exit 2
        fi
    done
fi

export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-mode0_no_shrink_baseline}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"${script_dir}"}
export RECORD_DIR=${RECORD_DIR:-"${OUTPUT_ROOT}/${OUTPUT_SUBDIR}"}
export TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-1}
export CONFIG_DIR="${CONFIG_DIR:-${script_dir}/verl/trainer/config}"
export PYTHONPATH="${script_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export HOME="${script_dir}"

# Keep these defaults local to the mode=0 baseline launcher. The internal
# wrapper enforces mode=0/no-shrink again so direct calls stay safe too.
export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=0
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=0
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE:-16}
export VLLM_ASCEND_SHRINK_AWARE_ENABLE=0
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS="${VLLM_ASCEND_REGISTER_CUSTOM_MODELS:-1}"
export VLLM_ASCEND_USE_NATIVE_QWEN3_MOE=0
export VLLM_MCORE_DIST_CKPT_RELAXED_LOAD="${VLLM_MCORE_DIST_CKPT_RELAXED_LOAD:-1}"
export VLLM_EAGER_BASELINE_NO_RESAMPLE=1
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=0
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
export VLLM_ASCEND_ELASTIC_UTIL_LOG=0
export MODE0_BASELINE_SAMPLE_NPU_UTIL="${MODE0_BASELINE_SAMPLE_NPU_UTIL:-0}"
export MODE0_SAVE_ROLLOUT_ARTIFACTS="${MODE0_SAVE_ROLLOUT_ARTIFACTS:-1}"
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-17408}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export DATA_SHUFFLE="${DATA_SHUFFLE:-True}"
export DATASET_FRACTION="${DATASET_FRACTION:-0.005}"

export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-47241}"
export MASTER_PORT="${MASTER_PORT:-26240}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-47241}"
export MODEL_PATH DISTCP_PATH TRAIN_FILE TEST_FILE

echo "[run mode0 baseline] record_dir=${RECORD_DIR} rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "[run mode0 baseline] model=${MODEL_PATH} distcp=${DISTCP_PATH} train=${TRAIN_FILE} test=${TEST_FILE}"
echo "[run mode0 baseline] mode=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE} floor=${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE} custom_models=${VLLM_ASCEND_REGISTER_CUSTOM_MODELS} native_qwen3=${VLLM_ASCEND_USE_NATIVE_QWEN3_MOE} relaxed_ckpt_load=${VLLM_MCORE_DIST_CKPT_RELAXED_LOAD}"
echo "[run mode0 baseline] dataset_fraction=${DATASET_FRACTION} data_shuffle=${DATA_SHUFFLE} moe_stats=${VLLM_MOE_PATTERN_STATS} moe_stats_timing=${VLLM_MOE_STATS_TIMING} npu_sampler=${MODE0_BASELINE_SAMPLE_NPU_UTIL} elastic_util_log=${VLLM_ASCEND_ELASTIC_UTIL_LOG} save_rollout_artifacts=${MODE0_SAVE_ROLLOUT_ARTIFACTS}"

bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager_baseline_util.sh "$@"
