#!/usr/bin/env bash
set -euo pipefail

# Shrink-to-8 redundant expert mode + low-priority Qwen2.5-1.5B sidecar.
# This keeps the same NPU AICore sampling and decode-step bucket markers used
# by the baseline/shrink8 utilization experiments, so the released-card
# utilization can be compared directly.

export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=1
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=8

export VERL_SIDECAR_ENABLE=1
export VERL_SIDECAR_MODEL_PATH=${VERL_SIDECAR_MODEL_PATH:-"/home/data/Qwen2.5-1.5B-Instruct"}
export VERL_SIDECAR_PROMPTS_FILE=${VERL_SIDECAR_PROMPTS_FILE:-"/home/qiuzy/verl_dev/data/gsm8k"}
export VERL_SIDECAR_DATA_SPLIT=${VERL_SIDECAR_DATA_SPLIT:-train}
export VERL_SIDECAR_STATE_DIR=${VERL_SIDECAR_STATE_DIR:-"sidecar_runs/state/qwen25_15b_gsm8k_train"}
export VERL_SIDECAR_PARALLEL_MODE=${VERL_SIDECAR_PARALLEL_MODE:-dp}

# Keep the sidecar small-model settings explicit for this utilization run.
# The watcher will infer VERL_SIDECAR_NPU_DEVICES from the inactive ranks after
# shrink-to-8; do not hard-code physical NPU ids here.
export VERL_SIDECAR_MAX_MODEL_LEN=${VERL_SIDECAR_MAX_MODEL_LEN:-2048}
export VERL_SIDECAR_MAX_TOKENS=${VERL_SIDECAR_MAX_TOKENS:-1024}
export VERL_SIDECAR_MAX_NUM_SEQS=${VERL_SIDECAR_MAX_NUM_SEQS:-128}
export VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=${VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS:-65536}
export VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE=${VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE:-128}
export VERL_SIDECAR_MAX_PROMPTS=${VERL_SIDECAR_MAX_PROMPTS:-1024}
export VERL_SIDECAR_GENERATE_CHUNK_SIZE=${VERL_SIDECAR_GENERATE_CHUNK_SIZE:-32}
export VERL_SIDECAR_REPEAT_UNTIL_KILLED=${VERL_SIDECAR_REPEAT_UNTIL_KILLED:-1}

export VLLM_ASCEND_ELASTIC_UTIL_LOG=1
export VLLM_ASCEND_ELASTIC_UTIL_BUCKET_STEPS=${VLLM_ASCEND_ELASTIC_UTIL_BUCKET_STEPS:-500}
export VLLM_ASCEND_DUMMY_WASTE_TIMING=0
export VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS=0

run_id=$(date +%Y%m%d%H%M%S)
export WJ_RECORDS_DIR=${WJ_RECORDS_DIR:-/home/sharedata/wj_records}
export VLLM_ASCEND_NPU_UTIL_CSV=${VLLM_ASCEND_NPU_UTIL_CSV:-"${WJ_RECORDS_DIR}/npu_util/${run_id}_shrink8_side_qwen25_15b.csv"}
export VLLM_ASCEND_NPU_UTIL_SAMPLE_INTERVAL=${VLLM_ASCEND_NPU_UTIL_SAMPLE_INTERVAL:-1.0}
mkdir -p "$(dirname "${VLLM_ASCEND_NPU_UTIL_CSV}")"

python3 "$(dirname "$0")/sample_npu_util.py" \
    --output "${VLLM_ASCEND_NPU_UTIL_CSV}" \
    --interval "${VLLM_ASCEND_NPU_UTIL_SAMPLE_INTERVAL}" &
sampler_pid=$!

cleanup_sampler() {
    if kill -0 "${sampler_pid}" 2>/dev/null; then
        kill "${sampler_pid}" 2>/dev/null || true
        wait "${sampler_pid}" 2>/dev/null || true
    fi
    echo "[npu util sampler] csv=${VLLM_ASCEND_NPU_UTIL_CSV}"
}
trap cleanup_sampler EXIT

bash "$(dirname "$0")/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh" "$@"
