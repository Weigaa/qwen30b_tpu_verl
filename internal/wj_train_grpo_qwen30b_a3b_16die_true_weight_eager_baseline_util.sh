#!/usr/bin/env bash
set -euo pipefail

# Baseline: no elastic shrink, no sidecar. Emits lightweight decode-step
# markers and samples real per-NPU AICore utilization with npu-smi.
export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=0
export VERL_SIDECAR_ENABLE=0
export VLLM_ASCEND_ELASTIC_UTIL_LOG=1
export VLLM_ASCEND_ELASTIC_UTIL_BUCKET_STEPS=${VLLM_ASCEND_ELASTIC_UTIL_BUCKET_STEPS:-500}
export VLLM_ASCEND_DUMMY_WASTE_TIMING=0
export VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS=0

run_id=$(date +%Y%m%d%H%M%S)
export WJ_RECORDS_DIR=${WJ_RECORDS_DIR:-/home/sharedata/wj_records}
export VLLM_ASCEND_NPU_UTIL_CSV=${VLLM_ASCEND_NPU_UTIL_CSV:-"${WJ_RECORDS_DIR}/npu_util/${run_id}_baseline.csv"}
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
