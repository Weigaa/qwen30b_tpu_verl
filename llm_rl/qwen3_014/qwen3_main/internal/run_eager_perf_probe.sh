#!/usr/bin/env bash
# Cheap, repeatable eager rollout validation wrapper.
#
# Usage:
#   bash internal/run_eager_perf_probe.sh [gen1|profile|layerprofile|npu_profile|threshold3|full16k]
#
# Modes:
#   gen1       One threshold-control generate-only step. Fastest reliable A/B.
#   profile    Same workload as gen1, with runner/MoE/attention debug timing.
#   layerprofile
#              One threshold-control generate-only step with Python layer
#              timers enabled and eager compile disabled.  This is for
#              bottleneck localization only, not throughput comparison.
#   npu_profile
#              One threshold-control generate-only step with torch_npu profiler
#              around vLLM generate on one rollout rank.
#   threshold3 Three threshold-control generate-only steps. Use after gen1 wins.
#   full16k    One full 16k rollout, aborting after generation. Use sparingly.
#
# The wrapper intentionally delegates all production configuration to
# wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_eager_fast.sh so A/B
# probes only vary the single hypothesis passed in by the caller.
set -euo pipefail

MODE=${1:-gen1}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
QWEN3_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
EAGER_FAST="${SCRIPT_DIR}/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_eager_fast.sh"
PROFILE_TOOL="${QWEN3_DIR}/tools/profile_rollout_logs.py"
STAMP=$(date +%Y%m%d%H%M%S)

cd "${QWEN3_DIR}"

export TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-1}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-256,512,640,768,896}
export VLLM_ROLLOUT_FAST_DEBUG=${VLLM_ROLLOUT_FAST_DEBUG:-1}
export VLLM_ROLLOUT_DEBUG_GENERATION=${VLLM_ROLLOUT_DEBUG_GENERATION:-1}
export VLLM_ROLLOUT_DEBUG_GENERATION_SAMPLES=${VLLM_ROLLOUT_DEBUG_GENERATION_SAMPLES:-8}
export VLLM_ROLLOUT_DEBUG_GROUPS=${VLLM_ROLLOUT_DEBUG_GROUPS:-8}

case "${MODE}" in
    gen1)
        export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_eager_probe_gen1_${STAMP}}
        export TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS:-1}
        export VLLM_ROLLOUT_DEBUG_GENERATE_ONLY=${VLLM_ROLLOUT_DEBUG_GENERATE_ONLY:-1}
        export VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE=${VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE:-0}
        ;;
    profile)
        export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_eager_probe_profile_${STAMP}}
        export TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS:-1}
        export VLLM_ROLLOUT_DEBUG_GENERATE_ONLY=${VLLM_ROLLOUT_DEBUG_GENERATE_ONLY:-1}
        export VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE=${VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE:-0}
        export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=${VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE:-1}
        export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE_INTERVAL=${VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE_INTERVAL:-128}
        export VLLM_ASCEND_MOE_STAGE_DEBUG=${VLLM_ASCEND_MOE_STAGE_DEBUG:-1}
        export VLLM_ASCEND_MOE_STAGE_DEBUG_INTERVAL=${VLLM_ASCEND_MOE_STAGE_DEBUG_INTERVAL:-256}
        export VLLM_ASCEND_ATTENTION_STAGE_DEBUG=${VLLM_ASCEND_ATTENTION_STAGE_DEBUG:-1}
        export VLLM_ASCEND_ATTENTION_STAGE_DEBUG_STATE_FILTER=${VLLM_ASCEND_ATTENTION_STAGE_DEBUG_STATE_FILTER:-DecodeOnly}
        export VLLM_ASCEND_ATTENTION_STAGE_DEBUG_INTERVAL=${VLLM_ASCEND_ATTENTION_STAGE_DEBUG_INTERVAL:-256}
        export VLLM_ASCEND_ATTENTION_STAGE_DEBUG_LIMIT=${VLLM_ASCEND_ATTENTION_STAGE_DEBUG_LIMIT:-256}
        ;;
    layerprofile)
        export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_eager_probe_layerprofile_${STAMP}}
        export TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS:-1}
        export VLLM_ROLLOUT_DEBUG_GENERATE_ONLY=${VLLM_ROLLOUT_DEBUG_GENERATE_ONLY:-1}
        export VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE=${VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE:-0}
        export VLLM_ASCEND_EAGER_COMPILE=0
        export VLLM_QWEN3_MOE_LAYER_PROFILE=1
        export VLLM_QWEN3_MOE_LAYER_PROFILE_FIRST_N=${VLLM_QWEN3_MOE_LAYER_PROFILE_FIRST_N:-96}
        export VLLM_QWEN3_MOE_LAYER_PROFILE_INTERVAL=${VLLM_QWEN3_MOE_LAYER_PROFILE_INTERVAL:-4096}
        ;;
    npu_profile)
        export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_eager_probe_npu_profile_${STAMP}}
        export TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS:-1}
        export VLLM_ROLLOUT_DEBUG_GENERATE_ONLY=${VLLM_ROLLOUT_DEBUG_GENERATE_ONLY:-1}
        export VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE=${VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE:-0}
        export VLLM_ROLLOUT_TORCH_NPU_PROFILE=1
        export VLLM_ROLLOUT_TORCH_NPU_PROFILE_RANK=${VLLM_ROLLOUT_TORCH_NPU_PROFILE_RANK:-0}
        export VLLM_ROLLOUT_TORCH_NPU_PROFILE_DIR=${VLLM_ROLLOUT_TORCH_NPU_PROFILE_DIR:-${QWEN3_DIR}/${OUTPUT_SUBDIR}/npu_profile_rank_${VLLM_ROLLOUT_TORCH_NPU_PROFILE_RANK}}
        export VLLM_ROLLOUT_TORCH_NPU_PROFILE_ACTIVE=${VLLM_ROLLOUT_TORCH_NPU_PROFILE_ACTIVE:-1}
        export VLLM_ROLLOUT_TORCH_NPU_PROFILE_STACK=${VLLM_ROLLOUT_TORCH_NPU_PROFILE_STACK:-0}
        ;;
    threshold3)
        export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_eager_probe_threshold3_${STAMP}}
        export TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS:-3}
        export VLLM_ROLLOUT_DEBUG_GENERATE_ONLY=${VLLM_ROLLOUT_DEBUG_GENERATE_ONLY:-1}
        export VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE=${VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE:-0}
        ;;
    full16k)
        export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_eager_probe_full16k_${STAMP}}
        unset VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS
        export VLLM_ROLLOUT_FAST_DEBUG=0
        export TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS:-1}
        export VLLM_ROLLOUT_DEBUG_GENERATE_ONLY=${VLLM_ROLLOUT_DEBUG_GENERATE_ONLY:-0}
        export VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE=${VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE:-1}
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        echo "Expected one of: gen1, profile, layerprofile, npu_profile, threshold3, full16k" >&2
        exit 2
        ;;
esac

echo "[probe] mode=${MODE}"
echo "[probe] output=${QWEN3_DIR}/${OUTPUT_SUBDIR}"
echo "[probe] steps=${TRAINER_TOTAL_TRAINING_STEPS} max_response=${MAX_RESPONSE_LENGTH}"
echo "[probe] tail_caps=${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-<unset>}"

bash "${EAGER_FAST}"

latest_log=$(ls -t "${OUTPUT_SUBDIR}"/logs/*.txt 2>/dev/null | head -n 1 || true)
if [ -n "${latest_log}" ]; then
    echo "[probe] latest_log=${QWEN3_DIR}/${latest_log}"
    if [ -x "${PROFILE_TOOL}" ] || [ -f "${PROFILE_TOOL}" ]; then
        python3 "${PROFILE_TOOL}" "${latest_log}"
    fi
else
    echo "[probe] no log found under ${OUTPUT_SUBDIR}/logs" >&2
fi
