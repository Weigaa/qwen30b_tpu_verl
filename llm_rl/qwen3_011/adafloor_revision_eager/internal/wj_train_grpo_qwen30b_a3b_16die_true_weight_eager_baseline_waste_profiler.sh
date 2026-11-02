#!/usr/bin/env bash
set -euo pipefail

# Baseline run with low-intrusion profiler markers for dummy waste analysis.
# Use MSTX ranges only: this keeps marker durations while avoiding full NPU
# kernel traces and CPU op_range traces, which are too large for a full rollout
# across all ranks.
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=0
export VERL_SIDECAR_ENABLE=0
export VLLM_ASCEND_DUMMY_WASTE_TIMING=0
export VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=0
export VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE=0
export VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS=1
export VLLM_ASCEND_DUMMY_WASTE_PROFILE_ROLLOUT_ONLY=1
export VLLM_ASCEND_DUMMY_WASTE_SELECTION_STATS=0
export VLLM_MOE_PATTERN_STATS=${VLLM_MOE_PATTERN_STATS:-0}
export VLLM_MOE_STATS=${VLLM_MOE_PATTERN_STATS}
# export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=4,8,12,16,20

unset VERL_SIDECAR_TRAIN_LOG
unset VERL_SIDECAR_LEASE_LOG
unset VERL_SIDECAR_LOG_FILE
unset VERL_SIDECAR_OUTPUT_FILE
unset VERL_SIDECAR_MONITOR_LOG

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
profile_time=$(date +%Y%m%d%H%M%S)
export WJ_RECORDS_DIR=${WJ_RECORDS_DIR:-"/home/sharedata/wj_records"}
export VERL_DUMMY_WASTE_PROFILE_DIR=${VERL_DUMMY_WASTE_PROFILE_DIR:-"${WJ_RECORDS_DIR}/dummy_waste_profiles/${profile_time}"}
mkdir -p "${VERL_DUMMY_WASTE_PROFILE_DIR}"

echo "[dummy waste profiler] wj_records_dir=${WJ_RECORDS_DIR} save_path=${VERL_DUMMY_WASTE_PROFILE_DIR}"

exec bash "${script_dir}/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh" \
    "global_profiler.tool=npu" \
    "global_profiler.steps=null" \
    global_profiler.save_path="${VERL_DUMMY_WASTE_PROFILE_DIR}" \
    "actor_rollout_ref.actor.profiler.enable=True" \
    "actor_rollout_ref.actor.profiler.all_ranks=True" \
    "actor_rollout_ref.actor.profiler.tool=npu" \
    "actor_rollout_ref.actor.profiler.tool_config.npu.contents=[mstx]" \
    "actor_rollout_ref.actor.profiler.tool_config.npu.level=level_none" \
    "actor_rollout_ref.actor.profiler.tool_config.npu.analysis=True" \
    "$@"
