#!/usr/bin/env bash
set -euo pipefail

# Baseline run for strict per-rank dummy waste accounting.
# This wrapper intentionally overrides any caller environment that would turn
# on elastic shrink or sidecar inference.

export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=0
export VERL_SIDECAR_ENABLE=0

# Enable low-intrusion dummy waste timing. This records host-side dummy forward
# and MoE intervals without device synchronize; set
# VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=1 only for a short debug run.
export VLLM_ASCEND_DUMMY_WASTE_TIMING=1
export VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=${VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC:-0}
export VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE=0
export VLLM_ASCEND_DUMMY_WASTE_SELECTION_STATS=${VLLM_ASCEND_DUMMY_WASTE_SELECTION_STATS:-0}

# Keep MoE pattern CSV optional. The dummy waste lines are sufficient for the
# rank waste table; turn this on only if you also want top-k pattern artifacts.
export VLLM_MOE_PATTERN_STATS=${VLLM_MOE_PATTERN_STATS:-0}
export VLLM_MOE_STATS=${VLLM_MOE_PATTERN_STATS}

# Avoid accidental stale sidecar output paths from previous shells.
unset VERL_SIDECAR_TRAIN_LOG
unset VERL_SIDECAR_LEASE_LOG
unset VERL_SIDECAR_LOG_FILE
unset VERL_SIDECAR_OUTPUT_FILE
unset VERL_SIDECAR_MONITOR_LOG

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${script_dir}/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh" "$@"
