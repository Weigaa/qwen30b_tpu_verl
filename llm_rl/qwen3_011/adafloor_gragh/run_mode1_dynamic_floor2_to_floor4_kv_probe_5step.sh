#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

# Five-step floor2 -> floor4 -> full-world correctness probe.
#
# Step floors:
#   1: floor2
#   2: floor2
#   3: floor4
#   4: floor4
#   5: floor16
#
# This wrapper intentionally writes to a fresh timestamped output directory by
# default, so stale rollout_data from earlier short probes cannot be mistaken
# for the current run.
export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_floor2_to_floor4_kv_probe_5step_${RUN_STAMP}}"
export DYNAMIC_PLAN_STEPS="${DYNAMIC_PLAN_STEPS:-5}"
export DYNAMIC_TRAIN_STEPS="${DYNAMIC_TRAIN_STEPS:-5}"
export DYNAMIC_FORCE_SELECTED_FLOORS="${DYNAMIC_FORCE_SELECTED_FLOORS:-2,2,4,4,16}"

# Keep all steps comparable: repeated-halving buckets [8,4,2,1,1] use
# [8,16,32,64,64] token caps at every step.
export DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP="${DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP:-8,16,32,64,64;8,16,32,64,64;8,16,32,64,64;8,16,32,64,64;8,16,32,64,64}"

# The one-step smoke test already validates the low-level loader write path.
# For the 5-step performance/correctness run, keep all heavyweight diagnostics
# off by default. Re-enable them explicitly only when debugging corruption.
export VLLM_ASCEND_MODE1_WEIGHT_LOADER_DIAG="${VLLM_ASCEND_MODE1_WEIGHT_LOADER_DIAG:-0}"
export VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_ROW_TRACE_LOG="${VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_ROW_TRACE_LOG:-0}"
export VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_VERIFY_LOG="${VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_VERIFY_LOG:-0}"
export VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_POSTCHECK_LOG="${VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_POSTCHECK_LOG:-0}"
export VLLM_ASCEND_MODE1_WEIGHT_FINGERPRINT_LOG="${VLLM_ASCEND_MODE1_WEIGHT_FINGERPRINT_LOG:-0}"
export VLLM_ASCEND_MODE1_WEIGHT_FINGERPRINT_REASONS="${VLLM_ASCEND_MODE1_WEIGHT_FINGERPRINT_REASONS:-pre_generate,execute_single_wave}"
export VLLM_ASCEND_MODE1_PRE_GENERATE_STATE_DUMP="${VLLM_ASCEND_MODE1_PRE_GENERATE_STATE_DUMP:-0}"
export VLLM_ASCEND_MODE1_PRE_GENERATE_STATE_STRICT="${VLLM_ASCEND_MODE1_PRE_GENERATE_STATE_STRICT:-1}"

echo "[floor2->floor4 kv probe 5step] run_name=$DYNAMIC_RUN_NAME"
echo "[floor2->floor4 kv probe 5step] floors=$DYNAMIC_FORCE_SELECTED_FLOORS steps=$DYNAMIC_TRAIN_STEPS"

exec "$SCRIPT_DIR/run_mode1_dynamic_floor2_to_floor4_kv_probe.sh" "$@"
