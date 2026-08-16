#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

# Keep offline planning identical to the full epoch1 step5:
# max_response_len=16384, max_num_batched_tokens=17408.
# Only the runtime tail-validation caps are shortened to avoid spending a full
# 16k decode when we only need the step5 scheduler/KV batch-start state.
export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_floor2_to_floor4_kv_diag_5step_planfull_runtimeshort_${RUN_STAMP}}"
export DYNAMIC_FULL_MAX_RESPONSE_LENGTH="${DYNAMIC_FULL_MAX_RESPONSE_LENGTH:-16384}"
export DYNAMIC_FULL_MAX_RESPONSE_LEN="${DYNAMIC_FULL_MAX_RESPONSE_LEN:-16384}"
export DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS="${DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS:-17408}"
export DYNAMIC_FULL_MAX_PROMPT_LENGTH="${DYNAMIC_FULL_MAX_PROMPT_LENGTH:-1024}"
export DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP="${DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP:-8,16,32,64,64;8,16,32,64,64;8,16,32,64,64;8,16,32,64,64;8,16,32,64,64}"
export VLLM_ASCEND_MODE1_KV_PREEMPT_DIAG="${VLLM_ASCEND_MODE1_KV_PREEMPT_DIAG:-1}"

echo "[floor2 planfull/runtimeshort] run_name=$DYNAMIC_RUN_NAME"
echo "[floor2 planfull/runtimeshort] planning_max_response=$DYNAMIC_FULL_MAX_RESPONSE_LENGTH runtime_caps=$DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP"

exec "$SCRIPT_DIR/run_mode1_dynamic_floor2_to_floor4_kv_probe_5step_fullworld_hot.sh" "$@"
