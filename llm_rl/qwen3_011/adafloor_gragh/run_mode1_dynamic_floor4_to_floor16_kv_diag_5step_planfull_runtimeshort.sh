#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

# Floor4-history control for the floor2 planfull/runtimeshort probe.
# Planning stays full length so step5 prompt->rank assignment matches the real
# floor4/floor2 epoch1 tail. Runtime caps stay short to expose batch-start KV
# free-block state without paying for the complete 16k decode.
export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_floor4_to_floor16_kv_diag_5step_planfull_runtimeshort_${RUN_STAMP}}"
export DYNAMIC_FULL_MAX_RESPONSE_LENGTH="${DYNAMIC_FULL_MAX_RESPONSE_LENGTH:-16384}"
export DYNAMIC_FULL_MAX_RESPONSE_LEN="${DYNAMIC_FULL_MAX_RESPONSE_LEN:-16384}"
export DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS="${DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS:-17408}"
export DYNAMIC_FULL_MAX_PROMPT_LENGTH="${DYNAMIC_FULL_MAX_PROMPT_LENGTH:-1024}"
export DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP="${DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP:-8,16,32,64,64;8,16,32,64,64;8,16,32,64,64;8,16,32,64,64;8,16,32,64,64}"
export VLLM_ASCEND_MODE1_KV_PREEMPT_DIAG="${VLLM_ASCEND_MODE1_KV_PREEMPT_DIAG:-1}"

echo "[floor4 planfull/runtimeshort] run_name=$DYNAMIC_RUN_NAME"
echo "[floor4 planfull/runtimeshort] planning_max_response=$DYNAMIC_FULL_MAX_RESPONSE_LENGTH runtime_caps=$DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP"

exec "$SCRIPT_DIR/run_mode1_dynamic_floor4_to_floor16_kv_diag_5step_step5full.sh" "$@"
