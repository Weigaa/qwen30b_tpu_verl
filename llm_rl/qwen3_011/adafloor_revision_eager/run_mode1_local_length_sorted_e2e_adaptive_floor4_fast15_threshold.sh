#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Fast smoke test for the expensive step1 -> step5 adaptive-KV path.
# Keep the selected batches identical to fast15, but cap generation lengths so
# KV planning and runtime sampling use the same small validation budget.
export FAST_STEP_SUBSET="${FAST_STEP_SUBSET:-1}"
export FAST_STEP_SUBSET_STEPS="${FAST_STEP_SUBSET_STEPS:-1,5}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_length_sorted_e2e_adaptive_floor4_fast15_threshold}"

export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-256,512,640,768,896}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-896}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export VLLM_ASCEND_MODE1_KV_RESIZE_LIVE_TENSOR_SCAN="${VLLM_ASCEND_MODE1_KV_RESIZE_LIVE_TENSOR_SCAN:-1}"

exec "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh" "$@"
