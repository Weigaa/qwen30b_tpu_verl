#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Fast floor=2 feasibility smoke test.
#
# This keeps generation short with per-step validation caps while exercising the
# correct halving shrink path: 16 -> 8 -> 4 -> 2. It is intended to validate
# floor2 communication/import/KV behavior, not full-quality rollout metrics.
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_length_sorted_e2e_adaptive_floor2_threshold}"
export PLAN_DIR="${PLAN_DIR:-$SCRIPT_DIR/$OUTPUT_SUBDIR/oracle}"

export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-256,512,640,768,896}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-896}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-896}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

export VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4,2
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2
export MIN_ADAPTIVE_FLOOR=2
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"

printf '[mode1 floor2 threshold] tail_validate_level_tokens=%s max_response=%s output=%s\n' \
    "$VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS" "$MAX_RESPONSE_LENGTH" "$OUTPUT_SUBDIR"
printf '[mode1 floor2 threshold] stages=%s min_floor=%s plan_dir=%s\n' \
    "$VLLM_ASCEND_SHRINK_AWARE_STAGES" "$MIN_ADAPTIVE_FLOOR" "$PLAN_DIR"

exec "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2.sh" "$@"
