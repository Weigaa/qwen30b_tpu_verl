#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_mode1_local_baseline_random_batch_threshold_multistep.sh [extra hydra args...]

Fast multi-step smoke test for mode=1 floor=4 baseline:
  - reuses the local baseline random-batch script
  - uses elastic tail validation level caps to create uneven rank tails
  - exercises 16->8->4 shrink/restore paths faster than a full rollout

Useful overrides:
  MAX_RESPONSE_LENGTH=640
  VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,640,640
  DATASET_FRACTION=0.005
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

: "${MAX_PROMPT_LENGTH:=1024}"
: "${MAX_RESPONSE_LENGTH:=640}"
export MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

export VLLM_ROLLOUT_EARLY_STOP_ENABLE=0
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-256,512,640,640,640}"

export DATASET_FRACTION="${DATASET_FRACTION:-0.005}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_baseline_random_batch_floor4_threshold_multistep}"

printf '[mode1 threshold multistep] max_prompt=%s max_response=%s max_batched_tokens=%s\n' \
    "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" "$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
printf '[mode1 threshold multistep] early_stop=%s tail_validate_level_tokens=%s dataset_fraction=%s total_epochs=%s\n' \
    "$VLLM_ROLLOUT_EARLY_STOP_ENABLE" \
    "$VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS" \
    "$DATASET_FRACTION" \
    "$TRAINER_TOTAL_EPOCHS"

exec "$SCRIPT_DIR/run_mode1_local_baseline_random_batch.sh" "$@"
