#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASELINE_DIR="${BASELINE_DIR:-$SCRIPT_DIR/mode1_dynamic_length_aware_adaptive_floor4_max4096_natural_3epoch_rerun/epoch_000_mode0_probe}"
RESUME_CKPT="${RESUME_CKPT:-$BASELINE_DIR/checkpoints/qwen3moe_for_eagle3/global_step_5}"

if [[ ! -d "$BASELINE_DIR/rollout_data" ]]; then
    echo "missing baseline rollout_data: $BASELINE_DIR/rollout_data" >&2
    exit 2
fi
if [[ ! -d "$RESUME_CKPT" ]]; then
    echo "missing resume checkpoint: $RESUME_CKPT" >&2
    exit 2
fi

cd "$SCRIPT_DIR"

env -u VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS \
DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_length_aware_adaptive_floor4_max4096_natural_quick2}" \
DYNAMIC_TOTAL_EPOCHS=2 \
DYNAMIC_SHRINK_POLICY=natural \
DYNAMIC_SKIP_MODE0_PROBE=1 \
DYNAMIC_INITIAL_BASELINE_DIR="$BASELINE_DIR" \
DYNAMIC_INITIAL_RESUME_CKPT="$RESUME_CKPT" \
DYNAMIC_PLAN_STEPS=5 \
DYNAMIC_TRAIN_STEPS=2 \
DYNAMIC_ENABLE_THRESHOLD_CONTROL=1 \
DYNAMIC_RESET_PROGRESS_AFTER_RESUME=1 \
TRAINER_SAVE_FREQ=2 \
MAX_ACTOR_CKPT_TO_KEEP=1 \
MAX_PROMPT_LENGTH=1024 \
MAX_RESPONSE_LENGTH=4096 \
MAX_RESPONSE_LEN=4096 \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=5120 \
./run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh "$@"
