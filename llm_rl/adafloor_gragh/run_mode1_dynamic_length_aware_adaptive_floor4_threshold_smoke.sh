#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_length_aware_adaptive_floor4_threshold_smoke}"
export DYNAMIC_TOTAL_EPOCHS="${DYNAMIC_TOTAL_EPOCHS:-2}"
export DYNAMIC_SHRINK_POLICY="${DYNAMIC_SHRINK_POLICY:-planned}"

# This smoke path validates the dynamic EMA -> plan -> mode1 rollout chain.
# By default it skips checkpointing for fast rollout validation. If
# DYNAMIC_ENABLE_CKPT_CHAIN=1 is requested, save one checkpoint so the smoke
# also covers the Megatron async dist-checkpoint writer.
export DYNAMIC_ENABLE_CKPT_CHAIN="${DYNAMIC_ENABLE_CKPT_CHAIN:-0}"

export DYNAMIC_SKIP_MODE0_PROBE="${DYNAMIC_SKIP_MODE0_PROBE:-1}"
export DYNAMIC_INITIAL_BASELINE_DIR="${DYNAMIC_INITIAL_BASELINE_DIR:-$SCRIPT_DIR/mode1_dynamic_length_aware_adaptive_floor4/epoch_000_mode0_probe}"
export DYNAMIC_PLAN_STEPS="${DYNAMIC_PLAN_STEPS:-1}"
export DYNAMIC_DATASET_FRACTION="${DYNAMIC_DATASET_FRACTION:-0.005}"
export DYNAMIC_LENGTH_EMA_DECAY="${DYNAMIC_LENGTH_EMA_DECAY:-0.3}"
export DYNAMIC_ENABLE_THRESHOLD_CONTROL=1

export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-$DYNAMIC_RUN_NAME}"
export PLAN_STEPS="$DYNAMIC_PLAN_STEPS"
if [[ "$DYNAMIC_ENABLE_CKPT_CHAIN" == "1" ]]; then
    export SAVE_CKPT_ENABLE="${SAVE_CKPT_ENABLE:-1}"
    export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-1}"
else
    export SAVE_CKPT_ENABLE="${SAVE_CKPT_ENABLE:-0}"
    export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:--1}"
fi

export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-256,512,640,768,896}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-896}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

echo "[dynamic threshold smoke] run_name=$DYNAMIC_RUN_NAME policy=$DYNAMIC_SHRINK_POLICY"
echo "[dynamic threshold smoke] initial_baseline=$DYNAMIC_INITIAL_BASELINE_DIR plan_steps=$DYNAMIC_PLAN_STEPS ema_decay=$DYNAMIC_LENGTH_EMA_DECAY"
echo "[dynamic threshold smoke] max_response=$MAX_RESPONSE_LENGTH tail_validate=$VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS checkpoint_chain=$DYNAMIC_ENABLE_CKPT_CHAIN"

exec "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh" "$@"
