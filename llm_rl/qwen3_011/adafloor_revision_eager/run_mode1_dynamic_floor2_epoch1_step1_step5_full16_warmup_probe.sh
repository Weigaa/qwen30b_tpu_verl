#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

# Validate whether the slow first floor16 step is a cold-runtime effect.
#
# Phase 1: run a tiny no-shrink/full-world mode0 warmup in this invocation.
#          It initializes the full-world KV/MC2/HCCL path without producing a
#          full epoch0 history.
# Phase 2: reuse the real floor4 epoch0 history, build the normal 5-step epoch1
#          plan, then compact it to original steps 1 and 5.  The resulting
#          2-step mode1 run executes the original first floor2 step followed by
#          the original final floor16 step.

BASELINE_EPOCH0="${DYNAMIC_INITIAL_BASELINE_DIR:-$SCRIPT_DIR/mode1_dynamic_length_aware_adaptive_floor4_natural_tailguard_full3/epoch_000_mode0_probe}"
if [[ ! -d "$BASELINE_EPOCH0/rollout_data" ]]; then
    echo "missing reusable epoch0 rollout data: $BASELINE_EPOCH0/rollout_data" >&2
    exit 2
fi

export RUN_FULL16_WARMUP="${RUN_FULL16_WARMUP:-1}"
export FULL_STEP5_DECODE="${FULL_STEP5_DECODE:-0}"
export WARMUP_RUN_NAME="${WARMUP_RUN_NAME:-mode1_dynamic_floor2_full16_warmup_${RUN_STAMP}}"
if [[ "$RUN_FULL16_WARMUP" == "1" ]]; then
    DEFAULT_MAIN_RUN_NAME="mode1_dynamic_floor2_step1_step5_after_full16_warmup_${RUN_STAMP}"
else
    DEFAULT_MAIN_RUN_NAME="mode1_dynamic_floor2_step1_step5_no_full16_warmup_${RUN_STAMP}"
fi
export MAIN_RUN_NAME="${DYNAMIC_RUN_NAME:-$DEFAULT_MAIN_RUN_NAME}"
export WARMUP_OUTPUT_ROOT="${WARMUP_OUTPUT_ROOT:-$SCRIPT_DIR}"
PORT_OFFSET="${PORT_OFFSET:-$(( (RANDOM % 500) * 20 ))}"
export WARMUP_MASTER_PORT="${WARMUP_MASTER_PORT:-$((25000 + PORT_OFFSET))}"
export WARMUP_HCCL_IF_BASE_PORT="${WARMUP_HCCL_IF_BASE_PORT:-$((38000 + PORT_OFFSET))}"
export WARMUP_VERL_HCCL_IF_BASE_PORT_START="${WARMUP_VERL_HCCL_IF_BASE_PORT_START:-$WARMUP_HCCL_IF_BASE_PORT}"
export MAIN_MASTER_PORT="${MAIN_MASTER_PORT:-$((27000 + PORT_OFFSET))}"
export MAIN_HCCL_IF_BASE_PORT="${MAIN_HCCL_IF_BASE_PORT:-$((50000 + PORT_OFFSET))}"
export MAIN_VERL_HCCL_IF_BASE_PORT_START="${MAIN_VERL_HCCL_IF_BASE_PORT_START:-$MAIN_HCCL_IF_BASE_PORT}"

if [[ "$RUN_FULL16_WARMUP" == "1" ]]; then
    echo "[floor2 step1/step5 warmup probe] phase=full16_warmup run_name=$WARMUP_RUN_NAME master_port=$WARMUP_MASTER_PORT hccl_base=$WARMUP_HCCL_IF_BASE_PORT"
    OUTPUT_ROOT="$WARMUP_OUTPUT_ROOT" \
    OUTPUT_SUBDIR="$WARMUP_RUN_NAME" \
    MASTER_PORT="$WARMUP_MASTER_PORT" \
    HCCL_IF_BASE_PORT="$WARMUP_HCCL_IF_BASE_PORT" \
    VERL_HCCL_IF_BASE_PORT_START="$WARMUP_VERL_HCCL_IF_BASE_PORT_START" \
    TRAINER_TOTAL_EPOCHS=1 \
    TRAINER_SAVE_FREQ=-1 \
    SAVE_CKPT_ENABLE=0 \
    DATA_SHUFFLE=False \
    DATASET_FRACTION="${WARMUP_DATASET_FRACTION:-0.001}" \
    TRAIN_BATCH_SIZE="${WARMUP_TRAIN_BATCH_SIZE:-32}" \
    ROLLOUT_N="${WARMUP_ROLLOUT_N:-1}" \
    MAX_PROMPT_LENGTH="${WARMUP_MAX_PROMPT_LENGTH:-1024}" \
    MAX_RESPONSE_LENGTH="${WARMUP_MAX_RESPONSE_LENGTH:-16384}" \
    ROLLOUT_MAX_NUM_SEQS="${WARMUP_ROLLOUT_MAX_NUM_SEQS:-32}" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="${WARMUP_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-17408}" \
    ROLLOUT_GPU_MEMORY_UTILIZATION="${WARMUP_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}" \
    VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${WARMUP_TAIL_VALIDATE_LEVEL_TOKENS:-8,16,32,64,64}" \
    MODE0_SAVE_ROLLOUT_ARTIFACTS=0 \
    "$SCRIPT_DIR/run_mode0_no_shrink_baseline.sh" \
        trainer.total_training_steps="${WARMUP_TRAIN_STEPS:-1}"

    WARMUP_DIR="$WARMUP_OUTPUT_ROOT/$WARMUP_RUN_NAME"
    WARMUP_LOG=$(find "$WARMUP_DIR" -type f -path '*/logs/*.txt' 2>/dev/null | sort | tail -n 1 || true)
    if [[ -z "$WARMUP_LOG" ]]; then
        echo "full16 warmup finished but no warmup log was found under $WARMUP_DIR" >&2
        exit 3
    fi
    if ! grep -q "GPU KV cache size: 380,800 tokens" "$WARMUP_LOG"; then
        echo "warning: warmup log did not show full-world KV=380,800 tokens: $WARMUP_LOG" >&2
    fi
    echo "[floor2 step1/step5 warmup probe] warmup_log=$WARMUP_LOG"
fi

echo "[floor2 step1/step5 warmup probe] phase=mode1_subset run_name=$MAIN_RUN_NAME baseline=$BASELINE_EPOCH0"

export MASTER_PORT="$MAIN_MASTER_PORT"
export HCCL_IF_BASE_PORT="$MAIN_HCCL_IF_BASE_PORT"
export VERL_HCCL_IF_BASE_PORT_START="$MAIN_VERL_HCCL_IF_BASE_PORT_START"
export DYNAMIC_RUN_NAME="$MAIN_RUN_NAME"
export DYNAMIC_TOTAL_EPOCHS=2
export DYNAMIC_START_EPOCH=1
export DYNAMIC_ENABLE_CKPT_CHAIN=0
export DYNAMIC_PLAN_STEPS=5
export DYNAMIC_TRAIN_STEPS=2
export DYNAMIC_FORCE_SELECTED_FLOORS="${DYNAMIC_FORCE_SELECTED_FLOORS:-2,2,4,4,16}"
export FAST_STEP_SUBSET=1
export FAST_STEP_SUBSET_STEPS="${FAST_STEP_SUBSET_STEPS:-1,5}"

export DYNAMIC_FULL_MAX_PROMPT_LENGTH="${DYNAMIC_FULL_MAX_PROMPT_LENGTH:-1024}"
if [[ "$FULL_STEP5_DECODE" == "1" ]]; then
    # Keep original step1 short, but let original step5 exercise the real
    # full-world long-decode path.  This is the useful mode for validating
    # whether full16 warmup removes the slow floor2 epoch1 final step.
    export DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP="${DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP:-8,16,32,64,64;16384,16384,16384,16384,16384}"
    export DYNAMIC_FULL_MAX_RESPONSE_LENGTH="${DYNAMIC_FULL_MAX_RESPONSE_LENGTH:-16384}"
    export DYNAMIC_FULL_MAX_RESPONSE_LEN="${DYNAMIC_FULL_MAX_RESPONSE_LEN:-16384}"
    export DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS="${DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS:-17408}"
else
    # Short-cap smoke test.  These per-step caps are indexed after the fast
    # subset remaps original step1 -> new step1 and original step5 -> new step2.
    export DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP="${DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP:-8,16,32,64,64;8,16,32,64,64}"
    export DYNAMIC_FULL_MAX_RESPONSE_LENGTH="${DYNAMIC_FULL_MAX_RESPONSE_LENGTH:-1024}"
    export DYNAMIC_FULL_MAX_RESPONSE_LEN="${DYNAMIC_FULL_MAX_RESPONSE_LEN:-1024}"
    export DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS="${DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS:-2048}"
fi
export DYNAMIC_SHORT_STEP_CAP_ENABLE=0
export DYNAMIC_TAIL_GUARD_MIN_CAP="${DYNAMIC_TAIL_GUARD_MIN_CAP:-1024}"
export DYNAMIC_TAIL_GUARD_ROUND_TO="${DYNAMIC_TAIL_GUARD_ROUND_TO:-64}"
export SAVE_CKPT_ENABLE=0
export TRAINER_SAVE_FREQ=-1

echo "[floor2 step1/step5 warmup probe] subset_steps=$FAST_STEP_SUBSET_STEPS floors=$DYNAMIC_FORCE_SELECTED_FLOORS full_step5_decode=$FULL_STEP5_DECODE master_port=$MASTER_PORT hccl_base=$HCCL_IF_BASE_PORT runtime_caps=$DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP"

exec "$SCRIPT_DIR/run_mode1_dynamic_floor2_to_floor4_kv_probe.sh" "$@"
