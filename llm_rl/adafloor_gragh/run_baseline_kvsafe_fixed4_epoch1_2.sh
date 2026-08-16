#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# KVSafeFixed4 baseline for the main policy comparison.
#
# The prompt buckets, EMA predictor, tail guard, floor KV budgets, mode1
# runtime, and checkpoint chain match the Natural floor4 lifecycle. Every step
# is forced to floor4. If a contiguous length bucket is not KV feasible, the
# planner swaps prompts with neighboring steps until all five floor4 plans fit
# the configured cap. Within each step it uses KV-constrained MinSkew rather
# than AdaFloor's quorum-aware release-area objective.

BASELINE_EPOCH0="${DYNAMIC_INITIAL_BASELINE_DIR:-$SCRIPT_DIR/mode1_dynamic_length_aware_adaptive_floor4_natural_tailguard_full3/epoch_000_mode0_probe}"
DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-baseline_kvsafe_fixed4_tailguard_reuse_epoch0_2epoch}"
OUTPUT_ROOT="${DYNAMIC_OUTPUT_ROOT:-$SCRIPT_DIR}"
OUTPUT_DIR="$OUTPUT_ROOT/$DYNAMIC_RUN_NAME"

if [[ ! -d "$BASELINE_EPOCH0/rollout_data" ]]; then
    echo "missing reusable epoch0 rollout data: $BASELINE_EPOCH0/rollout_data" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
    echo "output already exists: $OUTPUT_DIR" >&2
    echo "Set DYNAMIC_RUN_NAME to a new directory or ALLOW_EXISTING_OUTPUT=1." >&2
    exit 2
fi

echo "[KVSafeFixed4 baseline] output=$OUTPUT_DIR"
echo "[KVSafeFixed4 baseline] reusable_epoch0=$BASELINE_EPOCH0"
echo "[KVSafeFixed4 baseline] epochs=1,2 steps_per_epoch=5 matching=min_skew forced_floor=4 repair=neighbor_swap"

export RANK_MATCHING_POLICY=min_skew
export KV_SAFE_FIXED_FLOOR=0
export ACTIVE_PEAK_SAFETY_FACTOR="${BASELINE_ACTIVE_PEAK_SAFETY_FACTOR:-1.16}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.9}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4:-280576}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8:-315648}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16:-${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS:-380800}}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS:-$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16}"
export FLOOR_KV_CAPS="${FLOOR_KV_CAPS:-4:$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4,8:$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8,16:$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16}"
export VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4
export MIN_ADAPTIVE_FLOOR=4

exec env \
    DYNAMIC_RUN_NAME="$DYNAMIC_RUN_NAME" \
    DYNAMIC_OUTPUT_ROOT="$OUTPUT_ROOT" \
    DYNAMIC_INITIAL_BASELINE_DIR="$BASELINE_EPOCH0" \
    DYNAMIC_TOTAL_EPOCHS=3 \
    DYNAMIC_START_EPOCH=1 \
    DYNAMIC_SKIP_MODE0_PROBE=1 \
    DYNAMIC_INITIAL_RESUME_CKPT="${BASELINE_INITIAL_RESUME_CKPT:-}" \
    DYNAMIC_ENABLE_CKPT_CHAIN=1 \
    DYNAMIC_LENGTH_EMA_DECAY=0.3 \
    DYNAMIC_PLAN_STEPS=5 \
    DYNAMIC_TRAIN_STEPS=5 \
    DYNAMIC_ENABLE_THRESHOLD_CONTROL=0 \
    DYNAMIC_DISABLE_TAIL_GUARD=0 \
    DYNAMIC_SHORT_STEP_CAP_ENABLE=1 \
    DYNAMIC_SHORT_STEP_EXIT_THRESHOLD=4096 \
    DYNAMIC_SHORT_STEP_CAP_TOKENS=4096 \
    DYNAMIC_SHORT_STEP_CAP_FLOORS=4 \
    DYNAMIC_TAIL_GUARD_RATIO_QUANTILE=0.95 \
    DYNAMIC_TAIL_GUARD_RATIO_WINDOW=3 \
    DYNAMIC_TAIL_GUARD_DEFAULT_RATIO=1.20 \
    DYNAMIC_TAIL_GUARD_MIN_CAP=4096 \
    DYNAMIC_TAIL_GUARD_ROUND_TO=512 \
    FLOOR_KV_CAPS="$FLOOR_KV_CAPS" \
    DYNAMIC_FORCE_SELECTED_FLOOR=4 \
    DYNAMIC_FORCE_SELECTED_FLOORS= \
    FORCE_SELECTED_FLOOR= \
    FORCE_SELECTED_FLOORS= \
    "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_natural_full3.sh" \
    "$@"
