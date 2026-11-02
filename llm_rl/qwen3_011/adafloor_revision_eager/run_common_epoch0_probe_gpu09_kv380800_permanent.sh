#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Build the single shared epoch0 state used by paper comparisons. This run is
# intentionally mode0/no-shrink and does not apply any response-length guard.
# Keep both its rollout history and global_step_5 checkpoint permanently.

OUTPUT_ROOT="${COMMON_EPOCH0_OUTPUT_ROOT:-$SCRIPT_DIR}"
RUN_NAME="${COMMON_EPOCH0_RUN_NAME:-common_epoch0_probe_gpu09_kv380800_permanent}"
RUN_ROOT="$OUTPUT_ROOT/$RUN_NAME"
EPOCH_DIR="$RUN_ROOT/epoch_000_mode0_probe"
CHECKPOINT_PATH="$EPOCH_DIR/checkpoints/qwen3moe_for_eagle3/global_step_5"

MODEL_PATH="${MODEL_PATH:-/data/Qwen3-30B-A3B}"
DISTCP_PATH="${DISTCP_PATH:-/data/Qwen3-30B-A3B_megatron}"
TRAIN_FILE="${TRAIN_FILE:-/data/deepscaler/train.parquet}"
TEST_FILE="${TEST_FILE:-/data/deepscaler/test.parquet}"

TRAIN_STEPS="${COMMON_EPOCH0_TRAIN_STEPS:-5}"
DATASET_FRACTION="${COMMON_EPOCH0_DATASET_FRACTION:-0.005}"
KV_TOKENS_PER_RANK="${COMMON_EPOCH0_KV_TOKENS_PER_RANK:-380800}"
KV_BLOCK_SIZE="${VLLM_KV_BLOCK_SIZE:-128}"
MIN_FREE_GB="${COMMON_EPOCH0_MIN_FREE_GB:-80}"

if (( TRAIN_STEPS != 5 )); then
    echo "the shared paper epoch0 must contain exactly 5 training steps" >&2
    exit 2
fi
if (( KV_TOKENS_PER_RANK <= 0 || KV_BLOCK_SIZE <= 0 \
      || KV_TOKENS_PER_RANK % KV_BLOCK_SIZE != 0 )); then
    echo "KV token capacity must be positive and divisible by block size" >&2
    exit 2
fi
KV_BLOCKS=$((KV_TOKENS_PER_RANK / KV_BLOCK_SIZE))

for required_path in "$MODEL_PATH" "$DISTCP_PATH" "$TRAIN_FILE" "$TEST_FILE"; do
    if [[ ! -e "$required_path" ]]; then
        echo "missing required input: $required_path" >&2
        exit 2
    fi
done

if [[ -e "$RUN_ROOT" ]]; then
    echo "refusing to overwrite permanent common epoch0: $RUN_ROOT" >&2
    echo "Remove an incomplete directory explicitly or set COMMON_EPOCH0_RUN_NAME to a new name." >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
available_kb=$(df -Pk "$OUTPUT_ROOT" | awk 'NR == 2 {print $4}')
required_kb=$((MIN_FREE_GB * 1024 * 1024))
if (( available_kb < required_kb )); then
    echo "insufficient free space for permanent epoch0 checkpoint" >&2
    echo "available_kb=$available_kb required_kb=$required_kb output_root=$OUTPUT_ROOT" >&2
    exit 2
fi

mkdir -p "$EPOCH_DIR"
printf '%s\n' \
    "INCOMPLETE common epoch0 run" \
    "Do not use this directory unless the COMPLETE marker exists." \
    > "$RUN_ROOT/INCOMPLETE"

echo "[common epoch0] output=$EPOCH_DIR"
echo "[common epoch0] model=$MODEL_PATH train=$TRAIN_FILE dataset_fraction=$DATASET_FRACTION"
echo "[common epoch0] steps=$TRAIN_STEPS gpu_memory_utilization=0.9 kv_tokens_per_rank=$KV_TOKENS_PER_RANK blocks=$KV_BLOCKS"
echo "[common epoch0] tail_guard=disabled checkpoint=$CHECKPOINT_PATH"

set +e
env \
    -u VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS \
    -u VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    OUTPUT_SUBDIR="$RUN_NAME/epoch_000_mode0_probe" \
    RECORD_DIR="$EPOCH_DIR" \
    MODEL_PATH="$MODEL_PATH" \
    DISTCP_PATH="$DISTCP_PATH" \
    TRAIN_FILE="$TRAIN_FILE" \
    TEST_FILE="$TEST_FILE" \
    TRAINER_TOTAL_EPOCHS=1 \
    DATASET_FRACTION="$DATASET_FRACTION" \
    DATA_SHUFFLE=False \
    TRAIN_BATCH_SIZE=32 \
    MAX_PROMPT_LENGTH=1024 \
    MAX_RESPONSE_LENGTH=16384 \
    MAX_RESPONSE_LEN=16384 \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=17408 \
    ROLLOUT_MAX_NUM_SEQS=32 \
    ROLLOUT_N=16 \
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 \
    MODE0_SAVE_ROLLOUT_ARTIFACTS=1 \
    SAVE_CKPT_ENABLE=1 \
    TRAINER_SAVE_FREQ=5 \
    MAX_ACTOR_CKPT_TO_KEEP=1 \
    MAX_CRITIC_CKPT_TO_KEEP=1 \
    VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1 \
    "$SCRIPT_DIR/run_mode0_no_shrink_baseline.sh" \
        trainer.total_training_steps="$TRAIN_STEPS" \
        trainer.resume_mode=disable \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override="$KV_BLOCKS" \
        "$@"
run_rc=$?
set -e

if (( run_rc != 0 )); then
    echo "common epoch0 failed with exit_code=$run_rc; incomplete output retained at $RUN_ROOT" >&2
    exit "$run_rc"
fi

rollout_count=$(find "$EPOCH_DIR/rollout_data" -maxdepth 1 -type f -name '*.jsonl' 2>/dev/null | wc -l)
length_count=$(find "$EPOCH_DIR/rollout_length" -maxdepth 1 -type f -name 'length_*.txt' 2>/dev/null | wc -l)
if (( rollout_count != 5 || length_count != 5 )); then
    echo "common epoch0 artifact validation failed: rollout=$rollout_count length=$length_count" >&2
    exit 3
fi
if [[ ! -d "$CHECKPOINT_PATH/actor" ]]; then
    echo "common epoch0 checkpoint validation failed: $CHECKPOINT_PATH/actor" >&2
    exit 4
fi

rm -f "$RUN_ROOT/INCOMPLETE"
printf '%s\n' \
    "PERMANENT COMMON EPOCH0 CHECKPOINT" \
    "Keep this directory for all paper baseline and AdaFloor reruns." \
    "Rollout history: $EPOCH_DIR" \
    "Resume checkpoint: $CHECKPOINT_PATH" \
    > "$RUN_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT"
touch "$CHECKPOINT_PATH/.PRESERVE_COMMON_EPOCH0"

{
    printf 'export DYNAMIC_INITIAL_BASELINE_DIR=%q\n' "$EPOCH_DIR"
    printf 'export BASELINE_INITIAL_RESUME_CKPT=%q\n' "$CHECKPOINT_PATH"
    printf 'export DYNAMIC_INITIAL_RESUME_CKPT=%q\n' "$CHECKPOINT_PATH"
} > "$RUN_ROOT/reuse.env"

echo "[common epoch0] complete"
echo "[common epoch0] rollout_history=$EPOCH_DIR"
echo "[common epoch0] checkpoint=$CHECKPOINT_PATH"
echo "[common epoch0] reuse with: source $RUN_ROOT/reuse.env"
