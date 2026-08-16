#!/usr/bin/env bash
set -euo pipefail

if [[ "${ADAFLOOR_VANILLA_DRIVER_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    vanilla_driver_source=$(realpath "${BASH_SOURCE[0]}")
    vanilla_driver_snapshot=$(mktemp "${vanilla_driver_source}.run-snapshot.XXXXXX")
    cp -- "$vanilla_driver_source" "$vanilla_driver_snapshot"
    chmod 700 "$vanilla_driver_snapshot"
    set +e
    ADAFLOOR_VANILLA_DRIVER_SNAPSHOT_ACTIVE=1 \
        "$vanilla_driver_snapshot" "$@"
    vanilla_driver_rc=$?
    set -e
    rm -f -- "$vanilla_driver_snapshot"
    exit "$vanilla_driver_rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Run the full16 Vanilla control for sequential epochs 1 and 2 from the same
# public epoch0 checkpoint used by every paper policy. Epoch 2 resumes the
# checkpoint produced by Vanilla epoch 1.

COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
REUSE_ENV="$COMMON_EPOCH0_ROOT/reuse.env"
OUTPUT_ROOT="${FAIR_OUTPUT_ROOT:-/data/adafloor_shared_state/paper_fair_reruns_common_epoch0}"
RUN_NAME="${FAIR_RUN_NAME:-baseline_vanilla_common_epoch0_epoch1_2}"
RUN_ROOT="$OUTPUT_ROOT/$RUN_NAME"

MODEL_PATH="${MODEL_PATH:-/data/Qwen3-30B-A3B}"
DISTCP_PATH="${DISTCP_PATH:-/data/Qwen3-30B-A3B_megatron}"
TRAIN_FILE="${TRAIN_FILE:-/data/deepscaler/train.parquet}"
TEST_FILE="${TEST_FILE:-/data/deepscaler/test.parquet}"
DATASET_FRACTION="${FAIR_DATASET_FRACTION:-0.005}"
KV_TOKENS_PER_RANK="${VANILLA_KV_TOKENS_PER_RANK:-380800}"
KV_BLOCK_SIZE="${VLLM_KV_BLOCK_SIZE:-128}"
START_EPOCH="${DYNAMIC_START_EPOCH:-1}"
TOTAL_EPOCHS="${DYNAMIC_TOTAL_EPOCHS:-3}"
STEPS_PER_EPOCH="${DYNAMIC_TRAIN_STEPS:-5}"
TRAIN_BATCH_SIZE_EFFECTIVE="${FAIR_TRAIN_BATCH_SIZE:-32}"
ROLLOUT_N_EFFECTIVE="${FAIR_ROLLOUT_N:-16}"
MAX_RESPONSE_LENGTH_EFFECTIVE="${FAIR_MAX_RESPONSE_LENGTH:-16384}"
CHECKPOINT_MODEL_DIR_NAME="${CHECKPOINT_MODEL_DIR_NAME:-qwen3moe_for_eagle3}"
if [[ ! "$CHECKPOINT_MODEL_DIR_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid CHECKPOINT_MODEL_DIR_NAME=$CHECKPOINT_MODEL_DIR_NAME" >&2
    exit 2
fi
if ! [[ "$START_EPOCH" =~ ^[0-9]+$ && "$TOTAL_EPOCHS" =~ ^[0-9]+$ ]] \
   || (( START_EPOCH < 1 || TOTAL_EPOCHS <= START_EPOCH )); then
    echo "invalid Vanilla epoch interval: start=$START_EPOCH total=$TOTAL_EPOCHS" >&2
    exit 2
fi
if ! [[ "$STEPS_PER_EPOCH" =~ ^[1-9][0-9]*$ \
        && "$TRAIN_BATCH_SIZE_EFFECTIVE" =~ ^[1-9][0-9]*$ \
        && "$ROLLOUT_N_EFFECTIVE" =~ ^[1-9][0-9]*$ \
        && "$MAX_RESPONSE_LENGTH_EFFECTIVE" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid Vanilla workload setting" >&2
    exit 2
fi
MAX_BATCHED_TOKENS_EFFECTIVE=$((1024 + MAX_RESPONSE_LENGTH_EFFECTIVE))

if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
      || ! -f "$REUSE_ENV" ]]; then
    echo "common epoch0 is not complete: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "$REUSE_ENV"
resume_ckpt="${BASELINE_INITIAL_RESUME_CKPT:-}"
if [[ -z "$resume_ckpt" || ! -d "$resume_ckpt/actor" \
      || ! -f "$resume_ckpt/.PRESERVE_COMMON_EPOCH0" ]]; then
    echo "invalid preserved common epoch0 checkpoint: ${resume_ckpt:-<unset>}" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "refusing to overwrite fair Vanilla output: $RUN_ROOT" >&2
    exit 2
fi
if (( KV_TOKENS_PER_RANK <= 0 || KV_BLOCK_SIZE <= 0 \
      || KV_TOKENS_PER_RANK % KV_BLOCK_SIZE != 0 )); then
    echo "KV token capacity must be positive and divisible by block size" >&2
    exit 2
fi
KV_BLOCKS=$((KV_TOKENS_PER_RANK / KV_BLOCK_SIZE))

find_latest_checkpoint() {
    local epoch_dir="$1"
    find "$epoch_dir/checkpoints/$CHECKPOINT_MODEL_DIR_NAME" \
        -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
        | sort -V | tail -1
}

validate_epoch() {
    local epoch_dir="$1"
    local rollout_count length_count
    rollout_count=$(find "$epoch_dir/rollout_data" -maxdepth 1 \
        -type f -name '*.jsonl' 2>/dev/null | wc -l)
    length_count=$(find "$epoch_dir/rollout_length" -maxdepth 1 \
        -type f -name 'length_*.txt' 2>/dev/null | wc -l)
    if (( rollout_count != STEPS_PER_EPOCH \
          || length_count != STEPS_PER_EPOCH )); then
        echo "incomplete Vanilla epoch: dir=$epoch_dir rollout=$rollout_count length=$length_count" >&2
        exit 3
    fi
}

mkdir -p "$OUTPUT_ROOT"
echo "[fair Vanilla] common_epoch0=$COMMON_EPOCH0_ROOT"
echo "[fair Vanilla] initial_checkpoint=$resume_ckpt"
echo "[fair Vanilla] output=$RUN_ROOT epochs=${START_EPOCH}..$((TOTAL_EPOCHS - 1)) steps=$STEPS_PER_EPOCH"
echo "[fair Vanilla] gpu_memory_utilization=0.9 kv_tokens_per_rank=$KV_TOKENS_PER_RANK blocks=$KV_BLOCKS"

for (( epoch = START_EPOCH; epoch < TOTAL_EPOCHS; epoch++ )); do
    epoch_tag=$(printf '%03d' "$epoch")
    epoch_dir="$RUN_ROOT/epoch_${epoch_tag}_mode0_vanilla"
    echo "[fair Vanilla] epoch=$epoch resume=$resume_ckpt output=$epoch_dir"

    env \
        -u VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS \
        -u VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP \
        OUTPUT_ROOT="$OUTPUT_ROOT" \
        OUTPUT_SUBDIR="$RUN_NAME/epoch_${epoch_tag}_mode0_vanilla" \
        RECORD_DIR="$epoch_dir" \
        MODEL_PATH="$MODEL_PATH" \
        DISTCP_PATH="$DISTCP_PATH" \
        TRAIN_FILE="$TRAIN_FILE" \
        TEST_FILE="$TEST_FILE" \
        TRAINER_TOTAL_EPOCHS=1 \
        DATASET_FRACTION="$DATASET_FRACTION" \
        DATA_SHUFFLE=False \
        TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE_EFFECTIVE" \
        MAX_PROMPT_LENGTH=1024 \
        MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH_EFFECTIVE" \
        MAX_RESPONSE_LEN="$MAX_RESPONSE_LENGTH_EFFECTIVE" \
        ROLLOUT_MAX_NUM_BATCHED_TOKENS="$MAX_BATCHED_TOKENS_EFFECTIVE" \
        ROLLOUT_MAX_NUM_SEQS="$TRAIN_BATCH_SIZE_EFFECTIVE" \
        ROLLOUT_N="$ROLLOUT_N_EFFECTIVE" \
        ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 \
        MODE0_SAVE_ROLLOUT_ARTIFACTS=1 \
        SAVE_CKPT_ENABLE=1 \
        TRAINER_SAVE_FREQ="$STEPS_PER_EPOCH" \
        MAX_ACTOR_CKPT_TO_KEEP=1 \
        MAX_CRITIC_CKPT_TO_KEEP=1 \
        VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1 \
        "$SCRIPT_DIR/run_mode0_no_shrink_baseline.sh" \
            "trainer.total_training_steps=$STEPS_PER_EPOCH" \
            trainer.resume_mode=resume_path \
            "trainer.resume_from_path=$resume_ckpt" \
            +actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override="$KV_BLOCKS" \
            "$@"

    validate_epoch "$epoch_dir"
    resume_ckpt=$(find_latest_checkpoint "$epoch_dir")
    if [[ -z "$resume_ckpt" || ! -d "$resume_ckpt/actor" ]]; then
        echo "Vanilla epoch $epoch did not produce a resumable checkpoint" >&2
        exit 4
    fi
done

echo "[fair Vanilla] complete output=$RUN_ROOT"
