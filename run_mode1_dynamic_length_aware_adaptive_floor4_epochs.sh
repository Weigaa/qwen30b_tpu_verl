#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh [extra hydra args...]

Dynamic multi-epoch length-aware shrink driver:
  epoch 0: run mode=0/no-shrink, train normally, and collect rollout lengths.
  epoch N: rebuild an adaptive floor4 length-sorted plan from epoch N-1
           rollout lengths, then run mode=1 with natural or planned policy.

Key environment variables:
  DYNAMIC_TOTAL_EPOCHS=2        Total epochs including epoch0 probe.
  DYNAMIC_SHRINK_POLICY=planned planned or natural for mode=1 epochs.
  DYNAMIC_RUN_NAME=...          Output subdirectory prefix.
  DYNAMIC_SKIP_MODE0_PROBE=0    Set to 1 to start from DYNAMIC_INITIAL_BASELINE_DIR.
  DYNAMIC_INITIAL_BASELINE_DIR= Existing rollout directory when skipping probe.
  DYNAMIC_ENABLE_CKPT_CHAIN=1   Save/resume checkpoints across child epochs.
  DYNAMIC_LENGTH_EMA_DECAY=0.7  EMA decay for historical rollout lengths.
  DYNAMIC_ENABLE_THRESHOLD_CONTROL=0
                                 Keep short threshold/max-response overrides.
  DYNAMIC_TRAIN_STEPS=           Optional child-run training steps; keeps
                                 DYNAMIC_PLAN_STEPS unchanged for planning.
  TRAIN_FILE_ORIG=...           Original train parquet prompt pool.

The driver intentionally starts a fresh Python/vLLM process per epoch. This
keeps mode0/mode1 state, KV cache state, communication groups, and rank-plan
JSON caches isolated while still feeding each epoch's measured rollout lengths
into the next epoch's planner.
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

DYNAMIC_TOTAL_EPOCHS="${DYNAMIC_TOTAL_EPOCHS:-2}"
if (( DYNAMIC_TOTAL_EPOCHS < 2 )); then
    echo "DYNAMIC_TOTAL_EPOCHS must be at least 2: epoch0 probe + one mode1 epoch" >&2
    exit 2
fi

DYNAMIC_SHRINK_POLICY="${DYNAMIC_SHRINK_POLICY:-planned}"
case "$DYNAMIC_SHRINK_POLICY" in
    natural|planned|fixed|plan) ;;
    *)
        echo "unsupported DYNAMIC_SHRINK_POLICY=$DYNAMIC_SHRINK_POLICY; use natural or planned" >&2
        exit 2
        ;;
esac

DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_length_aware_adaptive_floor4}"
DYNAMIC_OUTPUT_ROOT="${DYNAMIC_OUTPUT_ROOT:-$REPO_ROOT}"
DYNAMIC_DATASET_FRACTION="${DYNAMIC_DATASET_FRACTION:-${DATASET_FRACTION_FOR_ORACLE:-0.005}}"
DYNAMIC_PLAN_STEPS="${DYNAMIC_PLAN_STEPS:-${PLAN_STEPS:-5}}"
DYNAMIC_TRAIN_STEPS="${DYNAMIC_TRAIN_STEPS:-}"
DYNAMIC_SKIP_MODE0_PROBE="${DYNAMIC_SKIP_MODE0_PROBE:-0}"
DYNAMIC_INITIAL_BASELINE_DIR="${DYNAMIC_INITIAL_BASELINE_DIR:-}"
DYNAMIC_ENABLE_CKPT_CHAIN="${DYNAMIC_ENABLE_CKPT_CHAIN:-1}"
DYNAMIC_INITIAL_RESUME_CKPT="${DYNAMIC_INITIAL_RESUME_CKPT:-}"
DYNAMIC_LENGTH_EMA_DECAY="${DYNAMIC_LENGTH_EMA_DECAY:-0.7}"
DYNAMIC_ENABLE_THRESHOLD_CONTROL="${DYNAMIC_ENABLE_THRESHOLD_CONTROL:-0}"

if [[ "$DYNAMIC_ENABLE_THRESHOLD_CONTROL" != "1" ]]; then
    # The dynamic planner must see and validate the true rollout-length
    # distribution.  Threshold smoke wrappers intentionally override this.
    unset VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS
    export MAX_PROMPT_LENGTH="${DYNAMIC_FULL_MAX_PROMPT_LENGTH:-1024}"
    export MAX_RESPONSE_LENGTH="${DYNAMIC_FULL_MAX_RESPONSE_LENGTH:-16384}"
    export MAX_RESPONSE_LEN="${DYNAMIC_FULL_MAX_RESPONSE_LEN:-$MAX_RESPONSE_LENGTH}"
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
fi

TRAIN_FILE_ORIG="${TRAIN_FILE_ORIG:-/data/deepscaler/train.parquet}"
TEST_FILE="${TEST_FILE:-/data/deepscaler/test.parquet}"
MODEL_PATH="${MODEL_PATH:-/data/Qwen3-30B-A3B}"
DISTCP_PATH="${DISTCP_PATH:-/data/Qwen3-30B-A3B_megatron}"

validate_rollout_dir() {
    local dir="$1"
    local label="$2"
    if [[ ! -d "$dir/rollout_data" ]]; then
        echo "$label missing rollout_data directory: $dir/rollout_data" >&2
        exit 3
    fi
    local count
    count=$(find "$dir/rollout_data" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
    if (( count < DYNAMIC_PLAN_STEPS )); then
        echo "$label has only $count rollout jsonl files, expected at least $DYNAMIC_PLAN_STEPS: $dir" >&2
        exit 3
    fi
}

find_latest_checkpoint() {
    local output_dir="$1"
    local ckpt_root="$output_dir/checkpoints/qwen3moe_for_eagle3"
    if [[ ! -d "$ckpt_root" ]]; then
        return 1
    fi
    find "$ckpt_root" -maxdepth 1 -type d -name 'global_step_*' \
        | sort -V \
        | tail -1
}

NEXT_ROLLOUT_DIR=""
NEXT_CKPT_PATH=""

run_mode0_probe() {
    local subdir="$DYNAMIC_RUN_NAME/epoch_000_mode0_probe"
    local record_dir="$DYNAMIC_OUTPUT_ROOT/$subdir"
    local save_ckpt_enable="${SAVE_CKPT_ENABLE:-0}"
    local train_steps="${DYNAMIC_TRAIN_STEPS:-$DYNAMIC_PLAN_STEPS}"
    local trainer_save_freq="${TRAINER_SAVE_FREQ:-$train_steps}"
    if [[ "$DYNAMIC_ENABLE_CKPT_CHAIN" == "1" ]]; then
        save_ckpt_enable=1
    fi

    echo "[dynamic length-aware] epoch=0 policy=mode0_no_shrink record_dir=$record_dir"
    OUTPUT_ROOT="$DYNAMIC_OUTPUT_ROOT" \
    OUTPUT_SUBDIR="$subdir" \
    RECORD_DIR="$record_dir" \
    TRAIN_FILE="$TRAIN_FILE_ORIG" \
    TEST_FILE="$TEST_FILE" \
    MODEL_PATH="$MODEL_PATH" \
    DISTCP_PATH="$DISTCP_PATH" \
    TRAINER_TOTAL_EPOCHS=1 \
    DATA_SHUFFLE=False \
    DATASET_FRACTION="$DYNAMIC_DATASET_FRACTION" \
    MODE0_SAVE_ROLLOUT_ARTIFACTS=1 \
    SAVE_CKPT_ENABLE="$save_ckpt_enable" \
    TRAINER_SAVE_FREQ="$trainer_save_freq" \
    "$REPO_ROOT/run_mode0_no_shrink_baseline.sh" "$@"

    validate_rollout_dir "$record_dir" "epoch0 probe"
    NEXT_ROLLOUT_DIR="$record_dir"
    NEXT_CKPT_PATH=""
    if [[ "$DYNAMIC_ENABLE_CKPT_CHAIN" == "1" ]]; then
        NEXT_CKPT_PATH=$(find_latest_checkpoint "$record_dir" || true)
        if [[ -z "$NEXT_CKPT_PATH" ]]; then
            echo "checkpoint chaining is enabled, but epoch0 wrote no checkpoint under $record_dir" >&2
            exit 4
        fi
        echo "[dynamic length-aware] epoch=0 latest_ckpt=$NEXT_CKPT_PATH"
    fi
}

run_mode1_epoch() {
    local epoch="$1"
    local baseline_dirs="$2"
    local resume_ckpt="$3"
    shift 3

    local epoch_tag
    epoch_tag=$(printf '%03d' "$epoch")
    local policy_slug="$DYNAMIC_SHRINK_POLICY"
    if [[ "$policy_slug" == "fixed" || "$policy_slug" == "plan" ]]; then
        policy_slug="planned"
    fi
    local subdir="$DYNAMIC_RUN_NAME/epoch_${epoch_tag}_mode1_${policy_slug}"
    local output_dir="$DYNAMIC_OUTPUT_ROOT/$subdir"
    local plan_dir="$output_dir/oracle"
    local child_args=("$@")
    local save_ckpt_enable="${SAVE_CKPT_ENABLE:-0}"
    local train_steps="${DYNAMIC_TRAIN_STEPS:-$DYNAMIC_PLAN_STEPS}"
    local trainer_save_freq="${TRAINER_SAVE_FREQ:-$train_steps}"

    if [[ "$DYNAMIC_ENABLE_CKPT_CHAIN" == "1" ]]; then
        save_ckpt_enable=1
        export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
        if [[ -n "$resume_ckpt" ]]; then
            child_args+=("trainer.resume_mode=resume_path")
            child_args+=("trainer.resume_from_path=$resume_ckpt")
        fi
    fi

    echo "[dynamic length-aware] epoch=$epoch policy=$DYNAMIC_SHRINK_POLICY baseline_dirs=$baseline_dirs output_dir=$output_dir"
    if [[ -n "$resume_ckpt" ]]; then
        echo "[dynamic length-aware] epoch=$epoch resume_ckpt=$resume_ckpt"
    fi
    OUTPUT_ROOT="$DYNAMIC_OUTPUT_ROOT" \
    OUTPUT_SUBDIR="$subdir" \
    BASELINE_DIRS="$baseline_dirs" \
    PLAN_DIR="$plan_dir" \
    TRAIN_FILE_ORIG="$TRAIN_FILE_ORIG" \
    TEST_FILE="$TEST_FILE" \
    MODEL_PATH="$MODEL_PATH" \
    DISTCP_PATH="$DISTCP_PATH" \
    TRAINER_TOTAL_EPOCHS=1 \
    DATASET_FRACTION_FOR_ORACLE="$DYNAMIC_DATASET_FRACTION" \
    LENGTH_EMA_DECAY="$DYNAMIC_LENGTH_EMA_DECAY" \
    PLAN_STEPS="$DYNAMIC_PLAN_STEPS" \
    SAVE_CKPT_ENABLE="$save_ckpt_enable" \
    TRAINER_SAVE_FREQ="$trainer_save_freq" \
    VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY="$DYNAMIC_SHRINK_POLICY" \
    VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME="${DYNAMIC_RESET_PROGRESS_AFTER_RESUME:-1}" \
    "$REPO_ROOT/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh" \
        "trainer.total_training_steps=$train_steps" "${child_args[@]}"

    validate_rollout_dir "$output_dir" "epoch $epoch mode1"
    NEXT_ROLLOUT_DIR="$output_dir"
    NEXT_CKPT_PATH=""
    if [[ "$DYNAMIC_ENABLE_CKPT_CHAIN" == "1" ]]; then
        NEXT_CKPT_PATH=$(find_latest_checkpoint "$output_dir" || true)
        if [[ -z "$NEXT_CKPT_PATH" ]]; then
            echo "checkpoint chaining is enabled, but no checkpoint was written under $output_dir" >&2
            exit 4
        fi
        echo "[dynamic length-aware] epoch=$epoch latest_ckpt=$NEXT_CKPT_PATH"
    fi
}

mkdir -p "$DYNAMIC_OUTPUT_ROOT/$DYNAMIC_RUN_NAME"

echo "[dynamic length-aware] total_epochs=$DYNAMIC_TOTAL_EPOCHS mode1_policy=$DYNAMIC_SHRINK_POLICY run_name=$DYNAMIC_RUN_NAME"
echo "[dynamic length-aware] dataset_fraction=$DYNAMIC_DATASET_FRACTION plan_steps=$DYNAMIC_PLAN_STEPS train=$TRAIN_FILE_ORIG"
echo "[dynamic length-aware] train_steps=${DYNAMIC_TRAIN_STEPS:-<plan_steps>}"
echo "[dynamic length-aware] checkpoint_chain=$DYNAMIC_ENABLE_CKPT_CHAIN length_ema_decay=$DYNAMIC_LENGTH_EMA_DECAY threshold_control=$DYNAMIC_ENABLE_THRESHOLD_CONTROL"
echo "[dynamic length-aware] max_prompt=$MAX_PROMPT_LENGTH max_response=$MAX_RESPONSE_LENGTH max_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS tail_validate=${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-<unset>}"

previous_rollout_dir=""
baseline_history=""
if [[ "$DYNAMIC_SKIP_MODE0_PROBE" == "1" ]]; then
    if [[ -z "$DYNAMIC_INITIAL_BASELINE_DIR" ]]; then
        echo "DYNAMIC_SKIP_MODE0_PROBE=1 requires DYNAMIC_INITIAL_BASELINE_DIR" >&2
        exit 2
    fi
    previous_rollout_dir="$DYNAMIC_INITIAL_BASELINE_DIR"
    validate_rollout_dir "$previous_rollout_dir" "initial baseline"
    baseline_history="$previous_rollout_dir"
else
    run_mode0_probe "$@"
    previous_rollout_dir="$NEXT_ROLLOUT_DIR"
    baseline_history="$previous_rollout_dir"
    DYNAMIC_INITIAL_RESUME_CKPT="$NEXT_CKPT_PATH"
fi

for (( epoch = 1; epoch < DYNAMIC_TOTAL_EPOCHS; epoch++ )); do
    run_mode1_epoch "$epoch" "$baseline_history" "$DYNAMIC_INITIAL_RESUME_CKPT" "$@"
    previous_rollout_dir="$NEXT_ROLLOUT_DIR"
    baseline_history="${baseline_history}:$previous_rollout_dir"
    DYNAMIC_INITIAL_RESUME_CKPT="$NEXT_CKPT_PATH"
done

echo "[dynamic length-aware] done final_rollout_dir=$previous_rollout_dir"
