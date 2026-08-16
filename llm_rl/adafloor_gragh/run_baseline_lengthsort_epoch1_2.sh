#!/usr/bin/env bash
set -euo pipefail

if [[ "${ADAFLOOR_LENGTHSORT_DRIVER_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    lengthsort_driver_source=$(realpath "${BASH_SOURCE[0]}")
    lengthsort_driver_snapshot=$(mktemp "${lengthsort_driver_source}.run-snapshot.XXXXXX")
    cp -- "$lengthsort_driver_source" "$lengthsort_driver_snapshot"
    chmod 700 "$lengthsort_driver_snapshot"
    set +e
    ADAFLOOR_LENGTHSORT_DRIVER_SNAPSHOT_ACTIVE=1 \
        "$lengthsort_driver_snapshot" "$@"
    lengthsort_driver_rc=$?
    set -e
    rm -f -- "$lengthsort_driver_snapshot"
    exit "$lengthsort_driver_rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Pure LengthSort baseline for the main policy comparison.
#
# Each epoch rebuilds five contiguous 32-prompt buckets from historical
# response lengths. Within each bucket, prompts remain in predicted length
# order. Rollout runs in mode0 on all 16 ranks, so there is no elastic shrink
# or optimized rank-plan execution. If adjacent long prompts exceed the fixed
# KV budget, vLLM is allowed to preempt them and the cost remains in the result.
# The final Vanilla KV capacity and the common tail-guard policy are retained.

TRAIN_FILE_ORIG="${TRAIN_FILE_ORIG:-/data/deepscaler/train.parquet}"
TEST_FILE="${TEST_FILE:-/data/deepscaler/test.parquet}"
MODEL_PATH="${MODEL_PATH:-/data/Qwen3-30B-A3B}"
DISTCP_PATH="${DISTCP_PATH:-/data/Qwen3-30B-A3B_megatron}"
BASELINE_EPOCH0="${DYNAMIC_INITIAL_BASELINE_DIR:-$SCRIPT_DIR/mode1_dynamic_length_aware_adaptive_floor4_natural_tailguard_full3/epoch_000_mode0_probe}"
OUTPUT_ROOT="${DYNAMIC_OUTPUT_ROOT:-$SCRIPT_DIR}"
DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-baseline_lengthsort_tailguard_reuse_epoch0_2epoch}"
OUTPUT_DIR="$OUTPUT_ROOT/$DYNAMIC_RUN_NAME"

PLAN_STEPS="${DYNAMIC_PLAN_STEPS:-5}"
TRAIN_STEPS="${DYNAMIC_TRAIN_STEPS:-5}"
BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-$BATCH_SIZE}"
DATASET_FRACTION="${DYNAMIC_DATASET_FRACTION:-0.005}"
LENGTH_EMA_DECAY="${DYNAMIC_LENGTH_EMA_DECAY:-0.3}"
ACTIVE_PEAK_SAFETY_FACTOR="${BASELINE_ACTIVE_PEAK_SAFETY_FACTOR:-1.16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
ROLLOUT_N="${ROLLOUT_N:-16}"
KV_TOKENS_PER_RANK="${VANILLA_KV_TOKENS_PER_RANK:-380800}"
KV_ADMISSION_TOKENS_PER_RANK="${VANILLA_KV_ADMISSION_TOKENS_PER_RANK:-$KV_TOKENS_PER_RANK}"
KV_BLOCK_SIZE="${VLLM_KV_BLOCK_SIZE:-128}"
ENABLE_TAIL_GUARD="${BASELINE_ENABLE_TAIL_GUARD:-1}"
ALLOW_INFEASIBLE_PLAN="${BASELINE_ALLOW_INFEASIBLE_PLAN:-1}"
TAIL_GUARD_RATIO_QUANTILE="${DYNAMIC_TAIL_GUARD_RATIO_QUANTILE:-0.95}"
TAIL_GUARD_RATIO_WINDOW="${DYNAMIC_TAIL_GUARD_RATIO_WINDOW:-3}"
TAIL_GUARD_DEFAULT_RATIO="${DYNAMIC_TAIL_GUARD_DEFAULT_RATIO:-1.20}"
TAIL_GUARD_MIN_CAP="${DYNAMIC_TAIL_GUARD_MIN_CAP:-4096}"
TAIL_GUARD_ROUND_TO="${DYNAMIC_TAIL_GUARD_ROUND_TO:-512}"
START_EPOCH="${DYNAMIC_START_EPOCH:-1}"
TOTAL_EPOCHS="${DYNAMIC_TOTAL_EPOCHS:-3}"
CHECKPOINT_MODEL_DIR_NAME="${CHECKPOINT_MODEL_DIR_NAME:-qwen3moe_for_eagle3}"
if [[ ! "$CHECKPOINT_MODEL_DIR_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid CHECKPOINT_MODEL_DIR_NAME=$CHECKPOINT_MODEL_DIR_NAME" >&2
    exit 2
fi

if ! [[ "$START_EPOCH" =~ ^[0-9]+$ && "$TOTAL_EPOCHS" =~ ^[0-9]+$ ]] \
   || (( START_EPOCH < 1 || TOTAL_EPOCHS <= START_EPOCH )); then
    echo "invalid LengthSort epoch interval: start=$START_EPOCH total=$TOTAL_EPOCHS" >&2
    exit 2
fi
if [[ "$ALLOW_INFEASIBLE_PLAN" != 0 && "$ALLOW_INFEASIBLE_PLAN" != 1 ]]; then
    echo "BASELINE_ALLOW_INFEASIBLE_PLAN must be 0 or 1" >&2
    exit 2
fi

FIXED_WORK_ACTIVE=0
fixed_contract_fields=(
    "${DEEPSEEK_FIXED_WORK_LAUNCH_PROTOCOL:-}"
    "${DEEPSEEK_FIXED_WORK_PHASE:-}"
    "${DEEPSEEK_FIXED_WORK_EXPECTED_STEPS:-}"
    "${DEEPSEEK_FIXED_WORK_EXECUTION_CODE_SHA256:-}"
    "${VERL_FIXED_WORK_REPLAY_TRACE:-}"
    "${VERL_FIXED_WORK_REPLAY_SHA256:-}"
)
for fixed_contract_field in "${fixed_contract_fields[@]}"; do
    if [[ -n "$fixed_contract_field" ]]; then
        FIXED_WORK_ACTIVE=1
        break
    fi
done
if (( FIXED_WORK_ACTIVE == 1 )); then
    if [[ "${DEEPSEEK_FIXED_WORK_LAUNCH_PROTOCOL:-}" \
          != deepseek_batch64_fixed_work_replay_v3 ]]; then
        echo "invalid DeepSeek fixed-work launch protocol" >&2
        exit 2
    fi
    case "${DEEPSEEK_FIXED_WORK_PHASE:-}" in
        gate) fixed_expected_steps=1 ;;
        epoch) fixed_expected_steps=5 ;;
        *) echo "DeepSeek fixed-work phase must be gate or epoch" >&2; exit 2 ;;
    esac
    if [[ "${DEEPSEEK_FIXED_WORK_EXPECTED_STEPS:-}" != "$fixed_expected_steps" ]] \
       || (( PLAN_STEPS != fixed_expected_steps \
             || TRAIN_STEPS != fixed_expected_steps )); then
        echo "DeepSeek fixed-work step contract is inconsistent" >&2
        exit 2
    fi
    if (( BATCH_SIZE != 64 || MAX_NUM_SEQS != 64 || ROLLOUT_N != 16 \
          || START_EPOCH != 1 || TOTAL_EPOCHS != 2 )); then
        echo "DeepSeek fixed-work workload must use batch64, n=16, max_num_seqs=64, and epoch1" >&2
        exit 2
    fi
    if [[ "${DEEPSEEK_KV_CAP_TRAIN_BATCH_SIZE:-}" != 64 \
          || "${DEEPSEEK_KV_CAP_ROLLOUT_N:-}" != 16 \
          || "${DEEPSEEK_KV_CAP_MAX_NUM_SEQS:-}" != 64 \
          || "${DEEPSEEK_KV_CAP_EXPECTED_RESPONSES_PER_STEP:-}" != 1024 ]]; then
        echo "DeepSeek fixed-work KV contract is not the authorized batch64 profile" >&2
        exit 2
    fi
    fixed_execution_sha256=$(python3 \
        "$SCRIPT_DIR/tools/hash_deepseek_execution_code.py" --root "$SCRIPT_DIR")
    if [[ "${DEEPSEEK_FIXED_WORK_EXECUTION_CODE_SHA256:-}" \
          != "$fixed_execution_sha256" ]]; then
        echo "DeepSeek fixed-work execution hash is stale" >&2
        exit 2
    fi
    if [[ "${VERL_FIXED_WORK_REPLAY_REQUIRE_PLAN_CAP:-}" != 0 ]]; then
        echo "LengthSort fixed-work replay must not require an AdaFloor runtime cap" >&2
        exit 2
    fi
    python3 - "${VERL_FIXED_WORK_REPLAY_TRACE:-}" \
        "${VERL_FIXED_WORK_REPLAY_SHA256:-}" "$fixed_expected_steps" <<'PY_FIXED_WORK'
import sys

from verl.utils.fixed_work_replay import load_fixed_work_replay

trace = load_fixed_work_replay(sys.argv[1], expected_sha256=sys.argv[2])
expected_steps = int(sys.argv[3])
if trace.steps != tuple(range(1, expected_steps + 1)):
    raise SystemExit(
        f"fixed-work trace steps={trace.steps}, expected 1..{expected_steps}"
    )
if trace.record_count != expected_steps * 1024:
    raise SystemExit(
        f"fixed-work trace records={trace.record_count}, "
        f"expected {expected_steps * 1024}"
    )
PY_FIXED_WORK
elif (( PLAN_STEPS != 5 || TRAIN_STEPS != 5 || BATCH_SIZE != 32 \
        || MAX_NUM_SEQS != 32 )); then
    echo "paper LengthSort baseline requires five steps, batch32, and max_num_seqs=32" >&2
    exit 2
fi
if (( KV_TOKENS_PER_RANK <= 0 || KV_BLOCK_SIZE <= 0 \
      || KV_TOKENS_PER_RANK % KV_BLOCK_SIZE != 0 )); then
    echo "KV_TOKENS_PER_RANK must be positive and divisible by KV_BLOCK_SIZE" >&2
    exit 2
fi
if (( KV_ADMISSION_TOKENS_PER_RANK <= 0 \
      || KV_ADMISSION_TOKENS_PER_RANK > KV_TOKENS_PER_RANK \
      || KV_ADMISSION_TOKENS_PER_RANK % KV_BLOCK_SIZE != 0 )); then
    echo "KV admission capacity must be a positive block-aligned value no larger than physical capacity" >&2
    exit 2
fi
KV_BLOCKS=$((KV_TOKENS_PER_RANK / KV_BLOCK_SIZE))

for required_path in \
    "$TRAIN_FILE_ORIG" "$TEST_FILE" "$MODEL_PATH" "$DISTCP_PATH"; do
    if [[ ! -e "$required_path" ]]; then
        echo "missing input: $required_path" >&2
        exit 2
    fi
done
if [[ ! -d "$BASELINE_EPOCH0/rollout_data" ]]; then
    echo "missing reusable epoch0 rollout data: $BASELINE_EPOCH0/rollout_data" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
    echo "output already exists: $OUTPUT_DIR" >&2
    echo "Set DYNAMIC_RUN_NAME to a new directory or ALLOW_EXISTING_OUTPUT=1." >&2
    exit 2
fi

find_latest_checkpoint() {
    local epoch_dir="$1"
    find "$epoch_dir/checkpoints/$CHECKPOINT_MODEL_DIR_NAME" \
        -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
        | sort -V | tail -1
}

validate_rollout_dir() {
    local epoch_dir="$1"
    local count
    count=$(find "$epoch_dir/rollout_data" -maxdepth 1 -type f -name '*.jsonl' 2>/dev/null | wc -l)
    if (( count != TRAIN_STEPS )); then
        echo "expected $TRAIN_STEPS rollout artifacts under $epoch_dir, found $count" >&2
        exit 3
    fi
}

mkdir -p "$OUTPUT_DIR"
history="$BASELINE_EPOCH0"
resume_ckpt="${BASELINE_INITIAL_RESUME_CKPT:-}"

echo "[LengthSort baseline] output=$OUTPUT_DIR"
echo "[LengthSort baseline] reusable_epoch0=$BASELINE_EPOCH0"
echo "[LengthSort baseline] epochs=${START_EPOCH}..$((TOTAL_EPOCHS - 1)) steps_per_epoch=$TRAIN_STEPS batch_size=$BATCH_SIZE mode=0 matching=contiguous allow_kv_preemption=true"
echo "[LengthSort baseline] gpu_memory_utilization=0.9 physical_kv_tokens_per_rank=$KV_TOKENS_PER_RANK admission_kv_tokens_per_rank=$KV_ADMISSION_TOKENS_PER_RANK num_gpu_blocks_override=$KV_BLOCKS"
echo "[LengthSort baseline] tail_guard=$ENABLE_TAIL_GUARD ema_decay=$LENGTH_EMA_DECAY allow_infeasible_plan=$ALLOW_INFEASIBLE_PLAN"

for (( epoch = START_EPOCH; epoch < TOTAL_EPOCHS; epoch++ )); do
    epoch_tag=$(printf '%03d' "$epoch")
    epoch_dir="$OUTPUT_DIR/epoch_${epoch_tag}_mode0_lengthsort"
    plan_dir="$epoch_dir/oracle"
    train_file="$plan_dir/length_sorted_train.parquet"
    rank_plan="$plan_dir/length_sorted_rank_plan.json"
    summary_file="$plan_dir/length_sorted_rank_plan_summary.json"
    oracle_file="$plan_dir/length_sorted_length_oracle.json"
    mkdir -p "$plan_dir"

    baseline_args=()
    IFS=':' read -r -a history_dirs <<< "$history"
    for history_dir in "${history_dirs[@]}"; do
        if [[ ! -d "$history_dir/rollout_data" ]]; then
            echo "missing rollout history: $history_dir/rollout_data" >&2
            exit 2
        fi
        baseline_args+=(--baseline-dir "$history_dir")
    done

    guard_quantile="$TAIL_GUARD_RATIO_QUANTILE"
    guard_default="$TAIL_GUARD_DEFAULT_RATIO"
    guard_min_cap="$TAIL_GUARD_MIN_CAP"
    if [[ "$ENABLE_TAIL_GUARD" != "1" ]]; then
        guard_quantile=1.0
        guard_default=999
        guard_min_cap="$MAX_RESPONSE_LENGTH"
    fi

    planner_extra_args=()
    if [[ -n "${PLANNER_TOKENIZER_PATH:-}" ]]; then
        planner_extra_args+=(--tokenizer-path "$PLANNER_TOKENIZER_PATH")
    fi
    if [[ "$ALLOW_INFEASIBLE_PLAN" == 1 ]]; then
        planner_extra_args+=(--allow-infeasible)
    fi

    echo "[LengthSort baseline] epoch=$epoch planning history=$history"
    python3 -u "$SCRIPT_DIR/tools/build_mode1_length_sorted_e2e_plan.py" \
        "${baseline_args[@]}" \
        "${planner_extra_args[@]}" \
        --length-ema-decay "$LENGTH_EMA_DECAY" \
        --train-file "$TRAIN_FILE_ORIG" \
        --output-train "$train_file" \
        --output-plan "$rank_plan" \
        --output-summary "$summary_file" \
        --output-oracle "$oracle_file" \
        --steps "$PLAN_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --responses-per-prompt "$ROLLOUT_N" \
        --dataset-fraction "$DATASET_FRACTION" \
        --max-rank-peak-tokens "$KV_ADMISSION_TOKENS_PER_RANK" \
        --adaptive-floor \
        --force-selected-floor 16 \
        --floor-kv-caps "16:$KV_ADMISSION_TOKENS_PER_RANK" \
        --rank-matching-policy contiguous \
        --active-peak-safety-factor "$ACTIVE_PEAK_SAFETY_FACTOR" \
        --max-response-len "$MAX_RESPONSE_LENGTH" \
        --tail-guard-ratio-quantile "$guard_quantile" \
        --tail-guard-ratio-window "$TAIL_GUARD_RATIO_WINDOW" \
        --tail-guard-default-ratio "$guard_default" \
        --tail-guard-min-cap "$guard_min_cap" \
        --tail-guard-round-to "$TAIL_GUARD_ROUND_TO" \
        --max-cross-step-repair-swaps 0 \
        --repair-candidate-limit 1

    step_caps=""
    if [[ "$ENABLE_TAIL_GUARD" == "1" ]]; then
        step_caps=$(python3 - "$summary_file" "$PLAN_STEPS" <<'PY'
import json
import sys
from pathlib import Path

steps = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_steps = int(sys.argv[2])
if len(steps) != expected_steps:
    raise SystemExit(f"expected {expected_steps} plan steps, got {len(steps)}")
if any(step.get("rank_matching_policy") != "contiguous" for step in steps):
    raise SystemExit("LengthSort plan did not preserve contiguous matching")
if any(int(step.get("selected_floor", -1)) != 16 for step in steps):
    raise SystemExit("LengthSort plan unexpectedly selected a shrink floor")
print(";".join(
    ",".join([str(int(step["tail_guard_response_cap"]))] * 16)
    for step in steps
))
PY
)
    fi

    child_args=(trainer.resume_mode=disable)
    if [[ -n "$resume_ckpt" ]]; then
        child_args=(
            trainer.resume_mode=resume_path
            "trainer.resume_from_path=$resume_ckpt"
        )
    fi

    echo "[LengthSort baseline] epoch=$epoch output=$epoch_dir resume=${resume_ckpt:-<base-model>}"
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    OUTPUT_SUBDIR="$DYNAMIC_RUN_NAME/epoch_${epoch_tag}_mode0_lengthsort" \
    RECORD_DIR="$epoch_dir" \
    TRAIN_FILE="$train_file" \
    TEST_FILE="$TEST_FILE" \
    MODEL_PATH="$MODEL_PATH" \
    DISTCP_PATH="$DISTCP_PATH" \
    TRAINER_TOTAL_EPOCHS=1 \
    DATASET_FRACTION=1.0 \
    DATA_SHUFFLE=False \
    TRAIN_BATCH_SIZE="$BATCH_SIZE" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
    MAX_RESPONSE_LEN="$MAX_RESPONSE_LENGTH" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    ROLLOUT_MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    ROLLOUT_N="$ROLLOUT_N" \
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 \
    VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP="$step_caps" \
    MODE0_SAVE_ROLLOUT_ARTIFACTS=1 \
    SAVE_CKPT_ENABLE=1 \
    TRAINER_SAVE_FREQ="$TRAIN_STEPS" \
    MAX_ACTOR_CKPT_TO_KEEP=1 \
    MAX_CRITIC_CKPT_TO_KEEP=1 \
    VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1 \
    "$SCRIPT_DIR/run_mode0_no_shrink_baseline.sh" \
        trainer.total_training_steps="$TRAIN_STEPS" \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override="$KV_BLOCKS" \
        "${child_args[@]}" \
        "$@"

    validate_rollout_dir "$epoch_dir"
    resume_ckpt=$(find_latest_checkpoint "$epoch_dir")
    if [[ -z "$resume_ckpt" ]]; then
        echo "epoch $epoch did not write a resumable checkpoint" >&2
        exit 4
    fi
    history="$history:$epoch_dir"
done

last_epoch_tag=$(printf '%03d' "$((TOTAL_EPOCHS - 1))")
echo "[LengthSort baseline] done final_rollout_dir=$OUTPUT_DIR/epoch_${last_epoch_tag}_mode0_lengthsort"
