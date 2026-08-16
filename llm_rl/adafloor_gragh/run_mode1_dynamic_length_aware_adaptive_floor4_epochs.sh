#!/usr/bin/env bash
set -euo pipefail

# Protect the epoch loop and its per-epoch validation from live edits to this
# file while a multi-hour experiment is running.
if [[ "${ADAFLOOR_DYNAMIC_DRIVER_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    dynamic_driver_source=$(realpath "${BASH_SOURCE[0]}")
    dynamic_driver_snapshot=$(mktemp "${dynamic_driver_source}.run-snapshot.XXXXXX")
    cp -- "$dynamic_driver_source" "$dynamic_driver_snapshot"
    chmod 700 "$dynamic_driver_snapshot"
    set +e
    ADAFLOOR_DYNAMIC_DRIVER_SNAPSHOT_ACTIVE=1 \
        "$dynamic_driver_snapshot" "$@"
    dynamic_driver_rc=$?
    set -e
    rm -f -- "$dynamic_driver_snapshot"
    exit "$dynamic_driver_rc"
fi

if [[ "${DYNAMIC_DRIVER_XTRACE:-0}" == "1" ]]; then
    set -x
fi

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
  DYNAMIC_START_EPOCH=1         First mode1 epoch index when skipping mode0.
  DYNAMIC_INITIAL_BASELINE_DIR= Existing rollout directory when skipping probe.
  DYNAMIC_ENABLE_CKPT_CHAIN=1   Save/resume checkpoints across child epochs.
  DYNAMIC_LENGTH_EMA_DECAY=0.3  EMA decay for historical rollout lengths.
  DYNAMIC_ENABLE_THRESHOLD_CONTROL=0
                                 Keep short threshold/max-response overrides.
  DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS=
                                 Optional runtime-only tail validation caps.
                                 This does not change offline length planning.
  DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP=
                                 Optional semicolon-separated per-step runtime
                                 caps, e.g. "64,64,32,32;64,64,32,32;32,32,32".
  DYNAMIC_SHORT_STEP_CAP_ENABLE=0
                                 Optional runtime guard for predicted-short
                                 shrink steps. When enabled, only selected
                                 floors in DYNAMIC_SHORT_STEP_CAP_FLOORS are
                                 capped if predicted step exit is below
                                 DYNAMIC_SHORT_STEP_EXIT_THRESHOLD.
  DYNAMIC_TRAIN_STEPS=           Optional child-run training steps; keeps
                                 DYNAMIC_PLAN_STEPS unchanged for planning.
  DYNAMIC_BUILD_OFFLINE_PLANNING_HISTORY=1
                                 Set to 0 for stress runs that intentionally
                                 repeat prompts and do not feed a later epoch.
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

# Planned floor groups retain dispatcher and expert runtime state across
# rollout transitions. Keep the verified shape-safe cleanup path mandatory at
# the shared driver layer so direct wrappers and resumed runs cannot silently
# inherit an unsafe shell environment.
if [[ "$DYNAMIC_SHRINK_POLICY" == "planned" || "$DYNAMIC_SHRINK_POLICY" == "plan" ]]; then
    require_explicit_planned_guard="${DYNAMIC_REQUIRE_EXPLICIT_PLANNED_MEMORY_GUARD:-0}"
    if [[ "$require_explicit_planned_guard" != 0 \
          && "$require_explicit_planned_guard" != 1 ]]; then
        echo "DYNAMIC_REQUIRE_EXPLICIT_PLANNED_MEMORY_GUARD must be 0 or 1" >&2
        exit 2
    fi
    planned_min_free_floor_mib="${VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB_FLOOR:-${FAIR_PLANNED_MIN_FREE_MIB_FLOOR:-28672}}"
    planned_min_free_mib="${VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB:-$planned_min_free_floor_mib}"
    if ! [[ "$planned_min_free_floor_mib" =~ ^[0-9]+$ \
            && "$planned_min_free_mib" =~ ^[0-9]+$ ]]; then
        echo "planned training HBM reserves must be nonnegative integers" >&2
        exit 2
    fi
    if [[ "$require_explicit_planned_guard" == 1 ]] \
       && (( planned_min_free_floor_mib <= 0 || planned_min_free_mib <= 0 )); then
        echo "this Planned runtime profile requires an explicitly measured positive training HBM reserve" >&2
        exit 2
    fi
    if (( planned_min_free_mib < planned_min_free_floor_mib )); then
        echo "[dynamic length-aware] raising planned training HBM reserve from ${planned_min_free_mib} MiB to ${planned_min_free_floor_mib} MiB" >&2
        planned_min_free_mib=$planned_min_free_floor_mib
    fi
    export VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING=1
    export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1
    export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=1
    export VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1
    export VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0
    export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB="$planned_min_free_mib"
fi

DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_length_aware_adaptive_floor4}"
DYNAMIC_OUTPUT_ROOT="${DYNAMIC_OUTPUT_ROOT:-$REPO_ROOT}"
DYNAMIC_DATASET_FRACTION="${DYNAMIC_DATASET_FRACTION:-${DATASET_FRACTION_FOR_ORACLE:-0.005}}"
DYNAMIC_PLAN_STEPS="${DYNAMIC_PLAN_STEPS:-${PLAN_STEPS:-5}}"
DYNAMIC_TRAIN_STEPS="${DYNAMIC_TRAIN_STEPS:-}"
DYNAMIC_SKIP_MODE0_PROBE="${DYNAMIC_SKIP_MODE0_PROBE:-0}"
DYNAMIC_START_EPOCH="${DYNAMIC_START_EPOCH:-1}"
DYNAMIC_INITIAL_BASELINE_DIR="${DYNAMIC_INITIAL_BASELINE_DIR:-}"
DYNAMIC_ENABLE_CKPT_CHAIN="${DYNAMIC_ENABLE_CKPT_CHAIN:-1}"
DYNAMIC_INITIAL_RESUME_CKPT="${DYNAMIC_INITIAL_RESUME_CKPT:-}"
DYNAMIC_LENGTH_EMA_DECAY="${DYNAMIC_LENGTH_EMA_DECAY:-0.3}"
DYNAMIC_ENABLE_THRESHOLD_CONTROL="${DYNAMIC_ENABLE_THRESHOLD_CONTROL:-0}"
DYNAMIC_IGNORE_TAIL_TIES_AT_RESPONSE_CAP="${DYNAMIC_IGNORE_TAIL_TIES_AT_RESPONSE_CAP:-auto}"
DYNAMIC_NATURAL_FLOOR8_RUNTIME_CAP="${DYNAMIC_NATURAL_FLOOR8_RUNTIME_CAP:-315648}"
DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS="${DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS:-}"
DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP="${DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP:-}"
DYNAMIC_SHORT_STEP_CAP_ENABLE="${DYNAMIC_SHORT_STEP_CAP_ENABLE:-0}"
DYNAMIC_SHORT_STEP_EXIT_THRESHOLD="${DYNAMIC_SHORT_STEP_EXIT_THRESHOLD:-4096}"
DYNAMIC_SHORT_STEP_CAP_TOKENS="${DYNAMIC_SHORT_STEP_CAP_TOKENS:-4096}"
DYNAMIC_SHORT_STEP_CAP_FLOORS="${DYNAMIC_SHORT_STEP_CAP_FLOORS:-4}"
DYNAMIC_TAIL_GUARD_RATIO_QUANTILE="${DYNAMIC_TAIL_GUARD_RATIO_QUANTILE:-0.95}"
DYNAMIC_TAIL_GUARD_RATIO_WINDOW="${DYNAMIC_TAIL_GUARD_RATIO_WINDOW:-3}"
DYNAMIC_TAIL_GUARD_DEFAULT_RATIO="${DYNAMIC_TAIL_GUARD_DEFAULT_RATIO:-1.20}"
DYNAMIC_TAIL_GUARD_MIN_CAP="${DYNAMIC_TAIL_GUARD_MIN_CAP:-4096}"
DYNAMIC_TAIL_GUARD_ROUND_TO="${DYNAMIC_TAIL_GUARD_ROUND_TO:-512}"
DYNAMIC_DISABLE_TAIL_GUARD="${DYNAMIC_DISABLE_TAIL_GUARD:-0}"
DYNAMIC_EXPECT_NO_RESPONSE_CAPS="${DYNAMIC_EXPECT_NO_RESPONSE_CAPS:-0}"
DYNAMIC_MODE1_CHILD_SCRIPT="${DYNAMIC_MODE1_CHILD_SCRIPT:-}"
DYNAMIC_BUILD_OFFLINE_PLANNING_HISTORY="${DYNAMIC_BUILD_OFFLINE_PLANNING_HISTORY:-1}"
CHECKPOINT_MODEL_DIR_NAME="${CHECKPOINT_MODEL_DIR_NAME:-qwen3moe_for_eagle3}"
if [[ ! "$CHECKPOINT_MODEL_DIR_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid CHECKPOINT_MODEL_DIR_NAME=$CHECKPOINT_MODEL_DIR_NAME" >&2
    exit 2
fi

case "$DYNAMIC_BUILD_OFFLINE_PLANNING_HISTORY" in
    0|1) ;;
    *)
        echo "DYNAMIC_BUILD_OFFLINE_PLANNING_HISTORY must be 0 or 1" >&2
        exit 2
        ;;
esac

if [[ "$DYNAMIC_SHRINK_POLICY" == "natural" && -z "${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8+x}" ]]; then
    # Natural policy does not keep planned floor groups resident, but after
    # real floor4 shrink/warmup the NPU runtime workspace still lowers the
    # observed floor8 KV capacity.  Use the measured per-rank minimum from the
    # full natural run instead of the optimistic cold-profile cap.
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8="$DYNAMIC_NATURAL_FLOOR8_RUNTIME_CAP"
fi

if [[ "$DYNAMIC_ENABLE_THRESHOLD_CONTROL" != "1" ]]; then
    # The dynamic planner must see and validate the true rollout-length
    # distribution.  Threshold smoke wrappers intentionally override this.
    if [[ -n "$DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS" ]]; then
        export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="$DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS"
    else
        unset VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS
    fi
    if [[ -n "$DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP" ]]; then
        export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP="$DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP"
    else
        unset VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP
    fi
    export MAX_PROMPT_LENGTH="${DYNAMIC_FULL_MAX_PROMPT_LENGTH:-1024}"
    export MAX_RESPONSE_LENGTH="${DYNAMIC_FULL_MAX_RESPONSE_LENGTH:-16384}"
    export MAX_RESPONSE_LEN="${DYNAMIC_FULL_MAX_RESPONSE_LEN:-$MAX_RESPONSE_LENGTH}"
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
else
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${MAX_RESPONSE_LEN:-4096}}"
    export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-$MAX_RESPONSE_LENGTH}"
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
fi

if [[ "$DYNAMIC_DISABLE_TAIL_GUARD" == "1" ]]; then
    DYNAMIC_TAIL_GUARD_MIN_CAP="$MAX_RESPONSE_LEN"
    DYNAMIC_TAIL_GUARD_DEFAULT_RATIO=999
    DYNAMIC_TAIL_GUARD_RATIO_QUANTILE=1.0
fi

if [[ "$DYNAMIC_IGNORE_TAIL_TIES_AT_RESPONSE_CAP" == "auto" ]]; then
    if (( MAX_RESPONSE_LEN < 16384 )); then
        DYNAMIC_IGNORE_TAIL_TIES_AT_RESPONSE_CAP=1
    else
        DYNAMIC_IGNORE_TAIL_TIES_AT_RESPONSE_CAP=0
    fi
fi

TRAIN_FILE_ORIG="${TRAIN_FILE_ORIG:-/data/deepscaler/train.parquet}"
TEST_FILE="${TEST_FILE:-/data/deepscaler/test.parquet}"
MODEL_PATH="${MODEL_PATH:-/data/Qwen3-30B-A3B}"
DISTCP_PATH="${DISTCP_PATH:-/data/Qwen3-30B-A3B_megatron}"

validate_rollout_dir() {
    local dir="$1"
    local label="$2"
    local expected_steps="${3:-$DYNAMIC_PLAN_STEPS}"
    if [[ ! -d "$dir/rollout_data" ]]; then
        echo "$label missing rollout_data directory: $dir/rollout_data" >&2
        exit 3
    fi
    local count
    count=$(find "$dir/rollout_data" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
    if (( count < expected_steps )); then
        echo "$label has only $count rollout jsonl files, expected at least $expected_steps: $dir" >&2
        exit 3
    fi
}

validate_planned_epoch_memory_safety() {
    local dir="$1"
    local label="$2"
    local expected_steps="$3"
    local log_file
    local guard_count
    local cleanup_count

    if [[ "$DYNAMIC_SHRINK_POLICY" != "planned" \
          && "$DYNAMIC_SHRINK_POLICY" != "plan" ]]; then
        return
    fi
    log_file=$(
        find "$dir/logs" -maxdepth 1 -type f -name '*.txt' \
            -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
    )
    if [[ -z "$log_file" || ! -f "$log_file" ]]; then
        echo "$label missing Planned memory-safety log under $dir/logs" >&2
        exit 3
    fi
    guard_count=$(grep -c 'Mode1 training memory guard: rank=0 ' \
        "$log_file" || true)
    cleanup_count=$(grep -cE \
        'Mode1 training-boundary full-world transient cleanup: rank=0 step=[0-9]+ ' \
        "$log_file" || true)
    if (( guard_count != expected_steps || cleanup_count != expected_steps )); then
        echo "$label failed Planned memory-safety validation: " \
             "guards=$guard_count cleanups=$cleanup_count expected=$expected_steps log=$log_file" >&2
        exit 3
    fi
    if grep -q 'Mode1 full-restore transient cleanup:.*canonical_offload_enabled=1' \
            "$log_file"; then
        echo "$label used shape-unsafe canonical loaded-weight offload: $log_file" >&2
        exit 3
    fi
    echo "[dynamic length-aware] $label Planned memory safety validated: guards=$guard_count cleanups=$cleanup_count"
}

build_offline_planning_history() {
    local dir="$1"
    local source_steps="${2:-$DYNAMIC_PLAN_STEPS}"
    local history_file="$dir/offline_planning_history.json"
    if [[ "$DYNAMIC_BUILD_OFFLINE_PLANNING_HISTORY" != "1" ]]; then
        echo "[dynamic length-aware] skipping offline planning history for $dir" >&2
        return
    fi
    if [[ -f "$history_file" ]]; then
        return
    fi
    echo "[dynamic length-aware] building offline planning history=$history_file" >&2
    python3 "$REPO_ROOT/tools/build_offline_planning_history.py" \
        --baseline-dir "$dir" \
        --steps "$source_steps" \
        --responses-per-prompt "${ROLLOUT_N:-16}" >&2
}

validate_rollout_history() {
    local history="$1"
    local label="$2"
    local normalized="${history//,/:}"
    local old_ifs="$IFS"
    local dir
    local source_steps

    IFS=':'
    for dir in $normalized; do
        if [[ -z "$dir" ]]; then
            continue
        fi
        source_steps="$DYNAMIC_PLAN_STEPS"
        if [[ -f "$dir/offline_planning_history.json" ]]; then
            source_steps=$(python3 -c \
                'import json,sys; print(int(json.load(open(sys.argv[1]))["steps"]))' \
                "$dir/offline_planning_history.json")
        fi
        validate_rollout_dir "$dir" "$label" "$source_steps"
        build_offline_planning_history "$dir" "$source_steps"
    done
    IFS="$old_ifs"
    printf '%s' "$normalized"
}

find_latest_checkpoint() {
    local output_dir="$1"
    local ckpt_root="$output_dir/checkpoints/$CHECKPOINT_MODEL_DIR_NAME"
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
    build_offline_planning_history "$record_dir"
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
    local child_script="$DYNAMIC_MODE1_CHILD_SCRIPT"

    if [[ -z "$child_script" ]]; then
        if [[ "${MIN_ADAPTIVE_FLOOR:-}" == "2" \
            || ",${VLLM_ASCEND_SHRINK_AWARE_STAGES:-}," == *,2,* \
            || ",${DYNAMIC_FORCE_SELECTED_FLOORS:-}," == *,2,* \
            || "${DYNAMIC_FORCE_SELECTED_FLOOR:-}" == "2" ]]; then
            child_script="$REPO_ROOT/run_mode1_local_length_sorted_e2e_adaptive_floor2.sh"
        else
            child_script="$REPO_ROOT/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"
        fi
    elif [[ "$child_script" != /* ]]; then
        child_script="$REPO_ROOT/$child_script"
    fi

    if [[ "$DYNAMIC_ENABLE_CKPT_CHAIN" == "1" ]]; then
        save_ckpt_enable=1
        export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
        if [[ -n "$resume_ckpt" ]]; then
            child_args+=("trainer.resume_mode=resume_path")
            child_args+=("trainer.resume_from_path=$resume_ckpt")
        fi
    fi

    echo "[dynamic length-aware] epoch=$epoch policy=$DYNAMIC_SHRINK_POLICY baseline_dirs=$baseline_dirs output_dir=$output_dir"
    echo "[dynamic length-aware] epoch=$epoch child_script=$child_script"
    if [[ -n "$resume_ckpt" ]]; then
        echo "[dynamic length-aware] epoch=$epoch resume_ckpt=$resume_ckpt"
    fi
    local child_rc=0
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
    IGNORE_TAIL_TIES_AT_RESPONSE_CAP="$DYNAMIC_IGNORE_TAIL_TIES_AT_RESPONSE_CAP" \
    TAIL_GUARD_RATIO_QUANTILE="$DYNAMIC_TAIL_GUARD_RATIO_QUANTILE" \
    TAIL_GUARD_RATIO_WINDOW="$DYNAMIC_TAIL_GUARD_RATIO_WINDOW" \
    TAIL_GUARD_DEFAULT_RATIO="$DYNAMIC_TAIL_GUARD_DEFAULT_RATIO" \
    TAIL_GUARD_MIN_CAP="$DYNAMIC_TAIL_GUARD_MIN_CAP" \
    TAIL_GUARD_ROUND_TO="$DYNAMIC_TAIL_GUARD_ROUND_TO" \
    EXPECT_NO_RESPONSE_CAPS="$DYNAMIC_EXPECT_NO_RESPONSE_CAPS" \
    PLAN_STEPS="$DYNAMIC_PLAN_STEPS" \
    SAVE_CKPT_ENABLE="$save_ckpt_enable" \
    TRAINER_SAVE_FREQ="$trainer_save_freq" \
    VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY="$DYNAMIC_SHRINK_POLICY" \
    VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE="$DYNAMIC_SHORT_STEP_CAP_ENABLE" \
    VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_EXIT_THRESHOLD="$DYNAMIC_SHORT_STEP_EXIT_THRESHOLD" \
    VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_TOKENS="$DYNAMIC_SHORT_STEP_CAP_TOKENS" \
    VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_FLOORS="$DYNAMIC_SHORT_STEP_CAP_FLOORS" \
    FORCE_SELECTED_FLOOR="${DYNAMIC_FORCE_SELECTED_FLOOR:-}" \
    FORCE_SELECTED_FLOORS="${DYNAMIC_FORCE_SELECTED_FLOORS:-}" \
    REQUIRE_COMPACT_HISTORY=1 \
    VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME="${DYNAMIC_RESET_PROGRESS_AFTER_RESUME:-1}" \
    "$child_script" \
        "trainer.total_training_steps=$train_steps" "${child_args[@]}" || child_rc=$?
    if (( child_rc != 0 )); then
        echo "[dynamic length-aware] epoch=$epoch child run failed with exit_code=$child_rc output_dir=$output_dir" >&2
        exit "$child_rc"
    fi

    validate_rollout_dir "$output_dir" "epoch $epoch mode1" "$train_steps"
    validate_planned_epoch_memory_safety \
        "$output_dir" "epoch $epoch mode1" "$train_steps"
    build_offline_planning_history "$output_dir" "$train_steps"
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
echo "[dynamic length-aware] ignore_tail_ties_at_response_cap=$DYNAMIC_IGNORE_TAIL_TIES_AT_RESPONSE_CAP"
echo "[dynamic length-aware] floor8_cap=${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8:-<local-default>}"
if [[ "$DYNAMIC_SHRINK_POLICY" == "planned" || "$DYNAMIC_SHRINK_POLICY" == "plan" ]]; then
    echo "[dynamic length-aware] planned_memory_safety=enabled transient_cleanup=$VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE canonical_offload=$VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE training_min_free_mib=$VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB"
fi
echo "[dynamic length-aware] short_step_cap_enable=$DYNAMIC_SHORT_STEP_CAP_ENABLE threshold=$DYNAMIC_SHORT_STEP_EXIT_THRESHOLD cap=$DYNAMIC_SHORT_STEP_CAP_TOKENS floors=$DYNAMIC_SHORT_STEP_CAP_FLOORS"
echo "[dynamic length-aware] disable_tail_guard=$DYNAMIC_DISABLE_TAIL_GUARD expect_no_response_caps=$DYNAMIC_EXPECT_NO_RESPONSE_CAPS tail_guard_ratio_q=$DYNAMIC_TAIL_GUARD_RATIO_QUANTILE window=$DYNAMIC_TAIL_GUARD_RATIO_WINDOW default_ratio=$DYNAMIC_TAIL_GUARD_DEFAULT_RATIO min_cap=$DYNAMIC_TAIL_GUARD_MIN_CAP round_to=$DYNAMIC_TAIL_GUARD_ROUND_TO"

previous_rollout_dir=""
baseline_history=""
if [[ "$DYNAMIC_SKIP_MODE0_PROBE" == "1" ]]; then
    if (( DYNAMIC_START_EPOCH < 1 )); then
        echo "DYNAMIC_START_EPOCH must be >= 1 when DYNAMIC_SKIP_MODE0_PROBE=1" >&2
        exit 2
    fi
    if (( DYNAMIC_START_EPOCH >= DYNAMIC_TOTAL_EPOCHS )); then
        echo "DYNAMIC_START_EPOCH=$DYNAMIC_START_EPOCH must be < DYNAMIC_TOTAL_EPOCHS=$DYNAMIC_TOTAL_EPOCHS" >&2
        exit 2
    fi
    if [[ -z "$DYNAMIC_INITIAL_BASELINE_DIR" ]]; then
        echo "DYNAMIC_SKIP_MODE0_PROBE=1 requires DYNAMIC_INITIAL_BASELINE_DIR" >&2
        exit 2
    fi
    baseline_history=$(validate_rollout_history "$DYNAMIC_INITIAL_BASELINE_DIR" "initial baseline")
    previous_rollout_dir="${baseline_history##*:}"
else
    DYNAMIC_START_EPOCH=1
    run_mode0_probe "$@"
    previous_rollout_dir="$NEXT_ROLLOUT_DIR"
    baseline_history="$previous_rollout_dir"
    DYNAMIC_INITIAL_RESUME_CKPT="$NEXT_CKPT_PATH"
fi

for (( epoch = DYNAMIC_START_EPOCH; epoch < DYNAMIC_TOTAL_EPOCHS; epoch++ )); do
    run_mode1_epoch "$epoch" "$baseline_history" "$DYNAMIC_INITIAL_RESUME_CKPT" "$@"
    previous_rollout_dir="$NEXT_ROLLOUT_DIR"
    baseline_history="${baseline_history}:$previous_rollout_dir"
    DYNAMIC_INITIAL_RESUME_CKPT="$NEXT_CKPT_PATH"
done

echo "[dynamic length-aware] done final_rollout_dir=$previous_rollout_dir"
