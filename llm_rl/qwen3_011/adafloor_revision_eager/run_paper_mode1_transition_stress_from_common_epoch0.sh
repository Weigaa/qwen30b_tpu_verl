#!/usr/bin/env bash
set -euo pipefail

if [[ "${ADAFLOOR_STRESS_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    source_path=$(realpath "${BASH_SOURCE[0]}")
    snapshot=$(mktemp "${source_path}.run-snapshot.XXXXXX")
    cp -- "$source_path" "$snapshot"
    chmod 700 "$snapshot"
    set +e
    ADAFLOOR_STRESS_SNAPSHOT_ACTIVE=1 "$snapshot" "$@"
    rc=$?
    set -e
    rm -f -- "$snapshot"
    exit "$rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
STRESS_OUTPUT_ROOT="${STRESS_OUTPUT_ROOT:-/data/adafloor_shared_state/mode1_transition_stress_$(date -u +%Y%m%dT%H%M%SZ)}"
STRESS_STEPS="${STRESS_STEPS:-20}"
STRESS_RUN_NAME="${STRESS_RUN_NAME:-natural_floor2_${STRESS_STEPS}step_transition_stress}"
STRESS_SEED="${STRESS_SEED:-707}"
STRESS_FINALIZE_EXISTING="${STRESS_FINALIZE_EXISTING:-0}"
REUSE_ENV="$COMMON_EPOCH0_ROOT/reuse.env"
RUN_DIR="$STRESS_OUTPUT_ROOT/$STRESS_RUN_NAME"

if ! [[ "$STRESS_STEPS" =~ ^[1-9][0-9]*$ && "$STRESS_SEED" =~ ^[0-9]+$ ]] \
   || [[ "$STRESS_FINALIZE_EXISTING" != "0" \
         && "$STRESS_FINALIZE_EXISTING" != "1" ]]; then
    echo "invalid transition stress setting" >&2
    exit 2
fi
if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
      || ! -f "$REUSE_ENV" ]]; then
    echo "preserved common epoch0 is incomplete" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$REUSE_ENV"
if [[ -e "$RUN_DIR" ]]; then
    if [[ "$STRESS_FINALIZE_EXISTING" != "1" ]]; then
        echo "refusing to overwrite transition stress output: $RUN_DIR" >&2
        exit 2
    fi
elif [[ "$STRESS_FINALIZE_EXISTING" == "1" ]]; then
    echo "cannot finalize missing transition stress output: $RUN_DIR" >&2
    exit 2
fi
if [[ "$STRESS_FINALIZE_EXISTING" != "1" ]] \
   && pgrep -f '[p]ython3 -m verl\.trainer\.main_ppo|[r]ay::TaskRunner\.run' >/dev/null; then
    echo "another training process is active" >&2
    exit 3
fi

floors=""
caps=""
expected_shrink_events=0
for (( step = 0; step < STRESS_STEPS; step++ )); do
    case $((step % 3)) in
        0) floor=2; expected_shrink_events=$((expected_shrink_events + 3)) ;;
        1) floor=4; expected_shrink_events=$((expected_shrink_events + 2)) ;;
        2) floor=8; expected_shrink_events=$((expected_shrink_events + 1)) ;;
    esac
    floors+="${floors:+,}$floor"
    caps+="${caps:+;}8,16,32,64,64"
done

if [[ "$STRESS_FINALIZE_EXISTING" != "1" ]]; then
    mkdir -p "$STRESS_OUTPUT_ROOT"
    {
        printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
        printf 'steps=%s\n' "$STRESS_STEPS"
        printf 'expected_shrink_events=%s\n' "$expected_shrink_events"
        printf 'expected_restore_events=%s\n' "$STRESS_STEPS"
        printf 'forced_floors=%s\n' "$floors"
        printf 'max_response_length=128\n'
        printf 'actor_frozen=true\n'
        printf 'checkpoint_save=false\n'
    } > "$STRESS_OUTPUT_ROOT/protocol.env"

    ray stop --force >/dev/null 2>&1 || true
    sleep 5

    DYNAMIC_OUTPUT_ROOT="$STRESS_OUTPUT_ROOT" \
    DYNAMIC_RUN_NAME="$STRESS_RUN_NAME" \
    DYNAMIC_INITIAL_BASELINE_DIR="$DYNAMIC_INITIAL_BASELINE_DIR" \
    DYNAMIC_PLAN_STEPS="$STRESS_STEPS" \
    DYNAMIC_HISTORY_STEPS=5 \
    DYNAMIC_TRAIN_STEPS="$STRESS_STEPS" \
    DYNAMIC_DATASET_FRACTION=0.10 \
    DYNAMIC_FORCE_SELECTED_FLOORS="$floors" \
    DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP="$caps" \
    DYNAMIC_FULL_MAX_RESPONSE_LENGTH=128 \
    DYNAMIC_FULL_MAX_RESPONSE_LEN=128 \
    DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS=1152 \
    DYNAMIC_TAIL_GUARD_MIN_CAP=128 \
    DYNAMIC_TAIL_GUARD_ROUND_TO=64 \
    DYNAMIC_BUILD_OFFLINE_PLANNING_HISTORY=0 \
    REPEAT_PROMPT_SET_TO_FILL=1 \
    VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
        "$SCRIPT_DIR/run_mode1_transition_stress_floor2_probe.sh" \
            trainer.resume_mode=resume_path \
            "trainer.resume_from_path=$DYNAMIC_INITIAL_RESUME_CKPT" \
            actor_rollout_ref.actor.optim.lr=0.0 \
            "actor_rollout_ref.rollout.seed=$STRESS_SEED" \
            data.train_batch_size=32 \
            data.max_prompt_length=1024 \
            data.max_response_length=128 \
            data.shuffle=False \
            actor_rollout_ref.rollout.n=16 \
            actor_rollout_ref.rollout.temperature=0.9 \
            actor_rollout_ref.rollout.top_p=0.9 \
            actor_rollout_ref.rollout.top_k=50
else
    echo "[transition stress] finalize-existing mode; training launch skipped"
fi

epoch_dir="$RUN_DIR/epoch_001_mode1_natural"
log_file=$(find "$epoch_dir/logs" -maxdepth 1 -type f -name '*.txt' \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
[[ -f "$log_file" ]] || { echo "transition stress log is missing" >&2; exit 4; }
rollout_count=$(find "$epoch_dir/rollout_data" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
length_count=$(find "$epoch_dir/rollout_length" -maxdepth 1 -type f -name 'length_*.txt' | wc -l)
metric_count=$(grep -cE 'step:[0-9]+ - ' "$log_file" || true)
timing_count=$(grep -cF 'rollout_output_time_s:' "$log_file" || true)
# Rank 15 remains active at floors 8, 4, and 2, so it observes every staged
# transition exactly once. Rank 0 exits at the first 16-to-8 transition and
# would undercount deeper shrink stages.
shrink_count=$(grep -c 'Elastic EP shrink rpc enter: global_rank=15 ' "$log_file" || true)
restore_count=$(grep -c 'Elastic parallel groups restored: global_rank=0 ' "$log_file" || true)

if (( rollout_count != STRESS_STEPS || length_count != STRESS_STEPS \
      || metric_count != STRESS_STEPS || timing_count != STRESS_STEPS \
      || shrink_count != expected_shrink_events || restore_count != STRESS_STEPS )); then
    echo "transition stress validation failed rollouts=$rollout_count lengths=$length_count metrics=$metric_count timings=$timing_count shrink=$shrink_count/$expected_shrink_events restore=$restore_count/$STRESS_STEPS" >&2
    exit 4
fi
if ! grep -qE "Training Progress: +100%.*${STRESS_STEPS}/${STRESS_STEPS}" "$log_file" \
   || ! grep -qF 'After trainer.fit' "$log_file" \
   || grep -qE 'response/aborted_ratio:(0\.[0-9]*[1-9]|[1-9])|OutOfMemoryError|NPU out of memory|HCCL.*timeout|collective.*timeout' "$log_file"; then
    echo "transition stress contains a completion or safety failure" >&2
    exit 4
fi
if find "$RUN_DIR" -type d -name checkpoints -print -quit | grep -q .; then
    echo "transition stress unexpectedly wrote checkpoints" >&2
    exit 4
fi

{
    printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'steps=%s\n' "$STRESS_STEPS"
    printf 'observed_shrink_events=%s\n' "$shrink_count"
    printf 'observed_restore_events=%s\n' "$restore_count"
    printf 'aborted_responses=0\n'
    printf 'oom=0\n'
    printf 'collective_timeout=0\n'
    printf 'log=%s\n' "$log_file"
} > "$STRESS_OUTPUT_ROOT/TRANSITION_STRESS_COMPLETE.txt"
python3 "$SCRIPT_DIR/analysis_eval/summarize_transition_stress.py" \
    --root "$STRESS_OUTPUT_ROOT" \
    --output "$STRESS_OUTPUT_ROOT/transition_stress_summary.md"
printf 'summary=%s\n' \
    "$STRESS_OUTPUT_ROOT/transition_stress_summary.md" \
    >> "$STRESS_OUTPUT_ROOT/TRANSITION_STRESS_COMPLETE.txt"
echo "[transition stress] complete output=$STRESS_OUTPUT_ROOT"
