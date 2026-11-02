#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
QUALITY_OUTPUT_ROOT="${QUALITY_OUTPUT_ROOT:-/data/adafloor_shared_state/p1_quality_50step_$(date -u +%Y%m%dT%H%M%SZ)}"
QUALITY_VARIANTS="${QUALITY_VARIANTS:-lengthsort lengthsort_guard adafloor_n_f2 adafloor_p_f4}"
QUALITY_SEED="${QUALITY_SEED:-505}"
QUALITY_START_EPOCH="${QUALITY_START_EPOCH:-1}"
QUALITY_TOTAL_EPOCHS="${QUALITY_TOTAL_EPOCHS:-11}"
QUALITY_MIN_FREE_GIB="${QUALITY_MIN_FREE_GIB:-180}"
REUSE_ENV="$COMMON_EPOCH0_ROOT/reuse.env"

if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
      || ! -f "$REUSE_ENV" ]]; then
    echo "preserved common epoch0 is incomplete: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi
if ! [[ "$QUALITY_SEED" =~ ^[0-9]+$ \
        && "$QUALITY_START_EPOCH" =~ ^[0-9]+$ \
        && "$QUALITY_TOTAL_EPOCHS" =~ ^[0-9]+$ \
        && "$QUALITY_MIN_FREE_GIB" =~ ^[0-9]+$ ]] \
   || (( QUALITY_START_EPOCH < 1 || QUALITY_TOTAL_EPOCHS <= QUALITY_START_EPOCH )); then
    echo "invalid quality-run setting" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "$REUSE_ENV"
if [[ ! -d "$DYNAMIC_INITIAL_BASELINE_DIR/rollout_data" \
      || ! -d "$DYNAMIC_INITIAL_RESUME_CKPT/actor" \
      || ! -f "$DYNAMIC_INITIAL_RESUME_CKPT/.PRESERVE_COMMON_EPOCH0" ]]; then
    echo "reuse.env does not reference a complete preserved epoch0" >&2
    exit 2
fi

mkdir -p "$QUALITY_OUTPUT_ROOT"
sha256sum \
    "$SCRIPT_DIR/run_paper_p1_quality_50step_from_common_epoch0.sh" \
    "$SCRIPT_DIR/analysis_eval/rotate_completed_checkpoints.py" \
    "$SCRIPT_DIR/analysis_eval/analyze_training_quality_curve.py" \
    "$SCRIPT_DIR/run_baseline_lengthsort_epoch1_2.sh" \
    "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh" \
    > "$QUALITY_OUTPUT_ROOT/code_sha256.txt"
{
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
    printf 'variants=%s\n' "$QUALITY_VARIANTS"
    printf 'seed=%s\n' "$QUALITY_SEED"
    printf 'epochs=%s\n' "$((QUALITY_TOTAL_EPOCHS - QUALITY_START_EPOCH))"
    printf 'steps=%s\n' "$((5 * (QUALITY_TOTAL_EPOCHS - QUALITY_START_EPOCH)))"
    printf 'actor_frozen=false\n'
    printf 'checkpoint_rotation=validated_successor\n'
} > "$QUALITY_OUTPUT_ROOT/protocol.env"

check_disk() {
    local free_kib required_kib
    free_kib=$(df --output=avail -k "$QUALITY_OUTPUT_ROOT" | tail -1 | tr -d '[:space:]')
    required_kib=$((QUALITY_MIN_FREE_GIB * 1024 * 1024))
    if ! [[ "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < required_kib )); then
        echo "insufficient disk for quality run: free_kib=${free_kib:-unknown}" >&2
        return 1
    fi
}

prepare_ray() {
    if pgrep -f '[p]ython3 -m verl\.trainer\.main_ppo|[r]ay::TaskRunner\.run' >/dev/null; then
        echo "refusing to clean Ray while another training process is active" >&2
        return 1
    fi
    ray stop --force >/dev/null 2>&1 || true
    sleep 5
}

validate_epoch_artifacts() {
    local epoch_dir="$1" file log_file count
    count=$(find "$epoch_dir/rollout_data" -maxdepth 1 -type f -name '*.jsonl' 2>/dev/null | wc -l)
    [[ "$count" == "5" ]] || return 1
    count=$(find "$epoch_dir/rollout_length" -maxdepth 1 -type f -name 'length_*.txt' 2>/dev/null | wc -l)
    [[ "$count" == "5" ]] || return 1
    while IFS= read -r file; do
        (( $(wc -l < "$file") == 512 )) || return 1
    done < <(find "$epoch_dir/rollout_data" "$epoch_dir/rollout_length" \
        -maxdepth 1 -type f \( -name '*.jsonl' -o -name 'length_*.txt' \) -print)
    log_file=$(find "$epoch_dir/logs" -maxdepth 1 -type f -name '*.txt' \
        -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    [[ -f "$log_file" ]] || return 1
    grep -qF 'Training Progress: 100%' "$log_file" || return 1
    grep -qF 'After trainer.fit' "$log_file" || return 1
    (( $(grep -cF 'rollout_output_time_s:' "$log_file" || true) == 5 )) || return 1
    ! grep -qE 'response/aborted_ratio:(0\.[0-9]*[1-9]|[1-9])|OutOfMemoryError|NPU out of memory' "$log_file"
}

validate_run() {
    local run_dir="$1" epoch tag epoch_dir found
    for (( epoch = QUALITY_START_EPOCH; epoch < QUALITY_TOTAL_EPOCHS; epoch++ )); do
        tag=$(printf '%03d' "$epoch")
        found=()
        mapfile -t found < <(find "$run_dir" -mindepth 1 -maxdepth 1 -type d \
            -name "epoch_${tag}_*" -print)
        (( ${#found[@]} == 1 )) || return 1
        epoch_dir="${found[0]}"
        validate_epoch_artifacts "$epoch_dir" || return 1
    done
}

variant_target() {
    case "$1" in
        lengthsort|lengthsort_guard)
            printf '%s' "$SCRIPT_DIR/run_baseline_lengthsort_epoch1_2.sh"
            ;;
        adafloor_n_f2)
            printf '%s' "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh"
            ;;
        adafloor_p_f4)
            printf '%s' "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh"
            ;;
        *) return 1 ;;
    esac
}

run_variant() {
    local variant="$1" target run_name run_dir target_pid monitor_pid rc checkpoint_dir
    target=$(variant_target "$variant") || {
        echo "unknown quality variant: $variant" >&2
        return 2
    }
    run_name="quality_${variant}_seed${QUALITY_SEED}_${QUALITY_START_EPOCH}_$((QUALITY_TOTAL_EPOCHS - 1))"
    run_dir="$QUALITY_OUTPUT_ROOT/$run_name"
    if [[ -f "$run_dir/QUALITY_RUN_COMPLETE.txt" ]]; then
        echo "[quality] already complete variant=$variant run=$run_dir"
        return 0
    fi
    if [[ -e "$run_dir" ]]; then
        echo "incomplete quality output requires inspection: $run_dir" >&2
        return 3
    fi
    check_disk
    prepare_ray
    mkdir -p "$run_dir"

    common_args=(
        data.train_batch_size=32
        data.max_prompt_length=1024
        data.max_response_length=16384
        data.shuffle=False
        actor_rollout_ref.rollout.n=16
        actor_rollout_ref.rollout.temperature=0.9
        actor_rollout_ref.rollout.top_p=0.9
        actor_rollout_ref.rollout.top_k=50
        actor_rollout_ref.rollout.seed="$QUALITY_SEED"
    )

    echo "[quality] start variant=$variant output=$run_dir"
    set +e
    if [[ "$variant" == lengthsort* ]]; then
        BASELINE_ENABLE_TAIL_GUARD=$([[ "$variant" == "lengthsort_guard" ]] && echo 1 || echo 0) \
        OUTPUT_ROOT="$QUALITY_OUTPUT_ROOT" \
        DYNAMIC_OUTPUT_ROOT="$QUALITY_OUTPUT_ROOT" \
        DYNAMIC_RUN_NAME="$run_name" \
        DYNAMIC_START_EPOCH="$QUALITY_START_EPOCH" \
        DYNAMIC_TOTAL_EPOCHS="$QUALITY_TOTAL_EPOCHS" \
        BASELINE_EPOCH0="$DYNAMIC_INITIAL_BASELINE_DIR" \
        BASELINE_INITIAL_RESUME_CKPT="$DYNAMIC_INITIAL_RESUME_CKPT" \
        VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
            "$target" "${common_args[@]}" &
    else
        planned_env=()
        if [[ "$variant" == "adafloor_p_f4" ]]; then
            planned_env=(
                RANK_MATCHING_POLICY=release_area
                VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING=1
                VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1
                VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=1
                VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1
                VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0
                VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB=28672
            )
        fi
        env "${planned_env[@]}" \
            DYNAMIC_OUTPUT_ROOT="$QUALITY_OUTPUT_ROOT" \
            DYNAMIC_RUN_NAME="$run_name" \
            DYNAMIC_SKIP_MODE0_PROBE=1 \
            DYNAMIC_INITIAL_BASELINE_DIR="$DYNAMIC_INITIAL_BASELINE_DIR" \
            DYNAMIC_INITIAL_RESUME_CKPT="$DYNAMIC_INITIAL_RESUME_CKPT" \
            DYNAMIC_START_EPOCH="$QUALITY_START_EPOCH" \
            DYNAMIC_TOTAL_EPOCHS="$QUALITY_TOTAL_EPOCHS" \
            DYNAMIC_ENABLE_CKPT_CHAIN=1 \
            DYNAMIC_PLAN_STEPS=5 \
            DYNAMIC_TRAIN_STEPS=5 \
            DYNAMIC_LENGTH_EMA_DECAY=0.3 \
            ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 \
            VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
            "$target" "${common_args[@]}" &
    fi
    target_pid=$!
    python3 "$SCRIPT_DIR/analysis_eval/rotate_completed_checkpoints.py" \
        --run-dir "$run_dir" --parent-pid "$target_pid" \
        > "$run_dir/checkpoint_rotation_stdout.log" 2>&1 &
    monitor_pid=$!
    wait "$target_pid"
    rc=$?
    wait "$monitor_pid" || true
    set -e
    if (( rc != 0 )); then
        echo "[quality] failed variant=$variant rc=$rc; newest checkpoint retained" >&2
        return "$rc"
    fi
    validate_run "$run_dir" || {
        echo "[quality] validation failed variant=$variant; checkpoint retained" >&2
        return 4
    }
    mapfile -t remaining < <(find "$run_dir" -mindepth 2 -maxdepth 2 \
        -type d -name checkpoints -print)
    for checkpoint_dir in "${remaining[@]}"; do
        [[ "$checkpoint_dir" == "$run_dir/"epoch_*/checkpoints ]] || {
            echo "unsafe final checkpoint path: $checkpoint_dir" >&2
            return 4
        }
    done
    if (( ${#remaining[@]} > 0 )); then
        rm -r -- "${remaining[@]}"
    fi
    {
        printf 'variant=%s\n' "$variant"
        printf 'seed=%s\n' "$QUALITY_SEED"
        printf 'validated_epochs=%s-%s\n' "$QUALITY_START_EPOCH" "$((QUALITY_TOTAL_EPOCHS - 1))"
        printf 'validated_steps=%s\n' "$((5 * (QUALITY_TOTAL_EPOCHS - QUALITY_START_EPOCH)))"
        printf 'actor_frozen=false\n'
        printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'common_epoch0_preserved=true\n'
    } > "$run_dir/QUALITY_RUN_COMPLETE.txt"
    echo "[quality] complete variant=$variant checkpoints_removed=true"
}

for variant in $QUALITY_VARIANTS; do
    run_variant "$variant"
done

analysis_args=()
for variant in $QUALITY_VARIANTS; do
    run_name="quality_${variant}_seed${QUALITY_SEED}_${QUALITY_START_EPOCH}_$((QUALITY_TOTAL_EPOCHS - 1))"
    analysis_args+=(--variant "$variant" "$QUALITY_OUTPUT_ROOT/$run_name")
done
python3 "$SCRIPT_DIR/analysis_eval/analyze_training_quality_curve.py" \
    "${analysis_args[@]}" --output-dir "$QUALITY_OUTPUT_ROOT/summary"
printf '[quality] all variants complete summary=%s\n' "$QUALITY_OUTPUT_ROOT/summary/quality_curve_summary.md"
