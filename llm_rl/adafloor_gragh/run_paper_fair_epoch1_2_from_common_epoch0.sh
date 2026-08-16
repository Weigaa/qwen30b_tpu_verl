#!/usr/bin/env bash
set -euo pipefail

# Bash reads top-level commands incrementally.  Run an immutable same-directory
# copy so editing the launcher during a multi-hour experiment cannot corrupt
# its post-run validation and cleanup phase.
if [[ "${ADAFLOOR_FAIR_RUNNER_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    fair_runner_source=$(realpath "${BASH_SOURCE[0]}")
    fair_runner_snapshot=$(mktemp "${fair_runner_source}.run-snapshot.XXXXXX")
    cp -- "$fair_runner_source" "$fair_runner_snapshot"
    chmod 700 "$fair_runner_snapshot"
    set +e
    ADAFLOOR_FAIR_RUNNER_SNAPSHOT_ACTIVE=1 \
        "$fair_runner_snapshot" "$@"
    fair_runner_rc=$?
    set -e
    rm -f -- "$fair_runner_snapshot"
    exit "$fair_runner_rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

usage() {
    cat <<'EOF'
Usage:
  ./run_paper_fair_epoch1_2_from_common_epoch0.sh VARIANT [extra hydra args...]

Variants:
  vanilla              Full16 random-order baseline
  lengthsort           Pure LengthSort without TailGuard
  lengthsort_guard     LengthSort with TailGuard
  fixed4               KV-safe FixedFloor at floor4 with TailGuard
  minskew              Adaptive floors with MinSkew matching and TailGuard
  adafloor_n_f4        AdaFloor Natural floor4 with TailGuard
  adafloor_n_f2        AdaFloor Natural floor2 with TailGuard
  adafloor_p_f4        AdaFloor Planned floor4 with TailGuard
  adafloor_p_f2        AdaFloor Planned floor2 with TailGuard
  adafloor_p_f4_minskew  Planned floor4 with MinSkew matching and TailGuard
  adafloor_n_f2_noguard  AdaFloor Natural floor2 without response caps

The launcher runs one variant only. Every variant reads the same public epoch0
rollout history and starts epoch1 from the same global_step_5 checkpoint.
After both epochs pass completion checks, their checkpoints are deleted by
default. Set FAIR_KEEP_COMPLETED_CHECKPOINTS=1 to retain them.

Workload sensitivity overrides keep their paper defaults when unset.
FAIR_TRAIN_BATCH_SIZE controls prompts per step, FAIR_ROLLOUT_N controls
responses per prompt, and FAIR_MAX_RESPONSE_LENGTH controls the response cap.
EOF
}

is_planned_variant() {
    [[ "${variant:-}" == "adafloor_p_f4" \
       || "${variant:-}" == "adafloor_p_f2" \
       || "${variant:-}" == "adafloor_p_f4_minskew" ]]
}

requires_zero_preemption() {
    case "${variant:-}" in
        lengthsort_guard|fixed4|minskew|adafloor_*) return 0 ;;
        *) return 1 ;;
    esac
}

validate_completed_epoch() {
    local run_dir="$1"
    local epoch_tag="$2"
    local epoch_dirs=()
    local rollout_files=()
    local length_files=()
    local log_file
    local marker
    local metric_count
    local rollout_count
    local preemption_count
    local oom_count
    local distcp_count
    local guard_count
    local guard_min_free_mib
    local boundary_transient_count
    local file

    mapfile -t epoch_dirs < <(
        find "$run_dir" -mindepth 1 -maxdepth 1 -type d \
            -name "epoch_${epoch_tag}_*" -print | sort
    )
    if (( ${#epoch_dirs[@]} != 1 )); then
        echo "[fair rerun] validation failed: expected one epoch_${epoch_tag} directory, found ${#epoch_dirs[@]}" >&2
        return 1
    fi

    local epoch_dir="${epoch_dirs[0]}"
    mapfile -t rollout_files < <(
        find "$epoch_dir/rollout_data" -maxdepth 1 -type f \
            -name '*.jsonl' -print 2>/dev/null | sort -V
    )
    mapfile -t length_files < <(
        find "$epoch_dir/rollout_length" -maxdepth 1 -type f \
            -name 'length_*.txt' -print 2>/dev/null | sort -V
    )
    if (( ${#rollout_files[@]} != FAIR_TRAIN_STEPS \
          || ${#length_files[@]} != FAIR_TRAIN_STEPS )); then
        echo "[fair rerun] validation failed: epoch_${epoch_tag} has ${#rollout_files[@]} rollout files and ${#length_files[@]} length files" >&2
        return 1
    fi
    for file in "${rollout_files[@]}" "${length_files[@]}"; do
        if (( $(wc -l < "$file") != FAIR_EXPECTED_RESPONSES_PER_STEP )); then
            echo "[fair rerun] validation failed: expected $FAIR_EXPECTED_RESPONSES_PER_STEP records in $file" >&2
            return 1
        fi
    done

    log_file=$(
        find "$epoch_dir/logs" -maxdepth 1 -type f -name '*.txt' \
            -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
    )
    if [[ -z "$log_file" || ! -f "$log_file" ]]; then
        echo "[fair rerun] validation failed: epoch_${epoch_tag} log is missing" >&2
        return 1
    fi
    rollout_count=$(grep -cF 'rollout_output_time_s:' "$log_file" || true)
    metric_count=$(grep -cE 'step:[0-9]+ - ' "$log_file" || true)
    if (( ${rollout_count:-0} != FAIR_TRAIN_STEPS \
          || ${metric_count:-0} != FAIR_TRAIN_STEPS )) \
       || ! grep -qF 'Training Progress: 100%' "$log_file" \
       || ! grep -qF 'After trainer.fit' "$log_file"; then
        echo "[fair rerun] validation failed: epoch_${epoch_tag} log is incomplete" >&2
        return 1
    fi
    aborted_values=$(sed -nE \
        's/.*response\/aborted_ratio:([0-9.eE+-]+).*/\1/p' "$log_file")
    aborted_count=$(wc -w <<< "$aborted_values")
    aborted_nonzero_count=$(awk \
        '$1 + 0 != 0 { count++ } END { print count + 0 }' \
        <<< "$aborted_values")
    if (( aborted_count != FAIR_TRAIN_STEPS \
          || aborted_nonzero_count != 0 )); then
        echo "[fair rerun] validation failed: epoch_${epoch_tag} has missing or nonzero aborted-response metrics" >&2
        return 1
    fi
    oom_count=$(grep -ciE \
        'NPU out of memory|Memory_Allocation_Failure|Failed to allocate.*NPU memory|OutOfMemoryError|ACL_ERROR_RT_MEMORY_ALLOCATION' \
        "$log_file" || true)
    if (( ${oom_count:-0} != 0 )); then
        echo "[fair rerun] validation failed: epoch_${epoch_tag} contains ${oom_count} NPU OOM events" >&2
        return 1
    fi
    if requires_zero_preemption; then
        preemption_count=$(grep -ciE \
            'preempting request|request preempted' "$log_file" || true)
        if (( ${preemption_count:-0} != 0 )); then
            echo "[fair rerun] validation failed: epoch_${epoch_tag} contains ${preemption_count} KV preemptions" >&2
            return 1
        fi
    fi
    if is_planned_variant; then
        guard_count=$(grep -c 'Mode1 training memory guard: rank=0 ' "$log_file" || true)
        if (( ${guard_count:-0} != FAIR_TRAIN_STEPS )); then
            echo "[fair rerun] validation failed: epoch_${epoch_tag} has ${guard_count:-0} Planned memory-guard boundaries, expected $FAIR_TRAIN_STEPS" >&2
            return 1
        fi
        while IFS= read -r guard_min_free_mib; do
            if ! [[ "$guard_min_free_mib" =~ ^[0-9]+$ ]] \
               || (( guard_min_free_mib < FAIR_PLANNED_MIN_FREE_MIB_FLOOR )); then
                echo "[fair rerun] validation failed: epoch_${epoch_tag} used unsafe Planned min_free_mib=${guard_min_free_mib:-<missing>}" >&2
                return 1
            fi
        done < <(
            sed -nE 's/.*Mode1 training memory guard: rank=0 min_free_mib=([0-9]+).*/\1/p' "$log_file"
        )
        boundary_transient_count=$(
            grep -cE 'Mode1 training-boundary full-world transient cleanup: rank=0 step=[0-9]+ ' \
                "$log_file" || true
        )
        if (( ${boundary_transient_count:-0} != FAIR_TRAIN_STEPS )); then
            echo "[fair rerun] validation failed: epoch_${epoch_tag} has ${boundary_transient_count:-0} Planned training-boundary transient cleanups, expected $FAIR_TRAIN_STEPS" >&2
            return 1
        fi
        if grep -q 'Mode1 full-restore transient cleanup:.*canonical_offload_enabled=1' "$log_file"; then
            echo "[fair rerun] validation failed: epoch_${epoch_tag} enabled shape-unsafe canonical weight offload" >&2
            return 1
        fi
    fi
    if [[ "${FAIR_FREEZE_ACTOR:-0}" == "1" ]]; then
        local frozen_lr_count
        frozen_lr_count=$(grep -cE 'step:[0-9]+ - .*actor/lr:0([.]0+)?([[:space:]]+-|$)' "$log_file" || true)
        if (( ${frozen_lr_count:-0} != FAIR_TRAIN_STEPS )); then
            echo "[fair rerun] validation failed: epoch_${epoch_tag} did not keep actor/lr at zero for all $FAIR_TRAIN_STEPS steps" >&2
            return 1
        fi
    fi
    if [[ "${VERL_PAIRED_REQUEST_SAMPLING_SEEDS:-0}" == "1" ]]; then
        if ! python3 - "$FAIR_EXPECTED_RESPONSES_PER_STEP" "${rollout_files[@]}" <<'PY'
import json
import sys

expected_count = int(sys.argv[1])
required = {
    "prompt_occurrence_ordinal",
    "rollout_prompt_hash",
    "rollout_sample_index",
    "rollout_request_seed",
}
for filename in sys.argv[2:]:
    with open(filename, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != expected_count:
        raise SystemExit(
            f"{filename}: expected {expected_count} rows, got {len(rows)}"
        )
    missing = [index for index, row in enumerate(rows) if not required <= row.keys()]
    if missing:
        raise SystemExit(f"{filename}: paired seed audit fields missing at rows {missing[:5]}")
    by_occurrence = {}
    stable_requests = set()
    for row in rows:
        occurrence = int(row["prompt_occurrence_ordinal"])
        sample = int(row["rollout_sample_index"])
        if occurrence < 0 or sample < 0:
            raise SystemExit(
                f"{filename}: occurrence and sample indices must be nonnegative"
            )
        stable_key = (occurrence, sample)
        if stable_key in stable_requests:
            raise SystemExit(
                f"{filename}: duplicate stable request identity {stable_key}"
            )
        stable_requests.add(stable_key)
        by_occurrence.setdefault(occurrence, []).append(row)
    expected_samples = set(range(16))
    invalid_occurrences = []
    for occurrence, occurrence_rows in by_occurrence.items():
        samples = {int(row["rollout_sample_index"]) for row in occurrence_rows}
        hashes = {row["rollout_prompt_hash"] for row in occurrence_rows}
        if len(occurrence_rows) != 16 or samples != expected_samples or len(hashes) != 1:
            invalid_occurrences.append(occurrence)
    if (
        len(stable_requests) != expected_count
        or len(by_occurrence) != expected_count // 16
        or invalid_occurrences
    ):
        raise SystemExit(
            f"{filename}: source occurrence coverage is incomplete, "
            f"occurrences={len(by_occurrence)} stable_requests="
            f"{len(stable_requests)} invalid={invalid_occurrences[:5]}"
        )
PY
        then
            echo "[fair rerun] validation failed: epoch_${epoch_tag} paired request seed audit is incomplete" >&2
            return 1
        fi
    fi

    marker="$epoch_dir/checkpoints/$CHECKPOINT_MODEL_DIR_NAME/latest_checkpointed_iteration.txt"
    if [[ ! -f "$marker" \
          || $(tr -d '[:space:]' < "$marker") != "$FAIR_TRAIN_STEPS" ]]; then
        echo "[fair rerun] validation failed: epoch_${epoch_tag} checkpoint marker is missing or invalid" >&2
        return 1
    fi
    distcp_count=$(
        find "$epoch_dir/checkpoints/$CHECKPOINT_MODEL_DIR_NAME/global_step_${FAIR_TRAIN_STEPS}/actor/dist_ckpt" \
            -maxdepth 1 -type f -name '*.distcp' 2>/dev/null | wc -l
    )
    if (( distcp_count <= 0 )); then
        echo "[fair rerun] validation failed: epoch_${epoch_tag} has no actor checkpoint shards" >&2
        return 1
    fi
    if [[ "$FAIR_EXPECTED_DISTCP_SHARDS" != auto ]] \
       && (( distcp_count != FAIR_EXPECTED_DISTCP_SHARDS )); then
        echo "[fair rerun] validation failed: epoch_${epoch_tag} has $distcp_count checkpoint shards, expected $FAIR_EXPECTED_DISTCP_SHARDS" >&2
        return 1
    fi

    VALIDATED_EPOCH_DIR="$epoch_dir"
}

list_matching_processes() {
    local pattern="$1"

    ps -eo pid=,cmd= \
        | grep -E -- "$pattern" \
        | grep -v -E '[g]rep -E|[g]rep -v' \
        || true
}

prepare_clean_ray_runtime() {
    local active_training
    local stale_ray
    local attempt

    active_training=$(list_matching_processes \
        'python3 -m verl\.trainer\.main_ppo|ray::TaskRunner\.run')
    if [[ -n "$active_training" ]]; then
        echo "[fair rerun] refusing to stop Ray while another VERL run is active:" >&2
        echo "$active_training" >&2
        return 1
    fi

    stale_ray=$(list_matching_processes \
        'ray::WorkerDict|ray::TaskRunner|gcs_server|raylet|dashboard\.py|runtime_env_agent|log_monitor\.py')
    if [[ -z "$stale_ray" ]]; then
        echo "[fair rerun] Ray preflight clean"
        return 0
    fi

    echo "[fair rerun] stopping stale Ray processes before HCCL initialization"
    ray stop --force >/dev/null 2>&1 || true
    for attempt in $(seq 1 15); do
        stale_ray=$(list_matching_processes \
            'ray::WorkerDict|ray::TaskRunner|gcs_server|raylet|dashboard\.py|runtime_env_agent|log_monitor\.py')
        if [[ -z "$stale_ray" ]]; then
            echo "[fair rerun] stale Ray processes cleared"
            return 0
        fi
        sleep 1
    done

    echo "[fair rerun] stale Ray processes remain after cleanup:" >&2
    echo "$stale_ray" >&2
    return 1
}

require_idle_npus() {
    local npu_processes
    npu_processes=$(npu-smi info | awk '
        /Process id/ { in_process_table=1; next }
        in_process_table && /^\|/ && !/No running processes found/ { print }
    ')
    if [[ -n "$npu_processes" ]]; then
        echo "[fair rerun] NPU processes remain after Ray preflight:" >&2
        printf '%s\n' "$npu_processes" >&2
        return 1
    fi
    echo "[fair rerun] NPU preflight clean"
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

variant="$1"
shift

COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
REUSE_ENV="$COMMON_EPOCH0_ROOT/reuse.env"
COMMON_EPOCH0_METADATA_ENV="$COMMON_EPOCH0_ROOT/common_epoch0_metadata.env"
FAIR_OUTPUT_ROOT="${FAIR_OUTPUT_ROOT:-/data/adafloor_shared_state/paper_fair_reruns_common_epoch0}"
export COMMON_EPOCH0_ROOT

if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
      || ! -f "$REUSE_ENV" ]]; then
    echo "common epoch0 is not complete: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "$REUSE_ENV"
if [[ -f "$COMMON_EPOCH0_METADATA_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$COMMON_EPOCH0_METADATA_ENV"
fi
if [[ -n "${EXPECTED_COMMON_EPOCH0_MODEL_PATH:-}" ]]; then
    if [[ -z "${COMMON_EPOCH0_MODEL_PATH:-}" ]]; then
        echo "common epoch0 lacks model identity metadata: $COMMON_EPOCH0_METADATA_ENV" >&2
        exit 2
    fi
    if [[ "$(realpath -m "$COMMON_EPOCH0_MODEL_PATH")" \
          != "$(realpath -m "$EXPECTED_COMMON_EPOCH0_MODEL_PATH")" ]]; then
        echo "common epoch0 model mismatch" >&2
        echo "recorded=$COMMON_EPOCH0_MODEL_PATH expected=$EXPECTED_COMMON_EPOCH0_MODEL_PATH" >&2
        exit 2
    fi
fi
if [[ -n "${EXPECTED_COMMON_EPOCH0_MODEL_REVISION:-}" \
      && "${COMMON_EPOCH0_MODEL_REVISION:-}" \
         != "$EXPECTED_COMMON_EPOCH0_MODEL_REVISION" ]]; then
    echo "common epoch0 model revision mismatch" >&2
    echo "recorded=${COMMON_EPOCH0_MODEL_REVISION:-<missing>} expected=$EXPECTED_COMMON_EPOCH0_MODEL_REVISION" >&2
    exit 2
fi
if [[ -n "${EXPECTED_COMMON_EPOCH0_DISTCP_PATH:-}" ]]; then
    if [[ -z "${COMMON_EPOCH0_DISTCP_PATH:-}" \
          || "$(realpath -m "$COMMON_EPOCH0_DISTCP_PATH")" \
             != "$(realpath -m "$EXPECTED_COMMON_EPOCH0_DISTCP_PATH")" ]]; then
        echo "common epoch0 distributed checkpoint mismatch" >&2
        echo "recorded=${COMMON_EPOCH0_DISTCP_PATH:-<missing>} expected=$EXPECTED_COMMON_EPOCH0_DISTCP_PATH" >&2
        exit 2
    fi
fi
if [[ -n "${EXPECTED_COMMON_EPOCH0_CHECKPOINT_MODEL_DIR_NAME:-}" \
      && "${COMMON_EPOCH0_CHECKPOINT_MODEL_DIR_NAME:-}" \
         != "$EXPECTED_COMMON_EPOCH0_CHECKPOINT_MODEL_DIR_NAME" ]]; then
    echo "common epoch0 checkpoint namespace mismatch" >&2
    echo "recorded=${COMMON_EPOCH0_CHECKPOINT_MODEL_DIR_NAME:-<missing>} expected=$EXPECTED_COMMON_EPOCH0_CHECKPOINT_MODEL_DIR_NAME" >&2
    exit 2
fi
if [[ -n "${EXPECTED_COMMON_EPOCH0_EXECUTION_PROFILE:-}" \
      && "${COMMON_EPOCH0_EXECUTION_PROFILE_USED:-}" \
         != "$EXPECTED_COMMON_EPOCH0_EXECUTION_PROFILE" ]]; then
    echo "common epoch0 execution profile mismatch" >&2
    echo "recorded=${COMMON_EPOCH0_EXECUTION_PROFILE_USED:-<missing>} expected=$EXPECTED_COMMON_EPOCH0_EXECUTION_PROFILE" >&2
    exit 2
fi
if [[ ! -d "$DYNAMIC_INITIAL_BASELINE_DIR/rollout_data" \
      || ! -d "$DYNAMIC_INITIAL_RESUME_CKPT/actor" \
      || ! -f "$DYNAMIC_INITIAL_RESUME_CKPT/.PRESERVE_COMMON_EPOCH0" ]]; then
    echo "reuse.env does not reference a complete preserved epoch0" >&2
    exit 2
fi

export FAIR_OUTPUT_ROOT
export DYNAMIC_OUTPUT_ROOT="$FAIR_OUTPUT_ROOT"
export DYNAMIC_SKIP_MODE0_PROBE=1
FAIR_START_EPOCH="${FAIR_START_EPOCH:-1}"
FAIR_TOTAL_EPOCHS="${FAIR_TOTAL_EPOCHS:-3}"
FAIR_TRAIN_BATCH_SIZE="${FAIR_TRAIN_BATCH_SIZE:-32}"
FAIR_ROLLOUT_N="${FAIR_ROLLOUT_N:-16}"
FAIR_MAX_RESPONSE_LENGTH="${FAIR_MAX_RESPONSE_LENGTH:-16384}"
FAIR_PROMPTS_PER_EPOCH="${FAIR_PROMPTS_PER_EPOCH:-160}"
CHECKPOINT_MODEL_DIR_NAME="${CHECKPOINT_MODEL_DIR_NAME:-qwen3moe_for_eagle3}"
FAIR_EXPECTED_DISTCP_SHARDS="${FAIR_EXPECTED_DISTCP_SHARDS:-32}"
FAIR_PLANNED_MIN_FREE_MIB_FLOOR="${FAIR_PLANNED_MIN_FREE_MIB_FLOOR:-28672}"
if [[ ! "$CHECKPOINT_MODEL_DIR_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid CHECKPOINT_MODEL_DIR_NAME=$CHECKPOINT_MODEL_DIR_NAME" >&2
    exit 2
fi
if [[ "$FAIR_EXPECTED_DISTCP_SHARDS" != auto ]] \
   && ! [[ "$FAIR_EXPECTED_DISTCP_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "FAIR_EXPECTED_DISTCP_SHARDS must be 'auto' or a positive integer" >&2
    exit 2
fi
if ! [[ "$FAIR_PLANNED_MIN_FREE_MIB_FLOOR" =~ ^[0-9]+$ ]]; then
    echo "FAIR_PLANNED_MIN_FREE_MIB_FLOOR must be a nonnegative integer" >&2
    exit 2
fi
export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB_FLOOR="$FAIR_PLANNED_MIN_FREE_MIB_FLOOR"
if ! [[ "$FAIR_START_EPOCH" =~ ^[0-9]+$ && "$FAIR_TOTAL_EPOCHS" =~ ^[0-9]+$ ]] \
   || (( FAIR_START_EPOCH < 1 || FAIR_TOTAL_EPOCHS <= FAIR_START_EPOCH )); then
    echo "invalid FAIR epoch interval: start=$FAIR_START_EPOCH total=$FAIR_TOTAL_EPOCHS" >&2
    exit 2
fi
if ! [[ "$FAIR_TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ \
        && "$FAIR_ROLLOUT_N" =~ ^[1-9][0-9]*$ \
        && "$FAIR_MAX_RESPONSE_LENGTH" =~ ^[1-9][0-9]*$ \
        && "$FAIR_PROMPTS_PER_EPOCH" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid fair workload override" >&2
    exit 2
fi
if (( FAIR_PROMPTS_PER_EPOCH % FAIR_TRAIN_BATCH_SIZE != 0 )); then
    echo "FAIR_PROMPTS_PER_EPOCH must be divisible by FAIR_TRAIN_BATCH_SIZE" >&2
    exit 2
fi
FAIR_STEPS_PER_EPOCH=$((FAIR_PROMPTS_PER_EPOCH / FAIR_TRAIN_BATCH_SIZE))
FAIR_EXPECTED_RESPONSES_PER_STEP=$((FAIR_TRAIN_BATCH_SIZE * FAIR_ROLLOUT_N))
FAIR_TRAIN_STEPS="${FAIR_TRAIN_STEPS:-$FAIR_STEPS_PER_EPOCH}"
if ! [[ "$FAIR_TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] \
   || (( FAIR_TRAIN_STEPS > FAIR_STEPS_PER_EPOCH )); then
    echo "FAIR_TRAIN_STEPS must be between 1 and $FAIR_STEPS_PER_EPOCH" >&2
    exit 2
fi
export DYNAMIC_START_EPOCH="$FAIR_START_EPOCH"
export DYNAMIC_TOTAL_EPOCHS="$FAIR_TOTAL_EPOCHS"
export DYNAMIC_ENABLE_CKPT_CHAIN=1
export DYNAMIC_PLAN_STEPS="$FAIR_STEPS_PER_EPOCH"
export DYNAMIC_TRAIN_STEPS="$FAIR_TRAIN_STEPS"
export DYNAMIC_LENGTH_EMA_DECAY=0.3
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.9
export TRAIN_BATCH_SIZE="$FAIR_TRAIN_BATCH_SIZE"
export ROLLOUT_N="$FAIR_ROLLOUT_N"
export DYNAMIC_FULL_MAX_RESPONSE_LENGTH="$FAIR_MAX_RESPONSE_LENGTH"
export DYNAMIC_FULL_MAX_RESPONSE_LEN="$FAIR_MAX_RESPONSE_LENGTH"

case "$variant" in
    vanilla)
        export FAIR_RUN_NAME="${FAIR_RUN_NAME:-baseline_vanilla_common_epoch0_epoch1_2}"
        target="$SCRIPT_DIR/run_baseline_vanilla_epoch1_2_from_common_epoch0.sh"
        ;;
    lengthsort)
        export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-baseline_lengthsort_noguard_common_epoch0_epoch1_2}"
        export BASELINE_ENABLE_TAIL_GUARD=0
        target="$SCRIPT_DIR/run_baseline_lengthsort_epoch1_2.sh"
        ;;
    lengthsort_guard)
        export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-baseline_lengthsort_tailguard_common_epoch0_epoch1_2}"
        export BASELINE_ENABLE_TAIL_GUARD=1
        target="$SCRIPT_DIR/run_baseline_lengthsort_epoch1_2.sh"
        ;;
    fixed4)
        export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-baseline_fixed4_tailguard_common_epoch0_epoch1_2}"
        target="$SCRIPT_DIR/run_baseline_kvsafe_fixed4_epoch1_2.sh"
        ;;
    minskew)
        export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-baseline_minskew_tailguard_common_epoch0_epoch1_2}"
        target="$SCRIPT_DIR/run_baseline_minskew_epoch1_2.sh"
        ;;
    adafloor_n_f4)
        export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-adafloor_natural_floor4_tailguard_common_epoch0_epoch1_2}"
        target="${FAIR_ADAFLOOR_TARGET:-$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_natural_full3.sh}"
        ;;
    adafloor_n_f2)
        export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-adafloor_natural_floor2_tailguard_common_epoch0_epoch1_2}"
        target="${FAIR_ADAFLOOR_TARGET:-$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh}"
        ;;
    adafloor_p_f4)
        export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-adafloor_planned_floor4_tailguard_common_epoch0_epoch1_2}"
        export RANK_MATCHING_POLICY=release_area
        # Planned groups and dispatcher warmup retain substantially more NPU
        # runtime state than the Natural lifecycle. Release that state at the
        # rollout-to-training boundary and fail early if the reclaimed memory
        # is still below the verified training reserve. The canonical-weight
        # offload experiment is intentionally disabled because it changes
        # tensor storage in a way that is not shape-safe for the next restore.
        export VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING=1
        export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1
        export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=1
        # Release the full-restore runtime views and buffers while retaining
        # the canonical expert-weight tensors. This is the shape-safe path
        # used by the successful Planned F4 validation run.
        planned_min_free_mib="${VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB:-$FAIR_PLANNED_MIN_FREE_MIB_FLOOR}"
        if ! [[ "$planned_min_free_mib" =~ ^[0-9]+$ ]]; then
            echo "invalid Planned F4 training reserve: $planned_min_free_mib" >&2
            exit 2
        fi
        if (( planned_min_free_mib < FAIR_PLANNED_MIN_FREE_MIB_FLOOR )); then
            echo "[fair rerun] raising Planned F4 training reserve from ${planned_min_free_mib} MiB to ${FAIR_PLANNED_MIN_FREE_MIB_FLOOR} MiB" >&2
            planned_min_free_mib=$FAIR_PLANNED_MIN_FREE_MIB_FLOOR
        fi
        export VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1
        export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB="$planned_min_free_mib"
        export VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0
        target="${FAIR_ADAFLOOR_TARGET:-$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh}"
        ;;
    adafloor_p_f2)
        export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-adafloor_planned_floor2_tailguard_common_epoch0_epoch1_2}"
        export RANK_MATCHING_POLICY=release_area
        export VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING=1
        export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1
        export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=1
        planned_min_free_mib="${VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB:-$FAIR_PLANNED_MIN_FREE_MIB_FLOOR}"
        if ! [[ "$planned_min_free_mib" =~ ^[0-9]+$ ]]; then
            echo "invalid Planned F2 training reserve: $planned_min_free_mib" >&2
            exit 2
        fi
        if (( planned_min_free_mib < FAIR_PLANNED_MIN_FREE_MIB_FLOOR )); then
            planned_min_free_mib=$FAIR_PLANNED_MIN_FREE_MIB_FLOOR
        fi
        export VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1
        export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB="$planned_min_free_mib"
        export VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0
        target="${FAIR_ADAFLOOR_TARGET:-$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh}"
        ;;
    adafloor_p_f4_minskew)
        export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-adafloor_planned_floor4_minskew_tailguard_common_epoch0_epoch1_2}"
        export RANK_MATCHING_POLICY=min_skew
        planned_min_free_mib="${VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB:-$FAIR_PLANNED_MIN_FREE_MIB_FLOOR}"
        if ! [[ "$planned_min_free_mib" =~ ^[0-9]+$ ]]; then
            echo "invalid Planned F4 training reserve: $planned_min_free_mib" >&2
            exit 2
        fi
        if (( planned_min_free_mib < FAIR_PLANNED_MIN_FREE_MIB_FLOOR )); then
            planned_min_free_mib=$FAIR_PLANNED_MIN_FREE_MIB_FLOOR
        fi
        export VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING=1
        export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1
        export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=1
        export VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1
        export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB="$planned_min_free_mib"
        export VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0
        target="$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh"
        ;;
    adafloor_n_f2_noguard)
        export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-adafloor_natural_floor2_noguard_common_epoch0_epoch1_2}"
        target="$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor2_natural_noguard_reuse_epoch0_2epoch.sh"
        ;;
    *)
        echo "unknown variant: $variant" >&2
        usage >&2
        exit 2
        ;;
esac

run_name="${FAIR_RUN_NAME:-$DYNAMIC_RUN_NAME}"
finalize_existing="${FAIR_FINALIZE_EXISTING:-0}"
if [[ "$finalize_existing" != "0" && "$finalize_existing" != "1" ]]; then
    echo "FAIR_FINALIZE_EXISTING must be 0 or 1: $finalize_existing" >&2
    exit 2
fi
if [[ -e "$FAIR_OUTPUT_ROOT/$run_name" && "$finalize_existing" != "1" ]]; then
    echo "refusing to overwrite fair rerun output: $FAIR_OUTPUT_ROOT/$run_name" >&2
    exit 2
fi
if [[ ! -d "$FAIR_OUTPUT_ROOT/$run_name" && "$finalize_existing" == "1" ]]; then
    echo "cannot finalize missing fair rerun output: $FAIR_OUTPUT_ROOT/$run_name" >&2
    exit 2
fi

echo "[fair rerun] variant=$variant"
echo "[fair rerun] epoch0_history=$DYNAMIC_INITIAL_BASELINE_DIR"
echo "[fair rerun] epoch0_checkpoint=$DYNAMIC_INITIAL_RESUME_CKPT"
echo "[fair rerun] output=$FAIR_OUTPUT_ROOT/$run_name"
echo "[fair rerun] target=$target"
echo "[fair rerun] workload=train_batch_size:$FAIR_TRAIN_BATCH_SIZE rollout_n:$FAIR_ROLLOUT_N max_response_length:$FAIR_MAX_RESPONSE_LENGTH planner_prompts:$FAIR_PROMPTS_PER_EPOCH plan_steps:$FAIR_STEPS_PER_EPOCH train_steps:$FAIR_TRAIN_STEPS expected_responses_per_step:$FAIR_EXPECTED_RESPONSES_PER_STEP"
if is_planned_variant; then
    echo "[fair rerun] training_boundary_transient_cleanup=unconditional"
    echo "[fair rerun] planned_training_runtime_release=$VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING"
    echo "[fair rerun] planned_training_memory_guard=$VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD"
    echo "[fair rerun] planned_training_memory_guard_strict=$VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT"
    echo "[fair rerun] planned_training_min_free_mib=$VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB"
    echo "[fair rerun] full_restore_transient_cleanup=$VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE"
    echo "[fair rerun] canonical_loaded_weight_offload=$VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE"
fi
if [[ "${FAIR_DRY_RUN:-0}" == "1" ]]; then
    echo "[fair rerun] dry run only"
    exit 0
fi

if [[ "$finalize_existing" != "1" ]]; then
    if [[ "${FAIR_SKIP_RAY_PREFLIGHT:-0}" != "1" ]]; then
        if ! prepare_clean_ray_runtime; then
            echo "[fair rerun] Ray preflight failed; refusing to start $variant" >&2
            exit 3
        fi
    fi
    if ! require_idle_npus; then
        echo "[fair rerun] NPU preflight failed; refusing to start $variant" >&2
        exit 3
    fi

    # Keep these overrides last so every policy uses the same workload and
    # sampling configuration even if a lower-level launcher has different defaults.
    fair_hydra_args=(
        data.train_batch_size="$FAIR_TRAIN_BATCH_SIZE"
        data.max_prompt_length=1024
        data.max_response_length="$FAIR_MAX_RESPONSE_LENGTH"
        data.shuffle=False
        actor_rollout_ref.rollout.n="$FAIR_ROLLOUT_N"
        actor_rollout_ref.rollout.temperature=0.9
        actor_rollout_ref.rollout.top_p=0.9
        actor_rollout_ref.rollout.top_k=50
    )
    if [[ "${FAIR_FREEZE_ACTOR:-0}" == "1" ]]; then
        fair_hydra_args+=(actor_rollout_ref.actor.optim.lr=0.0)
    fi

    set +e
    "$target" "$@" "${fair_hydra_args[@]}"
    run_rc=$?
    set -e
    if (( run_rc != 0 )); then
        echo "[fair rerun] variant=$variant failed with rc=$run_rc; checkpoints retained" >&2
        exit "$run_rc"
    fi
else
    echo "[fair rerun] finalize-existing mode; training launch skipped"
fi

keep_completed_checkpoints=${FAIR_KEEP_COMPLETED_CHECKPOINTS:-0}
if [[ "$keep_completed_checkpoints" != "0" \
      && "$keep_completed_checkpoints" != "1" ]]; then
    echo "FAIR_KEEP_COMPLETED_CHECKPOINTS must be 0 or 1" >&2
    exit 2
fi
if [[ "$finalize_existing" == "1" \
      && "$keep_completed_checkpoints" == "1" ]]; then
    echo "FAIR_FINALIZE_EXISTING=1 cannot be combined with FAIR_KEEP_COMPLETED_CHECKPOINTS=1" >&2
    exit 2
fi

run_dir=$(realpath -m "$FAIR_OUTPUT_ROOT/$run_name")
fair_output_root_real=$(realpath -m "$FAIR_OUTPUT_ROOT")
common_epoch0_root_real=$(realpath -m "$COMMON_EPOCH0_ROOT")
if [[ "$run_dir" != "$fair_output_root_real/"* \
      || "$run_dir" == "$common_epoch0_root_real" \
      || "$run_dir" == "$common_epoch0_root_real/"* ]]; then
    echo "[fair rerun] refusing unsafe checkpoint cleanup path: $run_dir" >&2
    exit 4
fi

cleanup_record="$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
cleanup_pending="$run_dir/.CHECKPOINTS_VALIDATED_PENDING_REMOVAL.txt"
retained_record="$run_dir/CHECKPOINTS_RETAINED_AFTER_VALIDATION.txt"
expected_epoch_tags=()
for (( epoch = FAIR_START_EPOCH; epoch < FAIR_TOTAL_EPOCHS; epoch++ )); do
    expected_epoch_tags+=("$(printf '%03d' "$epoch")")
done
expected_epochs_csv=$(IFS=,; echo "${expected_epoch_tags[*]}")
if [[ -f "$cleanup_record" ]]; then
    if ! grep -qFx "variant=$variant" "$cleanup_record" \
       || ! grep -qFx "validated_epochs=$expected_epochs_csv" "$cleanup_record" \
       || find "$run_dir" -mindepth 2 -maxdepth 2 -type d -name checkpoints \
            -print -quit | grep -q .; then
        echo "[fair rerun] existing cleanup record is inconsistent: $cleanup_record" >&2
        exit 4
    fi
    echo "[fair rerun] checkpoints were already validated and removed: $cleanup_record"
    exit 0
fi
if [[ -f "$cleanup_pending" ]]; then
    if ! grep -qFx "variant=$variant" "$cleanup_pending" \
       || ! grep -qFx "validated_epochs=$expected_epochs_csv" "$cleanup_pending"; then
        echo "[fair rerun] invalid pending cleanup transaction: $cleanup_pending" >&2
        exit 4
    fi
    mapfile -t pending_checkpoint_dirs < <(
        find "$run_dir" -mindepth 2 -maxdepth 2 -type d \
            -name checkpoints -print | sort
    )
    for checkpoint_dir in "${pending_checkpoint_dirs[@]}"; do
        if [[ "$checkpoint_dir" != "$run_dir/"epoch_*/checkpoints ]]; then
            echo "[fair rerun] refusing pending cleanup path: $checkpoint_dir" >&2
            exit 4
        fi
    done
    if (( ${#pending_checkpoint_dirs[@]} > 0 )); then
        rm -r -- "${pending_checkpoint_dirs[@]}"
    fi
    mv -f -- "$cleanup_pending" "$cleanup_record"
    rm -f -- "$retained_record"
    echo "[fair rerun] resumed and completed checkpoint cleanup: $cleanup_record"
    exit 0
fi

validated_epoch_dirs=()
for epoch_tag in "${expected_epoch_tags[@]}"; do
    if ! validate_completed_epoch "$run_dir" "$epoch_tag"; then
        echo "[fair rerun] variant=$variant did not pass cleanup validation; checkpoints retained" >&2
        exit 4
    fi
    validated_epoch_dirs+=("$VALIDATED_EPOCH_DIR")
done

checkpoint_dirs=()
for epoch_dir in "${validated_epoch_dirs[@]}"; do
    checkpoint_dir="$epoch_dir/checkpoints"
    if [[ "$checkpoint_dir" != "$run_dir/"epoch_*/checkpoints \
          || ! -d "$checkpoint_dir" ]]; then
        echo "[fair rerun] refusing unexpected checkpoint path: $checkpoint_dir" >&2
        exit 4
    fi
    checkpoint_dirs+=("$checkpoint_dir")
done

echo "[fair rerun] validation passed for epochs $expected_epochs_csv"
if [[ "$keep_completed_checkpoints" == "1" ]]; then
    retained_pending="$run_dir/.CHECKPOINTS_VALIDATED_RETAINED.tmp.$$"
    {
        printf 'variant=%s\n' "$variant"
        printf 'validated_epochs=%s\n' "$expected_epochs_csv"
        printf 'common_epoch0_checkpoint=%s\n' "$DYNAMIC_INITIAL_RESUME_CKPT"
        printf 'common_epoch0_preserved=true\n'
        printf 'checkpoints_retained=true\n'
        printf 'freeze_actor=%s\n' "${FAIR_FREEZE_ACTOR:-0}"
        printf 'paired_request_sampling_seeds=%s\n' \
            "${VERL_PAIRED_REQUEST_SAMPLING_SEEDS:-0}"
        printf 'train_batch_size=%s\n' "$FAIR_TRAIN_BATCH_SIZE"
        printf 'rollout_n=%s\n' "$FAIR_ROLLOUT_N"
        printf 'max_response_length=%s\n' "$FAIR_MAX_RESPONSE_LENGTH"
        printf 'prompts_per_epoch=%s\n' "$FAIR_PROMPTS_PER_EPOCH"
        printf 'plan_steps_per_epoch=%s\n' "$FAIR_STEPS_PER_EPOCH"
        printf 'steps_per_epoch=%s\n' "$FAIR_TRAIN_STEPS"
        printf 'executed_prompts=%s\n' "$((FAIR_TRAIN_STEPS * FAIR_TRAIN_BATCH_SIZE))"
    } > "$retained_pending"
    mv -f -- "$retained_pending" "$retained_record"
    echo "[fair rerun] validated checkpoints retained: $retained_record"
    exit 0
fi
{
    printf 'variant=%s\n' "$variant"
    printf 'validated_epochs=%s\n' "$expected_epochs_csv"
    printf 'removed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'common_epoch0_checkpoint=%s\n' "$DYNAMIC_INITIAL_RESUME_CKPT"
    printf 'common_epoch0_preserved=true\n'
    if is_planned_variant; then
        printf 'training_boundary_transient_cleanup=unconditional\n'
        printf 'planned_training_runtime_release=%s\n' "$VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING"
        printf 'planned_training_memory_guard=%s\n' "$VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD"
        printf 'planned_training_memory_guard_strict=%s\n' "$VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT"
        printf 'planned_training_min_free_mib=%s\n' "$VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB"
        printf 'full_restore_transient_cleanup=%s\n' "$VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE"
        printf 'canonical_loaded_weight_offload=%s\n' "$VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE"
    fi
    printf 'freeze_actor=%s\n' "${FAIR_FREEZE_ACTOR:-0}"
    printf 'paired_request_sampling_seeds=%s\n' "${VERL_PAIRED_REQUEST_SAMPLING_SEEDS:-0}"
    printf 'train_batch_size=%s\n' "$FAIR_TRAIN_BATCH_SIZE"
    printf 'rollout_n=%s\n' "$FAIR_ROLLOUT_N"
    printf 'max_response_length=%s\n' "$FAIR_MAX_RESPONSE_LENGTH"
    printf 'prompts_per_epoch=%s\n' "$FAIR_PROMPTS_PER_EPOCH"
    printf 'plan_steps_per_epoch=%s\n' "$FAIR_STEPS_PER_EPOCH"
    printf 'steps_per_epoch=%s\n' "$FAIR_TRAIN_STEPS"
    printf 'executed_prompts=%s\n' "$((FAIR_TRAIN_STEPS * FAIR_TRAIN_BATCH_SIZE))"
} > "$cleanup_pending"
echo "[fair rerun] deleting completed checkpoints under $run_dir"
rm -r -- "${checkpoint_dirs[@]}"
mv -f -- "$cleanup_pending" "$cleanup_record"
rm -f -- "$retained_record"
echo "[fair rerun] checkpoint cleanup complete; record=$cleanup_record"
