#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

RUN_ROOT=${RETRY_RUN_ROOT:-/data/adafloor_shared_state/adafloor_p_f4_training_guard_seed2_20260725T065522Z}
RUN_NAME=${RETRY_RUN_NAME:-adafloor_planned_floor4_tailguard_common_epoch0_epoch1_2}
RUN_DIR="$RUN_ROOT/$RUN_NAME"
SOURCE_RUN_ROOT=${RETRY_SOURCE_RUN_ROOT:-$RUN_ROOT}
SOURCE_RUN_NAME=${RETRY_SOURCE_RUN_NAME:-$RUN_NAME}
SOURCE_RUN_DIR="$SOURCE_RUN_ROOT/$SOURCE_RUN_NAME"
EPOCH1_DIR="$SOURCE_RUN_DIR/epoch_001_mode1_planned"
EPOCH2_DIR="$RUN_DIR/epoch_002_mode1_planned"
EPOCH1_CKPT="$EPOCH1_DIR/checkpoints/qwen3moe_for_eagle3/global_step_5"
COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}
REUSE_ENV="$COMMON_EPOCH0_ROOT/reuse.env"

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
        echo "[epoch2 retry] another VERL run is active:" >&2
        echo "$active_training" >&2
        return 1
    fi

    stale_ray=$(list_matching_processes \
        'ray::WorkerDict|ray::TaskRunner|gcs_server|raylet|dashboard\.py|runtime_env_agent|log_monitor\.py')
    if [[ -z "$stale_ray" ]]; then
        echo "[epoch2 retry] Ray preflight clean"
        return 0
    fi

    echo "[epoch2 retry] stopping stale Ray processes"
    ray stop --force >/dev/null 2>&1 || true
    for attempt in $(seq 1 15); do
        stale_ray=$(list_matching_processes \
            'ray::WorkerDict|ray::TaskRunner|gcs_server|raylet|dashboard\.py|runtime_env_agent|log_monitor\.py')
        if [[ -z "$stale_ray" ]]; then
            echo "[epoch2 retry] stale Ray processes cleared"
            return 0
        fi
        sleep 1
    done

    echo "[epoch2 retry] stale Ray processes remain:" >&2
    echo "$stale_ray" >&2
    return 1
}

validate_checkpoint() {
    local epoch_dir="$1"
    local marker="$epoch_dir/checkpoints/qwen3moe_for_eagle3/latest_checkpointed_iteration.txt"
    local distcp_dir="$epoch_dir/checkpoints/qwen3moe_for_eagle3/global_step_5/actor/dist_ckpt"
    local distcp_count

    if [[ ! -f "$marker" || $(tr -d '[:space:]' < "$marker") != 5 ]]; then
        echo "[epoch2 retry] invalid checkpoint marker under $epoch_dir" >&2
        return 1
    fi
    distcp_count=$(find "$distcp_dir" -maxdepth 1 -type f -name '*.distcp' 2>/dev/null | wc -l)
    if (( distcp_count != 32 )); then
        echo "[epoch2 retry] expected 32 checkpoint shards under $distcp_dir, found $distcp_count" >&2
        return 1
    fi
}

validate_completed_epoch() {
    local epoch_dir="$1"
    local rollout_count
    local length_count
    local metric_count
    local timing_count
    local log_file
    local file

    rollout_count=$(find "$epoch_dir/rollout_data" -maxdepth 1 -type f -name '*.jsonl' 2>/dev/null | wc -l)
    length_count=$(find "$epoch_dir/rollout_length" -maxdepth 1 -type f -name 'length_*.txt' 2>/dev/null | wc -l)
    if (( rollout_count != 5 || length_count != 5 )); then
        echo "[epoch2 retry] incomplete rollout artifacts under $epoch_dir" >&2
        return 1
    fi
    while IFS= read -r file; do
        if (( $(wc -l < "$file") != 512 )); then
            echo "[epoch2 retry] expected 512 records in $file" >&2
            return 1
        fi
    done < <(find "$epoch_dir/rollout_data" "$epoch_dir/rollout_length" \
        -maxdepth 1 -type f \( -name '*.jsonl' -o -name 'length_*.txt' \) | sort -V)

    log_file=$(find "$epoch_dir/logs" -maxdepth 1 -type f -name '*.txt' \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [[ -z "$log_file" || ! -f "$log_file" ]]; then
        echo "[epoch2 retry] log is missing under $epoch_dir" >&2
        return 1
    fi
    timing_count=$(grep -cF 'rollout_output_time_s:' "$log_file" || true)
    metric_count=$(grep -cE 'step:[1-5] - ' "$log_file" || true)
    if (( timing_count != 5 || metric_count != 5 )) \
       || ! grep -qF 'Training Progress: 100%' "$log_file" \
       || ! grep -qF 'After trainer.fit' "$log_file"; then
        echo "[epoch2 retry] completion markers are missing in $log_file" >&2
        return 1
    fi
    if grep -qE 'response/aborted_ratio:(0\.[0-9]*[1-9]|[1-9])' "$log_file"; then
        echo "[epoch2 retry] aborted responses found in $log_file" >&2
        return 1
    fi
    if grep -qE 'NPU out of memory|OutOfMemoryError' "$log_file"; then
        echo "[epoch2 retry] OOM failure found in $log_file" >&2
        return 1
    fi
    validate_checkpoint "$epoch_dir"
}

if [[ ! -f "$REUSE_ENV" || ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]]; then
    echo "[epoch2 retry] preserved common epoch0 is incomplete: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "$REUSE_ENV"
COMMON_EPOCH0_HISTORY="$DYNAMIC_INITIAL_BASELINE_DIR"
COMMON_EPOCH0_CKPT="$DYNAMIC_INITIAL_RESUME_CKPT"

if [[ ! -d "$COMMON_EPOCH0_HISTORY/rollout_data" \
      || ! -f "$COMMON_EPOCH0_CKPT/.PRESERVE_COMMON_EPOCH0" ]]; then
    echo "[epoch2 retry] reuse.env does not reference the preserved common epoch0" >&2
    exit 2
fi
if [[ ! -d "$EPOCH1_DIR" || ! -d "$EPOCH1_CKPT/actor" ]]; then
    echo "[epoch2 retry] retained epoch1 checkpoint is missing: $EPOCH1_CKPT" >&2
    exit 2
fi
validate_completed_epoch "$EPOCH1_DIR"

echo "[epoch2 retry] run_dir=$RUN_DIR"
echo "[epoch2 retry] history=$COMMON_EPOCH0_HISTORY:$EPOCH1_DIR"
echo "[epoch2 retry] resume_ckpt=$EPOCH1_CKPT"
echo "[epoch2 retry] forced_moe_runtime_release_before_training=1"
echo "[epoch2 retry] full_restore_transient_cleanup=1"
echo "[epoch2 retry] canonical_loaded_weight_offload=0"
if [[ "${RETRY_DRY_RUN:-0}" == 1 ]]; then
    echo "[epoch2 retry] dry run only"
    exit 0
fi

prepare_clean_ray_runtime

if [[ -e "$EPOCH2_DIR" ]]; then
    archive_dir="${EPOCH2_DIR}_failed_before_canonical_offload_$(date -u +%Y%m%dT%H%M%SZ)"
    echo "[epoch2 retry] archiving incomplete epoch2 to $archive_dir"
    mv -- "$EPOCH2_DIR" "$archive_dir"
fi

set +e
DYNAMIC_OUTPUT_ROOT="$RUN_ROOT" \
DYNAMIC_RUN_NAME="$RUN_NAME" \
DYNAMIC_SKIP_MODE0_PROBE=1 \
DYNAMIC_START_EPOCH=2 \
DYNAMIC_TOTAL_EPOCHS=3 \
DYNAMIC_ENABLE_CKPT_CHAIN=1 \
DYNAMIC_PLAN_STEPS=5 \
DYNAMIC_TRAIN_STEPS=5 \
DYNAMIC_LENGTH_EMA_DECAY=0.3 \
DYNAMIC_INITIAL_BASELINE_DIR="$COMMON_EPOCH0_HISTORY:$EPOCH1_DIR" \
DYNAMIC_INITIAL_RESUME_CKPT="$EPOCH1_CKPT" \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 \
VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING=1 \
VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1 \
VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=1 \
VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1 \
VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB="${VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB:-28672}" \
VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0 \
./run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh \
    actor_rollout_ref.rollout.seed=2 \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.shuffle=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=50 \
    "$@"
run_rc=$?
set -e

if (( run_rc != 0 )); then
    echo "[epoch2 retry] epoch2 failed with rc=$run_rc; epoch1 and epoch2 checkpoints retained" >&2
    exit "$run_rc"
fi

if ! validate_completed_epoch "$EPOCH2_DIR"; then
    echo "[epoch2 retry] epoch2 validation failed; checkpoints retained" >&2
    exit 4
fi

echo "[epoch2 retry] epoch2 validation passed"
if [[ "${RETRY_KEEP_COMPLETED_CHECKPOINTS:-0}" == 1 ]]; then
    echo "[epoch2 retry] completed checkpoints retained by request"
    exit 0
fi

rm -r -- "$EPOCH2_DIR/checkpoints"
source_epoch1_checkpoint_removed=false
if [[ "${RETRY_REMOVE_SOURCE_EPOCH1_CHECKPOINT:-0}" == 1 ]]; then
    rm -r -- "$EPOCH1_DIR/checkpoints"
    source_epoch1_checkpoint_removed=true
fi
cleanup_record="$RUN_DIR/CHECKPOINTS_REMOVED_AFTER_EPOCH2_RETRY_VALIDATION.txt"
{
    printf 'validated_epochs=001,002\n'
    printf 'forced_moe_runtime_release_before_training=1\n'
    printf 'full_restore_transient_cleanup=1\n'
    printf 'canonical_loaded_weight_offload=0\n'
    printf 'source_epoch1_checkpoint_removed=%s\n' "$source_epoch1_checkpoint_removed"
    printf 'removed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'common_epoch0_checkpoint=%s\n' "$COMMON_EPOCH0_CKPT"
    printf 'common_epoch0_preserved=true\n'
} > "$cleanup_record"
echo "[epoch2 retry] completed epoch2 checkpoint removed; common epoch0 preserved"
echo "[epoch2 retry] cleanup_record=$cleanup_record"
