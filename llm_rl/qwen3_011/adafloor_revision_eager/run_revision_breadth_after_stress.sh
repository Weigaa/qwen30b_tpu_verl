#!/usr/bin/env bash
set -euo pipefail

if [[ "${ADAFLOOR_BREADTH_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    source_path=$(realpath "${BASH_SOURCE[0]}")
    snapshot=$(mktemp "${source_path}.run-snapshot.XXXXXX")
    cp -- "$source_path" "$snapshot"
    chmod 700 "$snapshot"
    set +e
    ADAFLOOR_BREADTH_SNAPSHOT_ACTIVE=1 "$snapshot" "$@"
    rc=$?
    set -e
    rm -f -- "$snapshot"
    exit "$rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

P0_ROOT="${P0_ROOT:-/data/adafloor_shared_state/p0_matched_trials_common_epoch0_20260728T221830Z}"
STRESS_ROOT="${STRESS_ROOT:-$P0_ROOT/revision_stress}"
BREADTH_ROOT="${BREADTH_ROOT:-$P0_ROOT/revision_breadth_single_validation}"
COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
BREADTH_SEED="${BREADTH_SEED:-808}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-216000}"
MIN_FREE_GIB="${MIN_FREE_GIB:-180}"
STATUS_TSV="$BREADTH_ROOT/status.tsv"
FAILED_JOBS=0

if ! [[ "$BREADTH_SEED" =~ ^[0-9]+$ && "$MIN_FREE_GIB" =~ ^[0-9]+$ ]]; then
    echo "invalid breadth experiment setting" >&2
    exit 2
fi
if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]]; then
    echo "protected common epoch0 is missing" >&2
    exit 2
fi

waited=0
while [[ ! -f "$STRESS_ROOT/TRANSITION_STRESS_COMPLETE.txt" ]]; do
    if (( waited >= WAIT_TIMEOUT_SECONDS )); then
        echo "timed out waiting for transition stress" >&2
        exit 3
    fi
    printf '[revision breadth] waiting elapsed_s=%s\n' "$waited"
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
done

mkdir -p "$BREADTH_ROOT"
rm -f -- "$BREADTH_ROOT/BREADTH_SINGLE_VALIDATION_COMPLETE.txt"
if [[ ! -f "$STATUS_TSV" ]]; then
    printf 'job\tstarted_at_utc\tfinished_at_utc\trc\toutput\n' > "$STATUS_TSV"
fi

check_disk() {
    local free_kib required_kib
    free_kib=$(df --output=avail -k "$BREADTH_ROOT" | tail -1 | tr -d '[:space:]')
    required_kib=$((MIN_FREE_GIB * 1024 * 1024))
    if ! [[ "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < required_kib )); then
        echo "insufficient disk for breadth experiment: free_kib=${free_kib:-unknown}" >&2
        return 1
    fi
    echo "[revision breadth] disk_free_gib=$((free_kib / 1024 / 1024))"
}

record_job() {
    local job="$1" started="$2" finished="$3" rc="$4" output="$5"
    local status_tmp
    status_tmp=$(mktemp "${STATUS_TSV}.tmp.XXXXXX")
    awk -F '\t' -v job="$job" 'NR == 1 || $1 != job' "$STATUS_TSV" > "$status_tmp"
    printf '%s\t%s\t%s\t%s\t%s\n' "$job" "$started" "$finished" "$rc" "$output" >> "$status_tmp"
    mv -- "$status_tmp" "$STATUS_TSV"
}

remove_stale_failed_runs() {
    local run_dir="$1" stale
    for stale in "${run_dir}.failed."*; do
        [[ -e "$stale" ]] || continue
        rm -rf -- "$stale"
    done
}

run_fair_variant() {
    local label="$1" variant="$2" train_batch_size="$3" rollout_n="$4" max_response="$5"
    local ppo_mini_batch_size="${6:-$train_batch_size}"
    local job="${label}_${variant}"
    local output_root="$BREADTH_ROOT/workload_sensitivity/$label"
    local run_name="${job}_seed${BREADTH_SEED}_epoch1"
    local run_dir="$output_root/$run_name"
    local started finished rc
    if [[ -f "$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; then
        remove_stale_failed_runs "$run_dir"
        echo "[revision breadth] already complete job=$job"
        return 0
    fi
    if [[ -e "$run_dir" ]]; then
        mv -- "$run_dir" "${run_dir}.failed.$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    check_disk || return 1
    started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    set +e
    if [[ "$variant" == "vanilla" ]]; then
        env -u DYNAMIC_RUN_NAME \
            COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
            FAIR_OUTPUT_ROOT="$output_root" \
            FAIR_RUN_NAME="$run_name" \
            FAIR_START_EPOCH=1 \
            FAIR_TOTAL_EPOCHS=2 \
            FAIR_FREEZE_ACTOR=1 \
            FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
            FAIR_PROMPTS_PER_EPOCH=160 \
            FAIR_TRAIN_BATCH_SIZE="$train_batch_size" \
            FAIR_ROLLOUT_N="$rollout_n" \
            FAIR_MAX_RESPONSE_LENGTH="$max_response" \
            VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
                "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
                    "$variant" \
                    "actor_rollout_ref.rollout.seed=$BREADTH_SEED" \
                    "actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size"
    else
        env -u FAIR_RUN_NAME \
            COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
            FAIR_OUTPUT_ROOT="$output_root" \
            DYNAMIC_RUN_NAME="$run_name" \
            FAIR_START_EPOCH=1 \
            FAIR_TOTAL_EPOCHS=2 \
            FAIR_FREEZE_ACTOR=1 \
            FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
            FAIR_PROMPTS_PER_EPOCH=160 \
            FAIR_TRAIN_BATCH_SIZE="$train_batch_size" \
            FAIR_ROLLOUT_N="$rollout_n" \
            FAIR_MAX_RESPONSE_LENGTH="$max_response" \
            VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
                "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
                    "$variant" \
                    "actor_rollout_ref.rollout.seed=$BREADTH_SEED" \
                    "actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size"
    fi
    rc=$?
    set -e
    finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    record_job "$job" "$started" "$finished" "$rc" "$run_dir"
    if (( rc != 0 )); then
        FAILED_JOBS=$((FAILED_JOBS + 1))
        echo "[revision breadth] failed job=$job rc=$rc output=$run_dir" >&2
    else
        remove_stale_failed_runs "$run_dir"
    fi
    return 0
}

# One paired epoch per sensitivity point is enough to establish direction.
run_fair_variant response_cap8192 vanilla 32 16 8192
run_fair_variant response_cap8192 adafloor_n_f2 32 16 8192
run_fair_variant rollout_n8 vanilla 32 8 16384
run_fair_variant rollout_n8 adafloor_n_f2 32 8 16384
run_fair_variant batch16 vanilla 16 16 16384
run_fair_variant batch16 adafloor_n_f2 16 16 16384

python3 "$SCRIPT_DIR/analysis_eval/summarize_revision_breadth.py" \
    --root "$BREADTH_ROOT" --output "$BREADTH_ROOT/summary.md"
if (( FAILED_JOBS != 0 )); then
    rm -f -- "$BREADTH_ROOT/BREADTH_SINGLE_VALIDATION_COMPLETE.txt"
    {
        printf 'failed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'failed_jobs=%s\n' "$FAILED_JOBS"
        printf 'status=%s\n' "$STATUS_TSV"
    } > "$BREADTH_ROOT/BREADTH_SINGLE_VALIDATION_INCOMPLETE.txt"
    echo "[revision breadth] incomplete failed_jobs=$FAILED_JOBS" >&2
    exit 4
fi
rm -f -- "$BREADTH_ROOT/BREADTH_SINGLE_VALIDATION_INCOMPLETE.txt"
{
    printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'status=%s\n' "$STATUS_TSV"
    printf 'summary=%s\n' "$BREADTH_ROOT/summary.md"
} > "$BREADTH_ROOT/BREADTH_SINGLE_VALIDATION_COMPLETE.txt"
echo "[revision breadth] complete output=$BREADTH_ROOT"
