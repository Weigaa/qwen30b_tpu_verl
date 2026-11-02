#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

P0_ROOT="${P0_ROOT:-/data/adafloor_shared_state/p0_matched_trials_common_epoch0_20260728T221830Z}"
COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
QUEUE_LOG="${QUEUE_LOG:-$SCRIPT_DIR/analysis_eval/runtime_logs/revision_missing_experiments_queue.log}"
QUEUE_ROOT="$P0_ROOT/revision_queue"
CAPTURE_PID=""

mkdir -p "$(dirname "$QUEUE_LOG")" "$QUEUE_ROOT"
exec > >(tee -a "$QUEUE_LOG") 2>&1

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -n "$CAPTURE_PID" ]] && kill -0 "$CAPTURE_PID" 2>/dev/null; then
        kill "$CAPTURE_PID" 2>/dev/null || true
        wait "$CAPTURE_PID" 2>/dev/null || true
    fi
    if (( rc != 0 )); then
        {
            printf 'failed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'rc=%s\n' "$rc"
            printf 'log=%s\n' "$QUEUE_LOG"
        } > "$QUEUE_ROOT/QUEUE_INCOMPLETE.txt"
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]]; then
    echo "protected common epoch0 is missing: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi
if pgrep -f '[p]ython3 -m verl\.trainer\.main_ppo|[r]ay::TaskRunner\.run' >/dev/null; then
    echo "another training process is active" >&2
    exit 3
fi

rm -f -- "$QUEUE_ROOT/QUEUE_INCOMPLETE.txt" "$QUEUE_ROOT/QUEUE_COMPLETE.txt"
{
    printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
    printf 'policy=missing_evidence_only_single_validation\n'
    printf 'log=%s\n' "$QUEUE_LOG"
} > "$QUEUE_ROOT/protocol.env"

echo "[revision queue] start missing-evidence-only queue"
P0_ROOT="$P0_ROOT" \
COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    "$SCRIPT_DIR/capture_hccs_hbm_from_ranktime_natural.sh" &
CAPTURE_PID=$!

P0_ROOT="$P0_ROOT" \
COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
EXPECTED_TRIALS=4 \
ORACLE_TRIAL_SEEDS=101 \
ACCOUNTING_SEED=404 \
ACCOUNTING_ARCHIVE_INCOMPLETE=1 \
    "$SCRIPT_DIR/run_revision_followups_after_p0.sh"

wait "$CAPTURE_PID"
CAPTURE_PID=""

P0_ROOT="$P0_ROOT" \
COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
RUN_QUALITY_50STEP=0 \
    "$SCRIPT_DIR/run_revision_remaining_after_followups.sh"

P0_ROOT="$P0_ROOT" \
COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    "$SCRIPT_DIR/run_revision_stress_after_remaining.sh"

P0_ROOT="$P0_ROOT" \
COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    "$SCRIPT_DIR/run_revision_breadth_after_stress.sh"

{
    printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'followups=%s\n' "$P0_ROOT/revision_followups/FOLLOWUPS_COMPLETE.txt"
    printf 'architecture=%s\n' "$P0_ROOT/revision_remaining/architecture_hccs_hbm/ARCHITECTURE_PROBE_COMPLETE.txt"
    printf 'stress=%s\n' "$P0_ROOT/revision_stress/TRANSITION_STRESS_COMPLETE.txt"
    printf 'breadth=%s\n' "$P0_ROOT/revision_breadth_single_validation/BREADTH_SINGLE_VALIDATION_COMPLETE.txt"
} > "$QUEUE_ROOT/QUEUE_COMPLETE.txt"
echo "[revision queue] complete marker=$QUEUE_ROOT/QUEUE_COMPLETE.txt"
