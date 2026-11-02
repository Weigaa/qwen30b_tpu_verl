#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

P0_ROOT="${P0_ROOT:-/data/adafloor_shared_state/p0_matched_trials_common_epoch0_20260728T221830Z}"
FOLLOWUP_ROOT="${FOLLOWUP_ROOT:-$P0_ROOT/revision_followups}"
REMAINING_ROOT="${REMAINING_ROOT:-$P0_ROOT/revision_remaining}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-604800}"
COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
RUN_QUALITY_50STEP="${RUN_QUALITY_50STEP:-0}"
EXISTING_E2E_SUMMARY="${EXISTING_E2E_SUMMARY:-$SCRIPT_DIR/analysis_eval/existing_e2e_repetitions_20260729/summary.md}"

if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ \
        && "$WAIT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ \
        && "$RUN_QUALITY_50STEP" =~ ^[01]$ ]]; then
    echo "invalid remaining-queue timing" >&2
    exit 2
fi
if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]]; then
    echo "missing preserved common epoch0" >&2
    exit 2
fi
mkdir -p "$REMAINING_ROOT"

waited=0
while [[ ! -f "$FOLLOWUP_ROOT/FOLLOWUPS_COMPLETE.txt" ]]; do
    if (( waited >= WAIT_TIMEOUT_SECONDS )); then
        echo "timed out waiting for revision follow-ups" >&2
        exit 3
    fi
    printf '[revision remaining] waiting for follow-ups elapsed_s=%s\n' "$waited"
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
done

echo "[revision remaining] follow-ups complete"
architecture_marker="$REMAINING_ROOT/architecture_hccs_hbm/ARCHITECTURE_PROBE_COMPLETE.txt"
# The sampler reuses the Natural F2 rank-time run and may need a few seconds to
# finish its summaries after the follow-up runner writes its completion marker.
if [[ ! -f "$architecture_marker" ]]; then
    for _ in $(seq 1 30); do
        [[ -f "$architecture_marker" ]] && break
        sleep 2
    done
fi
if [[ -f "$architecture_marker" ]]; then
    echo "[revision remaining] reused HCCS/HBM evidence from rank-time Natural F2"
else
    COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    ARCH_OUTPUT_ROOT="$REMAINING_ROOT/architecture_hccs_hbm" \
        "$SCRIPT_DIR/run_paper_hccs_hbm_probe_from_common_epoch0.sh"
fi

if [[ "$RUN_QUALITY_50STEP" == "1" ]]; then
    COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    QUALITY_OUTPUT_ROOT="$REMAINING_ROOT/quality_50step" \
        "$SCRIPT_DIR/run_paper_p1_quality_50step_from_common_epoch0.sh"
    quality_summary="$REMAINING_ROOT/quality_50step/summary/quality_curve_summary.md"
else
    if [[ ! -s "$EXISTING_E2E_SUMMARY" ]]; then
        echo "existing three-seed quality summary is missing: $EXISTING_E2E_SUMMARY" >&2
        exit 4
    fi
    quality_summary="$EXISTING_E2E_SUMMARY"
    {
        printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'source=%s\n' "$EXISTING_E2E_SUMMARY"
        printf 'new_npu_runs=0\n'
        printf 'reason=deadline_reuse_of_three_seed_ten_step_training_trajectories\n'
    } > "$REMAINING_ROOT/QUALITY_REUSED_EXISTING.txt"
    echo "[revision remaining] reused existing 30-step-per-policy quality evidence"
fi

{
    printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'architecture=%s\n' "$architecture_marker"
    printf 'quality=%s\n' "$quality_summary"
} > "$REMAINING_ROOT/REMAINING_PHASE_COMPLETE.txt"
echo "[revision remaining] phase complete marker=$REMAINING_ROOT/REMAINING_PHASE_COMPLETE.txt"
