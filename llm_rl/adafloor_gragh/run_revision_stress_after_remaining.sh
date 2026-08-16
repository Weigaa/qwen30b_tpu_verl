#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

P0_ROOT="${P0_ROOT:-/data/adafloor_shared_state/p0_matched_trials_common_epoch0_20260728T221830Z}"
REMAINING_ROOT="${REMAINING_ROOT:-$P0_ROOT/revision_remaining}"
STRESS_ROOT="${STRESS_ROOT:-$P0_ROOT/revision_stress}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-1209600}"

waited=0
while [[ ! -f "$REMAINING_ROOT/REMAINING_PHASE_COMPLETE.txt" ]]; do
    if (( waited >= WAIT_TIMEOUT_SECONDS )); then
        echo "timed out waiting for remaining revision phase" >&2
        exit 3
    fi
    printf '[revision stress] waiting elapsed_s=%s\n' "$waited"
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
done

STRESS_OUTPUT_ROOT="$STRESS_ROOT" \
    "$SCRIPT_DIR/run_paper_mode1_transition_stress_from_common_epoch0.sh"
echo "[revision stress] complete marker=$STRESS_ROOT/TRANSITION_STRESS_COMPLETE.txt"
