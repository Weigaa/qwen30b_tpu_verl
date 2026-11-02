#!/usr/bin/env bash
set -euo pipefail

P0_ROOT="${P0_ROOT:-/data/adafloor_shared_state/p0_matched_trials_common_epoch0_20260728T221830Z}"
P0_PROCESS_GROUP="${P0_PROCESS_GROUP:-395958}"
TARGET_COMPLETE="${TARGET_COMPLETE:-4}"
POLL_SECONDS="${POLL_SECONDS:-0.2}"
MANIFEST="$P0_ROOT/trial_manifest.tsv"
MARKER="$P0_ROOT/STOPPED_AFTER_SINGLE_SEED.txt"

if [[ ! -f "$MANIFEST" || ! "$P0_PROCESS_GROUP" =~ ^[1-9][0-9]*$ \
      || ! "$TARGET_COMPLETE" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid single-seed stop configuration" >&2
    exit 2
fi
if (( P0_PROCESS_GROUP == $(ps -o pgid= -p $$ | tr -d ' ') )); then
    echo "refusing to stop the monitor's own process group" >&2
    exit 2
fi

completed_count() {
    awk -F '\t' 'NR > 1 && $4 == "complete" {count++} END {print count + 0}' "$MANIFEST"
}

echo "[P0 budget monitor] waiting target=$TARGET_COMPLETE pgid=$P0_PROCESS_GROUP"
while (( $(completed_count) < TARGET_COMPLETE )); do
    if grep -q $'\tincomplete$' "$MANIFEST"; then
        echo "P0 manifest contains an incomplete trial" >&2
        exit 3
    fi
    kill -0 -- "-$P0_PROCESS_GROUP" 2>/dev/null || {
        echo "P0 process group exited before target completion" >&2
        exit 3
    }
    sleep "$POLL_SECONDS"
done

# run_policy records complete only after artifact validation and checkpoint
# cleanup. Freeze the whole group before it can materially enter the next seed.
kill -STOP -- "-$P0_PROCESS_GROUP"
if (( $(completed_count) < TARGET_COMPLETE )); then
    kill -CONT -- "-$P0_PROCESS_GROUP"
    echo "completion count changed while stopping P0" >&2
    exit 3
fi
kill -TERM -- "-$P0_PROCESS_GROUP" 2>/dev/null || true
kill -CONT -- "-$P0_PROCESS_GROUP" 2>/dev/null || true
for _ in $(seq 1 50); do
    kill -0 -- "-$P0_PROCESS_GROUP" 2>/dev/null || break
    sleep 0.2
done
if kill -0 -- "-$P0_PROCESS_GROUP" 2>/dev/null; then
    echo "P0 process group did not terminate promptly" >&2
    exit 4
fi
{
    printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'completed_trials=%s\n' "$(completed_count)"
    printf 'reason=60_hour_budget_single_seed_breadth_first\n'
} > "$MARKER"
echo "[P0 budget monitor] stopped after validated single seed marker=$MARKER"
