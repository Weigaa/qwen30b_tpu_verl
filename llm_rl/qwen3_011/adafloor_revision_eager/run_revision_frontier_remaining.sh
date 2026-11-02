#!/usr/bin/env bash
# Continue the two missing arms only, never rerun the primary-only baseline.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${REVISION_FRONTIER_ROOT:-/workspace/adafloor_revision_results/frontier_windows_v1}
WAIT_PID=${REVISION_FRONTIER_WAIT_PID:-}
if [[ -n "$WAIT_PID" ]]; then
    [[ "$WAIT_PID" =~ ^[0-9]+$ ]] || exit 2
    while [[ -r "/proc/$WAIT_PID/stat" ]]; do
        state=$(awk '{print $3}' "/proc/$WAIT_PID/stat" 2>/dev/null || true)
        [[ "$state" != Z ]] || break
        sleep 10
    done
fi
[[ -f "$ROOT/primary_only/protocol.env" ]] || exit 3
rg -q '^exit_code=0$' "$ROOT/primary_only/protocol.env" || { echo 'existing primary-only arm did not finish cleanly' >&2; exit 3; }
for arm in static8 dynamic4; do
    if [[ -e "$ROOT/$arm" ]]; then
        echo "arm already exists; inspect rather than silently rerun:$arm" >&2
        exit 3
    fi
    bash "$REPO/run_revision_frontier_window.sh" "$arm" > "$ROOT/${arm}.launch.log" 2>&1
done
python3 "$REPO/analysis_eval/analyze_revision_frontier_windows.py" --root "$ROOT" --output-dir "$ROOT/final_audit"
