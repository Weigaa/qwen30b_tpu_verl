#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=/workspace/adafloor_revision_results/frontier_windows_v1
PID=${REVISION_FRONTIER_WAIT_PID:?confirmed live dynamic handle required}
[[ "$PID" =~ ^[0-9]+$ ]] || exit 2
while [[ -r "/proc/$PID/stat" ]]; do
    state=$(awk '{print $3}' "/proc/$PID/stat" 2>/dev/null || true)
    [[ "$state" != Z ]] || break
    sleep 10
done
rg -q '^exit_code=0$' "$ROOT/dynamic4/protocol.env" || { echo 'dynamic arm failed; stop and retain evidence' >&2; exit 3; }
python3 "$REPO/analysis_eval/analyze_revision_frontier_windows.py" --root "$ROOT" --output-dir "$ROOT/final_audit"
bash "$REPO/run_revision_resident_pressure_smoke.sh" run > /workspace/adafloor_revision_resident_pressure_npu_v1.launch.log 2>&1
