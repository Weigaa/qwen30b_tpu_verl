#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
LOG_PATH="${1:-}"
SLOW_MS="${MODE1_ANALYZE_SLOW_MS:-1000}"
REGRESSION_RATIO="${MODE1_ANALYZE_REGRESSION_RATIO:-1.5}"

cd "$REPO_DIR"

if [[ -n "$LOG_PATH" ]]; then
    echo "[mode1 analyze] log=$LOG_PATH slow_ms=$SLOW_MS regression_ratio=$REGRESSION_RATIO"
else
    echo "[mode1 analyze] log=<latest threshold log> slow_ms=$SLOW_MS regression_ratio=$REGRESSION_RATIO"
    LOG_PATH="$(python3 - <<'PY'
from pathlib import Path
root = Path("mode1_baseline_random_batch_floor4_threshold_multistep")
candidates = sorted(
    (p for p in root.rglob("*.txt")
     if p.name.startswith(("wjqwen30b-", "wjeagerqwen30b-"))
     and not p.name.endswith("_analysis.txt")),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not candidates:
    raise SystemExit("no rollout .txt logs found")
print(candidates[0])
PY
)"
fi

ANALYSIS_PATH="${LOG_PATH%.txt}_analysis.manual.txt"
MODE1_ANALYZE_SLOW_MS="$SLOW_MS" \
MODE1_ANALYZE_REGRESSION_RATIO="$REGRESSION_RATIO" \
    tools/write_mode1_threshold_analysis.sh "$LOG_PATH" "$ANALYSIS_PATH" >/dev/null
SUMMARY_PATH="${ANALYSIS_PATH%.txt}_summary.txt"
if [[ -f "$SUMMARY_PATH" ]]; then
    cat "$SUMMARY_PATH"
    echo
fi
cat "$ANALYSIS_PATH"
echo
echo "[mode1 analyze] analysis_file=$ANALYSIS_PATH"
if [[ -f "$SUMMARY_PATH" ]]; then
    echo "[mode1 analyze] summary_file=$SUMMARY_PATH"
fi
