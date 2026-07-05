#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || -z "${1:-}" ]]; then
    echo "usage: $0 <log_path> [analysis_path]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
LOG_PATH="$1"
ANALYSIS_PATH="${2:-${LOG_PATH%.txt}_analysis.txt}"
SUMMARY_PATH="${ANALYSIS_PATH%.txt}_summary.txt"
SLOW_MS="${MODE1_ANALYZE_SLOW_MS:-1000}"
REGRESSION_RATIO="${MODE1_ANALYZE_REGRESSION_RATIO:-1.5}"

cd "$REPO_DIR"
mkdir -p "$(dirname "$ANALYSIS_PATH")"

{
    echo "[mode1 auto-analysis] time=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "[mode1 auto-analysis] log=$LOG_PATH"
    echo "[mode1 auto-analysis] slow_ms=$SLOW_MS regression_ratio=$REGRESSION_RATIO"
    echo
    python3 tools/analyze_group_seq_digest.py "$LOG_PATH" --config-only
    echo
    python3 tools/analyze_group_seq_digest.py "$LOG_PATH" --verdict-only --around-slow "$SLOW_MS"
    echo
    python3 tools/analyze_group_seq_digest.py "$LOG_PATH" --by-rollout --around-slow "$SLOW_MS" --rollout-regression-ratio "$REGRESSION_RATIO"
    echo
    python3 tools/analyze_group_seq_digest.py "$LOG_PATH" --seq-mismatch-only
    echo
    python3 tools/analyze_group_seq_digest.py "$LOG_PATH" --moe-dispatch-only --around-slow "$SLOW_MS"
    echo
    python3 tools/summarize_mode1_memory.py "$LOG_PATH"
} > "$ANALYSIS_PATH" 2>&1

python3 tools/extract_mode1_analysis_summary.py "$ANALYSIS_PATH" > "$SUMMARY_PATH"

echo "$ANALYSIS_PATH"
