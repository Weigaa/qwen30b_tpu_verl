#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_PATH=""
if [[ $# -gt 0 && -f "$1" ]]; then
    LOG_PATH="$1"
    shift
else
    LOG_PATH=$(find "$SCRIPT_DIR" -maxdepth 2 -type f \
        \( -name '*_elastic.txt' -o -name 'wjqwen30b-a3b-record_graph_save4eagle3_*.txt' \) \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {sub(/^[^ ]+ /, ""); print; exit}')
fi

if [[ -z "${LOG_PATH:-}" || ! -f "$LOG_PATH" ]]; then
    echo "No log file found. Pass a log path explicitly." >&2
    exit 2
fi

python "$SCRIPT_DIR/analyze_floor4_mc2_log.py" "$LOG_PATH" "$@"
