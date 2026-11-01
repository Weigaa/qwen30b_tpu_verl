#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'USAGE'
Usage:
  bash analyze_latest_floor4_livepool_log.sh [log_path] [-- analyzer args...]

Find and analyze the newest log from run_mode1_floor4_livepool_overlap4group_test.sh.
Pass log_path explicitly to avoid auto-discovery.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

LOG_PATH=""
if [[ $# -gt 0 && -f "$1" ]]; then
    LOG_PATH="$1"
    shift
else
    SEARCH_ROOT="${LIVEPOOL_RUN_ROOT:-$SCRIPT_DIR/mode1_floor4_livepool_overlap4group_runs}"
    if [[ -d "$SEARCH_ROOT" ]]; then
        LOG_PATH=$(
            find "$SEARCH_ROOT" -type f \
                \( -name '*_elastic.txt' -o -name 'wjqwen30b-a3b-record_graph_save4eagle3_*.txt' -o -name 'launcher.log' \) \
                -printf '%T@ %p\n' 2>/dev/null |
                sort -nr |
                awk 'NR==1 {sub(/^[^ ]+ /, ""); print; exit}'
        )
    fi
    if [[ -z "${LOG_PATH:-}" && -f "$SCRIPT_DIR/.current_true_mode5_log" ]]; then
        candidate=$(cat "$SCRIPT_DIR/.current_true_mode5_log" 2>/dev/null || true)
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            LOG_PATH="$candidate"
        elif [[ -n "$candidate" && -f "$SCRIPT_DIR/$candidate" ]]; then
            LOG_PATH="$SCRIPT_DIR/$candidate"
        fi
    fi
fi

if [[ -z "${LOG_PATH:-}" || ! -f "$LOG_PATH" ]]; then
    echo "No live-pool log file found. Pass a log path explicitly." >&2
    exit 2
fi

echo "[floor4 livepool analyzer] log=$LOG_PATH" >&2
python "$SCRIPT_DIR/analyze_floor4_mc2_log.py" "$LOG_PATH" "$@"
