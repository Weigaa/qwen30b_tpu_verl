#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SESSION_NAME="${REVISION_TMUX_SESSION:-adafloor-revision-missing}"
QUEUE_LOG="${QUEUE_LOG:-$SCRIPT_DIR/analysis_eval/runtime_logs/revision_missing_experiments_queue.log}"

mkdir -p "$(dirname "$QUEUE_LOG")"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "revision queue is already running in tmux session: $SESSION_NAME" >&2
    exit 3
fi

tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR" \
    "exec env QUEUE_LOG='$QUEUE_LOG' '$SCRIPT_DIR/run_revision_missing_experiments_queue.sh'"
echo "[revision queue] tmux_session=$SESSION_NAME"
echo "[revision queue] log=$QUEUE_LOG"
echo "[revision queue] inspect=tmux attach -t $SESSION_NAME"
