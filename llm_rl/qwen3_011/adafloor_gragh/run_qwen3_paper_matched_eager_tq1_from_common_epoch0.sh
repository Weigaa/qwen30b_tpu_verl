#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    cat <<'EOF'
Usage:
  ./run_qwen3_paper_matched_eager_tq1_from_common_epoch0.sh dry-run
  ./run_qwen3_paper_matched_eager_tq1_from_common_epoch0.sh run
  ./run_qwen3_paper_matched_eager_tq1_from_common_epoch0.sh verify

The formal run executes only Vanilla, Natural-F2, and Planned-F4 for global
seeds 0, 2, and 3 at epoch 1. All controls use the same reduced KV caps as the
FULL_DECODE_ONLY graph arms. Completed arms are validated, checkpoint-cleaned,
and skipped on a later invocation with the same immutable result root.
EOF
    exit 0
fi

export GRAPH_MAIN_EXECUTION_MODE=eager_matched
export GRAPH_MAIN_REUSE_SEED0_VANILLA=0

exec "$SCRIPT_DIR/run_qwen3_paper_graph_main_from_common_epoch0.sh" "$@"
