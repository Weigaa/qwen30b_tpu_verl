#!/usr/bin/env bash
set -euo pipefail

# Same-seed end-to-end complement to the formal k=10 core3 study. The two
# capture sizes are selected independently for every epoch step from predicted
# lengths. The k=10 4-GiB reserve and KV caps are intentionally retained so
# graph count is the only capacity-side protocol change.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ACTION=${1:-dry-run}
case "$ACTION" in
    dry-run|smoke|run|verify) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_paper_graph_core3_k2_seed0_from_common_epoch0.sh dry-run
  ./run_qwen3_paper_graph_core3_k2_seed0_from_common_epoch0.sh smoke
  ./run_qwen3_paper_graph_core3_k2_seed0_from_common_epoch0.sh run
  ./run_qwen3_paper_graph_core3_k2_seed0_from_common_epoch0.sh verify

The formal run executes seed 0 for Vanilla Full16, AdaFloor-N F2, and
AdaFloor-P F4 over updated-policy epochs 1 and 2. Each epoch step selects the
minimum-padding two-graph set from predicted lengths. It uses the same common
epoch-0 checkpoint, 4-GiB graph reserve, KV caps, admission rules, and training
settings as the completed k=10 core3 experiment.
EOF
        exit 0
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac

MAIN_RUNNER="$SCRIPT_DIR/run_qwen3_paper_graph_main_from_common_epoch0.sh"
[[ -f "$MAIN_RUNNER" ]] || {
    echo "missing main runner: $MAIN_RUNNER" >&2
    exit 2
}

if [[ "$ACTION" == smoke ]]; then
    default_root="/workspace/adafloor_graph_results/qwen3_graph_core3_k2_seed0_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
else
    default_root=/workspace/adafloor_graph_results/qwen3_paper_graph_core3_k2_minpadding_seed0_tq1_v1
fi
RESULT_ROOT=${K2_CORE3_SEED0_ROOT:-$default_root}

exec env \
    GRAPH_MAIN_ROOT="$RESULT_ROOT" \
    GRAPH_MAIN_POLICY_PROFILE=core3 \
    GRAPH_MAIN_SEEDS_CSV=0 \
    GRAPH_MAIN_CAPTURE_SIZES='[32]' \
    GRAPH_MAIN_ACLGRAPH_BUCKET_COUNT=2 \
    GRAPH_MAIN_GRAPH_RESERVE_BYTES=4294967296 \
    GRAPH_MAIN_GRAPH_RESERVE_BASIS=matched_k10_4gib_for_bucket_count_isolation \
    GRAPH_MAIN_PROTOCOL_WRAPPER="$SCRIPT_DIR/run_qwen3_paper_graph_core3_k2_seed0_from_common_epoch0.sh" \
    GRAPH_MAIN_REUSE_SEED0_VANILLA=0 \
    "$MAIN_RUNNER" "$ACTION"
