#!/usr/bin/env bash
set -euo pipefail

# Epoch-planned k=10 FULL_DECODE_ONLY study for the three paper operating
# points. Each epoch computes a different minimum-padding set for each step.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ACTION=${1:-dry-run}
case "$ACTION" in
    dry-run|smoke|run|verify) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_paper_graph_core3_k10_from_common_epoch0.sh dry-run
  ./run_qwen3_paper_graph_core3_k10_from_common_epoch0.sh smoke
  ./run_qwen3_paper_graph_core3_k10_from_common_epoch0.sh run
  ./run_qwen3_paper_graph_core3_k10_from_common_epoch0.sh verify

`run` executes Vanilla Full16, AdaFloor-N F2, and AdaFloor-P F4 for global
seeds 0/2/3 and sequential updated-policy epochs 1/2. Before each epoch, every
step gets its own 10-graph set selected from predicted response lengths and the
step's final prompt-to-rank mapping. No current-run response length is used.
Completed arms are validated and their checkpoints/caches are removed before
the next arm.

`smoke` uses a fresh non-authoritative root and executes one step for each of
the three policies. It does not modify the formal result root.
EOF
        exit 0
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac

MAIN_RUNNER="$SCRIPT_DIR/run_qwen3_paper_graph_main_from_common_epoch0.sh"
[[ -x "$MAIN_RUNNER" || -f "$MAIN_RUNNER" ]] || {
    echo "missing main runner: $MAIN_RUNNER" >&2; exit 2;
}

if [[ "$ACTION" == smoke ]]; then
    default_root="/workspace/adafloor_graph_results/qwen3_graph_core3_k10_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
else
    default_root=/workspace/adafloor_graph_results/qwen3_paper_graph_core3_k10_minpadding_tq1_v2
fi
RESULT_ROOT=${K10_CORE3_ROOT:-$default_root}

exec env \
    GRAPH_MAIN_ROOT="$RESULT_ROOT" \
    GRAPH_MAIN_POLICY_PROFILE=core3 \
    GRAPH_MAIN_CAPTURE_SIZES='[32]' \
    GRAPH_MAIN_ACLGRAPH_BUCKET_COUNT=10 \
    GRAPH_MAIN_PROTOCOL_WRAPPER="$SCRIPT_DIR/run_qwen3_paper_graph_core3_k10_from_common_epoch0.sh" \
    GRAPH_MAIN_REUSE_SEED0_VANILLA=0 \
    "$MAIN_RUNNER" "$ACTION"
