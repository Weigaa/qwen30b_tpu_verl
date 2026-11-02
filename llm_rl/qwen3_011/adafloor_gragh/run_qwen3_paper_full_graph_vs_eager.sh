#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-dry-run}
case "$ACTION" in
    dry-run|run|verify) ;;
    *) echo "usage: $0 [dry-run|run|verify]" >&2; exit 2 ;;
esac

GRAPH_ROOT=${QWEN3_PAPER_GRAPH_ROOT:-/workspace/adafloor_graph_results/qwen3_paper_graph_main_full_decode_preemption_safe_tq1_buckets16_32_v2}
EAGER_ROOT=${QWEN3_PAPER_MATCHED_EAGER_ROOT:-/workspace/adafloor_graph_results/qwen3_paper_matched_eager_tq1_main}
COMPARISON_ROOT=${QWEN3_PAPER_GRAPH_EAGER_COMPARISON_ROOT:-/workspace/adafloor_graph_results/qwen3_paper_full_graph_vs_matched_eager_tq1}

GRAPH_RUNNER="$SCRIPT_DIR/run_qwen3_paper_graph_main_from_common_epoch0.sh"
EAGER_RUNNER="$SCRIPT_DIR/run_qwen3_paper_matched_eager_tq1_from_common_epoch0.sh"
COMPARATOR="$SCRIPT_DIR/analysis_eval/compare_qwen3_paper_graph_vs_matched_eager.py"

if [[ "$ACTION" == dry-run ]]; then
    echo "graph_root=$GRAPH_ROOT"
    echo "matched_eager_root=$EAGER_ROOT"
    echo "comparison_root=$COMPARISON_ROOT"
    echo "matrix=7 policies x 3 seeds x 2 epochs per execution mode"
    echo "matched_difference=FULL_DECODE_ONLY_and_formula_driven_KV_reservation_only"
    echo "run_order=finish_or_resume_graph_then_matched_eager_then_fail_closed_comparison"
    exit 0
fi

if [[ "$ACTION" == run ]]; then
    GRAPH_MAIN_ROOT="$GRAPH_ROOT" GRAPH_MAIN_EXECUTION_MODE=graph \
        "$GRAPH_RUNNER" run
    GRAPH_MAIN_ROOT="$EAGER_ROOT" "$EAGER_RUNNER" run
else
    GRAPH_MAIN_ROOT="$GRAPH_ROOT" GRAPH_MAIN_EXECUTION_MODE=graph \
        "$GRAPH_RUNNER" verify
    GRAPH_MAIN_ROOT="$EAGER_ROOT" "$EAGER_RUNNER" verify
fi

mkdir -p "$COMPARISON_ROOT"
python3 "$COMPARATOR" \
    --graph-summary "$GRAPH_ROOT/analysis_eval/summary.json" \
    --eager-summary "$EAGER_ROOT/analysis_eval/summary.json" \
    --graph-protocol "$GRAPH_ROOT/protocol.env" \
    --eager-protocol "$EAGER_ROOT/protocol.env" \
    --graph-code-manifest "$GRAPH_ROOT/code_sha256.txt" \
    --eager-code-manifest "$EAGER_ROOT/code_sha256.txt" \
    --graph-pass "$GRAPH_ROOT/GRAPH_MAIN_PASS.txt" \
    --eager-pass "$EAGER_ROOT/GRAPH_MAIN_PASS.txt" \
    --output-dir "$COMPARISON_ROOT"

sha256sum \
    "$COMPARISON_ROOT/full_graph_vs_matched_eager.json" \
    "$COMPARISON_ROOT/full_graph_vs_matched_eager.md" \
    > "$COMPARISON_ROOT/SHA256SUMS"
printf 'status=PASS\ncomparison_json_sha256=%s\n' \
    "$(sha256sum "$COMPARISON_ROOT/full_graph_vs_matched_eager.json" | awk '{print $1}')" \
    > "$COMPARISON_ROOT/PASS.txt"
echo "[paper graph/eager] PASS comparison=$COMPARISON_ROOT/full_graph_vs_matched_eager.md"
