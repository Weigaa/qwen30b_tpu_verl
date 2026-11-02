#!/usr/bin/env bash
set -euo pipefail

# Current-code matched comparison between fixed [16,32] and per-step
# minimum-padding two-graph selection. Both protocols reserve exactly
# 2.5 GiB/NPU for ACLGraph and therefore use identical KV/admission caps.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ACTION=${1:-dry-run}
case "$ACTION" in
    dry-run|smoke|run|verify) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_paper_graph_core3_k2_matched_2p5g_seed0_from_common_epoch0.sh dry-run
  ./run_qwen3_paper_graph_core3_k2_matched_2p5g_seed0_from_common_epoch0.sh smoke
  ./run_qwen3_paper_graph_core3_k2_matched_2p5g_seed0_from_common_epoch0.sh run
  ./run_qwen3_paper_graph_core3_k2_matched_2p5g_seed0_from_common_epoch0.sh verify

Runs two current-code seed-0 core3 protocols from the same common epoch0:
fixed capture sizes [16,32], then per-step predicted minimum-padding k=2.
Each protocol runs Vanilla Full16, AdaFloor-N F2, and AdaFloor-P F4 over
updated-policy epochs 1 and 2. Both reserve 2.5 GiB/NPU for ACLGraph.
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
    default_root="/workspace/adafloor_graph_results/qwen3_graph_core3_k2_matched_2p5g_seed0_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
else
    default_root=/workspace/adafloor_graph_results/qwen3_paper_graph_core3_k2_matched_2p5g_seed0_tq1_v1
fi
RESULT_ROOT=${K2_MATCHED_2P5G_ROOT:-$default_root}

run_protocol() {
    local label=$1 bucket_count=$2 capture_sizes=$3
    env \
        GRAPH_MAIN_ROOT="$RESULT_ROOT/$label" \
        GRAPH_MAIN_POLICY_PROFILE=core3 \
        GRAPH_MAIN_SEEDS_CSV=0 \
        GRAPH_MAIN_CAPTURE_SIZES="$capture_sizes" \
        GRAPH_MAIN_ACLGRAPH_BUCKET_COUNT="$bucket_count" \
        GRAPH_MAIN_PROTOCOL_WRAPPER="$SCRIPT_DIR/run_qwen3_paper_graph_core3_k2_matched_2p5g_seed0_from_common_epoch0.sh" \
        GRAPH_MAIN_REUSE_SEED0_VANILLA=0 \
        "$MAIN_RUNNER" "$ACTION"
}

run_protocol fixed_16_32 0 '[16,32]'
run_protocol minpadding_k2 2 '[32]'
