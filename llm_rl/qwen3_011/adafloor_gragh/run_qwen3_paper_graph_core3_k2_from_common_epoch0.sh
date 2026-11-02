#!/usr/bin/env bash
set -euo pipefail

# Rebuttal-grade two-bucket graph comparison. This keeps the submitted eager
# admission/floor semantics and changes only graph execution. It is deliberately
# separate from the k8 prompt-adaptive result root.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
ACTION=${1:-dry-run}
case "$ACTION" in dry-run|smoke|run|verify) ;; -h|--help)
    cat <<'EOF'
Usage: ./run_qwen3_paper_graph_core3_k2_from_common_epoch0.sh [dry-run|run|verify]

`run` evaluates Vanilla, AdaFloor-N F2 and AdaFloor-P F4 for seeds 0, 2 and 3,
with epochs 1 and 2. It uses the submitted synchronized response-length KV
admission (ACTIVE_PEAK_SAFETY_FACTOR=1.16), two capture buckets [16,32], and
the common epoch0 checkpoint. No prompt-adaptive admission is enabled.
EOF
    exit 0 ;; *) echo "unknown action: $ACTION" >&2; exit 2 ;; esac

RESULT_ROOT=${K2_CORE3_ROOT:-/data/adafloor_graph_results/qwen3_paper_graph_core3_k2_prompt_kv_rebuttal_v1}
exec env \
    GRAPH_MAIN_ROOT="$RESULT_ROOT" \
    GRAPH_MAIN_POLICY_PROFILE=core3 \
    GRAPH_MAIN_CAPTURE_SIZES='[16,32]' \
    GRAPH_MAIN_ACLGRAPH_BUCKET_COUNT=2 \
    GRAPH_MAIN_KV_ADMISSION_MODEL=synchronized \
    GRAPH_MAIN_GRAPH_RESERVE_BYTES=4294967296 \
    GRAPH_MAIN_GRAPH_RESERVE_BASIS=two_bucket_profile_4gib_rebuttal \
    GRAPH_MAIN_PROTOCOL_WRAPPER="$SCRIPT_DIR/run_qwen3_paper_graph_core3_k2_from_common_epoch0.sh" \
    GRAPH_MAIN_REUSE_SEED0_VANILLA=0 \
    "$SCRIPT_DIR/run_qwen3_paper_graph_main_from_common_epoch0.sh" "$ACTION"
