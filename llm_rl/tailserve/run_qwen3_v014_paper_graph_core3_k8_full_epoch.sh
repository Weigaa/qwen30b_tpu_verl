#!/usr/bin/env bash
# v0.14 full-epoch core3/k8 comparison, using the local qwen3 runtime.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export K8_CORE3_ROOT="${K8_CORE3_ROOT:-/workspace/adafloor_graph_results/qwen3_v014_paper_graph_core3_k8_$(date -u +%Y%m%dT%H%M%SZ)}"
export GRAPH_MAIN_POLICY_PROFILE=core3
export GRAPH_MAIN_CAPTURE_SIZES='[32]'
export GRAPH_MAIN_ACLGRAPH_BUCKET_COUNT=8
export GRAPH_MAIN_REUSE_SEED0_VANILLA=0
export COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-dry-run}" in
  dry-run|smoke|run|verify) ;;
  -h|--help) echo "usage: $0 {dry-run|smoke|run|verify}"; exit 0 ;;
  *) echo "usage: $0 {dry-run|smoke|run|verify}" >&2; exit 2 ;;
esac

exec "$ROOT/run_qwen3_paper_graph_core3_k8_from_common_epoch0.sh" "$@"
