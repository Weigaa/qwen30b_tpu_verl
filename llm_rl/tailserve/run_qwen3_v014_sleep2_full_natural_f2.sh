#!/usr/bin/env bash
# Formal v0.14 native sleep(level=2) + AdaFloor Natural-F2 + FULL graph.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-dry-run}
case "$ACTION" in dry-run|smoke|run|verify) ;; *) echo "usage: $0 {dry-run|smoke|run|verify}" >&2; exit 2 ;; esac

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
export V014_SLEEP2_PROFILE=core3
export V014_SLEEP2_POLICY=natural_f2
export V014_SLEEP2_SEEDS=${V014_SLEEP2_SEEDS:-0,2,3}
export V014_SLEEP2_ONLY_ARM=${V014_SLEEP2_ONLY_ARM:-}
export V014_SLEEP2_ROOT=${V014_SLEEP2_ROOT:-/workspace/adafloor_graph_results/qwen3_v014_sleep2_natural_f2_${STAMP}}

if [[ -n "$V014_SLEEP2_ONLY_ARM" ]]; then
    export GRAPH_MAIN_ONLY_ARM="$V014_SLEEP2_ONLY_ARM"
else
    export GRAPH_MAIN_ONLY_ARM=""
fi

exec "$ROOT/run_qwen3_v014_full_decode_sleep2_adafloor.sh" "$ACTION"
