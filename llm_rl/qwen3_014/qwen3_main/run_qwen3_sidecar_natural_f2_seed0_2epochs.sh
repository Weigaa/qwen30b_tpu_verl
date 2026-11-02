#!/usr/bin/env bash
# Main-experiment Ada-N2 protocol, with gated Sidecar; one seed, two updates epochs.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-run}
STAMP=${N2_SIDECAR_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT=${N2_SIDECAR_OUTPUT_ROOT:-/data/adafloor_graph_results/mlsys_v014_graph_sidecar_natural_f2_seed0_${STAMP}}
exec python3 "$ROOT/tools/run_v014_sidecar_training.py" "$MODE" --output "$OUTPUT"
