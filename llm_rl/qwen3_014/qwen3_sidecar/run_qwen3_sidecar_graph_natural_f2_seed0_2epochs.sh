#!/usr/bin/env bash
# Primary AND Sidecar use native FULL_DECODE_ONLY/TQ1. Two real RL epochs.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-run}
GATE_SECONDS=${N2_SIDECAR_WINDOW_MIN_SECONDS:-60}
GATE_COMPARISON=${N2_SIDECAR_WINDOW_COMPARISON:-gt}
STAMP=${N2_SIDECAR_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT=${N2_SIDECAR_OUTPUT_ROOT:-/data/adafloor_graph_results/mlsys_v014_fullgraph_sidecar_natural_f2_gate${GATE_SECONDS}_seed0_${STAMP}}
PROOF=${N2_SIDECAR_PARITY_PROOF:-}
if [[ "$MODE" == run && -z "$PROOF" ]]; then
    python3 "$ROOT/tools/run_sidecar_graph_parity.py" --output "${OUTPUT}_parity"
    PROOF="${OUTPUT}_parity/PARITY_RESULT.json"
fi
EXTRA=()
EXTRA+=(--window-minimum "$GATE_SECONDS" --window-comparison "$GATE_COMPARISON")
STORAGE=${N2_SIDECAR_EPOCH2_CHECKPOINT_ROOT:-/workspace/adafloor_graph_results/$(basename "$OUTPUT")_epoch2_checkpoints}
EXTRA+=(--epoch2-checkpoint-root "$STORAGE")
if [[ -n "$PROOF" ]]; then
    EXTRA+=(--sidecar-parity-proof "$PROOF")
fi
exec python3 "$ROOT/tools/run_v014_sidecar_training.py" "$MODE" --output "$OUTPUT" --sidecar-execution graph "${EXTRA[@]}"
