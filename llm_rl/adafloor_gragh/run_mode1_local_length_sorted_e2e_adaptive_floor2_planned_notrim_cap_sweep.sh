#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[floor2 planned notrim cap sweep] deprecated: use planned headroom sweep instead." >&2
echo "[floor2 planned notrim cap sweep] keeping nominal floor2 KV cap fixed and sweeping planned headroom." >&2

exec "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2_planned_headroom_sweep.sh" "$@"
