#!/usr/bin/env bash
# Compatibility alias. New runs use the release-aware, backend-matched matrix.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$ROOT/run_qwen3_v014_fair_one_step_matrix.sh" "$@"
