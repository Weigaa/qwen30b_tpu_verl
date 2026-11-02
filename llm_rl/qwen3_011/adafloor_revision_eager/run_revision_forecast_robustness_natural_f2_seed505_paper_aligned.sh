#!/usr/bin/env bash
# No performance, KV, communication, TailGuard, or profile overrides here.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$HERE/tools/run_paper_aligned_natural_f2_forecast.py" "${1:-dry-run}" \
    --output-root "${REVISION_FORECAST_ROOT:-/workspace/adafloor_revision_results/forecast_robustness_natural_f2_seed505_paper_aligned_v1}"
