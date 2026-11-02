#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REVISION_SMOKE_ROOT=${REVISION_WEIGHT_TIMING_ROOT:-/workspace/adafloor_revision_results/weight_refresh_timing_v1}
export ADAFLOOR_REVISION_WEIGHT_TIMING=1 ADAFLOOR_REVISION_EXPERT_AUDIT=0
export VLLM_ASCEND_MODE1_UPDATE_WEIGHTS_DIAG=0 ADAFLOOR_REVISION_COMPLETION_AUDIT=0 VERL_SIDECAR_ENABLE=0
# Same structural two-real-update eager test used for expert freshness.
# Limits force real shrink stages; they are not a quality/speedup benchmark.
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,1024,1024,1024
exec bash "$REPO/run_revision_hard_guard_smoke.sh"
