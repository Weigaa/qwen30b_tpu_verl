#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)

if (( $# != 0 )); then
    echo "DeepSeek Planned floor2 threshold smoke does not accept overrides" >&2
    exit 2
fi
if [[ "${ALLOW_INFEASIBLE_PLAN:-0}" != 0 ]]; then
    echo "ALLOW_INFEASIBLE_PLAN is forbidden for the DeepSeek Planned floor2 threshold smoke" >&2
    exit 2
fi
unset ALLOW_INFEASIBLE_PLAN
export BASELINE_ALLOW_INFEASIBLE_PLAN=0

if ! [[ "${DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB must be an explicitly measured positive integer" >&2
    exit 2
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/internal/deepseek_v2_lite_planned_f2_runtime_profile.sh"

export DEEPSEEK_ADAFLOOR_SMOKE_POLICY=planned
export DEEPSEEK_ADAFLOOR_SMOKE_FLOOR=2
export DEEPSEEK_ADAFLOOR_SMOKE_RUN_NAME="${DEEPSEEK_ADAFLOOR_SMOKE_RUN_NAME:-adafloor_ep16_planned_floor2_threshold_2step_$timestamp}"

echo "[DeepSeek Planned floor2 threshold smoke] topology=EP16 stages=8,4,2 steps=2 n=16"
exec "$SCRIPT_DIR/run_deepseek_v2_lite_floor2_threshold_two_step_smoke.sh"
