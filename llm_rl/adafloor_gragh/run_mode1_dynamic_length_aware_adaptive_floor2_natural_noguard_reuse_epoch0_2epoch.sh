#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Full floor2 natural-policy control run with no response-length truncation.
# Reuse epoch0, run epoch1/2, disable both planner tail guard and the separate
# runtime short-step cap, and fail before training if the generated plan still
# contains any response cap below the full response budget.
export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_length_aware_adaptive_floor2_natural_noguard_control_reuse_epoch0_2epoch}"
export DYNAMIC_TOTAL_EPOCHS="${DYNAMIC_TOTAL_EPOCHS:-3}"
export DYNAMIC_START_EPOCH="${DYNAMIC_START_EPOCH:-1}"
export DYNAMIC_DISABLE_TAIL_GUARD=1
export DYNAMIC_SHORT_STEP_CAP_ENABLE=0
export DYNAMIC_EXPECT_NO_RESPONSE_CAPS=1

exec "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh" "$@"
