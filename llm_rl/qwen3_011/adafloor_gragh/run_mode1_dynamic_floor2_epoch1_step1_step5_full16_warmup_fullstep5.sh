#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_STAMP
export RUN_FULL16_WARMUP="${RUN_FULL16_WARMUP:-1}"
export FULL_STEP5_DECODE=1
if [[ "$RUN_FULL16_WARMUP" == "1" ]]; then
    export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_floor2_step1_step5_after_full16_warmup_${RUN_STAMP}}"
else
    export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_floor2_step1_step5_no_full16_warmup_${RUN_STAMP}}"
fi
exec "$SCRIPT_DIR/run_mode1_dynamic_floor2_epoch1_step1_step5_full16_warmup_probe.sh" "$@"
