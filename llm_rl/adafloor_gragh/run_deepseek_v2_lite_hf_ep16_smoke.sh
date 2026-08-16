#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export DEEPSEEK_ACTOR_PROBE_ROLLOUT_LOAD_FORMAT=auto
export DEEPSEEK_ACTOR_PROBE_PRESERVE_INITIAL_HF_WEIGHTS=1
export DEEPSEEK_WEIGHT_SYNC_SMOKE_RUN_NAME=${DEEPSEEK_HF_EP16_SMOKE_RUN_NAME:-hf_ep16_smoke_$(date -u +%Y%m%dT%H%M%SZ)}

exec "$SCRIPT_DIR/run_deepseek_v2_lite_weight_sync_smoke.sh" "$@"
