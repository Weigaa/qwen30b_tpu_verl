#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Floor2 natural force-floor2 stress test with reload allocator trim disabled.
#
# This isolates whether the step2+ slowdown is caused by the per-layer
# synchronize/gc/empty_cache calls in mode1 reload post-processing.

export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_length_sorted_e2e_adaptive_floor2_natural_forcefloor2_cap50000_notrim}"
export PLAN_DIR="${PLAN_DIR:-$SCRIPT_DIR/$OUTPUT_SUBDIR/oracle}"
export VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR="${VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR:-0}"
export VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM="${VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM:-0}"
export VLLM_ASCEND_MODE1_LOAD_WEIGHTS_DEEP_DIAG="${VLLM_ASCEND_MODE1_LOAD_WEIGHTS_DEEP_DIAG:-1}"
export VLLM_ASCEND_MODE1_UPDATE_WEIGHTS_DIAG="${VLLM_ASCEND_MODE1_UPDATE_WEIGHTS_DIAG:-1}"

printf '[mode1 floor2 natural forcefloor2 cap50000 notrim] output=%s\n' "$OUTPUT_SUBDIR"
printf '[mode1 floor2 natural forcefloor2 cap50000 notrim] reload_trim_allocator=%s reload_sync_on_trim=%s\n' \
    "$VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR" \
    "$VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM"

exec "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2_natural_forcefloor2_cap50000.sh" "$@"
