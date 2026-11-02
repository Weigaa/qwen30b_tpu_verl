#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Extreme low-KV floor2 stress test.
#
# This intentionally leaves much more NPU memory to non-KV runtime workspace
# than the 131072/114688 variants. If this still fails, the failure is unlikely
# to be caused by KV cache reservation alone.

export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_length_sorted_e2e_adaptive_floor2_natural_forcefloor2_cap50000}"
export PLAN_DIR="${PLAN_DIR:-$SCRIPT_DIR/$OUTPUT_SUBDIR/oracle}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2:-50000}"
export FLOOR_KV_CAPS="${FLOOR_KV_CAPS:-2:${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2},4:280576,8:377344,16:377344}"
export VLLM_ASCEND_MODE1_UPDATE_WEIGHTS_DIAG="${VLLM_ASCEND_MODE1_UPDATE_WEIGHTS_DIAG:-1}"
export VLLM_ASCEND_MODE1_LOAD_WEIGHTS_DEEP_DIAG="${VLLM_ASCEND_MODE1_LOAD_WEIGHTS_DEEP_DIAG:-1}"
export VLLM_ASCEND_MODE1_PRIMARY_RELOAD_MAP_AFTER_RESTORE="${VLLM_ASCEND_MODE1_PRIMARY_RELOAD_MAP_AFTER_RESTORE:-1}"

printf '[mode1 floor2 natural forcefloor2 cap50000] output=%s\n' "$OUTPUT_SUBDIR"
printf '[mode1 floor2 natural forcefloor2 cap50000] floor2_kv_cap=%s floor_kv_caps=%s\n' \
    "$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2" "$FLOOR_KV_CAPS"
printf '[mode1 floor2 natural forcefloor2 cap50000] update_weights_diag=%s\n' \
    "$VLLM_ASCEND_MODE1_UPDATE_WEIGHTS_DIAG"
printf '[mode1 floor2 natural forcefloor2 cap50000] load_weights_deep_diag=%s\n' \
    "$VLLM_ASCEND_MODE1_LOAD_WEIGHTS_DEEP_DIAG"
printf '[mode1 floor2 natural forcefloor2 cap50000] primary_reload_map_after_restore=%s\n' \
    "$VLLM_ASCEND_MODE1_PRIMARY_RELOAD_MAP_AFTER_RESTORE"

exec "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2_natural_forcefloor2_cap131072.sh" "$@"
