#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# A/B variant of the stable force-floor2 natural run.  It keeps the same MC2
# lifecycle as the known-good cap131072 script, but lowers the floor2 KV budget
# by one 16K-token block to leave extra NPU headroom for HCCL/MC2 runtime
# workspace after the first shrink/restore cycle.

export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_length_sorted_e2e_adaptive_floor2_natural_forcefloor2_cap114688}"
export PLAN_DIR="${PLAN_DIR:-$SCRIPT_DIR/$OUTPUT_SUBDIR/oracle}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2:-114688}"
export FLOOR_KV_CAPS="${FLOOR_KV_CAPS:-2:${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2},4:280576,8:377344,16:377344}"

printf '[mode1 floor2 natural forcefloor2 cap114688] output=%s\n' "$OUTPUT_SUBDIR"
printf '[mode1 floor2 natural forcefloor2 cap114688] floor2_kv_cap=%s floor_kv_caps=%s\n' \
    "$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2" "$FLOOR_KV_CAPS"

exec "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2_natural_forcefloor2_cap131072.sh" "$@"
