#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Fast natural-policy floor=2 smoke test with every rollout step forced to
# selected_floor=2.  This keeps the comparison clean: no step is allowed to
# fall back to floor4/8/16 because of tail ties or planner heuristics.

export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_length_sorted_e2e_adaptive_floor2_natural_forcefloor2_cap131072}"
export PLAN_DIR="${PLAN_DIR:-$SCRIPT_DIR/$OUTPUT_SUBDIR/oracle}"

export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=natural
export VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE=0
export VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS=0
export VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS=0
export VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE=0
export VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP=0
export VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS=0
export VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=0
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE=0

export VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4,2
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2
export MIN_ADAPTIVE_FLOOR=2
export FORCE_SELECTED_FLOOR=2
export IGNORE_TAIL_TIES_AT_RESPONSE_CAP=1

# Largest floor2 KV cap observed to pass the natural threshold cap sweep.
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2:-131072}"
export FLOOR_KV_CAPS="${FLOOR_KV_CAPS:-2:${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2},4:280576,8:377344,16:377344}"

# Keep the validation short and bounded. Override these env vars if you want a
# longer run after the force-floor2 behavior is confirmed.
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-256,512,640,768,896}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-896}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-896}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"

printf '[mode1 floor2 natural forcefloor2] output=%s\n' "$OUTPUT_SUBDIR"
printf '[mode1 floor2 natural forcefloor2] floor2_kv_cap=%s floor_kv_caps=%s\n' \
    "$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2" "$FLOOR_KV_CAPS"
printf '[mode1 floor2 natural forcefloor2] levels=%s max_response=%s force_selected_floor=%s\n' \
    "$VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS" "$MAX_RESPONSE_LENGTH" "$FORCE_SELECTED_FLOOR"
printf '[mode1 floor2 natural forcefloor2] target_policy=%s fixed_reuse=%s cache_floor=%s keep_mc2=%s\n' \
    "$VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY" \
    "$VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE" \
    "$VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS" \
    "$VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE"

exec "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2.sh" "$@"
