#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Fast natural-policy floor=2 feasibility smoke test.
#
# This removes planned-topology residency/reuse from the experiment and keeps
# generation bounded by short validation caps. It is meant to answer whether the
# natural 16 -> 8 -> 4 -> 2 path itself is viable before testing planned reuse.

export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_length_sorted_e2e_adaptive_floor2_natural_threshold}"
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

export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-256,512,640,768,896}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-896}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-896}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

export VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4,2
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2
export MIN_ADAPTIVE_FLOOR=2
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"

# The default floor2 cap in the generic floor2 script (147456 tokens) is enough
# for offline active-token feasibility, but the short natural smoke test showed
# that repeated 16 -> 8 -> 4 -> 2 runtime warmup can leave too little HCCL/MC2
# workspace headroom. Keep this wrapper conservative by default; override it
# when running the cap-sweep script below.
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2:-98304}"

printf '[mode1 floor2 natural threshold] output=%s levels=%s max_response=%s\n' \
    "$OUTPUT_SUBDIR" "$VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS" "$MAX_RESPONSE_LENGTH"
printf '[mode1 floor2 natural threshold] floor2_kv_cap=%s\n' \
    "$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2"
printf '[mode1 floor2 natural threshold] target_policy=%s fixed_reuse=%s cache_floor=%s keep_mc2=%s\n' \
    "$VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY" \
    "$VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE" \
    "$VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS" \
    "$VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE"

exec "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2.sh" "$@"
