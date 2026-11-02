#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Diagnostic floor=2 smoke test.
#
# The normal floor2 threshold script already exercises the 16 -> 8 -> 4 -> 2
# path, but it can still OOM if HCCL/MC2/TBE runtime workspace leaves too little
# non-KV headroom. This wrapper intentionally uses a conservative KV budget and
# disables post-shrink warmup by default, so we can separate two questions:
#   1. Can the floor2 topology and reload path run with enough headroom?
#   2. Does MC2 warmup/runtime workspace require a larger floor2 reservation?

export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_length_sorted_e2e_adaptive_floor2_threshold_workspace_probe}"
export PLAN_DIR="${PLAN_DIR:-$SCRIPT_DIR/$OUTPUT_SUBDIR/oracle}"

export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-256,512,640,768,896}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-896}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-896}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

export VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4,2
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2
export MIN_ADAPTIVE_FLOOR=2

# Conservative runtime caps for the probe. The previous 147456-token floor2 cap
# left only about 1.6-2.0 GiB free after MC2 warmup in the failing run.
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2:-65536}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4:-196608}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8:-262144}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS:-315648}"

# Disable warmup first to isolate whether the topology itself is viable. Set this
# to 1 when probing the additional workspace required by MC2 dispatcher warmup.
export VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP="${VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP:-0}"
export VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG="${VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG:-0}"

printf '[floor2 workspace probe] output=%s levels=%s max_response=%s warmup=%s\n' \
    "$OUTPUT_SUBDIR" "$VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS" "$MAX_RESPONSE_LENGTH" \
    "$VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP"
printf '[floor2 workspace probe] caps floor2=%s floor4=%s floor8=%s full=%s\n' \
    "$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2" \
    "$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4" \
    "$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8" \
    "$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS"

exec "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2.sh" "$@"
