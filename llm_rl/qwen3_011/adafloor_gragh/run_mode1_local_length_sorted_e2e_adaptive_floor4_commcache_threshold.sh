#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Fast diagnostics for planned floor-group residency.
#
# Run only step 1/4/5 with a short validation generation budget:
#   step 1: floor4, planned groups [8..15] and [12..15]
#   step 4: floor8, should keep/reuse [8..15] and release [12..15]
#   step 5: floor16, should release planned floor groups and restore full world
#
# The comm-cache state log prints cached communicator groups, registry entries,
# MoE topology-cache entries, and per-rank non_torch/free memory at each KV
# resize phase.
export FAST_STEP_SUBSET="${FAST_STEP_SUBSET:-1}"
export FAST_STEP_SUBSET_STEPS="${FAST_STEP_SUBSET_STEPS:-1,4,5}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_length_sorted_e2e_adaptive_floor4_commcache_threshold}"

# Keep the Ascend platform plugin enabled while skipping unrelated/outdated
# general entry-point plugins that only add noisy, non-fatal ERROR logs.
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"

export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY="${VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY:-planned}"
export VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS="${VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS:-1}"
export VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS="${VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS:-1}"
export VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE="${VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE:-1}"
export VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP="${VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP:-1}"
export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS="${VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS:-147456}"
export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR2="${VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR2:-147456}"
export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR4="${VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR4:-147456}"
export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8="${VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8:-114688}"
export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR16="${VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR16:-0}"
export VLLM_ASCEND_MODE1_STEP_KV_CAP_INCLUDES_PLANNED_HEADROOM="${VLLM_ASCEND_MODE1_STEP_KV_CAP_INCLUDES_PLANNED_HEADROOM:-1}"
export VLLM_ASCEND_MODE1_COMM_CACHE_STATE_LOG="${VLLM_ASCEND_MODE1_COMM_CACHE_STATE_LOG:-1}"
export VLLM_ASCEND_MODE1_PARITY_FORCE_DESTROY_PLANNED_PRUNE="${VLLM_ASCEND_MODE1_PARITY_FORCE_DESTROY_PLANNED_PRUNE:-1}"

export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-256,512,640,768,896}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-896}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

set +e
"$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh" "$@"
status=$?
set -e

latest_log=$(ls -t "$SCRIPT_DIR/$OUTPUT_SUBDIR"/logs/*.txt 2>/dev/null | head -1 || true)
if [[ -n "$latest_log" ]]; then
    summary_path="${latest_log%.txt}.comm_cache_summary.tsv"
    python3 "$SCRIPT_DIR/tools/summarize_mode1_comm_cache_log.py" \
        "$latest_log" > "$summary_path" || true
    echo "[mode1 comm-cache threshold] latest_log=$latest_log"
    echo "[mode1 comm-cache threshold] comm_cache_summary=$summary_path"
fi

exit "$status"
