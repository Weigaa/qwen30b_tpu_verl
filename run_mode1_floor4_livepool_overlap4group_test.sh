#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash run_mode1_floor4_livepool_overlap4group_test.sh

Purpose:
  Test the experimental floor4 MC2 live-pool workaround. 4-rank MC2 groups are
  cached by rank tuple and kept resident during the process, so later steps do
  not hot-path destroy/recreate overlapping floor4 MC2 communicators.

Default pattern:
  - step 1 floor4 survivor is forced to [12,13,14,15]
  - step 2 floor4 survivor is forced to [10,11,12,13]
  - overlap is [12,13]

Useful overrides:
  BASELINE_TOTAL_TRAINING_STEPS=3
  VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES=floor4:12-15,floor4:10-13
  VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=280576
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BASELINE_TOTAL_TRAINING_STEPS="${BASELINE_TOTAL_TRAINING_STEPS:-3}"
export VERL_ELASTIC_TAIL_VALIDATE_ROTATE_BUCKETS="${VERL_ELASTIC_TAIL_VALIDATE_ROTATE_BUCKETS:-1}"
export VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES="${VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES:-floor4:12-15,floor4:10-13}"

# Core workaround: pool floor4 MC2 groups by exact rank tuple for the process
# lifetime. This intentionally avoids hot-path floor4 MC2 PG destroy/recreate.
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL:-1}"
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE="${VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE:-1}"
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS="${VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS:-mc2}"
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS:-12-15;10-13}"
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_MAX_GROUPS="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_MAX_GROUPS:-8}"
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_STRICT_RANK_SETS="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_STRICT_RANK_SETS:-1}"
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_UNIQUE_GROUP_NAME="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_UNIQUE_GROUP_NAME:-1}"

# Keep this as a clean live-pool experiment. If a floor4 MC2 group reaches the
# old retire/quarantine path, do not hide it behind the previous destroyed-PG
# ref retention workaround; make the miss visible in the analyzer instead.
export VLLM_ASCEND_MODE1_PARITY_DEFER_FLOOR4_MC2_DEVICE_PG_DESTROY=0
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_KEEP_DESTROYED_PG_REF=0
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_ASYNC_TRIM_DESTROYED_PG_REF=0
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_ASYNC_TRIM_DELAY_MS=0
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_DESTROYED_PG_REF_LIMIT=0

# Avoid extra quiesce collectives; live-pool should prevent floor4 MC2 retirement
# rather than trying to drain and destroy it more carefully.
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_QUIESCE_BEFORE_DESTROY=0

# Keep only slow timings and cache-hit proof logs. Verbose lifecycle logs can
# distort Ray stdout timing.
export VLLM_ASCEND_ELASTIC_GROUP_LIFECYCLE_LOG="${VLLM_ASCEND_ELASTIC_GROUP_LIFECYCLE_LOG:-0}"
export VLLM_ASCEND_ELASTIC_GROUP_CACHE_HIT_LOG="${VLLM_ASCEND_ELASTIC_GROUP_CACHE_HIT_LOG:-1}"
export VLLM_ASCEND_ELASTIC_SLOW_TIMING_LOG_MS="${VLLM_ASCEND_ELASTIC_SLOW_TIMING_LOG_MS:-1000}"
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_QUARANTINE_TRACE="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_QUARANTINE_TRACE:-0}"
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_LOG="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_LOG:-1}"
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_WARN_SIZE="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_WARN_SIZE:-8}"
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_BINDING_LOG="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_BINDING_LOG:-1}"

export RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/mode1_floor4_livepool_overlap4group_runs/$(date -u +%Y%m%dT%H%M%SZ)}"

printf '[mode1 floor4 livepool overlap4 test] modes=%s steps=%s run_root=%s live_pool=%s max_kv=%s\n' \
    "$VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES" \
    "$BASELINE_TOTAL_TRAINING_STEPS" \
    "$RUN_ROOT" \
    "$VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL" \
    "${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS:-288000}"
printf '[mode1 floor4 livepool overlap4 test] clean_livepool defer_floor4_destroy=%s keep_destroyed_pg_ref=%s quiesce=%s\n' \
    "$VLLM_ASCEND_MODE1_PARITY_DEFER_FLOOR4_MC2_DEVICE_PG_DESTROY" \
    "$VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_KEEP_DESTROYED_PG_REF" \
    "$VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_QUIESCE_BEFORE_DESTROY"
printf '[mode1 floor4 livepool overlap4 test] live_pool_rank_sets=%s\n' \
    "$VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS"
printf '[mode1 floor4 livepool overlap4 test] live_pool_max_groups=%s\n' \
    "$VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_MAX_GROUPS"
printf '[mode1 floor4 livepool overlap4 test] strict_rank_sets=%s\n' \
    "$VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_STRICT_RANK_SETS"
printf '[mode1 floor4 livepool overlap4 test] unique_group_name=%s\n' \
    "$VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_UNIQUE_GROUP_NAME"

exec bash "$SCRIPT_DIR/run_mode1_floor4_rotate4group_destroy_test.sh"
