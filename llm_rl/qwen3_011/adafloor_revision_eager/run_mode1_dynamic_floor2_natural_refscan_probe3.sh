#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Minimal floor2 -> floor4 reference-retention probe.
#
# This reuses the existing epoch0 rollout history from the natural tailguard
# floor4 run, builds the floor2-capable dynamic plan, and executes only the
# first three mode1 steps.  The first three steps in the current oracle are
# floor2, floor2, floor4, so this is enough to hit the suspected KV-resize
# leak path without spending time on a full epoch.

BASELINE_EPOCH0="${DYNAMIC_INITIAL_BASELINE_DIR:-$SCRIPT_DIR/mode1_dynamic_length_aware_adaptive_floor4_natural_tailguard_full3/epoch_000_mode0_probe}"
if [[ ! -d "$BASELINE_EPOCH0/rollout_data" ]]; then
    echo "missing reusable epoch0 rollout data: $BASELINE_EPOCH0/rollout_data" >&2
    exit 2
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_floor2_natural_refscan_probe3_$stamp}"

export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=natural
export VLLM_ASCEND_SHRINK_AWARE_STAGES="${VLLM_ASCEND_SHRINK_AWARE_STAGES:-8,4,2}"
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE:-2}"
export MIN_ADAPTIVE_FLOOR="${MIN_ADAPTIVE_FLOOR:-2}"

# Use the no-trim floor2 cap that survived the sweep.  Keep floor4/floor8 caps
# at the currently measured natural-policy values so the third step attempts the
# same floor2 -> floor4 KV expansion that exposed the stale references.
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2:-131072}"
export FLOOR_KV_CAPS="${FLOOR_KV_CAPS:-2:${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2},4:280576,8:315648,16:380800}"

# Natural mode should not keep planned floor communicators resident.
export VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE=0
export VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS=0
export VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS=0
export VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE=0
export VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP=0
export VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS=0
export VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=0
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE=0

# Keep the allocator stable and enable only the logs needed to identify stale
# tensor owners during KV resize.  Leave the very hot MC2 per-op logs off.
export VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR="${VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR:-0}"
export VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM="${VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM:-0}"
export VLLM_ASCEND_MODE1_CLEAR_STALE_PARAM_DICTS_AFTER_OLD_KV="${VLLM_ASCEND_MODE1_CLEAR_STALE_PARAM_DICTS_AFTER_OLD_KV:-1}"
export VLLM_ASCEND_MODE1_KV_RESIZE_LIVE_TENSOR_SCAN="${VLLM_ASCEND_MODE1_KV_RESIZE_LIVE_TENSOR_SCAN:-1}"
export VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG=0
export VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG_ALL=0
export VLLM_ASCEND_MODE1_COMM_CACHE_STATE_DETAILED=0
export VLLM_ASCEND_CUSTOM_MODE1_GLOBAL_TENSOR_SCAN=0

# Runtime caps keep the probe short while preserving the offline plan generated
# from full epoch0 lengths.
export DYNAMIC_ENABLE_THRESHOLD_CONTROL="${DYNAMIC_ENABLE_THRESHOLD_CONTROL:-0}"
export DYNAMIC_SHORT_STEP_CAP_ENABLE="${DYNAMIC_SHORT_STEP_CAP_ENABLE:-1}"
export DYNAMIC_SHORT_STEP_EXIT_THRESHOLD="${DYNAMIC_SHORT_STEP_EXIT_THRESHOLD:-16384}"
export DYNAMIC_SHORT_STEP_CAP_TOKENS="${DYNAMIC_SHORT_STEP_CAP_TOKENS:-1024}"
export DYNAMIC_SHORT_STEP_CAP_FLOORS="${DYNAMIC_SHORT_STEP_CAP_FLOORS:-2,4}"

echo "[floor2 refscan probe] run_name=$DYNAMIC_RUN_NAME"
echo "[floor2 refscan probe] reusable_epoch0=$BASELINE_EPOCH0"
echo "[floor2 refscan probe] stages=$VLLM_ASCEND_SHRINK_AWARE_STAGES floor_caps=$FLOOR_KV_CAPS"
echo "[floor2 refscan probe] train_steps=3 live_tensor_scan=$VLLM_ASCEND_MODE1_KV_RESIZE_LIVE_TENSOR_SCAN"

exec env \
    DYNAMIC_TOTAL_EPOCHS=2 \
    DYNAMIC_START_EPOCH=1 \
    DYNAMIC_SHRINK_POLICY=natural \
    DYNAMIC_SKIP_MODE0_PROBE=1 \
    DYNAMIC_INITIAL_BASELINE_DIR="$BASELINE_EPOCH0" \
    DYNAMIC_ENABLE_CKPT_CHAIN=0 \
    DYNAMIC_PLAN_STEPS=5 \
    DYNAMIC_TRAIN_STEPS=3 \
    SAVE_CKPT_ENABLE=0 \
    "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh" "$@"
