#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
SOURCE_ROOT=${SOURCE_ROOT:-/data/adafloor_shared_state/adafloor_p_f4_training_guard_seed2_20260725T065522Z}
SOURCE_NAME=${SOURCE_NAME:-adafloor_planned_floor4_tailguard_common_epoch0_epoch1_2}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/adafloor_shared_state/adafloor_p_f4_low_peak_optimizer_epoch2_${RUN_STAMP}}
OUTPUT_NAME=${OUTPUT_NAME:-adafloor_planned_floor4_tailguard_epoch2_low_peak_optimizer}

echo "[low-peak optimizer validation] source=$SOURCE_ROOT/$SOURCE_NAME"
echo "[low-peak optimizer validation] output=$OUTPUT_ROOT/$OUTPUT_NAME"
echo "[low-peak optimizer validation] canonical_loaded_weight_offload=0"
echo "[low-peak optimizer validation] full_restore_transient_cleanup=1"
echo "[low-peak optimizer validation] disabled_zero_grad_stat_is_skipped=1"

RETRY_SOURCE_RUN_ROOT="$SOURCE_ROOT" \
RETRY_SOURCE_RUN_NAME="$SOURCE_NAME" \
RETRY_RUN_ROOT="$OUTPUT_ROOT" \
RETRY_RUN_NAME="$OUTPUT_NAME" \
RETRY_REMOVE_SOURCE_EPOCH1_CHECKPOINT=0 \
VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1 \
VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0 \
VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB="${VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB:-28672}" \
./run_adafloor_p_f4_epoch2_from_retained_epoch1.sh "$@"
