#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
RETRY_OUTPUT_ROOT=${RETRY_OUTPUT_ROOT:-/data/adafloor_shared_state/adafloor_p_f4_seed3_transient_cleanup_retry_${RUN_STAMP}}

echo "[Planned F4 seed3 retry] output_root=$RETRY_OUTPUT_ROOT"
echo "[Planned F4 seed3 retry] full_restore_transient_cleanup=1"
echo "[Planned F4 seed3 retry] canonical_loaded_weight_offload=0"
echo "[Planned F4 seed3 retry] training_min_free_mib=${VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB:-28672}"

FAIR_OUTPUT_ROOT="$RETRY_OUTPUT_ROOT" \
VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1 \
VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0 \
VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB="${VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB:-28672}" \
./run_paper_fair_epoch1_2_from_common_epoch0.sh \
    adafloor_p_f4 \
    actor_rollout_ref.rollout.seed=3 \
    "$@"
