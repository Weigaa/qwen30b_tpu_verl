#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${FAIR_OUTPUT_ROOT:-/data/adafloor_shared_state/adafloor_p_f4_training_guard_seed2_${RUN_STAMP}}

echo "[adafloor_p_f4 guard rerun] output_root=$OUTPUT_ROOT"
echo "[adafloor_p_f4 guard rerun] rollout_seed=2"
echo "[adafloor_p_f4 guard rerun] training_min_free_mib=${VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB:-28672}"
echo "[adafloor_p_f4 guard rerun] forced_moe_runtime_release_before_training=1"
echo "[adafloor_p_f4 guard rerun] canonical_loaded_weight_offload=0"
echo "[adafloor_p_f4 guard rerun] full_restore_transient_cleanup=1"

FAIR_OUTPUT_ROOT="$OUTPUT_ROOT" \
FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING=1 \
VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1 \
VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=1 \
VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1 \
VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB="${VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB:-28672}" \
VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0 \
./run_paper_fair_epoch1_2_from_common_epoch0.sh \
    adafloor_p_f4 \
    actor_rollout_ref.rollout.seed=2 \
    "$@"
