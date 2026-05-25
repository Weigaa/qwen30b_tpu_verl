#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_TS="${RUN_TS:-$(date +%Y%m%d%H%M%S)}"
CUSTOM_MODE1_KV_HEADROOM_BYTES=${CUSTOM_MODE1_KV_HEADROOM_BYTES:-2147483648}
KV_CACHE_INIT_HEADROOM_BYTES=${KV_CACHE_INIT_HEADROOM_BYTES:-0}
TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-3}
OUTER_LOG="wjeagerqwen30b-a3b-custom_mode1_floor8_no_base_init_kvguard2048m_${TRAINER_TOTAL_EPOCHS}epochs_${RUN_TS}_elastic.txt"

{
    echo "[run] start_time=$(date -Iseconds) log=${OUTER_LOG}"
    echo "[run] custom=1 mode=1 floor=8 sidecar=0 headroom_all_zero=1 custom_kv_materialize_headroom=${CUSTOM_MODE1_KV_HEADROOM_BYTES} kv_cache_init_headroom=${KV_CACHE_INIT_HEADROOM_BYTES} total_epochs=${TRAINER_TOTAL_EPOCHS} no_base_fusedmoe_init=1"

    export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1
    export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
    export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=8
    export VERL_SIDECAR_ENABLE=0
    export VLLM_ASCEND_FULL_REDUNDANCY_EXPERIMENT_LOG=1

    # Keep the test focused: disable the older generic shrink/restore guards and
    # only leave the custom mode=1 KV materialization guard enabled.
    export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_HEADROOM_BYTES=0
    export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_LOW_FLOOR_HEADROOM_BYTES=0
    export VLLM_ASCEND_POST_SHRINK_PREFILL_ALLTOALL_HEADROOM_BYTES=0
    export VLLM_ASCEND_POST_RESTORE_DP_HEADROOM_BYTES=0
    export VLLM_ASCEND_POST_RESTORE_EP_HEADROOM_BYTES=0
    export VLLM_ASCEND_POST_RESTORE_MOE_DISPATCH_HEADROOM_BYTES=0
    export VLLM_ASCEND_FIRST_LIVE_PREFILL_HEADROOM_BYTES=0
    export VLLM_ASCEND_FIRST_LIVE_PREFILL_LOW_FLOOR_HEADROOM_BYTES=0
    export VLLM_ASCEND_EXTRA_ELASTIC_SAFETY_HEADROOM_BYTES=0
    export VLLM_ASCEND_FLOOR_PREALLOC_HEADROOM_SAFETY_BYTES=0
    export VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES="${CUSTOM_MODE1_KV_HEADROOM_BYTES}"
    export VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES="${KV_CACHE_INIT_HEADROOM_BYTES}"
    export TRAINER_TOTAL_EPOCHS

    set +e
    bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh
    run_exit_code=$?
    set -e

    echo "[run] end_time=$(date -Iseconds) exit_code=${run_exit_code}"
    exit "${run_exit_code}"
} 2>&1 | tee "${OUTER_LOG}"
