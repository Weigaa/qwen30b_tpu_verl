#!/usr/bin/env bash

# Shared workload contract for mode=1/2/3/4/5 performance comparisons.
# Mode wrappers should only add execution-mode-specific runtime settings.

export TRAIN_BATCH_SIZE="${PERF_CLEAN_TRAIN_BATCH_SIZE:-32}"
export MAX_PROMPT_LENGTH="${PERF_CLEAN_MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${PERF_CLEAN_MAX_RESPONSE_LENGTH:-16384}"
export ROLLOUT_N="${PERF_CLEAN_ROLLOUT_N:-16}"
export ROLLOUT_MAX_NUM_SEQS="${PERF_CLEAN_MAX_NUM_SEQS:-32}"
export ROLLOUT_MAX_MODEL_LEN="${PERF_CLEAN_MAX_MODEL_LEN:-17408}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${PERF_CLEAN_MAX_NUM_BATCHED_TOKENS:-17408}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${PERF_CLEAN_GPU_MEMORY_UTILIZATION:-0.85}"
export TRAINER_TOTAL_EPOCHS=1
export VERL_SIDECAR_ENABLE=0

# RLHFDataset selects rows [32, 64) after dataset_fraction for every mode.
export PERF_CLEAN_TRAINING_STEPS="${PERF_CLEAN_TRAINING_STEPS:-1}"

# Keep the natural tail policy identical across modes. The floor itself remains
# mode-wrapper-specific.
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${PERF_CLEAN_TAIL_LEVEL_TOKENS:-256,512,640,768,896}"

# Disable diagnostics that add synchronization or per-layer host logging.
export VLLM_ASCEND_CUSTOM_MODE1_DEBUG=0
export VLLM_ASCEND_CUSTOM_MODE1_TIMING_EVENTS=0
export VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG=0
export VLLM_ASCEND_MODE3_TRANSFER_LOG=0
export VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG=0
export VLLM_ASCEND_MODE3_TIMING_LOG=0
export VLLM_ASCEND_MODE3_TIMING_SYNC=0
export VLLM_ASCEND_MODE3_TIMING_EVERY=1000000
export VLLM_ASCEND_MODE3_TIMING_FIRST_N=0
export VLLM_ASCEND_MODE3_MOE_FORWARD_TIMING=0
export VLLM_ASCEND_MC2_HOST_TIMING_LOG=0
export VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS=0
export VLLM_ASCEND_BUCKET_OP_PROFILE=0
export VLLM_ASCEND_BUCKET_OP_PROFILE_BY_STAGE=0
export VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS=""
export VLLM_ASCEND_DUMMY_WASTE_TIMING=0
export VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=0
export VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE=0
export VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS=0

# Use the same socket allocation policy in every comparison wrapper.
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"

PERF_CLEAN_LAUNCHER_ARGS=(
    "trainer.total_training_steps=${PERF_CLEAN_TRAINING_STEPS}"
)

print_perf_clean_common_config() {
    printf '[perf clean common] dataset_slice=rl_dataset:[32,64) steps=%s tail_levels=%s\n' \
        "$PERF_CLEAN_TRAINING_STEPS" \
        "$VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS"
    printf '[perf clean common] batch=%s rollout_n=%s max_prompt=%s max_response=%s max_seqs=%s max_model_len=%s max_batched_tokens=%s gpu_memory_utilization=%s sidecar=%s\n' \
        "$TRAIN_BATCH_SIZE" "$ROLLOUT_N" "$MAX_PROMPT_LENGTH" \
        "$MAX_RESPONSE_LENGTH" "$ROLLOUT_MAX_NUM_SEQS" \
        "$ROLLOUT_MAX_MODEL_LEN" "$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
        "$ROLLOUT_GPU_MEMORY_UTILIZATION" "$VERL_SIDECAR_ENABLE"
}
