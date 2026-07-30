#!/usr/bin/env bash
set -euo pipefail
mode="$1"
run_root="$2"
state_dir="$run_root/state_mode${mode}"
log_capture="$run_root/mode${mode}.launcher.out"
rm -rf "$state_dir"
mkdir -p "$state_dir"

export TRAINER_TOTAL_EPOCHS=1
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896
export VERL_SIDECAR_ENABLE=1
export VERL_SIDECAR_MODEL_PATH=/home/data/Qwen2.5-1.5B-Instruct
export VERL_SIDECAR_PARALLEL_MODE=dp
export VERL_SIDECAR_STATE_DIR="$state_dir"
export VERL_SIDECAR_LOG_DIR="$run_root/sidecar_mode${mode}"
export VERL_SIDECAR_EXPECTED_ACTIVE_RANKS=8
export VERL_SIDECAR_START_ONCE=1
export VERL_SIDECAR_MAX_SHRINK_TOTAL_MS=0
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=8
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE="$mode"
export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=1
unset VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE
unset VLLM_ASCEND_MODE5_RUNTIME_MIN_COMPUTE_GROUP_SIZE
unset VLLM_ASCEND_MODE4_STABILITY_FORCE_FLOOR
if [[ "$mode" == "4" ]]; then
  export VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE=8
  export VLLM_ASCEND_MODE4_BLOCK_PREFETCH_LAYERS=8
fi
bash /workspace/cann-recipes-train/llm_rl/qwen3/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh >"$log_capture" 2>&1
