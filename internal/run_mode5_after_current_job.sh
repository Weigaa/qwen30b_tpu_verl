#!/usr/bin/env bash
set -euo pipefail
cd /workspace/cann-recipes-train/llm_rl/qwen3

echo "[mode5-watch] start_time=$(date '+%Y-%m-%dT%H:%M:%S%z') cwd=$(pwd)"
while pgrep -f "python3 -m verl.trainer.main_ppo" >/dev/null 2>&1; do
  echo "[mode5-watch] waiting_for_existing_main_ppo time=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  sleep 60
done

echo "[mode5-watch] launch_mode5 time=$(date '+%Y-%m-%dT%H:%M:%S%z')"
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=5
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=1
export VLLM_ASCEND_MODE4_STABILITY_PROFILE=1
export VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY=fixed
export VLLM_ASCEND_MODE5_BALANCE_REMOTE_SOURCE_FANOUT=0
export VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC=1
export VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=1
export VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION=0.75
export VLLM_ASCEND_MODE3_TIMING_LOG=1
export VLLM_ASCEND_MODE3_TIMING_SYNC=1
export VLLM_ASCEND_MODE3_TIMING_EVERY=1
export VLLM_ASCEND_MODE3_TIMING_FIRST_N=8
export VLLM_ASCEND_MODE3_TIMING_LAYERS=all
export VLLM_ASCEND_MODE3_TRANSFER_LOG=0
export VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG=0
bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh
