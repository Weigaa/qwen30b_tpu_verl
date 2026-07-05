#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Diagnostic: same model/runtime and forced-MC2 MoE, but disable shrink-aware
# scheduling and keep all 16 ranks active. If step2+ still slows down, the
# issue is not shrink/restore lifecycle; it is in rollout runtime/update sync.
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_baseline_random_batch_floor2_diag_no_shrink_mc2}"
export MODEL_PATH="${MODEL_PATH:-/home/data/Qwen3-30B-A3B}"
export DISTCP_PATH="${DISTCP_PATH:-/home/data/Qwen3-30B-A3B_megatron}"
export TRAIN_FILE="${TRAIN_FILE:-/workspace/data/deepscaler/train.parquet}"
export TEST_FILE="${TEST_FILE:-/workspace/data/deepscaler/test.parquet}"
export MODE1_FLOOR=16
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=16
export VLLM_ASCEND_SHRINK_AWARE_ENABLE=0
export ROLLOUT_SHRINK_AWARE_ENABLE=false
export ROLLOUT_SHRINK_AWARE_MODE=off
export VLLM_ASCEND_SHRINK_AWARE_STAGES="${VLLM_ASCEND_SHRINK_AWARE_STAGES:-}"
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-640,640,640,640,640}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-640}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export KV_TOKENS="${KV_TOKENS:-100000}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export DATASET_FRACTION="${DATASET_FRACTION:-0.005}"
export VLLM_ASCEND_FORCE_MC2_MOE="${VLLM_ASCEND_FORCE_MC2_MOE:-1}"
export VLLM_ASCEND_FORCE_ALLTOALL_MOE=0
export VLLM_ASCEND_AVOID_ALLTOALL_ON_SHRINK_STAGE=1
export VLLM_ASCEND_RESERVED_MC2_MASK_TOKENS="${VLLM_ASCEND_RESERVED_MC2_MASK_TOKENS:-4096}"
PORT_BASE="${PORT_BASE:-$((30000 + ($$ % 20000)))}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-$((PORT_BASE))-$((PORT_BASE + 511))}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-$((PORT_BASE + 512))-$((PORT_BASE + 1023))}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-$((PORT_BASE + 1024))}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-$HCCL_IF_BASE_PORT}"
export MASTER_PORT="${MASTER_PORT:-$((PORT_BASE + 1536))}"

printf '[diag no shrink mc2] output=%s kv=%s floor=%s shrink_enable=%s rollout_shrink=%s max_response=%s max_batched=%s mask=%s ports=%s/%s master=%s\n' \
  "$OUTPUT_SUBDIR" "$KV_TOKENS" "$MODE1_FLOOR" "$VLLM_ASCEND_SHRINK_AWARE_ENABLE" \
  "$ROLLOUT_SHRINK_AWARE_ENABLE" "$MAX_RESPONSE_LENGTH" "$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  "$VLLM_ASCEND_RESERVED_MC2_MASK_TOKENS" "$HCCL_NPU_SOCKET_PORT_RANGE" \
  "$HCCL_HOST_SOCKET_PORT_RANGE" "$MASTER_PORT"

exec "$SCRIPT_DIR/run_mode1_local_baseline_random_batch.sh" \
  data.sampler.class_path=null \
  data.sampler.class_name=null \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_scheduling=false \
  actor_rollout_ref.rollout.shrink_aware.shrink_aware_mode=off \
  "$@"
