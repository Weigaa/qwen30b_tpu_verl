#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Diagnostic: keep rollout/reward/old_log_prob/ref paths, but skip the actor
# parameter update. If step2+ decode speed no longer drops, the regression is
# caused by update_actor or the weight-sync/runtime state that follows it.
export VERL_DIAG_SKIP_ACTOR_UPDATE=1
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_baseline_random_batch_floor2_diag_skip_actor_update}"
export KV_TOKENS="${KV_TOKENS:-100000}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export DATASET_FRACTION="${DATASET_FRACTION:-0.005}"
export VLLM_ASCEND_MOE_DISPATCH_SEQ_DIGEST_SAMPLE_EVERY="${VLLM_ASCEND_MOE_DISPATCH_SEQ_DIGEST_SAMPLE_EVERY:-64}"
PORT_BASE="${PORT_BASE:-$((30000 + ($$ % 20000)))}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-$((PORT_BASE))-$((PORT_BASE + 511))}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-$((PORT_BASE + 512))-$((PORT_BASE + 1023))}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-$((PORT_BASE + 1024))}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-$HCCL_IF_BASE_PORT}"
export MASTER_PORT="${MASTER_PORT:-$((PORT_BASE + 1536))}"

printf '[diag skip actor update] output=%s kv=%s epochs=%s dataset_fraction=%s ports=%s/%s master=%s\n' \
  "$OUTPUT_SUBDIR" "$KV_TOKENS" "$TRAINER_TOTAL_EPOCHS" "$DATASET_FRACTION" \
  "$HCCL_NPU_SOCKET_PORT_RANGE" "$HCCL_HOST_SOCKET_PORT_RANGE" "$MASTER_PORT"

exec "$SCRIPT_DIR/run_mode1_local_baseline_random_batch_floor2_threshold_multistep.sh" "$@"
