#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Diagnostic baseline: exactly the current floor2 shrink-aware MC2 path, with a
# stable output dir name for A/B comparison against skip-update and no-shrink.
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_baseline_random_batch_floor2_diag_current_baseline}"
export KV_TOKENS="${KV_TOKENS:-100000}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export DATASET_FRACTION="${DATASET_FRACTION:-0.005}"
PORT_BASE="${PORT_BASE:-$((30000 + ($$ % 20000)))}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-$((PORT_BASE))-$((PORT_BASE + 511))}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-$((PORT_BASE + 512))-$((PORT_BASE + 1023))}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-$((PORT_BASE + 1024))}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-$HCCL_IF_BASE_PORT}"
export MASTER_PORT="${MASTER_PORT:-$((PORT_BASE + 1536))}"

printf '[diag current baseline] output=%s kv=%s epochs=%s dataset_fraction=%s ports=%s/%s master=%s\n' \
  "$OUTPUT_SUBDIR" "$KV_TOKENS" "$TRAINER_TOTAL_EPOCHS" "$DATASET_FRACTION" \
  "$HCCL_NPU_SOCKET_PORT_RANGE" "$HCCL_HOST_SOCKET_PORT_RANGE" "$MASTER_PORT"

exec "$SCRIPT_DIR/run_mode1_local_baseline_random_batch_floor2_threshold_multistep.sh" "$@"
