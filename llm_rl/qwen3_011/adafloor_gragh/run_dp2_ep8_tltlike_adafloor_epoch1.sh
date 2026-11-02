#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# This arm implements TLT-style worker-granularity reuse accounting without
# TLT speculative decoding. AdaFloor first length-groups the global steps and
# then independently selects the KV-safe floor inside each EP8 worker.
DP2_EP8_GROUPING=length_sorted \
DP2_EP8_RUN_NAME="${DP2_EP8_RUN_NAME:-dp2_ep8_tltlike_adafloor_epoch1}" \
HCCL_IF_BASE_PORT="${DP2_EP8_TLTLIKE_HCCL_IF_BASE_PORT:-36000}" \
VERL_HCCL_IF_BASE_PORT_START="${DP2_EP8_TLTLIKE_HCCL_IF_BASE_PORT:-36000}" \
MASTER_PORT="${DP2_EP8_TLTLIKE_MASTER_PORT:-15000}" \
VERL_MASTER_PORT_START="${DP2_EP8_TLTLIKE_MASTER_PORT:-15000}" \
"$SCRIPT_DIR/run_dp2_ep8_adafloor_one_epoch.sh" "$@"
