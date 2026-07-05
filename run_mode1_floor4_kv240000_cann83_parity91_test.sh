#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Old software stack, but runtime knobs aligned with the successful CANN 9.1 run.
# This wrapper intentionally does not source/use CANN 9.1 or the torch 2.9 venv.
export KV_TOKENS="${KV_TOKENS:-240000}"
export RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_cann83_parity91_kv${KV_TOKENS}_floor4}"

# Match the CANN 9.1 successful run behavior while keeping CANN/torch/ATB old.
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE="${VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE:-0}"
export VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC="${VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC:-1}"
export VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE="${VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE:-1}"
export VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_NON_MEMBER_PLACEHOLDER="${VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_NON_MEMBER_PLACEHOLDER:-1}"
export VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_RETIRE="${VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_RETIRE:-0}"
export VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_NON_MEMBER_PLACEHOLDER="${VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_NON_MEMBER_PLACEHOLDER:-0}"

# Keep the rest identical to the old-stack wrapper/test path.
exec bash "$SCRIPT_DIR/run_mode1_floor4_kv240000_cann83_test.sh" "$@"
