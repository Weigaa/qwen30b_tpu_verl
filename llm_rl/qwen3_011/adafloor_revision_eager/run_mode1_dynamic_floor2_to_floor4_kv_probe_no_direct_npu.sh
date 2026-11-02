#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Diagnostic variant of the floor2 -> floor4 KV probe.
# It keeps the same 2,2,4 schedule and short per-rank decode caps, but disables
# the direct NPU expert import path during shrink preload.  If this crosses the
# previous 8->4 failure point, the bug is isolated to direct NPU import / its
# transient HCCL workspace rather than the floor4 payload or MC2 mapping.

export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_floor2_to_floor4_kv_probe_no_direct_npu}"
export VLLM_ASCEND_MODE1_DISABLE_DIRECT_NPU_IMPORT=1

echo "[floor2->floor4 kv probe no-direct-npu] disable_direct_npu_import=${VLLM_ASCEND_MODE1_DISABLE_DIRECT_NPU_IMPORT}"
echo "[floor2->floor4 kv probe no-direct-npu] run_name=${DYNAMIC_RUN_NAME}"

exec "$SCRIPT_DIR/run_mode1_dynamic_floor2_to_floor4_kv_probe.sh" "$@"
