#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ "${QWEN3_RUNTIME_STACK:-}" != vllm0141_vllm_ascend0140rc1_torch29_cann850_a3 ]; then
  source "${ROOT}/tools/activate_cann85_a3_torch29.sh"
fi

unset ASCEND_RT_VISIBLE_DEVICES NPU_VISIBLE_DEVICES
export HCCL_OP_EXPANSION_MODE=HOST

NPROC=${QWEN3_HCCL_SMOKE_NPROC:-16}
TIMEOUT_SECONDS=${QWEN3_HCCL_SMOKE_TIMEOUT_SECONDS:-300}
# Use a per-invocation port block so a recently terminated HCCL job cannot
# leave the default 60000 listener unavailable to this short-lived probe.
SMOKE_PORT_BLOCK=$((52000 + (($$ % 200) * 32)))
export HCCL_IF_BASE_PORT=${QWEN3_HCCL_SMOKE_IF_BASE_PORT:-${SMOKE_PORT_BLOCK}}
if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]] || [ "${NPROC}" -gt 16 ]; then
  printf '[hccl smoke] invalid QWEN3_HCCL_SMOKE_NPROC=%s (expected 1..16)\n' \
    "${NPROC}" >&2
  exit 2
fi

printf '[hccl smoke] start ranks=%s expansion=%s port=%s timeout=%ss\n' \
  "${NPROC}" "${HCCL_OP_EXPANSION_MODE}" "${HCCL_IF_BASE_PORT}" \
  "${TIMEOUT_SECONDS}"
timeout --signal=TERM --kill-after=30 "${TIMEOUT_SECONDS}" \
  python3 -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC}" \
    "${ROOT}/tools/hccl_collective_smoke.py"
printf '[hccl smoke] PASS ranks=%s (including communicator teardown)\n' "${NPROC}"
