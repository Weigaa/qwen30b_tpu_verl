#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

choose_free_port_block() {
  local start="${1:?start}"
  local width="${2:?width}"
  local upper="${3:-61000}"
  python3 - "$start" "$width" "$upper" <<'PY'
import os
import random
import socket
import sys

start = int(sys.argv[1])
width = int(sys.argv[2])
upper = int(sys.argv[3]) if len(sys.argv) > 3 else 61000

def block_is_free(base: int) -> bool:
    sockets = []
    try:
        for port in range(base, base + width):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            sockets.append(sock)
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()

rng = random.Random((os.getpid() << 16) ^ int.from_bytes(os.urandom(4), "little"))
candidates = list(range(start, upper - width, max(width, 8)))
rng.shuffle(candidates)
for base in candidates:
    if block_is_free(base):
        print(base)
        raise SystemExit(0)
raise SystemExit(f"failed to find a free port block start>={start} width={width}")
PY
}

choose_free_hccl_base_port() {
  local start="${1:-20000}"
  local block="${2:-12288}"
  python3 - "$start" "$block" <<'PY'
import os
import random
import socket
import sys

start = int(sys.argv[1])
block = int(sys.argv[2])
limit = 65535 - block
offsets = (0, 4096, 8192)

def can_bind(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False

start = max(1024, min(start, limit))
candidates = list(range(start, limit + 1, block))
if not candidates or candidates[0] != 20000:
    candidates.extend(range(20000, limit + 1, block))
rng = random.Random((os.getpid() << 16) ^ int.from_bytes(os.urandom(4), "little"))
rng.shuffle(candidates)
seen = set()
for base in candidates:
    if base in seen:
        continue
    seen.add(base)
    if all(can_bind(base + offset) for offset in offsets):
        print(base)
        raise SystemExit(0)
raise SystemExit(
    f"failed to find a free HCCL base port start>={start} block={block} "
    f"limit={limit} offsets={offsets}"
)
PY
}

# Do not reuse the low default ports from the training script. HCCL may consume
# a range starting at the base port, so reserve a block instead of probing only
# one port.
export MASTER_PORT="${MASTER_PORT:-$(choose_free_port_block 24000 8)}"
export VERL_MASTER_PORT_START="${VERL_MASTER_PORT_START:-${MASTER_PORT}}"
export VERL_HCCL_IF_BASE_PORT_BLOCK="${VERL_HCCL_IF_BASE_PORT_BLOCK:-12288}"
if [[ -n "${HCCL_IF_BASE_PORT:-}" ]] \
    && (( HCCL_IF_BASE_PORT >= 1024 )) \
    && (( HCCL_IF_BASE_PORT <= 65535 - VERL_HCCL_IF_BASE_PORT_BLOCK )); then
  export HCCL_IF_BASE_PORT
else
  export HCCL_IF_BASE_PORT="$(choose_free_hccl_base_port 20000 "${VERL_HCCL_IF_BASE_PORT_BLOCK}")"
fi
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-${HCCL_IF_BASE_PORT}}"
echo "[mode1_floor2_test] MASTER_PORT=${MASTER_PORT} VERL_MASTER_PORT_START=${VERL_MASTER_PORT_START} HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT} VERL_HCCL_IF_BASE_PORT_START=${VERL_HCCL_IF_BASE_PORT_START} VERL_HCCL_IF_BASE_PORT_BLOCK=${VERL_HCCL_IF_BASE_PORT_BLOCK}"

# Mode=1 floor=2 parity smoke test aligned with the validated nm_mode1_final
# path. Keep the validated native KV cap and avoid local staging experiments so
# direct NPU imports and cleanup boundaries match the reference branch.
env \
  -u VLLM_TORCH_PROFILER_DIR \
  -u VLLM_ASCEND_BUCKET_OP_PROFILE \
  -u VLLM_ASCEND_BUCKET_OP_PROFILE_DIR \
  -u VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS \
  -u VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS \
  -u VLLM_ASCEND_NATIVE_MOE_TOPK_DEBUG \
  -u VLLM_ASCEND_MODE3_TIMING_LOG \
  -u VLLM_ASCEND_MODE3_TIMING_SYNC \
  VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1 \
  VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1 \
  VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2 \
  VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1 \
  VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=377344 \
  VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=1 \
  VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=0 \
  VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK=0 \
  VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP=0 \
  VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE=0 \
  VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP=0 \
  VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP=0 \
  VLLM_ASCEND_MODE1_DIRECT_NPU_STAGING=0 \
  VLLM_ASCEND_MODE1_PRE_SHRINK_NPU_DRAIN="${VLLM_ASCEND_MODE1_PRE_SHRINK_NPU_DRAIN:-1}" \
  VLLM_ASCEND_MODE1_PRE_SHRINK_DEVICE_BARRIER="${VLLM_ASCEND_MODE1_PRE_SHRINK_DEVICE_BARRIER:-1}" \
  VLLM_ASCEND_ELASTIC_INCLUDE_ENGINE_RUNNING_IN_ACTIVE_RANKS="${VLLM_ASCEND_ELASTIC_INCLUDE_ENGINE_RUNNING_IN_ACTIVE_RANKS:-0}" \
  VLLM_ASCEND_MODE1_WAIT_DUMMY_BOUNDARY_BEFORE_SHRINK="${VLLM_ASCEND_MODE1_WAIT_DUMMY_BOUNDARY_BEFORE_SHRINK:-0}" \
  VLLM_ASCEND_MODE1_SYNC_DUMMY_BOUNDARY_BEFORE_SHRINK="${VLLM_ASCEND_MODE1_SYNC_DUMMY_BOUNDARY_BEFORE_SHRINK:-0}" \
  VLLM_ASCEND_MODE1_WAIT_DUMMY_BOUNDARY_MAX_GROUP_SIZE="${VLLM_ASCEND_MODE1_WAIT_DUMMY_BOUNDARY_MAX_GROUP_SIZE:-4}" \
  VLLM_ASCEND_MODE1_EXPORT_DIAG="${VLLM_ASCEND_MODE1_EXPORT_DIAG:-0}" \
  VLLM_ASCEND_MODE1_EXPORT_STEP_DIAG="${VLLM_ASCEND_MODE1_EXPORT_STEP_DIAG:-1}" \
  VLLM_ASCEND_MODE1_EXPORT_STEP_DIAG_SLOW_MS="${VLLM_ASCEND_MODE1_EXPORT_STEP_DIAG_SLOW_MS:-1000}" \
  VLLM_ASCEND_MODE1_P2P_EXPORT_COPY=0 \
  VLLM_ASCEND_FULL_REDUNDANCY_EXPERIMENT_LOG=0 \
  VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG=0 \
  VLLM_ASCEND_MODE1_DIRECT_NPU_DIAG=0 \
  MASTER_PORT="${MASTER_PORT}" \
  VERL_MASTER_PORT_START="${VERL_MASTER_PORT_START}" \
  HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT}" \
  VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START}" \
  VERL_HCCL_IF_BASE_PORT_BLOCK="${VERL_HCCL_IF_BASE_PORT_BLOCK}" \
  TRAINER_TOTAL_EPOCHS=1 \
  bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh \
  2>&1 | tee mode1_floor2_first_step_$(date -u +%Y%m%dT%H%M%SZ).log
