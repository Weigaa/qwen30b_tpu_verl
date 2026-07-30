#!/usr/bin/env bash
set -euo pipefail

CALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="${SOURCE_REPO:-${CALLER_DIR}}"
REMOTE_REF="${REMOTE_REF:-origin/nm_mode1_final}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_PARENT="${RUN_PARENT:-/tmp}"
RUN_DIR="${RUN_DIR:-${RUN_PARENT}/qwen3_nm_mode1_final_${TS}}"
OUT_DIR="${OUT_DIR:-${CALLER_DIR}/nm_mode1_final_runs/${TS}}"

MODEL_PATH="${MODEL_PATH:-/home/data/Qwen3-30B-A3B}"
DISTCP_PATH="${DISTCP_PATH:-/home/data/Qwen3-30B-A3B_megatron}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/data/deepscaler/train.parquet}"
TEST_FILE="${TEST_FILE:-/workspace/data/deepscaler/test.parquet}"

choose_free_port_block() {
  local start="${1:?start}"
  local width="${2:?width}"
  python3 - "$start" "$width" <<'PY'
import os
import random
import socket
import sys

start = int(sys.argv[1])
width = int(sys.argv[2])
upper = 61000

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

MASTER_PORT="${MASTER_PORT:-$(choose_free_port_block 24000 8)}"
VERL_MASTER_PORT_START="${VERL_MASTER_PORT_START:-${MASTER_PORT}}"
HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-$(choose_free_port_block 42000 128)}"
VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-${HCCL_IF_BASE_PORT}}"

mkdir -p "${RUN_DIR}" "${OUT_DIR}"

echo "[nm_mode1_final] source_repo=${SOURCE_REPO}"
echo "[nm_mode1_final] remote_ref=${REMOTE_REF}"
echo "[nm_mode1_final] run_dir=${RUN_DIR}"
echo "[nm_mode1_final] out_dir=${OUT_DIR}"
echo "[nm_mode1_final] model_path=${MODEL_PATH}"
echo "[nm_mode1_final] distcp_path=${DISTCP_PATH}"
echo "[nm_mode1_final] train_file=${TRAIN_FILE}"
echo "[nm_mode1_final] test_file=${TEST_FILE}"
echo "[nm_mode1_final] MASTER_PORT=${MASTER_PORT} VERL_MASTER_PORT_START=${VERL_MASTER_PORT_START}"
echo "[nm_mode1_final] HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT} VERL_HCCL_IF_BASE_PORT_START=${VERL_HCCL_IF_BASE_PORT_START}"

git -C "${SOURCE_REPO}" rev-parse --verify "${REMOTE_REF}^{commit}" >/dev/null
git -C "${SOURCE_REPO}" archive "${REMOTE_REF}" | tar -x -C "${RUN_DIR}"
git -C "${SOURCE_REPO}" rev-parse "${REMOTE_REF}^{commit}" > "${OUT_DIR}/source_commit.txt"

copy_logs() {
  local rc=$?
  shopt -s nullglob
  cp -a "${RUN_DIR}"/wjeagerqwen30b-a3b-with_draft_*.txt "${OUT_DIR}/" 2>/dev/null || true
  cp -a "${RUN_DIR}"/mode1_floor2_*.log "${OUT_DIR}/" 2>/dev/null || true
  shopt -u nullglob
  echo "[nm_mode1_final] exit_code=${rc}"
  echo "[nm_mode1_final] run_dir=${RUN_DIR}"
  echo "[nm_mode1_final] copied_logs_to=${OUT_DIR}"
  exit "${rc}"
}
trap copy_logs EXIT

cd "${RUN_DIR}"

env \
  -u VLLM_TORCH_PROFILER_DIR \
  -u VLLM_ASCEND_BUCKET_OP_PROFILE \
  -u VLLM_ASCEND_BUCKET_OP_PROFILE_DIR \
  -u VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS \
  -u VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS \
  -u VLLM_ASCEND_NATIVE_MOE_TOPK_DEBUG \
  -u VLLM_ASCEND_MODE3_TIMING_LOG \
  -u VLLM_ASCEND_MODE3_TIMING_SYNC \
  MODEL_PATH="${MODEL_PATH}" \
  DISTCP_PATH="${DISTCP_PATH}" \
  TRAIN_FILE="${TRAIN_FILE}" \
  TEST_FILE="${TEST_FILE}" \
  RECORD_DIR="${OUT_DIR}/record" \
  VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1 \
  VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2 \
  VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1 \
  VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=377344 \
  VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=1 \
  VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=0 \
  VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK=0 \
  VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP=0 \
  VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE=0 \
  VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP=0 \
  TRAINER_TOTAL_EPOCHS=1 \
  MASTER_PORT="${MASTER_PORT}" \
  VERL_MASTER_PORT_START="${VERL_MASTER_PORT_START}" \
  HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT}" \
  VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START}" \
  bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh \
  2>&1 | tee "${OUT_DIR}/launcher.log"
