#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29641}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-20000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

TS="$(date +%Y%m%dT%H%M%S%N)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/rank_pair_p2p_alias_runs/${TS}}"
mkdir -p "${OUT_DIR}"
LOGFILE="${OUT_DIR}/bench.log"

LAYERS="${LAYERS:-48}"
EXPERTS="${EXPERTS:-64}"
CAPACITY="${CAPACITY:-64}"
PAIRS="${PAIRS:-12:14,13:15,12:15,13:14}"
GROUP_RANKS="${GROUP_RANKS:-12,13,14,15}"

{
  echo "[bench] start_time=$(date -Is) out_dir=${OUT_DIR}"
  echo "[bench] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT}"
  echo "[bench] LAYERS=${LAYERS} EXPERTS=${EXPERTS} CAPACITY=${CAPACITY} PAIRS=${PAIRS} GROUP_RANKS=${GROUP_RANKS}"
  python3 - <<'PY'
import sys
print('[bench] python', sys.version.replace('\n', ' '))
try:
    import torch
    print('[bench] torch', torch.__version__)
except Exception as exc:
    print('[bench] torch import failed', repr(exc))
try:
    import torch_npu
    print('[bench] torch_npu', getattr(torch_npu, '__version__', 'unknown'))
except Exception as exc:
    print('[bench] torch_npu import failed', repr(exc))
PY
  torchrun \
    --nnodes=1 \
    --nproc_per_node=16 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/rank_pair_p2p_alias_bench.py" \
      --pairs "${PAIRS}" \
      --group-ranks "${GROUP_RANKS}" \
      --layers "${LAYERS}" \
      --experts "${EXPERTS}" \
      --capacity "${CAPACITY}" \
      "$@"
  echo "[bench] end_time=$(date -Is)"
} 2>&1 | tee "${LOGFILE}"

echo "[bench] logfile=${LOGFILE}"
