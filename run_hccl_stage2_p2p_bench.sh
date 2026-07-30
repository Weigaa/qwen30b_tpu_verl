#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/hccl_stage2_p2p_bench/${TS}}"
mkdir -p "${OUT_DIR}"

choose_free_port() {
  python3 - <<'PY'
import os
import random
import socket

rng = random.Random((os.getpid() << 16) ^ int.from_bytes(os.urandom(4), "little"))
ports = list(range(24000, 61000))
rng.shuffle(ports)
for port in ports:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        continue
    finally:
        sock.close()
    print(port)
    raise SystemExit(0)
raise SystemExit("no free port")
PY
}

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

def is_free(base):
    socks = []
    try:
        for port in range(base, base + width):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            socks.append(sock)
        return True
    except OSError:
        return False
    finally:
        for sock in socks:
            sock.close()

rng = random.Random((os.getpid() << 16) ^ int.from_bytes(os.urandom(4), "little"))
candidates = list(range(start, upper - width, max(width, 8)))
rng.shuffle(candidates)
for base in candidates:
    if is_free(base):
        print(base)
        raise SystemExit(0)
raise SystemExit("no free port block")
PY
}

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-12,13,14,15}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(choose_free_port)}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-$(choose_free_port_block 42000 128)}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp23s0f3}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-120}"
if [[ "${HCCL_CONNECT_TIMEOUT}" =~ ^[0-9]+$ ]] && (( HCCL_CONNECT_TIMEOUT < 120 )); then
  echo "[bench] HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT} is invalid for HCCL; clamping to 120"
  export HCCL_CONNECT_TIMEOUT=120
fi
export BENCH_DIST_TIMEOUT_S="${BENCH_DIST_TIMEOUT_S:-150}"
export BENCH_MB_LIST="${BENCH_MB_LIST:-16,64,288}"
export BENCH_REPEATS="${BENCH_REPEATS:-3}"

echo "[bench] OUT_DIR=${OUT_DIR}"
echo "[bench] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "[bench] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[bench] HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT} HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT}"
echo "[bench] BENCH_MB_LIST=${BENCH_MB_LIST} BENCH_REPEATS=${BENCH_REPEATS}"

bash "${ROOT_DIR}/internal/collect_npu_hccl_env.sh" > "${OUT_DIR}/env.log" 2>&1 || true

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${ROOT_DIR}/internal/hccl_stage2_p2p_bench.py" \
  2>&1 | tee "${OUT_DIR}/bench.log"
