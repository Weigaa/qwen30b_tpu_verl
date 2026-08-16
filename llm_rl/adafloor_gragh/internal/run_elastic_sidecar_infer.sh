#!/usr/bin/env bash
set -euo pipefail

# Offline low-priority sidecar inference for resources released by elastic shrink.
# Required:
#   VERL_SIDECAR_MODEL_PATH=/path/to/model
# Optional:
#   VERL_SIDECAR_NPU_DEVICES=comma-separated released NPU ids
#   VERL_SIDECAR_PROMPTS_FILE=/path/to/prompts.txt|jsonl|json|parquet|dataset_dir
#   VERL_SIDECAR_OUTPUT_FILE=/path/to/output.jsonl
#   VERL_SIDECAR_MAX_SECONDS=60
# Parallel layout:
#   VERL_SIDECAR_PARALLEL_MODE=dp|tp|hybrid|replica_tp|auto
#   VERL_SIDECAR_TENSOR_PARALLEL_SIZE=<cards per replica>
#   VERL_SIDECAR_REPLICA_COUNT=<number of independent replicas>
# Examples with 8 released NPUs:
#   DP8:         TENSOR_PARALLEL_SIZE=1, REPLICA_COUNT=8
#   TP8:         TENSOR_PARALLEL_SIZE=8, REPLICA_COUNT=1
#   4 replicas:  TENSOR_PARALLEL_SIZE=2, REPLICA_COUNT=4
# As long as the selected model fits in each replica group, the sidecar does
# not assume a fixed model size or fixed parallelism.
# Expert parallel:
#   VERL_SIDECAR_ENABLE_EXPERT_PARALLEL=1 only permits EP.
#   EP is enabled effectively only when the sidecar model is detected as MoE.
#   Override detection with VERL_SIDECAR_MODEL_IS_MOE=0|1 if needed.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TS=$(date +%Y%m%d%H%M%S)

: "${VERL_SIDECAR_MODEL_PATH:?VERL_SIDECAR_MODEL_PATH must point to the sidecar inference model}"
: "${VERL_SIDECAR_NPU_DEVICES:?VERL_SIDECAR_NPU_DEVICES must be set by the shrink watcher or manually for direct use}"

detect_sidecar_host_ip() {
    python3 - <<'PY'
import socket

def detect() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect does not require the address to be reachable; it only
        # asks the kernel which local address would be used.
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"
    finally:
        sock.close()

print(detect())
PY
}

VERL_SIDECAR_MASTER_PORT=${VERL_SIDECAR_MASTER_PORT:-24300}
VERL_SIDECAR_MASTER_ADDR=${VERL_SIDECAR_MASTER_ADDR:-$(detect_sidecar_host_ip)}
VERL_SIDECAR_HCCL_IF_BASE_PORT=${VERL_SIDECAR_HCCL_IF_BASE_PORT:-52000}
VERL_SIDECAR_HCCL_IF_BASE_PORT_FIXED=${VERL_SIDECAR_HCCL_IF_BASE_PORT_FIXED:-0}
VERL_SIDECAR_MASTER_PORT_STRIDE=${VERL_SIDECAR_MASTER_PORT_STRIDE:-32}
VERL_SIDECAR_HCCL_PORT_STRIDE=${VERL_SIDECAR_HCCL_PORT_STRIDE:-2048}
VERL_SIDECAR_HCCL_PORT_WINDOW=${VERL_SIDECAR_HCCL_PORT_WINDOW:-64}
VERL_SIDECAR_FORCE_HOST_IP_FOR_TP=${VERL_SIDECAR_FORCE_HOST_IP_FOR_TP:-1}
VERL_SIDECAR_LOG_FILE=${VERL_SIDECAR_LOG_FILE:-"${ROOT_DIR}/sidecar_infer_${TS}.log"}
VERL_SIDECAR_OUTPUT_FILE=${VERL_SIDECAR_OUTPUT_FILE:-"${ROOT_DIR}/sidecar_infer_${TS}.jsonl"}
VERL_SIDECAR_MAX_SECONDS=${VERL_SIDECAR_MAX_SECONDS:-0}
VERL_SIDECAR_DEVICE_COUNT=$(python3 - "${VERL_SIDECAR_NPU_DEVICES}" <<'PY'
import sys

devices = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(len(devices))
PY
)
if [[ "${VERL_SIDECAR_DEVICE_COUNT}" == "0" ]]; then
    echo "VERL_SIDECAR_NPU_DEVICES does not contain any usable device id: ${VERL_SIDECAR_NPU_DEVICES}" >&2
    exit 2
fi
VERL_SIDECAR_PARALLEL_MODE=${VERL_SIDECAR_PARALLEL_MODE:-dp}
VERL_SIDECAR_DATA_PARALLEL_SIZE=${VERL_SIDECAR_DATA_PARALLEL_SIZE:-${VERL_SIDECAR_DP_SIZE:-1}}
SIDECAR_PARALLEL_PLAN=$(python3 - \
    "${VERL_SIDECAR_NPU_DEVICES}" \
    "${VERL_SIDECAR_PARALLEL_MODE}" \
    "${VERL_SIDECAR_TENSOR_PARALLEL_SIZE:-}" \
    "${VERL_SIDECAR_REPLICA_COUNT:-}" \
    "${VERL_SIDECAR_DATA_PARALLEL_SIZE}" <<'PY'
import shlex
import sys

devices_arg, mode, tp_arg, replica_arg, dp_arg = sys.argv[1:6]
devices = [item.strip() for item in devices_arg.split(",") if item.strip()]
device_count = len(devices)
mode = (mode or "dp").strip().lower()
valid_modes = {"dp", "tp", "hybrid", "replica_tp", "auto"}
if mode not in valid_modes:
    raise SystemExit(
        f"Unsupported VERL_SIDECAR_PARALLEL_MODE={mode}; "
        f"expected one of {sorted(valid_modes)}")

if tp_arg.strip():
    tp_size = int(tp_arg)
elif mode == "tp":
    tp_size = device_count
else:
    tp_size = 1
if tp_size <= 0:
    raise SystemExit(f"Invalid VERL_SIDECAR_TENSOR_PARALLEL_SIZE={tp_size}")

dp_size = int(dp_arg) if dp_arg.strip() else 1
if dp_size <= 0:
    raise SystemExit(f"Invalid VERL_SIDECAR_DATA_PARALLEL_SIZE={dp_size}")
group_size = tp_size * dp_size
if group_size <= 0:
    raise SystemExit(f"Invalid sidecar group size: tp={tp_size}, dp={dp_size}")

if replica_arg.strip():
    replica_count = int(replica_arg)
    if replica_count <= 0:
        raise SystemExit(f"Invalid VERL_SIDECAR_REPLICA_COUNT={replica_count}")
    used_devices = replica_count * group_size
    if used_devices > device_count:
        raise SystemExit(
            "VERL_SIDECAR_REPLICA_COUNT * VERL_SIDECAR_TENSOR_PARALLEL_SIZE "
            "* VERL_SIDECAR_DATA_PARALLEL_SIZE exceeds available devices: "
            f"{replica_count} * {tp_size} * {dp_size} > {device_count}")
else:
    if device_count % group_size != 0:
        raise SystemExit(
            f"Device count {device_count} is not divisible by sidecar group "
            f"size tp*dp={tp_size}*{dp_size}; set VERL_SIDECAR_REPLICA_COUNT "
            "explicitly if you intentionally want to leave some devices unused.")
    replica_count = device_count // group_size
    used_devices = device_count

groups = [
    ",".join(devices[start:start + group_size])
    for start in range(0, used_devices, group_size)
]
unused = devices[used_devices:]
print(f"VERL_SIDECAR_TENSOR_PARALLEL_SIZE={tp_size}")
print(f"VERL_SIDECAR_DATA_PARALLEL_SIZE={dp_size}")
print(f"VERL_SIDECAR_REPLICA_COUNT={replica_count}")
print("VERL_SIDECAR_DEVICE_GROUPS=" + shlex.quote(";".join(groups)))
print("VERL_SIDECAR_UNUSED_DEVICES=" + shlex.quote(",".join(unused)))
PY
)
eval "${SIDECAR_PARALLEL_PLAN}"
SIDECAR_GLOBAL_SHARD_PLAN=$(python3 - \
    "${VERL_SIDECAR_REPLICA_COUNT}" \
    "${VERL_SIDECAR_GLOBAL_NUM_SHARDS:-}" \
    "${VERL_SIDECAR_GLOBAL_SHARD_INDICES:-}" <<'PY'
import shlex
import sys

replica_count = int(sys.argv[1])
num_shards_arg = sys.argv[2].strip()
indices_arg = sys.argv[3].strip()
num_shards = int(num_shards_arg) if num_shards_arg else replica_count
if num_shards <= 0:
    raise SystemExit(
        f"Invalid VERL_SIDECAR_GLOBAL_NUM_SHARDS={num_shards}")
if indices_arg:
    indices = [int(item.strip()) for item in indices_arg.split(",")
               if item.strip()]
else:
    indices = list(range(replica_count))
if len(indices) != replica_count:
    raise SystemExit(
        "VERL_SIDECAR_GLOBAL_SHARD_INDICES must contain one index per "
        f"replica: indices={indices} replicas={replica_count}")
if len(set(indices)) != len(indices):
    raise SystemExit(
        f"VERL_SIDECAR_GLOBAL_SHARD_INDICES contains duplicates: {indices}")
if any(index < 0 or index >= num_shards for index in indices):
    raise SystemExit(
        "VERL_SIDECAR_GLOBAL_SHARD_INDICES must be in "
        f"[0, {num_shards}): {indices}")
print(f"VERL_SIDECAR_GLOBAL_NUM_SHARDS={num_shards}")
print("VERL_SIDECAR_GLOBAL_SHARD_INDICES=" +
      shlex.quote(",".join(map(str, indices))))
PY
)
eval "${SIDECAR_GLOBAL_SHARD_PLAN}"
SIDECAR_EP_PLAN=$(python3 - \
    "${VERL_SIDECAR_MODEL_PATH}" \
    "${VERL_SIDECAR_ENABLE_EXPERT_PARALLEL:-0}" \
    "${VERL_SIDECAR_MODEL_IS_MOE:-}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

model_path, ep_allowed_arg, override_arg = sys.argv[1:4]

def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}

def detect_moe(path: str) -> tuple[bool, str]:
    config_path = Path(path) / "config.json"
    data = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    haystack = " ".join([
        str(data.get("model_type", "")),
        " ".join(str(item) for item in data.get("architectures", []) or []),
        Path(path).name,
    ]).lower()
    moe_keys = {
        "num_experts",
        "num_local_experts",
        "moe_intermediate_size",
        "decoder_sparse_step",
        "moe_layer_freq",
    }
    is_moe = "moe" in haystack or any(key in data for key in moe_keys)
    reason = "override" if override_arg.strip() else (
        "config_or_name" if is_moe else "dense_or_unknown")
    return is_moe, reason

if override_arg.strip():
    model_is_moe = as_bool(override_arg)
    reason = "override"
else:
    model_is_moe, reason = detect_moe(model_path)

ep_allowed = as_bool(ep_allowed_arg)
effective = ep_allowed and model_is_moe
print(f"VERL_SIDECAR_ENABLE_EXPERT_PARALLEL={int(ep_allowed)}")
print(f"VERL_SIDECAR_MODEL_IS_MOE={int(model_is_moe)}")
print(f"VERL_SIDECAR_MODEL_IS_MOE_REASON={shlex.quote(reason)}")
print(f"VERL_SIDECAR_ENABLE_EXPERT_PARALLEL_EFFECTIVE={int(effective)}")
PY
)
eval "${SIDECAR_EP_PLAN}"
VERL_SIDECAR_NEEDS_HCCL=0
if [[ "${VERL_SIDECAR_TENSOR_PARALLEL_SIZE}" -gt 1 || "${VERL_SIDECAR_DATA_PARALLEL_SIZE}" -gt 1 || "${VERL_SIDECAR_ENABLE_EXPERT_PARALLEL_EFFECTIVE}" == "1" ]]; then
    VERL_SIDECAR_NEEDS_HCCL=1
fi

if [[ "${VERL_SIDECAR_NEEDS_HCCL}" == "1" ]]; then
    if [[ "${VERL_SIDECAR_HCCL_IF_BASE_PORT_FIXED,,}" == "1" \
        || "${VERL_SIDECAR_HCCL_IF_BASE_PORT_FIXED,,}" == "true" \
        || "${VERL_SIDECAR_HCCL_IF_BASE_PORT_FIXED,,}" == "yes" \
        || "${VERL_SIDECAR_HCCL_IF_BASE_PORT_FIXED,,}" == "on" ]]; then
        SIDECAR_PORT_PLAN=$(python3 - \
            "${VERL_SIDECAR_REPLICA_COUNT}" \
            "${VERL_SIDECAR_DATA_PARALLEL_SIZE}" \
            "${VERL_SIDECAR_MASTER_PORT}" \
            "${VERL_SIDECAR_MASTER_PORT_STRIDE}" \
            "${VERL_SIDECAR_HCCL_IF_BASE_PORT}" \
            "${VERL_SIDECAR_HCCL_PORT_STRIDE}" \
            "${VERL_SIDECAR_HCCL_PORT_WINDOW}" <<'PY'
import shlex
import sys

replica_count, data_parallel_size, master_base, master_stride, hccl_base, hccl_stride, window = map(int, sys.argv[1:8])
master_ports = [master_base + index * master_stride for index in range(replica_count)]
rpc_ports = [
    master_base + index * master_stride + 1
    for index in range(replica_count)
] if data_parallel_size > 1 else []
hccl_ports = [hccl_base + index * hccl_stride for index in range(replica_count)]
max_master = max(master_ports + rpc_ports) if (master_ports or rpc_ports) else master_base
max_hccl = max((base + 2 * window - 1 for base in hccl_ports), default=hccl_base)
if max_master > 65535 or max_hccl > 65535:
    raise SystemExit(
        "Sidecar fixed TP/EP port range exceeds 65535: "
        f"master max={max_master}, HCCL max={max_hccl}. "
        "Lower *_BASE_PORT or *_PORT_STRIDE.")
print("VERL_SIDECAR_MASTER_PORTS=" + shlex.quote(",".join(map(str, master_ports))))
print("VERL_SIDECAR_DATA_PARALLEL_RPC_PORTS=" + shlex.quote(",".join(map(str, rpc_ports))))
print("VERL_SIDECAR_HCCL_IF_BASE_PORTS=" + shlex.quote(",".join(map(str, hccl_ports))))
print(f"VERL_SIDECAR_MASTER_PORT={master_ports[0]}")
print(f"VERL_SIDECAR_HCCL_IF_BASE_PORT={hccl_ports[0]}")
PY
        )
    else
        SIDECAR_PORT_PLAN=$(python3 - \
            "${VERL_SIDECAR_REPLICA_COUNT}" \
            "${VERL_SIDECAR_DATA_PARALLEL_SIZE}" \
            "${VERL_SIDECAR_MASTER_PORT}" \
            "${VERL_SIDECAR_MASTER_PORT_STRIDE}" \
            "${VERL_SIDECAR_HCCL_IF_BASE_PORT}" \
            "${VERL_SIDECAR_HCCL_PORT_WINDOW}" <<'PY'
import shlex
import socket
import sys

replica_count = int(sys.argv[1])
data_parallel_size = int(sys.argv[2])
master_hint = int(sys.argv[3])
master_stride = int(sys.argv[4])
hccl_hint = int(sys.argv[5])
window = int(sys.argv[6])
if replica_count <= 0:
    raise SystemExit(f"Invalid sidecar replica count: {replica_count}")
if window <= 0:
    raise SystemExit(f"Invalid HCCL port window: {window}")

used: set[int] = set()

def port_free(port: int) -> bool:
    if port in used or port < 1 or port > 65535:
        return False
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()

def window_free(base: int, width: int) -> bool:
    if base < 1 or base + width - 1 > 65535:
        return False
    ports = range(base, base + width)
    if any(port in used for port in ports):
        return False
    sockets = []
    try:
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", port))
            sockets.append(sock)
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()

def reserve_port(start: int, end: int) -> int:
    candidates = []
    if start <= end:
        candidates.extend(range(start, end + 1))
    # Fallback range keeps the run alive if the preferred range is fragmented.
    candidates.extend(range(20000, 30000))
    for port in candidates:
        if port_free(port):
            used.add(port)
            return port
    raise SystemExit("No free master port found for sidecar shard")

def reserve_window(start: int, end: int, width: int) -> int:
    candidates = []
    if start <= end:
        candidates.extend(range(start, end + 1))
    # HCCL ports are best kept away from common Ray/vLLM control ports.
    candidates.extend(range(30000, 62000))
    for base in candidates:
        if window_free(base, width):
            used.update(range(base, base + width))
            return base
    raise SystemExit("No free per-shard HCCL port windows found for sidecar TP/EP")

master_ports = []
rpc_ports = []
hccl_ports = []
hccl_width = 2 * window
for index in range(replica_count):
    master_pref = master_hint + index * max(1, master_stride)
    master_ports.append(reserve_port(master_pref, 29999))
    if data_parallel_size > 1:
        rpc_ports.append(reserve_port(master_ports[-1] + 1, 29999))
    # Allocate each replica independently. This avoids the old all-or-nothing
    # equal-stride layout, which failed when only one shard window was occupied.
    hccl_pref = hccl_hint + index * hccl_width
    hccl_ports.append(reserve_window(hccl_pref, 62000 - hccl_width + 1, hccl_width))

print("VERL_SIDECAR_MASTER_PORTS=" + shlex.quote(",".join(map(str, master_ports))))
print("VERL_SIDECAR_DATA_PARALLEL_RPC_PORTS=" + shlex.quote(",".join(map(str, rpc_ports))))
print("VERL_SIDECAR_HCCL_IF_BASE_PORTS=" + shlex.quote(",".join(map(str, hccl_ports))))
print(f"VERL_SIDECAR_MASTER_PORT={master_ports[0]}")
print(f"VERL_SIDECAR_HCCL_IF_BASE_PORT={hccl_ports[0]}")
PY
        )
    fi
    eval "${SIDECAR_PORT_PLAN}"
else
    SIDECAR_PORT_PLAN=$(python3 - \
        "${VERL_SIDECAR_REPLICA_COUNT}" \
        "${VERL_SIDECAR_DATA_PARALLEL_SIZE}" \
        "${VERL_SIDECAR_MASTER_PORT}" \
        "${VERL_SIDECAR_MASTER_PORT_STRIDE}" \
        "${VERL_SIDECAR_HCCL_IF_BASE_PORT}" \
        "${VERL_SIDECAR_HCCL_PORT_STRIDE}" <<'PY'
import shlex
import sys

replica_count, data_parallel_size, master_base, master_stride, hccl_base, hccl_stride = map(int, sys.argv[1:7])
master_ports = [master_base + index * max(1, master_stride) for index in range(replica_count)]
rpc_ports = [
    master_base + index * max(1, master_stride) + 1
    for index in range(replica_count)
] if data_parallel_size > 1 else []
# TP1/DP1 dense replicas do not initialize HCCL, but the launcher still indexes
# this array per shard when exporting common environment variables.
hccl_ports = [hccl_base for _ in range(replica_count)]
print("VERL_SIDECAR_MASTER_PORTS=" + shlex.quote(",".join(map(str, master_ports))))
print("VERL_SIDECAR_DATA_PARALLEL_RPC_PORTS=" + shlex.quote(",".join(map(str, rpc_ports))))
print("VERL_SIDECAR_HCCL_IF_BASE_PORTS=" + shlex.quote(",".join(map(str, hccl_ports))))
print(f"VERL_SIDECAR_MASTER_PORT={master_ports[0]}")
print(f"VERL_SIDECAR_HCCL_IF_BASE_PORT={hccl_ports[0]}")
PY
    )
    eval "${SIDECAR_PORT_PLAN}"
fi
VERL_SIDECAR_GPU_MEMORY_UTILIZATION=${VERL_SIDECAR_GPU_MEMORY_UTILIZATION:-0.90}
VERL_SIDECAR_MAX_MODEL_LEN=${VERL_SIDECAR_MAX_MODEL_LEN:-2048}
VERL_SIDECAR_MAX_NUM_SEQS=${VERL_SIDECAR_MAX_NUM_SEQS:-128}
VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=${VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS:-65536}
VERL_SIDECAR_MAX_TOKENS=${VERL_SIDECAR_MAX_TOKENS:-1024}
VERL_SIDECAR_TEMPERATURE=${VERL_SIDECAR_TEMPERATURE:-0.0}
VERL_SIDECAR_TOP_P=${VERL_SIDECAR_TOP_P:-1.0}
VERL_SIDECAR_N=${VERL_SIDECAR_N:-1}
VERL_SIDECAR_TRUST_REMOTE_CODE=${VERL_SIDECAR_TRUST_REMOTE_CODE:-1}
VERL_SIDECAR_PROMPT=${VERL_SIDECAR_PROMPT:-"Explain elastic resource sharing in one sentence."}
VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA=${VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA:-"${VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE:-${VERL_SIDECAR_MAX_NUM_SEQS}}"}
# Backward-compatible alias: in DP mode a replica is one device, but with TP>1
# scheduling capacity is controlled per replica rather than per physical card.
VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE=${VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE:-"${VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA}"}
VERL_SIDECAR_MAX_PROMPTS=${VERL_SIDECAR_MAX_PROMPTS:-$((VERL_SIDECAR_REPLICA_COUNT * VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA))}
VERL_SIDECAR_REPEAT_UNTIL_KILLED=${VERL_SIDECAR_REPEAT_UNTIL_KILLED:-1}
VERL_SIDECAR_MAX_ITERATIONS=${VERL_SIDECAR_MAX_ITERATIONS:-0}
VERL_SIDECAR_ITERATION_SLEEP_SECONDS=${VERL_SIDECAR_ITERATION_SLEEP_SECONDS:-0}
VERL_SIDECAR_GENERATE_CHUNK_SIZE=${VERL_SIDECAR_GENERATE_CHUNK_SIZE:-32}
VERL_SIDECAR_STREAM_CHECKPOINT=${VERL_SIDECAR_STREAM_CHECKPOINT:-1}
VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS=${VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS:-0}
VERL_SIDECAR_STOP_FILE=${VERL_SIDECAR_STOP_FILE:-"${VERL_SIDECAR_OUTPUT_FILE}.stop_requested"}
VERL_SIDECAR_PRIMARY_DEVICE_GROUP=${VERL_SIDECAR_DEVICE_GROUPS%%;*}
if [[ -z "${VERL_SIDECAR_STATE_DIR:-}" ]]; then
    VERL_SIDECAR_STATE_DIR=$(python3 - "${ROOT_DIR}" "${VERL_SIDECAR_MODEL_PATH}" "${VERL_SIDECAR_PROMPTS_FILE:-default}" "${VERL_SIDECAR_DATA_SPLIT:-train}" <<'PY'
import os
import re
import sys
from pathlib import Path

root, model, prompts, split = sys.argv[1:5]

def safe(value: str) -> str:
    name = Path(value).name or value
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return name or "default"

print(os.path.join(root, "sidecar_runs", "state",
                   f"{safe(model)}_{safe(prompts)}_{safe(split)}"))
PY
)
fi

mkdir -p "$(dirname "${VERL_SIDECAR_LOG_FILE}")" \
    "$(dirname "${VERL_SIDECAR_OUTPUT_FILE}")" \
    "${VERL_SIDECAR_STATE_DIR}"

export ASCEND_RT_VISIBLE_DEVICES="${VERL_SIDECAR_NPU_DEVICES}"
export MASTER_ADDR="${VERL_SIDECAR_MASTER_ADDR}"
export MASTER_PORT="${VERL_SIDECAR_MASTER_PORT}"
export HCCL_IF_BASE_PORT="${VERL_SIDECAR_HCCL_IF_BASE_PORT}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_IF_BASE_PORT}-$((HCCL_IF_BASE_PORT + VERL_SIDECAR_HCCL_PORT_WINDOW - 1))"
export HCCL_NPU_SOCKET_PORT_RANGE="$((HCCL_IF_BASE_PORT + VERL_SIDECAR_HCCL_PORT_WINDOW))-$((HCCL_IF_BASE_PORT + 2 * VERL_SIDECAR_HCCL_PORT_WINDOW - 1))"
export VLLM_DP_MASTER_PORT="${VERL_SIDECAR_MASTER_PORT}"
export VERL_SIDECAR_LOG_FILE
export VERL_SIDECAR_OUTPUT_FILE
export VERL_SIDECAR_MODEL_PATH
export VERL_SIDECAR_PROMPTS_FILE
export VERL_SIDECAR_MAX_PROMPTS
export VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE
export VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA
export VERL_SIDECAR_DEVICE_COUNT
export VERL_SIDECAR_REPLICA_COUNT
export VERL_SIDECAR_DEVICE_GROUPS
export VERL_SIDECAR_UNUSED_DEVICES
export VERL_SIDECAR_PRIMARY_DEVICE_GROUP
export VERL_SIDECAR_GLOBAL_NUM_SHARDS
export VERL_SIDECAR_GLOBAL_SHARD_INDICES
export VERL_SIDECAR_MASTER_ADDR
export VERL_SIDECAR_MASTER_PORT
export VERL_SIDECAR_MASTER_PORTS
export VERL_SIDECAR_MASTER_PORT_STRIDE
export VERL_SIDECAR_DATA_PARALLEL_RPC_PORTS
export VERL_SIDECAR_HCCL_IF_BASE_PORT
export VERL_SIDECAR_HCCL_IF_BASE_PORTS
export VERL_SIDECAR_HCCL_PORT_STRIDE
export VERL_SIDECAR_HCCL_PORT_WINDOW
export VERL_SIDECAR_HCCL_IF_BASE_PORT_FIXED
export VERL_SIDECAR_NEEDS_HCCL
export VERL_SIDECAR_FORCE_HOST_IP_FOR_TP
export VERL_SIDECAR_PARALLEL_MODE
export VERL_SIDECAR_TENSOR_PARALLEL_SIZE
export VERL_SIDECAR_DATA_PARALLEL_SIZE
export VERL_SIDECAR_ENABLE_EXPERT_PARALLEL
export VERL_SIDECAR_MODEL_IS_MOE
export VERL_SIDECAR_MODEL_IS_MOE_REASON
export VERL_SIDECAR_ENABLE_EXPERT_PARALLEL_EFFECTIVE
export VERL_SIDECAR_GPU_MEMORY_UTILIZATION
export VERL_SIDECAR_MAX_MODEL_LEN
export VERL_SIDECAR_MAX_NUM_SEQS
export VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS
export VERL_SIDECAR_MAX_TOKENS
export VERL_SIDECAR_TEMPERATURE
export VERL_SIDECAR_TOP_P
export VERL_SIDECAR_N
export VERL_SIDECAR_TRUST_REMOTE_CODE
export VERL_SIDECAR_PROMPT
export VERL_SIDECAR_REPEAT_UNTIL_KILLED
export VERL_SIDECAR_MAX_ITERATIONS
export VERL_SIDECAR_ITERATION_SLEEP_SECONDS
export VERL_SIDECAR_GENERATE_CHUNK_SIZE
export VERL_SIDECAR_STREAM_CHECKPOINT
export VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS
export VERL_SIDECAR_STOP_FILE
export VERL_SIDECAR_STATE_DIR
export VERL_SIDECAR_DATA_SPLIT="${VERL_SIDECAR_DATA_SPLIT:-train}"
export VERL_SIDECAR_USE_SHORT_DATA="${VERL_SIDECAR_USE_SHORT_DATA:-0}"
export VERL_SIDECAR_ENFORCE_EAGER="${VERL_SIDECAR_ENFORCE_EAGER:-1}"
# Do not inherit the training rollout DP=16 into the sidecar unless explicitly requested.
export VLLM_DP_SIZE="${VERL_SIDECAR_DATA_PARALLEL_SIZE}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_LOGGING_LEVEL="${VERL_SIDECAR_VLLM_LOGGING_LEVEL:-INFO}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export VLLM_ENABLE_EXPERT_PARALLEL="${VERL_SIDECAR_ENABLE_EXPERT_PARALLEL_EFFECTIVE}"

# The sidecar is an independent low-priority inference job. Training-side
# elastic reservations would reduce its KV budget without protecting rollout.
unset VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK
unset VLLM_ASCEND_ELASTIC_EXECUTION_MODE
unset VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE
unset VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE
unset VLLM_ASCEND_MODE5_RUNTIME_MIN_COMPUTE_GROUP_SIZE
unset VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES
unset VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES
unset VLLM_ASCEND_MODE4_MOE_DISPATCH_HEADROOM_BYTES
unset VLLM_ASCEND_MODE4_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES
unset VLLM_ASCEND_MODE5_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES
unset VLLM_ASCEND_MODE3_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES
unset VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS
if [[ "${VERL_SIDECAR_FORCE_HOST_IP_FOR_TP,,}" != "0" \
    && "${VERL_SIDECAR_FORCE_HOST_IP_FOR_TP,,}" != "false" \
    && "${VERL_SIDECAR_FORCE_HOST_IP_FOR_TP,,}" != "no" \
    && "${VERL_SIDECAR_FORCE_HOST_IP_FOR_TP,,}" != "off" \
    && ( "${VERL_SIDECAR_TENSOR_PARALLEL_SIZE}" -gt 1 || "${VERL_SIDECAR_DATA_PARALLEL_SIZE}" -gt 1 || "${VERL_SIDECAR_ENABLE_EXPERT_PARALLEL_EFFECTIVE}" == "1" ) ]]; then
    export VLLM_HOST_IP="${VERL_SIDECAR_MASTER_ADDR}"
    export VLLM_MULTIPROC_DISTRIBUTED_HOST_IP="${VERL_SIDECAR_MASTER_ADDR}"
    export VLLM_MULTIPROC_USE_HOST_IP=1
    export VLLM_MULTIPROC_SET_WORKER_RANK_ENVS=1
    export VLLM_REUSE_WORLD_GROUP_FOR_FULL_MODEL_PARALLEL=1
fi

# The sidecar is launched from a training process tree, so make sure vLLM TP
# does not inherit training/Ray distributed rank metadata.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE
unset ROLE_RANK ROLE_WORLD_SIZE NODE_RANK
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_RANK
unset PMI_RANK PMI_SIZE PMI_LOCAL_RANK

if [[ "${VERL_SIDECAR_CONFIG_ONLY:-0}" == "1" ]]; then
    [[ -f "${VERL_SIDECAR_MODEL_PATH}/config.json" ]] || {
        echo "sidecar config check failed: missing ${VERL_SIDECAR_MODEL_PATH}/config.json" >&2
        exit 2
    }
    [[ -e "${VERL_SIDECAR_PROMPTS_FILE}" ]] || {
        echo "sidecar config check failed: missing ${VERL_SIDECAR_PROMPTS_FILE}" >&2
        exit 2
    }
    echo "sidecar_config_check=ok"
    echo "sidecar_model=${VERL_SIDECAR_MODEL_PATH}"
    echo "sidecar_prompts=${VERL_SIDECAR_PROMPTS_FILE}"
    echo "sidecar_devices=${VERL_SIDECAR_NPU_DEVICES}"
    echo "sidecar_device_groups=${VERL_SIDECAR_DEVICE_GROUPS}"
    echo "sidecar_tp=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE}"
    echo "sidecar_replicas=${VERL_SIDECAR_REPLICA_COUNT}"
    echo "sidecar_master_ports=${VERL_SIDECAR_MASTER_PORTS}"
    echo "sidecar_hccl_ports=${VERL_SIDECAR_HCCL_IF_BASE_PORTS}"
    exit 0
fi

SIDECAR_SHARD_OUTPUTS=()
PY_SCRIPT=$(mktemp /tmp/elastic_sidecar_infer.XXXXXX.py)
cleanup_sidecar() {
    set +e
    if [[ "${#SIDECAR_SHARD_OUTPUTS[@]}" -gt 0 ]]; then
        : > "${VERL_SIDECAR_OUTPUT_FILE}"
        for shard_output in "${SIDECAR_SHARD_OUTPUTS[@]}"; do
            if [[ -f "${shard_output}" ]]; then
                cat "${shard_output}" >> "${VERL_SIDECAR_OUTPUT_FILE}"
            fi
        done
    fi
    rm -f "${PY_SCRIPT}"
}
trap cleanup_sidecar EXIT

cat > "${PY_SCRIPT}" <<'PY'
from copy import copy
import gc
import json
import os
import signal
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _prompt_to_text(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("prompt", "text", "content", "question"):
            if key in value:
                return _prompt_to_text(value[key])
        if "messages" in value:
            return _prompt_to_text(value["messages"])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, dict) and "content" in item:
                role = item.get("role")
                content = _prompt_to_text(item.get("content"))
                parts.append(f"{role}: {content}" if role else content)
            else:
                parts.append(_prompt_to_text(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def _resolve_prompt_path(path: Path) -> Path:
    if not path.is_dir():
        return path
    split = os.environ.get("VERL_SIDECAR_DATA_SPLIT", "train").strip() or "train"
    use_short = _as_bool(os.environ.get("VERL_SIDECAR_USE_SHORT_DATA", "0"))
    splits = _dedupe([split, "train", "test"])
    candidates = []
    for item in splits:
        if use_short:
            candidates.extend([f"{item}_short.parquet", f"{item}.parquet"])
        else:
            candidates.extend([f"{item}.parquet", f"{item}_short.parquet"])
    for name in candidates:
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No supported prompt parquet found under dataset dir: {path}; "
        f"checked={candidates}")


def _record(prompt_id: int, prompt: str, source: str) -> dict:
    return {"prompt_id": int(prompt_id), "prompt": prompt, "source": source}


def _load_parquet_records(path: Path) -> list[dict]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Reading parquet prompts requires pandas/pyarrow in the sidecar environment.") from exc
    df = pd.read_parquet(path)
    records = []
    for row_idx, row in df.iterrows():
        prompt = ""
        if "prompt" in row and row["prompt"] is not None:
            prompt = _prompt_to_text(row["prompt"])
        elif "text" in row and row["text"] is not None:
            prompt = _prompt_to_text(row["text"])
        elif "question" in row and row["question"] is not None:
            prompt = _prompt_to_text(row["question"])
        else:
            extra_info = row.get("extra_info") if "extra_info" in row else None
            if isinstance(extra_info, dict) and extra_info.get("question") is not None:
                prompt = _prompt_to_text(extra_info["question"])
        if prompt:
            records.append(_record(len(records), prompt, f"{path}:{row_idx}"))
    return records


def load_prompt_records() -> list[dict]:
    prompt_file = os.environ.get("VERL_SIDECAR_PROMPTS_FILE", "").strip()
    fallback = os.environ.get("VERL_SIDECAR_PROMPT", "Explain elastic resource sharing in one sentence.")
    if not prompt_file:
        return [_record(0, fallback, "fallback")]
    path = _resolve_prompt_path(Path(prompt_file))
    if not path.exists():
        raise FileNotFoundError(f"VERL_SIDECAR_PROMPTS_FILE does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return _load_parquet_records(path)
    text = path.read_text(encoding="utf-8")
    records = []
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    prompt = item
                elif isinstance(item, dict):
                    prompt = _prompt_to_text(item.get("prompt", item.get("text", item)))
                else:
                    prompt = str(item)
                if prompt:
                    records.append(_record(len(records), prompt, f"{path}:{len(records)}"))
            return records
        if isinstance(data, dict):
            value = data.get("prompts", data.get("prompt", data.get("text", fallback)))
            if isinstance(value, list):
                for item in value:
                    prompt = _prompt_to_text(item)
                    if prompt:
                        records.append(_record(len(records), prompt, f"{path}:{len(records)}"))
                return records
            return [_record(0, _prompt_to_text(value), str(path))]
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            obj = json.loads(line)
            prompt = _prompt_to_text(obj.get("prompt", obj.get("text", obj)))
        else:
            prompt = line
        if prompt:
            records.append(_record(len(records), prompt, f"{path}:{len(records)}"))
    return records


def _read_jsonl_ids(path: Path) -> set[int]:
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("prompt_id") is not None:
                ids.add(int(item["prompt_id"]))
    return ids


def _read_epoch(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    return int(data.get("sidecar_epoch", 0) or 0)


def _write_epoch(path: Path, sidecar_epoch: int) -> None:
    _atomic_write_json(path, {
        "time": time.time(),
        "sidecar_epoch": int(sidecar_epoch),
        "shard_index": shard_index,
        "num_shards": num_shards,
    })


def _read_inflight_ids(path: Path) -> list[int]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    ids = data.get("prompt_ids", [])
    return [int(item) for item in ids if item is not None]


def _read_resume_records(path: Path) -> dict[int, dict]:
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("prompt_id") is not None:
                records[int(item["prompt_id"])] = item
    return records


def _shard_prompt_ids(records: list[dict]) -> set[int]:
    return {
        int(item["prompt_id"])
        for item in records
        if int(item["prompt_id"]) % num_shards == shard_index
    }


def _is_shard_epoch_complete(records: list[dict], completed_ids: set[int],
                             inflight_ids: list[int],
                             resume_records: dict[int, dict]) -> bool:
    shard_ids = _shard_prompt_ids(records)
    if not shard_ids:
        return False
    if inflight_ids or resume_records:
        return False
    return shard_ids.issubset(completed_ids)


def _reset_shard_state_for_next_epoch(
        completed_file: Path, inflight_file: Path, resume_file: Path,
        partials_file: Path, epoch_file: Path, sidecar_epoch: int,
        records: list[dict], completed_ids: set[int]) -> int:
    next_epoch = sidecar_epoch + 1
    for path in (completed_file, inflight_file, resume_file, partials_file):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    _write_epoch(epoch_file, next_epoch)
    print(json.dumps({
        "event": "sidecar_epoch_rollover",
        "completed_sidecar_epoch": sidecar_epoch,
        "next_sidecar_epoch": next_epoch,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "total_prompts": len(records),
        "shard_prompts": len(_shard_prompt_ids(records)),
        "completed_prompts": len(completed_ids),
        "state_dir": str(epoch_file.parent),
    }, ensure_ascii=False), flush=True)
    return next_epoch


def _write_resume_records(path: Path, records: dict[int, dict]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        for prompt_id in sorted(records):
            f.write(json.dumps(records[prompt_id], ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _stable_records(records: list[dict]) -> list[dict]:
    offset = int(os.environ.get("VERL_SIDECAR_PROMPT_OFFSET", "0"))
    if records and offset > 0:
        offset = offset % len(records)
        records = records[offset:] + records[:offset]
    return records


def _select_pending_records(records: list[dict], completed_ids: set[int],
                            inflight_ids: list[int], resume_records: dict[int, dict],
                            max_records: int) -> list[dict]:
    by_id = {int(item["prompt_id"]): item for item in records}
    shard_records = [
        item for item in _stable_records(records)
        if int(item["prompt_id"]) % num_shards == shard_index
    ]
    selected = []
    seen = set()
    priority_ids = list(inflight_ids) + sorted(resume_records)
    for prompt_id in priority_ids:
        if prompt_id in completed_ids or prompt_id in seen:
            continue
        item = by_id.get(prompt_id)
        if item is not None and int(item["prompt_id"]) % num_shards == shard_index:
            resume = resume_records.get(prompt_id)
            item = dict(item)
            if resume:
                item["resume_prompt"] = resume.get("resume_prompt", item["prompt"])
                item["resume_prefix_text"] = resume.get("partial_text", "")
                item["resume_token_ids_len"] = int(resume.get("token_ids_len", 0) or 0)
            selected.append(item)
            seen.add(prompt_id)
    for item in shard_records:
        prompt_id = int(item["prompt_id"])
        if prompt_id in completed_ids or prompt_id in seen:
            continue
        selected.append(item)
        seen.add(prompt_id)
        if max_records > 0 and len(selected) >= max_records:
            break
    if max_records > 0:
        selected = selected[:max_records]
    return selected


def _chunks(items: list[dict], size: int):
    if size <= 0:
        size = len(items) or 1
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


_shutdown_requested = False
_stop_file_logged = False


def _stop_file_path() -> Path | None:
    value = os.environ.get("VERL_SIDECAR_STOP_FILE", "").strip()
    return Path(value) if value else None


def _stop_requested() -> bool:
    global _shutdown_requested, _stop_file_logged
    if _shutdown_requested:
        return True
    path = _stop_file_path()
    if path is not None and path.exists():
        _shutdown_requested = True
        if not _stop_file_logged:
            _stop_file_logged = True
            print(json.dumps({
                "event": "sidecar_soft_stop_observed",
                "time": time.time(),
                "stop_file": str(path),
                "shard_index": shard_index,
                "num_shards": num_shards,
            }, ensure_ascii=False), flush=True)
        return True
    return False


def _request_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _request_shutdown)
signal.signal(signal.SIGINT, _request_shutdown)


def _call_if_present(obj, label: str, method_names: tuple[str, ...]) -> dict:
    if obj is None:
        return {"label": label, "status": "missing"}
    for method_name in method_names:
        method = getattr(obj, method_name, None)
        if callable(method):
            start = time.perf_counter()
            try:
                method()
                return {
                    "label": label,
                    "method": method_name,
                    "status": "ok",
                    "time_s": time.perf_counter() - start,
                }
            except Exception as exc:
                return {
                    "label": label,
                    "method": method_name,
                    "status": "error",
                    "error": repr(exc),
                    "time_s": time.perf_counter() - start,
                }
    return {"label": label, "status": "no_method"}


def _shutdown_llm_engine(llm) -> None:
    start = time.perf_counter()
    print(json.dumps({
        "event": "sidecar_engine_shutdown_start",
        "time": time.time(),
        "shard_index": shard_index,
        "num_shards": num_shards,
    }, ensure_ascii=False), flush=True)

    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    model_executor = getattr(engine, "model_executor", None)
    results = []
    seen = set()
    for label, obj, methods in (
            ("llm", llm, ("shutdown", "close")),
            ("llm.llm_engine", engine, ("shutdown", "close")),
            ("llm.llm_engine.engine_core", engine_core, ("shutdown", "close")),
            ("llm.llm_engine.model_executor", model_executor, ("shutdown", "close")),
    ):
        if obj is not None and id(obj) in seen:
            continue
        if obj is not None:
            seen.add(id(obj))
        results.append(_call_if_present(obj, label, methods))

    cache_result = {"label": "torch.npu.empty_cache", "status": "skipped"}
    try:
        import torch
        npu = getattr(torch, "npu", None)
        if npu is not None:
            try:
                npu.empty_cache()
                cache_result = {"label": "torch.npu.empty_cache", "status": "ok"}
            except Exception as exc:
                cache_result = {
                    "label": "torch.npu.empty_cache",
                    "status": "error",
                    "error": repr(exc),
                }
    except Exception as exc:
        cache_result = {
            "label": "torch.npu.empty_cache",
            "status": "import_error",
            "error": repr(exc),
        }
    results.append(cache_result)
    gc.collect()

    print(json.dumps({
        "event": "sidecar_engine_shutdown_done",
        "time": time.time(),
        "shutdown_time_s": time.perf_counter() - start,
        "results": results,
        "shard_index": shard_index,
        "num_shards": num_shards,
    }, ensure_ascii=False), flush=True)


def _completion_payload(out, prefix_text: str = "") -> tuple[dict, int, str, int]:
    token_ids = getattr(out, "token_ids", None) or []
    text = prefix_text + out.text
    payload = {
        "text": text,
        "delta_text": out.text,
        "token_ids_len": len(token_ids),
        "resume_prefix_text_len": len(prefix_text),
        "finish_reason": getattr(out, "finish_reason", None),
    }
    return payload, len(token_ids), text, len(token_ids)


def _sampling_params_for_source(base_sampling_params, source: dict):
    params = copy(base_sampling_params)
    if not _as_bool(os.environ.get("VERL_SIDECAR_RESUME_REMAINING_TOKENS", "1")):
        return params

    max_response_tokens = int(os.environ.get("VERL_SIDECAR_MAX_TOKENS", "1024"))
    already_generated = int(source.get("resume_token_ids_len", 0) or 0)
    if already_generated <= 0:
        return params

    # Resume prompts already include the generated prefix. Continue only for the
    # remaining response budget instead of granting another full max_tokens.
    params.max_tokens = max(1, max_response_tokens - already_generated)
    return params


def _write_final_output(output_file: Path, completed_file: Path, resume_file: Path,
                        resume_records: dict[int, dict], source: dict,
                        output, completions: list[dict], iteration: int,
                        chunk_start: int) -> None:
    prompt_id = int(source["prompt_id"])
    with output_file.open("a", encoding="utf-8") as out_f:
        out_f.write(json.dumps({
            "sidecar_epoch": sidecar_epoch,
            "iteration": iteration,
            "chunk_start": chunk_start,
            "prompt_id": prompt_id,
            "prompt_source": source.get("source", ""),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "prompt": source["prompt"],
            "resume_prompt": output.prompt,
            "prompt_token_ids_len": len(getattr(output, "prompt_token_ids", None) or []),
            "outputs": completions,
        }, ensure_ascii=False) + "\n")
        out_f.flush()
        os.fsync(out_f.fileno())
    _append_jsonl(completed_file, [{
        "time": time.time(),
        "sidecar_epoch": sidecar_epoch,
        "iteration": iteration,
        "chunk_start": chunk_start,
        "prompt_id": prompt_id,
        "output_file": str(output_file),
    }])
    resume_records.pop(prompt_id, None)
    _write_resume_records(resume_file, resume_records)


def _write_skipped_output(output_file: Path, completed_file: Path,
                          resume_file: Path, resume_records: dict[int, dict],
                          source: dict, iteration: int, chunk_start: int,
                          reason: str, error: str) -> None:
    """Quarantine an un-runnable resume prompt so one bad item does not kill a shard."""
    prompt_id = int(source["prompt_id"])
    prefix_text = source.get("resume_prefix_text", "")
    token_ids_len = int(source.get("resume_token_ids_len", 0) or 0)
    prompt = source.get("resume_prompt", source["prompt"])
    with output_file.open("a", encoding="utf-8") as out_f:
        out_f.write(json.dumps({
            "sidecar_epoch": sidecar_epoch,
            "iteration": iteration,
            "chunk_start": chunk_start,
            "prompt_id": prompt_id,
            "prompt_source": source.get("source", ""),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "prompt": source["prompt"],
            "resume_prompt": prompt,
            "prompt_chars_len": len(prompt),
            "sidecar_status": reason,
            "sidecar_error": error,
            "outputs": [{
                "text": prefix_text,
                "delta_text": "",
                "token_ids_len": token_ids_len,
                "resume_prefix_text_len": len(prefix_text),
                "finish_reason": reason,
            }],
        }, ensure_ascii=False) + "\n")
        out_f.flush()
        os.fsync(out_f.fileno())
    _append_jsonl(completed_file, [{
        "time": time.time(),
        "sidecar_epoch": sidecar_epoch,
        "iteration": iteration,
        "chunk_start": chunk_start,
        "prompt_id": prompt_id,
        "status": reason,
        "output_file": str(output_file),
    }])
    resume_records.pop(prompt_id, None)
    _write_resume_records(resume_file, resume_records)
    print(json.dumps({
        "event": "sidecar_prompt_skipped",
        "reason": reason,
        "error": error,
        "prompt_id": prompt_id,
        "prompt_chars_len": len(prompt),
        "resume_token_ids_len": token_ids_len,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "output_file": str(output_file),
    }, ensure_ascii=False), flush=True)


def _generate_chunk_blocking(llm, chunk_records: list[dict], sampling_params,
                             output_file: Path, completed_file: Path,
                             resume_file: Path, resume_records: dict[int, dict],
                             iteration: int, chunk_start: int) -> tuple[float, int, int]:
    infer_start = time.perf_counter()
    request_sampling_params = [
        _sampling_params_for_source(sampling_params, item)
        for item in chunk_records
    ]
    outputs = llm.generate([item.get("resume_prompt", item["prompt"]) for item in chunk_records],
                           request_sampling_params, use_tqdm=False)
    infer_s = time.perf_counter() - infer_start
    chunk_output_tokens = 0
    for output_idx, output in enumerate(outputs):
        source = chunk_records[output_idx] if output_idx < len(chunk_records) else None
        if source is None:
            continue
        prefix_text = source.get("resume_prefix_text", "")
        completions = []
        for out in output.outputs:
            payload, token_count, _, _ = _completion_payload(out, prefix_text)
            chunk_output_tokens += token_count
            completions.append(payload)
        _write_final_output(output_file, completed_file, resume_file, resume_records,
                            source, output, completions, iteration, chunk_start)
    return infer_s, chunk_output_tokens, len(outputs)


def _generate_chunk_streaming(llm, chunk_records: list[dict], sampling_params,
                              output_file: Path, completed_file: Path,
                              resume_file: Path, partials_file: Path,
                              resume_records: dict[int, dict],
                              iteration: int, chunk_start: int) -> tuple[float, int, int]:
    request_to_record = {}
    partial_rows = []
    latest_by_request = {}
    skipped_requests = 0
    for item in chunk_records:
        params = _sampling_params_for_source(sampling_params, item)
        params.output_kind = RequestOutputKind.CUMULATIVE
        request_id = str(next(llm.request_counter))
        prompt = item.get("resume_prompt", item["prompt"])
        try:
            llm.llm_engine.add_request(request_id, prompt, params, tokenization_kwargs={})
        except ValueError as exc:
            error = str(exc)
            if "maximum model length" not in error and "longer than" not in error:
                raise
            _write_skipped_output(
                output_file, completed_file, resume_file, resume_records,
                item, iteration, chunk_start, "context_overflow_skipped", error)
            skipped_requests += 1
            continue
        request_to_record[request_id] = item

    infer_start = time.perf_counter()
    chunk_output_tokens = 0
    finished_requests = skipped_requests
    step_idx = 0
    partial_sync_every_steps = int(os.environ.get("VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS", "0"))

    while request_to_record:
        step_outputs = llm.llm_engine.step()
        step_idx += 1
        for output in step_outputs:
            request_id = str(output.request_id)
            source = request_to_record.get(request_id)
            if source is None:
                continue
            prompt_id = int(source["prompt_id"])
            prefix_text = source.get("resume_prefix_text", "")
            completions = []
            latest_text = prefix_text
            latest_tokens = int(source.get("resume_token_ids_len", 0) or 0)
            step_tokens = 0
            for out in output.outputs:
                payload, token_count, full_text, latest_token_count = _completion_payload(out, prefix_text)
                completions.append(payload)
                step_tokens += token_count
                latest_text = full_text
                latest_tokens = latest_token_count
            partial_row = {
                "time": time.time(),
                "sidecar_epoch": sidecar_epoch,
                "iteration": iteration,
                "chunk_start": chunk_start,
                "request_id": request_id,
                "prompt_id": prompt_id,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "finished": bool(output.finished),
                "prompt": source["prompt"],
                "resume_prompt": output.prompt,
                "outputs": completions,
            }
            partial_rows.append(partial_row)
            latest_by_request[request_id] = partial_row

            if output.finished:
                _write_final_output(output_file, completed_file, resume_file,
                                    resume_records, source, output, completions,
                                    iteration, chunk_start)
                request_to_record.pop(request_id, None)
                latest_by_request.pop(request_id, None)
                finished_requests += 1
                chunk_output_tokens += step_tokens
            else:
                resume_records[prompt_id] = {
                    "time": time.time(),
                    "sidecar_epoch": sidecar_epoch,
                    "iteration": iteration,
                    "chunk_start": chunk_start,
                    "prompt_id": prompt_id,
                    "prompt": source["prompt"],
                    "resume_prompt": source["prompt"] + latest_text,
                    "partial_text": latest_text,
                    "token_ids_len": latest_tokens,
                    "request_id": request_id,
                }

        if partial_sync_every_steps > 0 and step_idx % partial_sync_every_steps == 0:
            _append_jsonl(partials_file, partial_rows)
            _write_resume_records(resume_file, resume_records)
            partial_rows.clear()

        if _stop_requested():
            if latest_by_request:
                _append_jsonl(partials_file, list(latest_by_request.values()))
            _write_resume_records(resume_file, resume_records)
            print(json.dumps({
                "event": "sidecar_soft_stop_checkpointed",
                "time": time.time(),
                "iteration": iteration,
                "chunk_start": chunk_start,
                "sidecar_epoch": sidecar_epoch,
                "active_requests": len(request_to_record),
                "checkpointed_requests": len(latest_by_request),
                "resume_prompts": len(resume_records),
                "shard_index": shard_index,
                "num_shards": num_shards,
                "resume_file": str(resume_file),
                "partials_file": str(partials_file),
            }, ensure_ascii=False), flush=True)
            if request_to_record:
                try:
                    llm.llm_engine.abort_request(list(request_to_record))
                except Exception:
                    pass
            break

    if partial_rows and (partial_sync_every_steps > 0 or _shutdown_requested):
        _append_jsonl(partials_file, partial_rows)
    if _shutdown_requested:
        _write_resume_records(resume_file, resume_records)

    infer_s = time.perf_counter() - infer_start
    return infer_s, chunk_output_tokens, finished_requests

model_path = os.environ["VERL_SIDECAR_MODEL_PATH"]
output_file = Path(os.environ["VERL_SIDECAR_OUTPUT_FILE"])
output_file.parent.mkdir(parents=True, exist_ok=True)

shard_index = int(os.environ.get("VERL_SIDECAR_SHARD_INDEX", "0"))
num_shards = int(os.environ.get("VERL_SIDECAR_NUM_SHARDS", "1"))
if shard_index < 0 or num_shards <= 0 or shard_index >= num_shards:
    raise ValueError(f"Invalid sidecar shard: index={shard_index}, num_shards={num_shards}")
repeat_until_killed = _as_bool(os.environ.get("VERL_SIDECAR_REPEAT_UNTIL_KILLED", "1"))
max_iterations = int(os.environ.get("VERL_SIDECAR_MAX_ITERATIONS", "0"))
iteration_sleep_s = float(os.environ.get("VERL_SIDECAR_ITERATION_SLEEP_SECONDS", "0"))
max_prompts = int(os.environ.get("VERL_SIDECAR_MAX_PROMPTS", "32"))
max_prompts_per_device = int(os.environ.get("VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE", "0"))
if num_shards > 1:
    max_records_per_iteration = max_prompts_per_device or ((max_prompts + num_shards - 1) // num_shards)
else:
    max_records_per_iteration = max_prompts
chunk_size = int(os.environ.get("VERL_SIDECAR_GENERATE_CHUNK_SIZE", "32"))
stream_checkpoint = _as_bool(os.environ.get("VERL_SIDECAR_STREAM_CHECKPOINT", "1"))

state_dir = Path(os.environ.get("VERL_SIDECAR_STATE_DIR", str(output_file.parent / "state")))
state_dir.mkdir(parents=True, exist_ok=True)
completed_file = state_dir / f"completed.shard{shard_index}.jsonl"
inflight_file = state_dir / f"inflight.shard{shard_index}.json"
resume_file = state_dir / f"resume.shard{shard_index}.jsonl"
partials_file = state_dir / f"partials.shard{shard_index}.jsonl"
epoch_file = state_dir / f"epoch.shard{shard_index}.json"
sidecar_epoch = _read_epoch(epoch_file)

records = load_prompt_records()
completed_ids = _read_jsonl_ids(completed_file)
inflight_ids = _read_inflight_ids(inflight_file)
resume_records = _read_resume_records(resume_file)
initial_records = _select_pending_records(records, completed_ids, inflight_ids,
                                          resume_records, max_records_per_iteration)
if (not initial_records and repeat_until_killed
        and _is_shard_epoch_complete(records, completed_ids, inflight_ids,
                                     resume_records)):
    sidecar_epoch = _reset_shard_state_for_next_epoch(
        completed_file, inflight_file, resume_file, partials_file, epoch_file,
        sidecar_epoch, records, completed_ids)
    completed_ids = _read_jsonl_ids(completed_file)
    inflight_ids = _read_inflight_ids(inflight_file)
    resume_records = _read_resume_records(resume_file)
    initial_records = _select_pending_records(records, completed_ids,
                                              inflight_ids, resume_records,
                                              max_records_per_iteration)
if not initial_records:
    output_file.touch()
    print(json.dumps({
        "event": "sidecar_no_work",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "total_prompts": len(records),
        "completed_prompts": len(completed_ids),
        "resume_prompts": len(resume_records),
        "sidecar_epoch": sidecar_epoch,
        "state_dir": str(state_dir),
        "output_file": str(output_file),
    }, ensure_ascii=False), flush=True)
    raise SystemExit(0)
if output_file.exists() and _as_bool(os.environ.get("VERL_SIDECAR_RESET_OUTPUT_ON_START", "0")):
    output_file.unlink()

start_total = time.perf_counter()
load_start = time.perf_counter()
engine_kwargs = {
    "model": model_path,
    "tensor_parallel_size": int(os.environ.get("VERL_SIDECAR_TENSOR_PARALLEL_SIZE", "1")),
    "enable_expert_parallel": _as_bool(os.environ.get("VERL_SIDECAR_ENABLE_EXPERT_PARALLEL_EFFECTIVE", "0")),
    "gpu_memory_utilization": float(os.environ.get("VERL_SIDECAR_GPU_MEMORY_UTILIZATION", "0.80")),
    "max_num_seqs": int(os.environ.get("VERL_SIDECAR_MAX_NUM_SEQS", "16")),
    "max_num_batched_tokens": int(os.environ.get("VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS", "1024")),
    "trust_remote_code": _as_bool(os.environ.get("VERL_SIDECAR_TRUST_REMOTE_CODE", "1")),
    "enforce_eager": _as_bool(os.environ.get("VERL_SIDECAR_ENFORCE_EAGER", "1")),
}
max_model_len = os.environ.get("VERL_SIDECAR_MAX_MODEL_LEN", "").strip()
if max_model_len:
    engine_kwargs["max_model_len"] = int(max_model_len)

data_parallel_size = int(os.environ.get("VERL_SIDECAR_DATA_PARALLEL_SIZE", "1"))
if data_parallel_size > 1:
    engine_kwargs["data_parallel_size"] = data_parallel_size
    engine_kwargs["data_parallel_backend"] = os.environ.get(
        "VERL_SIDECAR_DATA_PARALLEL_BACKEND", "mp")
    data_parallel_rpc_port = os.environ.get("VERL_SIDECAR_DATA_PARALLEL_RPC_PORT", "").strip()
    if data_parallel_rpc_port:
        engine_kwargs["data_parallel_rpc_port"] = int(data_parallel_rpc_port)

print(json.dumps({
    "event": "sidecar_load_start",
    "model_path": model_path,
    "devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
    "master_port": os.environ.get("MASTER_PORT", ""),
    "master_addr": os.environ.get("MASTER_ADDR", ""),
    "hccl_if_base_port": os.environ.get("HCCL_IF_BASE_PORT", ""),
    "hccl_host_socket_port_range": os.environ.get("HCCL_HOST_SOCKET_PORT_RANGE", ""),
    "hccl_npu_socket_port_range": os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE", ""),
    "sidecar_hccl_if_base_port_fixed": os.environ.get("VERL_SIDECAR_HCCL_IF_BASE_PORT_FIXED", ""),
    "sidecar_hccl_port_stride": os.environ.get("VERL_SIDECAR_HCCL_PORT_STRIDE", ""),
    "sidecar_hccl_port_window": os.environ.get("VERL_SIDECAR_HCCL_PORT_WINDOW", ""),
    "vllm_host_ip": os.environ.get("VLLM_HOST_IP", ""),
    "vllm_multiproc_distributed_host_ip": os.environ.get("VLLM_MULTIPROC_DISTRIBUTED_HOST_IP", ""),
    "vllm_multiproc_use_host_ip": os.environ.get("VLLM_MULTIPROC_USE_HOST_IP", ""),
    "vllm_multiproc_set_worker_rank_envs": os.environ.get("VLLM_MULTIPROC_SET_WORKER_RANK_ENVS", ""),
    "vllm_reuse_world_group_for_full_model_parallel": os.environ.get("VLLM_REUSE_WORLD_GROUP_FOR_FULL_MODEL_PARALLEL", ""),
    "num_prompts": len(initial_records),
    "total_prompts": len(records),
    "completed_prompts": len(completed_ids),
    "inflight_prompts": len(inflight_ids),
    "resume_prompts": len(resume_records),
    "sidecar_epoch": sidecar_epoch,
    "stream_checkpoint": stream_checkpoint,
    "shard_index": shard_index,
    "num_shards": num_shards,
    "repeat_until_killed": repeat_until_killed,
    "max_iterations": max_iterations,
    "prompts_source": os.environ.get("VERL_SIDECAR_PROMPTS_FILE", ""),
    "data_split": os.environ.get("VERL_SIDECAR_DATA_SPLIT", ""),
    "use_short_data": os.environ.get("VERL_SIDECAR_USE_SHORT_DATA", ""),
    "state_dir": str(state_dir),
    "resume_file": str(resume_file),
    "partials_file": str(partials_file),
    "stop_file": os.environ.get("VERL_SIDECAR_STOP_FILE", ""),
    "expert_parallel_allowed": os.environ.get("VERL_SIDECAR_ENABLE_EXPERT_PARALLEL", ""),
    "expert_parallel_effective": os.environ.get("VERL_SIDECAR_ENABLE_EXPERT_PARALLEL_EFFECTIVE", ""),
    "model_is_moe": os.environ.get("VERL_SIDECAR_MODEL_IS_MOE", ""),
    "model_is_moe_reason": os.environ.get("VERL_SIDECAR_MODEL_IS_MOE_REASON", ""),
    "vllm_enable_expert_parallel": os.environ.get("VLLM_ENABLE_EXPERT_PARALLEL", ""),
    "elastic_mode_env": os.environ.get("VLLM_ASCEND_ELASTIC_EXECUTION_MODE", ""),
    "elastic_shrink_env": os.environ.get("VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK", ""),
    "kv_cache_init_headroom_env": os.environ.get("VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES", ""),
    "custom_mode1_kv_headroom_env": os.environ.get("VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES", ""),
    "max_prompts": max_prompts,
    "max_prompts_per_device": max_prompts_per_device,
    "max_records_per_iteration": max_records_per_iteration,
    "generate_chunk_size": chunk_size,
    "engine_kwargs": {k: v for k, v in engine_kwargs.items() if k != "model"},
}, ensure_ascii=False), flush=True)

llm = LLM(**engine_kwargs)
load_s = time.perf_counter() - load_start

sampling_params = SamplingParams(
    n=int(os.environ.get("VERL_SIDECAR_N", "1")),
    temperature=float(os.environ.get("VERL_SIDECAR_TEMPERATURE", "0.0")),
    top_p=float(os.environ.get("VERL_SIDECAR_TOP_P", "1.0")),
    max_tokens=int(os.environ.get("VERL_SIDECAR_MAX_TOKENS", "1024")),
)
print(json.dumps({
    "event": "sidecar_sampling_params",
    "n": int(os.environ.get("VERL_SIDECAR_N", "1")),
    "temperature": float(os.environ.get("VERL_SIDECAR_TEMPERATURE", "0.0")),
    "top_p": float(os.environ.get("VERL_SIDECAR_TOP_P", "1.0")),
    "max_tokens": int(os.environ.get("VERL_SIDECAR_MAX_TOKENS", "1024")),
    "max_model_len": engine_kwargs.get("max_model_len"),
    "max_num_seqs": engine_kwargs.get("max_num_seqs"),
    "max_num_batched_tokens": engine_kwargs.get("max_num_batched_tokens"),
    "shard_index": shard_index,
    "num_shards": num_shards,
}, ensure_ascii=False), flush=True)

iteration = 0
total_requests = 0
total_output_tokens = 0
total_infer_s = 0.0
last_total_prompts = len(records)
while True:
    if _stop_requested():
        print(json.dumps({
            "event": "sidecar_stop_before_iteration",
            "iteration": iteration,
            "sidecar_epoch": sidecar_epoch,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "state_dir": str(state_dir),
        }, ensure_ascii=False), flush=True)
        break

    completed_ids = _read_jsonl_ids(completed_file)
    inflight_ids = _read_inflight_ids(inflight_file)
    resume_records = _read_resume_records(resume_file)
    records = load_prompt_records()
    last_total_prompts = len(records)
    selected_records = _select_pending_records(records, completed_ids, inflight_ids,
                                               resume_records, max_records_per_iteration)
    if (not selected_records and repeat_until_killed
            and not _shutdown_requested
            and _is_shard_epoch_complete(records, completed_ids, inflight_ids,
                                         resume_records)):
        sidecar_epoch = _reset_shard_state_for_next_epoch(
            completed_file, inflight_file, resume_file, partials_file,
            epoch_file, sidecar_epoch, records, completed_ids)
        completed_ids = _read_jsonl_ids(completed_file)
        inflight_ids = _read_inflight_ids(inflight_file)
        resume_records = _read_resume_records(resume_file)
        selected_records = _select_pending_records(records, completed_ids,
                                                   inflight_ids, resume_records,
                                                   max_records_per_iteration)
    prompt_offset = int(os.environ.get("VERL_SIDECAR_PROMPT_OFFSET", "0"))
    if not selected_records:
        print(json.dumps({
            "event": "sidecar_iteration_no_work",
            "iteration": iteration,
            "prompt_offset": prompt_offset,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "total_prompts": len(records),
            "completed_prompts": len(completed_ids),
            "resume_prompts": len(resume_records),
            "sidecar_epoch": sidecar_epoch,
            "state_dir": str(state_dir),
            "output_file": str(output_file),
        }, ensure_ascii=False), flush=True)
        break

    iteration_requests = 0
    iteration_output_tokens = 0
    iteration_infer_s = 0.0
    for chunk_start, chunk_records in _chunks(selected_records, chunk_size):
        if _stop_requested():
            print(json.dumps({
                "event": "sidecar_stop_before_chunk",
                "iteration": iteration,
                "chunk_start": chunk_start,
                "sidecar_epoch": sidecar_epoch,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "state_dir": str(state_dir),
            }, ensure_ascii=False), flush=True)
            break

        chunk_ids = [int(item["prompt_id"]) for item in chunk_records]
        _atomic_write_json(inflight_file, {
            "time": time.time(),
            "iteration": iteration,
            "chunk_start": chunk_start,
            "sidecar_epoch": sidecar_epoch,
            "prompt_ids": chunk_ids,
            "shard_index": shard_index,
            "num_shards": num_shards,
        })
        print(json.dumps({
            "event": "sidecar_chunk_start",
            "iteration": iteration,
            "chunk_start": chunk_start,
            "sidecar_epoch": sidecar_epoch,
            "chunk_size": len(chunk_records),
            "prompt_ids_first": chunk_ids[:8],
            "stream_checkpoint": stream_checkpoint,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "state_dir": str(state_dir),
        }, ensure_ascii=False), flush=True)

        resume_records = _read_resume_records(resume_file)
        if stream_checkpoint:
            infer_s, chunk_output_tokens, finished_requests = _generate_chunk_streaming(
                llm, chunk_records, sampling_params, output_file, completed_file,
                resume_file, partials_file, resume_records, iteration, chunk_start)
        else:
            infer_s, chunk_output_tokens, finished_requests = _generate_chunk_blocking(
                llm, chunk_records, sampling_params, output_file, completed_file,
                resume_file, resume_records, iteration, chunk_start)
        try:
            inflight_file.unlink()
        except FileNotFoundError:
            pass

        iteration_requests += finished_requests
        iteration_output_tokens += chunk_output_tokens
        iteration_infer_s += infer_s
        print(json.dumps({
            "event": "sidecar_chunk_done",
            "iteration": iteration,
            "chunk_start": chunk_start,
            "sidecar_epoch": sidecar_epoch,
            "inference_time_s": infer_s,
            "num_requests": finished_requests,
            "submitted_requests": len(chunk_records),
            "prompt_ids_first": chunk_ids[:8],
            "shutdown_requested": _shutdown_requested,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "num_output_tokens": chunk_output_tokens,
            "tokens_per_s": (chunk_output_tokens / infer_s) if infer_s > 0 else 0.0,
            "completed_file": str(completed_file),
            "resume_file": str(resume_file),
            "output_file": str(output_file),
        }, ensure_ascii=False), flush=True)
        if _shutdown_requested:
            break

    total_requests += iteration_requests
    total_output_tokens += iteration_output_tokens
    total_infer_s += iteration_infer_s
    print(json.dumps({
        "event": "sidecar_iteration_done",
        "iteration": iteration,
        "prompt_offset": prompt_offset,
        "sidecar_epoch": sidecar_epoch,
        "inference_time_s": iteration_infer_s,
        "num_requests": iteration_requests,
        "total_prompts": len(records),
        "completed_prompts": len(_read_jsonl_ids(completed_file)),
        "resume_prompts": len(_read_resume_records(resume_file)),
        "shutdown_requested": _shutdown_requested,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "num_output_tokens": iteration_output_tokens,
        "tokens_per_s": (iteration_output_tokens / iteration_infer_s) if iteration_infer_s > 0 else 0.0,
        "state_dir": str(state_dir),
        "output_file": str(output_file),
    }, ensure_ascii=False), flush=True)

    iteration += 1
    if _shutdown_requested:
        break
    if not repeat_until_killed:
        break
    if max_iterations > 0 and iteration >= max_iterations:
        break
    if iteration_sleep_s > 0:
        time.sleep(iteration_sleep_s)

total_s = time.perf_counter() - start_total
print(json.dumps({
    "event": "sidecar_done",
    "model_load_time_s": load_s,
    "inference_time_s": total_infer_s,
    "total_time_s": total_s,
    "num_requests": total_requests,
    "iterations": iteration,
    "total_prompts_last_iteration": last_total_prompts,
    "completed_prompts": len(_read_jsonl_ids(completed_file)),
    "resume_prompts": len(_read_resume_records(resume_file)),
    "sidecar_epoch": sidecar_epoch,
    "shutdown_requested": _shutdown_requested,
    "shard_index": shard_index,
    "num_shards": num_shards,
    "num_output_tokens": total_output_tokens,
    "tokens_per_s": (total_output_tokens / total_infer_s) if total_infer_s > 0 else 0.0,
    "state_dir": str(state_dir),
    "output_file": str(output_file),
}, ensure_ascii=False), flush=True)

try:
    _shutdown_llm_engine(llm)
finally:
    llm = None
    gc.collect()

PY

{
    echo "sidecar_start_time=$(date +%s.%N)"
    echo "sidecar_devices=${VERL_SIDECAR_NPU_DEVICES}"
    echo "sidecar_device_count=${VERL_SIDECAR_DEVICE_COUNT}"
    echo "sidecar_replica_count=${VERL_SIDECAR_REPLICA_COUNT}"
    echo "sidecar_global_num_shards=${VERL_SIDECAR_GLOBAL_NUM_SHARDS}"
    echo "sidecar_global_shard_indices=${VERL_SIDECAR_GLOBAL_SHARD_INDICES}"
    echo "sidecar_parallel_mode=${VERL_SIDECAR_PARALLEL_MODE}"
    echo "sidecar_tensor_parallel_size=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE}"
    echo "sidecar_data_parallel_size=${VERL_SIDECAR_DATA_PARALLEL_SIZE}"
    echo "sidecar_device_groups=${VERL_SIDECAR_DEVICE_GROUPS}"
    echo "sidecar_unused_devices=${VERL_SIDECAR_UNUSED_DEVICES}"
    echo "sidecar_master_addr=${VERL_SIDECAR_MASTER_ADDR}"
    echo "sidecar_force_host_ip_for_tp=${VERL_SIDECAR_FORCE_HOST_IP_FOR_TP}"
    echo "sidecar_vllm_host_ip=${VLLM_HOST_IP:-}"
    echo "sidecar_vllm_multiproc_host_ip=${VLLM_MULTIPROC_DISTRIBUTED_HOST_IP:-}"
    echo "sidecar_vllm_reuse_world_group=${VLLM_REUSE_WORLD_GROUP_FOR_FULL_MODEL_PARALLEL:-}"
    echo "sidecar_master_port_base=${VERL_SIDECAR_MASTER_PORT}"
    echo "sidecar_master_ports=${VERL_SIDECAR_MASTER_PORTS}"
    echo "sidecar_master_port_stride=${VERL_SIDECAR_MASTER_PORT_STRIDE}"
    echo "sidecar_data_parallel_rpc_ports=${VERL_SIDECAR_DATA_PARALLEL_RPC_PORTS}"
    echo "sidecar_hccl_if_base_port=${VERL_SIDECAR_HCCL_IF_BASE_PORT}"
    echo "sidecar_hccl_if_base_ports=${VERL_SIDECAR_HCCL_IF_BASE_PORTS}"
    echo "sidecar_hccl_host_socket_port_range=${HCCL_HOST_SOCKET_PORT_RANGE}"
    echo "sidecar_hccl_npu_socket_port_range=${HCCL_NPU_SOCKET_PORT_RANGE}"
    echo "sidecar_hccl_if_base_port_fixed=${VERL_SIDECAR_HCCL_IF_BASE_PORT_FIXED}"
    echo "sidecar_hccl_port_stride=${VERL_SIDECAR_HCCL_PORT_STRIDE}"
    echo "sidecar_hccl_port_window=${VERL_SIDECAR_HCCL_PORT_WINDOW}"
    echo "sidecar_vllm_set_worker_rank_envs=${VLLM_MULTIPROC_SET_WORKER_RANK_ENVS:-}"
    echo "sidecar_ep_allowed=${VERL_SIDECAR_ENABLE_EXPERT_PARALLEL}"
    echo "sidecar_model_is_moe=${VERL_SIDECAR_MODEL_IS_MOE}"
    echo "sidecar_model_is_moe_reason=${VERL_SIDECAR_MODEL_IS_MOE_REASON}"
    echo "sidecar_ep_effective=${VERL_SIDECAR_ENABLE_EXPERT_PARALLEL_EFFECTIVE}"
    echo "sidecar_model=${VERL_SIDECAR_MODEL_PATH}"
    echo "sidecar_output=${VERL_SIDECAR_OUTPUT_FILE}"
    echo "sidecar_state_dir=${VERL_SIDECAR_STATE_DIR}"
    echo "sidecar_generate_chunk_size=${VERL_SIDECAR_GENERATE_CHUNK_SIZE}"
    echo "sidecar_stream_checkpoint=${VERL_SIDECAR_STREAM_CHECKPOINT}"
    echo "sidecar_partial_sync_every_steps=${VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS}"
    echo "sidecar_stop_file=${VERL_SIDECAR_STOP_FILE}"
    set +e
    IFS=',' read -r -a sidecar_master_ports <<< "${VERL_SIDECAR_MASTER_PORTS}"
    IFS=',' read -r -a sidecar_dp_rpc_ports <<< "${VERL_SIDECAR_DATA_PARALLEL_RPC_PORTS}"
    IFS=',' read -r -a sidecar_hccl_base_ports <<< "${VERL_SIDECAR_HCCL_IF_BASE_PORTS}"
    IFS=',' read -r -a sidecar_global_shard_indices <<< "${VERL_SIDECAR_GLOBAL_SHARD_INDICES}"
    for ((port_index=${#sidecar_master_ports[@]}; port_index<VERL_SIDECAR_REPLICA_COUNT; port_index++)); do
        sidecar_master_ports+=("$((VERL_SIDECAR_MASTER_PORT + port_index * VERL_SIDECAR_MASTER_PORT_STRIDE))")
    done
    for ((port_index=${#sidecar_hccl_base_ports[@]}; port_index<VERL_SIDECAR_REPLICA_COUNT; port_index++)); do
        sidecar_hccl_base_ports+=("$((VERL_SIDECAR_HCCL_IF_BASE_PORT + port_index * VERL_SIDECAR_HCCL_PORT_STRIDE))")
    done
    if [[ "${VERL_SIDECAR_DATA_PARALLEL_SIZE}" -gt 1 ]]; then
        for ((port_index=${#sidecar_dp_rpc_ports[@]}; port_index<VERL_SIDECAR_REPLICA_COUNT; port_index++)); do
            sidecar_dp_rpc_ports+=("$((sidecar_master_ports[port_index] + 1))")
        done
    fi
    if [[ "${VERL_SIDECAR_REPLICA_COUNT}" -gt 1 ]]; then
        IFS=';' read -r -a sidecar_device_groups <<< "${VERL_SIDECAR_DEVICE_GROUPS}"
        sidecar_pids=()
        replica_index=0
        for raw_group in "${sidecar_device_groups[@]}"; do
            device_group=$(echo "${raw_group}" | xargs)
            [[ -n "${device_group}" ]] || continue
            global_shard_index=${sidecar_global_shard_indices[$replica_index]}
            shard_output="${VERL_SIDECAR_OUTPUT_FILE}.shard${global_shard_index}"
            SIDECAR_SHARD_OUTPUTS+=("${shard_output}")
            (
                export ASCEND_RT_VISIBLE_DEVICES="${device_group}"
                export MASTER_ADDR="${VERL_SIDECAR_MASTER_ADDR}"
                export VERL_SIDECAR_SHARD_INDEX="${global_shard_index}"
                export VERL_SIDECAR_NUM_SHARDS="${VERL_SIDECAR_GLOBAL_NUM_SHARDS}"
                export VERL_SIDECAR_OUTPUT_FILE="${shard_output}"
                export MASTER_PORT="${sidecar_master_ports[$replica_index]}"
                if [[ "${VERL_SIDECAR_DATA_PARALLEL_SIZE}" -gt 1 ]]; then
                    export VERL_SIDECAR_DATA_PARALLEL_RPC_PORT="${sidecar_dp_rpc_ports[$replica_index]}"
                else
                    unset VERL_SIDECAR_DATA_PARALLEL_RPC_PORT
                fi
                export HCCL_IF_BASE_PORT="${sidecar_hccl_base_ports[$replica_index]}"
                export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_IF_BASE_PORT}-$((HCCL_IF_BASE_PORT + VERL_SIDECAR_HCCL_PORT_WINDOW - 1))"
                export HCCL_NPU_SOCKET_PORT_RANGE="$((HCCL_IF_BASE_PORT + VERL_SIDECAR_HCCL_PORT_WINDOW))-$((HCCL_IF_BASE_PORT + 2 * VERL_SIDECAR_HCCL_PORT_WINDOW - 1))"
                export VLLM_DP_MASTER_PORT="${MASTER_PORT}"
                echo "sidecar_shard_start_time=$(date +%s.%N) replica=${replica_index} shard=${global_shard_index}/${VERL_SIDECAR_GLOBAL_NUM_SHARDS} device_group=${device_group} tp=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE} dp=${VERL_SIDECAR_DATA_PARALLEL_SIZE} master_port=${MASTER_PORT} data_parallel_rpc_port=${VERL_SIDECAR_DATA_PARALLEL_RPC_PORT:-} hccl_if_base_port=${HCCL_IF_BASE_PORT} hccl_host_socket_port_range=${HCCL_HOST_SOCKET_PORT_RANGE} hccl_npu_socket_port_range=${HCCL_NPU_SOCKET_PORT_RANGE} output=${shard_output}"
                if [[ "${VERL_SIDECAR_MAX_SECONDS}" != "0" ]]; then
                    timeout --kill-after=10s "${VERL_SIDECAR_MAX_SECONDS}s" python3 -u "${PY_SCRIPT}" 2>&1
                    shard_rc=$?
                else
                    python3 -u "${PY_SCRIPT}" 2>&1
                    shard_rc=$?
                fi
                echo "sidecar_shard_end_time=$(date +%s.%N) replica=${replica_index} shard=${global_shard_index}/${VERL_SIDECAR_GLOBAL_NUM_SHARDS} device_group=${device_group} exit_code=${shard_rc}"
                exit "${shard_rc}"
            ) &
            sidecar_pids+=("$!")
            replica_index=$((replica_index + 1))
        done
        rc=0
        for pid in "${sidecar_pids[@]}"; do
            wait "${pid}"
            shard_wait_rc=$?
            if [[ "${shard_wait_rc}" != "0" && "${rc}" == "0" ]]; then
                rc="${shard_wait_rc}"
            fi
        done
        : > "${VERL_SIDECAR_OUTPUT_FILE}"
        for shard_output in "${SIDECAR_SHARD_OUTPUTS[@]}"; do
            if [[ -f "${shard_output}" ]]; then
                cat "${shard_output}" >> "${VERL_SIDECAR_OUTPUT_FILE}"
            fi
        done
    elif [[ "${VERL_SIDECAR_MAX_SECONDS}" != "0" ]]; then
        export ASCEND_RT_VISIBLE_DEVICES="${VERL_SIDECAR_PRIMARY_DEVICE_GROUP}"
        export VERL_SIDECAR_SHARD_INDEX="${sidecar_global_shard_indices[0]}"
        export VERL_SIDECAR_NUM_SHARDS="${VERL_SIDECAR_GLOBAL_NUM_SHARDS}"
        if [[ "${VERL_SIDECAR_DATA_PARALLEL_SIZE}" -gt 1 && -n "${sidecar_dp_rpc_ports[0]:-}" ]]; then
            export VERL_SIDECAR_DATA_PARALLEL_RPC_PORT="${sidecar_dp_rpc_ports[0]}"
        fi
        echo "sidecar_shard_start_time=$(date +%s.%N) shard=${VERL_SIDECAR_SHARD_INDEX}/${VERL_SIDECAR_NUM_SHARDS} device_group=${VERL_SIDECAR_PRIMARY_DEVICE_GROUP} tp=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE} dp=${VERL_SIDECAR_DATA_PARALLEL_SIZE} master_port=${MASTER_PORT} data_parallel_rpc_port=${VERL_SIDECAR_DATA_PARALLEL_RPC_PORT:-} hccl_if_base_port=${HCCL_IF_BASE_PORT} hccl_host_socket_port_range=${HCCL_HOST_SOCKET_PORT_RANGE} hccl_npu_socket_port_range=${HCCL_NPU_SOCKET_PORT_RANGE} output=${VERL_SIDECAR_OUTPUT_FILE}"
        timeout --kill-after=10s "${VERL_SIDECAR_MAX_SECONDS}s" python3 -u "${PY_SCRIPT}" 2>&1
        rc=$?
    else
        export ASCEND_RT_VISIBLE_DEVICES="${VERL_SIDECAR_PRIMARY_DEVICE_GROUP}"
        export VERL_SIDECAR_SHARD_INDEX="${sidecar_global_shard_indices[0]}"
        export VERL_SIDECAR_NUM_SHARDS="${VERL_SIDECAR_GLOBAL_NUM_SHARDS}"
        if [[ "${VERL_SIDECAR_DATA_PARALLEL_SIZE}" -gt 1 && -n "${sidecar_dp_rpc_ports[0]:-}" ]]; then
            export VERL_SIDECAR_DATA_PARALLEL_RPC_PORT="${sidecar_dp_rpc_ports[0]}"
        fi
        echo "sidecar_shard_start_time=$(date +%s.%N) shard=${VERL_SIDECAR_SHARD_INDEX}/${VERL_SIDECAR_NUM_SHARDS} device_group=${VERL_SIDECAR_PRIMARY_DEVICE_GROUP} tp=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE} dp=${VERL_SIDECAR_DATA_PARALLEL_SIZE} master_port=${MASTER_PORT} data_parallel_rpc_port=${VERL_SIDECAR_DATA_PARALLEL_RPC_PORT:-} hccl_if_base_port=${HCCL_IF_BASE_PORT} hccl_host_socket_port_range=${HCCL_HOST_SOCKET_PORT_RANGE} hccl_npu_socket_port_range=${HCCL_NPU_SOCKET_PORT_RANGE} output=${VERL_SIDECAR_OUTPUT_FILE}"
        python3 -u "${PY_SCRIPT}" 2>&1
        rc=$?
    fi
    set -e
    echo "sidecar_end_time=$(date +%s.%N)"
    echo "sidecar_exit_code=${rc}"
    if [[ "${rc}" == "124" || "${rc}" == "137" ]]; then
        echo "sidecar_killed_by_deadline=1"
    else
        echo "sidecar_killed_by_deadline=0"
    fi
    exit "${rc}"
} | tee -a "${VERL_SIDECAR_LOG_FILE}"
