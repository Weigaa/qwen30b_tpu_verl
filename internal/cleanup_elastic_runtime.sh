#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  internal/cleanup_elastic_runtime.sh [--dry-run|--kill] [--no-ray] [--kill-port-owners]

Purpose:
  Clean stale local Ray / vLLM sidecar / HCCL runtime state before an elastic
  rollout experiment. The default mode is --dry-run.

Environment:
  VERL_CLEANUP_PORTS
    Comma-separated ports to inspect. Default:
    20000,23300,24300,52000,53024,64021

  VERL_CLEANUP_EXTRA_PATTERNS
    Extra process grep patterns separated by '|'.

Notes:
  --kill runs "ray stop --force" and kills matching stale sidecar/vLLM
  processes. It intentionally does not kill arbitrary port owners unless
  --kill-port-owners is also provided.
EOF
}

mode="dry-run"
stop_ray=1
kill_port_owners=0
while (($#)); do
    case "$1" in
        --dry-run)
            mode="dry-run"
            ;;
        --kill)
            mode="kill"
            ;;
        --no-ray)
            stop_ray=0
            ;;
        --kill-port-owners)
            kill_port_owners=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

ports_csv=${VERL_CLEANUP_PORTS:-"20000,23300,24300,52000,53024,64021"}
base_patterns='elastic_sidecar_infer|run_elastic_sidecar_infer|watch_elastic_shrink_and_run_sidecar|EngineCore|VllmWorker'
if [[ -n "${VERL_CLEANUP_EXTRA_PATTERNS:-}" ]]; then
    proc_patterns="${base_patterns}|${VERL_CLEANUP_EXTRA_PATTERNS}"
else
    proc_patterns="${base_patterns}"
fi

echo "[cleanup] mode=${mode} stop_ray=${stop_ray} kill_port_owners=${kill_port_owners}"
echo "[cleanup] ports=${ports_csv}"
echo "[cleanup] patterns=${proc_patterns}"

list_matching_processes() {
    ps -eo pid=,ppid=,stat=,etime=,cmd= \
        | grep -E "${proc_patterns}" \
        | grep -v -E 'grep -E|cleanup_elastic_runtime.sh' \
        || true
}

echo "[cleanup] matching processes before cleanup:"
matching_processes=$(list_matching_processes)
if [[ -n "${matching_processes}" ]]; then
    echo "${matching_processes}"
else
    echo "[cleanup] none"
fi

echo "[cleanup] listening ports before cleanup:"
python3 - "${ports_csv}" <<'PY'
import os
import socket
import sys
from pathlib import Path

ports = {int(p) for p in sys.argv[1].split(",") if p.strip()}
if not ports:
    raise SystemExit(0)

def parse_tcp_table(path):
    rows = []
    try:
        lines = Path(path).read_text().splitlines()[1:]
    except FileNotFoundError:
        return rows
    for line in lines:
        cols = line.split()
        local_addr = cols[1]
        state = cols[3]
        inode = cols[9]
        if state != "0A":
            continue
        host_hex, port_hex = local_addr.rsplit(":", 1)
        port = int(port_hex, 16)
        if port in ports:
            rows.append((port, inode))
    return rows

inode_to_port = {}
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    for port, inode in parse_tcp_table(table):
        inode_to_port[inode] = port

if not inode_to_port:
    print("[cleanup] no matching listening ports")
    raise SystemExit(0)

owners = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    fd_dir = proc / "fd"
    try:
        fds = list(fd_dir.iterdir())
    except (FileNotFoundError, PermissionError):
        continue
    owned_ports = set()
    for fd in fds:
        try:
            target = os.readlink(fd)
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if not target.startswith("socket:["):
            continue
        inode = target[len("socket:["):-1]
        if inode in inode_to_port:
            owned_ports.add(inode_to_port[inode])
    if not owned_ports:
        continue
    try:
        cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore").strip()
    except (FileNotFoundError, PermissionError):
        cmd = ""
    owners.append((int(proc.name), sorted(owned_ports), cmd))

for pid, owned_ports, cmd in sorted(owners):
    print(f"[cleanup] port_owner pid={pid} ports={','.join(map(str, owned_ports))} cmd={cmd[:240]}")
PY

if [[ "${mode}" != "kill" ]]; then
    echo "[cleanup] dry-run only. Re-run with --kill to clean stale runtime state."
    exit 0
fi

if [[ "${stop_ray}" == "1" ]]; then
    echo "[cleanup] running ray stop --force"
    ray stop --force || true
fi

if [[ -n "${matching_processes}" ]]; then
    echo "[cleanup] terminating matching stale processes"
    echo "${matching_processes}" | awk '{print $1}' | xargs -r kill -TERM || true
    sleep 2
    remaining=$(list_matching_processes)
    if [[ -n "${remaining}" ]]; then
        echo "[cleanup] force-killing remaining stale processes"
        echo "${remaining}" | awk '{print $1}' | xargs -r kill -KILL || true
    fi
fi

if [[ "${kill_port_owners}" == "1" ]]; then
    echo "[cleanup] killing owners of selected listening ports"
    python3 - "${ports_csv}" <<'PY' | xargs -r kill -TERM || true
import os
import sys
from pathlib import Path

ports = {int(p) for p in sys.argv[1].split(",") if p.strip()}
inode_to_port = {}
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        lines = Path(table).read_text().splitlines()[1:]
    except FileNotFoundError:
        continue
    for line in lines:
        cols = line.split()
        if cols[3] != "0A":
            continue
        port = int(cols[1].rsplit(":", 1)[1], 16)
        if port in ports:
            inode_to_port[cols[9]] = port
owners = set()
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        fds = list((proc / "fd").iterdir())
    except (FileNotFoundError, PermissionError):
        continue
    for fd in fds:
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:[") and target[len("socket:["):-1] in inode_to_port:
            owners.add(proc.name)
for pid in sorted(owners, key=int):
    print(pid)
PY
fi

echo "[cleanup] done"
