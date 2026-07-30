#!/usr/bin/env bash
set -euo pipefail

section() {
  printf '\n===== %s =====\n' "$*"
}

run_cmd() {
  printf '\n$ %s\n' "$*"
  "$@" 2>&1 || true
}

section "basic"
run_cmd date -u
run_cmd pwd
run_cmd uname -a
if [[ -f /etc/os-release ]]; then
  run_cmd cat /etc/os-release
fi

section "git"
run_cmd git rev-parse --show-toplevel
run_cmd git rev-parse HEAD
run_cmd git branch --show-current
run_cmd git status --short

section "python torch torch_npu"
run_cmd which python3
run_cmd which torchrun
python3 - <<'PY' || true
import os
import platform
import sys

print("python_executable:", sys.executable)
print("python_version:", sys.version.replace("\n", " "))
print("platform:", platform.platform())

for mod_name in ("torch", "torch_npu", "ray", "vllm"):
    try:
        mod = __import__(mod_name)
        print(f"{mod_name}_version:", getattr(mod, "__version__", "<missing __version__>"))
        print(f"{mod_name}_file:", getattr(mod, "__file__", "<missing __file__>"))
    except Exception as exc:
        print(f"{mod_name}_import_error:", repr(exc))

for key in (
    "PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH",
    "TOOLCHAIN_HOME",
):
    print(f"{key}={os.environ.get(key, '')}")
PY
run_cmd python3 -m pip show torch torch-npu ray vllm vllm-ascend

section "ascend install"
run_cmd ls -la /usr/local/Ascend
run_cmd find /usr/local/Ascend -maxdepth 4 -type f -name version.info -print -exec cat {} ';'

section "npu status"
run_cmd npu-smi info
run_cmd npu-smi info -t topo

section "network"
run_cmd ip -br addr
run_cmd ip route
run_cmd ss -ltnp

section "relevant env"
env | sort | grep -E '^(ASCEND|HCCL|GLOO|MASTER|RANK|WORLD|LOCAL_RANK|VLLM|VERL|PYTORCH|TASK_QUEUE|RAY|NCCL|OMP|MKL|LD_LIBRARY_PATH|PYTHONPATH)=' || true
