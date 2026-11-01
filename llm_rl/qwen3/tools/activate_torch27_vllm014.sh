#!/usr/bin/env bash

# Source this file to select the repository-local Torch 2.7/vLLM 0.14 runtime.
QWEN3_RUNTIME_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
QWEN3_RUNTIME_VENV=${QWEN3_TORCH27_VLLM014_VENV:-${QWEN3_RUNTIME_ROOT}/.venv_torch27_vllm014}

if [ ! -x "${QWEN3_RUNTIME_VENV}/bin/python" ]; then
  printf 'missing qwen3 Torch 2.7/vLLM 0.14 venv: %s\n' \
    "${QWEN3_RUNTIME_VENV}" >&2
  return 1 2>/dev/null || exit 1
fi

bash "${QWEN3_RUNTIME_ROOT}/tools/select_vllm_ascend_native.sh" \
  torch27_cann83

# Vendor environment scripts are not consistently nounset-safe.
QWEN3_RUNTIME_RESTORE_NOUNSET=0
case $- in
  *u*)
    QWEN3_RUNTIME_RESTORE_NOUNSET=1
    set +u
    ;;
esac

source "${QWEN3_RUNTIME_VENV}/bin/activate"
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1

QWEN3_PYTORCH_LIB_PATHS=$("${QWEN3_RUNTIME_VENV}/bin/python" - <<'PY'
import importlib.util
from pathlib import Path

paths = []
for package in ("torch", "torch_npu"):
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"missing required package: {package}")
    lib_dir = Path(spec.origin).resolve().parent / "lib"
    if not lib_dir.is_dir():
        raise RuntimeError(f"missing native library directory: {lib_dir}")
    paths.append(str(lib_dir))
print(":".join(paths))
PY
)

export QWEN3_LOCAL_IMPORT_ROOT="${QWEN3_RUNTIME_ROOT}"
export QWEN3_RUNTIME_STACK=torch27_cann83
export PYTHONPATH="${QWEN3_RUNTIME_ROOT}/tools:${QWEN3_RUNTIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${QWEN3_RUNTIME_ROOT}/vllm_ascend:${QWEN3_RUNTIME_ROOT}/vllm_ascend/lib64:${QWEN3_PYTORCH_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [ "${QWEN3_RUNTIME_RESTORE_NOUNSET}" = 1 ]; then
  set -u
fi
unset QWEN3_RUNTIME_RESTORE_NOUNSET
unset QWEN3_PYTORCH_LIB_PATHS

printf '[qwen3 runtime] venv=%s python=%s\n' \
  "${QWEN3_RUNTIME_VENV}" "$(command -v python3)"
