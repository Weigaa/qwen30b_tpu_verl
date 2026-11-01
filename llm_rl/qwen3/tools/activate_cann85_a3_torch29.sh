#!/usr/bin/env bash

# Source this file to select the repository-local vLLM 0.14 runtime.  Start
# from clean path variables so CANN 8.3/9.1 or another checkout cannot leak in.
QWEN3_RUNTIME_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
QWEN3_RUNTIME_VENV=${QWEN3_CANN85_TORCH29_VENV:-${QWEN3_RUNTIME_ROOT}/.venv_cann85_a3_torch29}
QWEN3_CANN85_ROOT=${QWEN3_CANN85_ROOT:-/workspace/cann-recipes-train/llm_rl/qwen3/cann_8.5.0_a3}
QWEN3_CANN85_HOME="${QWEN3_CANN85_ROOT}/cann-8.5.0"
QWEN3_NNAL85_ATB_ENV="${QWEN3_CANN85_ROOT}/nnal/atb/set_env.sh"
QWEN3_DRIVER_ROOT=${QWEN3_DRIVER_ROOT:-/usr/local/Ascend/driver}

for required in \
  "${QWEN3_RUNTIME_VENV}/bin/python" \
  "${QWEN3_CANN85_HOME}/set_env.sh" \
  "${QWEN3_NNAL85_ATB_ENV}" \
  "${QWEN3_DRIVER_ROOT}/version.info" \
  "${QWEN3_DRIVER_ROOT}/lib64/driver/libascend_hal.so"
do
  if [ ! -e "${required}" ]; then
    printf 'missing qwen3 vLLM 0.14 runtime component: %s\n' "${required}" >&2
    return 1 2>/dev/null || exit 1
  fi
done

if ! bash "${QWEN3_RUNTIME_ROOT}/tools/select_vllm_ascend_native.sh" torch29_cann85; then
  printf 'failed to select the Torch 2.9/CANN 8.5 vLLM-Ascend native artifacts\n' >&2
  return 1 2>/dev/null || exit 1
fi

# Vendor environment scripts are not consistently nounset-safe.
QWEN3_RUNTIME_RESTORE_NOUNSET=0
case $- in
  *u*)
    QWEN3_RUNTIME_RESTORE_NOUNSET=1
    set +u
    ;;
esac

if [ -n "${VIRTUAL_ENV:-}" ] && command -v deactivate >/dev/null 2>&1; then
  deactivate
fi
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LIBRARY_PATH
unset CPLUS_INCLUDE_PATH CPATH ASCEND_HOME_PATH ASCEND_OPP_PATH
unset ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME ATB_HOME_PATH ATB_HOME

# Keep only operating-system paths before loading the selected stack.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source "${QWEN3_RUNTIME_VENV}/bin/activate"
if ! source "${QWEN3_CANN85_HOME}/set_env.sh"; then
  printf 'failed to load CANN 8.5 environment: %s\n' \
    "${QWEN3_CANN85_HOME}/set_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi
# NNAL warns against sourcing ATB and ASDSIP together. vLLM needs ATB only.
if ! source "${QWEN3_NNAL85_ATB_ENV}" --cxx_abi=1; then
  printf 'failed to load NNAL ATB 8.5 environment: %s\n' \
    "${QWEN3_NNAL85_ATB_ENV}" >&2
  return 1 2>/dev/null || exit 1
fi

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
export QWEN3_RUNTIME_STACK=vllm0141_vllm_ascend0140rc1_torch29_cann850_a3
export QWEN3_RUNTIME_VENV QWEN3_CANN85_ROOT QWEN3_CANN85_HOME
export QWEN3_NNAL85_ATB_ENV QWEN3_DRIVER_ROOT
export QWEN3_CANN_VERSION_FILE="${QWEN3_CANN85_HOME}/share/info/runtime/version.info"
export SOC_VERSION=${QWEN3_SOC_VERSION:-ascend910_9392}
# On this host, CANN 8.5's AIV/AICPU collective expansion fails with 507018
# even in a standalone torchrun probe. HOST expansion uses the working HCCL
# path and must be selected before torch_npu initializes the runtime.
export HCCL_OP_EXPANSION_MODE=HOST
export PYTHONNOUSERSITE=1
export PIP_REQUIRE_VIRTUALENV=1
QWEN3_CANN_PYTHONPATH="${QWEN3_CANN85_HOME}/python/site-packages:${QWEN3_CANN85_HOME}/opp/built-in/op_impl/ai_core/tbe"
export QWEN3_CANN_PYTHONPATH
export PYTHONPATH="${QWEN3_RUNTIME_ROOT}/tools:${QWEN3_RUNTIME_ROOT}:${QWEN3_CANN_PYTHONPATH}"

# Real driver libraries must precede CANN devlib stubs. Local sources and the
# selected venv must precede every optional CANN component.
export LD_LIBRARY_PATH="${QWEN3_DRIVER_ROOT}/lib64/driver:${QWEN3_DRIVER_ROOT}/lib64/common:${QWEN3_DRIVER_ROOT}/lib64:${QWEN3_RUNTIME_ROOT}/vllm_ascend:${QWEN3_RUNTIME_ROOT}/vllm_ascend/lib64:${QWEN3_PYTORCH_LIB_PATHS}:${LD_LIBRARY_PATH:-}"

QWEN3_FORBIDDEN_RUNTIME_PATHS='(/usr/local/Ascend/ascend-toolkit/|/usr/local/Ascend/nnal/|/workspace/vllm([/:]|$)|/workspace/vllm-ascend([/:]|$)|/cann_master_9\.1/|/cann_8\.5\.0_a3_stable/|/\.venv_torch27_vllm014/)'
for variable in PATH PYTHONPATH LD_LIBRARY_PATH ASCEND_OPP_PATH; do
  value=${!variable:-}
  if printf '%s\n' "${value}" | grep -Eq "${QWEN3_FORBIDDEN_RUNTIME_PATHS}"; then
    printf 'mixed runtime path detected in %s: %s\n' "${variable}" "${value}" >&2
    return 1 2>/dev/null || exit 1
  fi
done

if [ "${QWEN3_RUNTIME_RESTORE_NOUNSET}" = 1 ]; then
  set -u
fi
unset QWEN3_RUNTIME_RESTORE_NOUNSET QWEN3_PYTORCH_LIB_PATHS
unset QWEN3_FORBIDDEN_RUNTIME_PATHS

printf '[qwen3 runtime] stack=%s venv=%s cann=%s atb=%s python=%s\n' \
  "${QWEN3_RUNTIME_STACK}" "${QWEN3_RUNTIME_VENV}" \
  "${QWEN3_CANN85_HOME}" "${QWEN3_NNAL85_ATB_ENV}" "$(command -v python3)"
