#!/usr/bin/env bash
set -euo pipefail

# Create an isolated Python environment for validating CANN 9.1 A3 + HCCL master
# without touching the system Python or other repositories.
#
# Selected minimal-change stack:
#   torch      2.7.1
#   torch_npu  2.7.1.post2.dev20251120 (cp311/aarch64)
# This torch_npu wheel has the newer MoeDistributeDispatchV2 schema fields
# required by newer CANN OPP: elastic_info and comm_alg.

REPO_ROOT="${REPO_ROOT:-/workspace/cann-recipes-train/llm_rl/qwen3_true_mode5_shrinkaware_staged}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv_cann91_a3_torch271}"
WHEEL_CACHE="${WHEEL_CACHE:-/workspace/cann-recipes-train/llm_rl/qwen3/torch_npu_wheel_probe}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PYPI_INDEX="${PYPI_INDEX:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
ASCEND_PYPI="${ASCEND_PYPI:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.7.1.post2.dev20251120}"
TORCH_NPU_WHEEL="torch_npu-${TORCH_NPU_VERSION}-cp311-cp311-manylinux_2_28_aarch64.whl"
TORCH_NPU_URL="${ASCEND_PYPI}/torch-npu/${TORCH_NPU_WHEEL}"
CANN91_ENV="${CANN91_ENV:-/workspace/cann-recipes-train/llm_rl/qwen3/use_hccl_master_9_1_a3.sh}"

mkdir -p "${WHEEL_CACHE}"
cd "${REPO_ROOT}"

if [[ ! -f "${CANN91_ENV}" ]]; then
  echo "missing CANN 9.1 A3 env wrapper: ${CANN91_ENV}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Python 3.11 is required for the selected torch_npu cp311 wheel, got {sys.version}")
PY

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel -i "${PYPI_INDEX}" --trusted-host mirrors.huaweicloud.com

if [[ ! -f "${WHEEL_CACHE}/${TORCH_NPU_WHEEL}" ]]; then
  curl -L --retry 3 --connect-timeout 10 -o "${WHEEL_CACHE}/${TORCH_NPU_WHEEL}" "${TORCH_NPU_URL}"
fi

python -m pip install --upgrade --no-cache-dir \
  "torch==${TORCH_VERSION}" \
  -i "${PYPI_INDEX}" --trusted-host mirrors.huaweicloud.com

# torch_npu imports optional helper modules at import time. Keep these inside
# the venv so the system Python and other repos are untouched.
python -m pip install --upgrade --no-cache-dir \
  "numpy==1.26.4" "fsspec==2025.3.0" PyYAML psutil decorator attrs cloudpickle \
  -i "${PYPI_INDEX}" --trusted-host mirrors.huaweicloud.com

python -m pip install --upgrade --no-cache-dir \
  "${WHEEL_CACHE}/${TORCH_NPU_WHEEL}"

# Basic import and schema validation under the CANN 9.1 A3 environment.
# Source CANN after installing so imports resolve against the intended runtime.
# shellcheck disable=SC1090
source "${CANN91_ENV}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_ROOT}/.cache/torch_extensions_cann91_a3_torch271}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

python - <<'PY'
import os
import torch
import torch_npu
print("python", os.sys.executable)
print("torch", torch.__version__, torch.__file__)
print("torch_npu", getattr(torch_npu, "__version__", None), torch_npu.__file__)
print("ASCEND_HOME_PATH", os.environ.get("ASCEND_HOME_PATH"))
print("ASCEND_OPP_PATH", os.environ.get("ASCEND_OPP_PATH"))
print("TORCH_EXTENSIONS_DIR", os.environ.get("TORCH_EXTENSIONS_DIR"))
schema = str(torch.ops.npu.npu_moe_distribute_dispatch_v2.default._schema)
print("dispatch_v2_schema", schema)
missing = [name for name in ("elastic_info", "comm_alg") if name not in schema]
if missing:
    raise SystemExit(f"torch_npu schema is still missing required fields: {missing}")
print("schema_check=OK")
PY

cat <<EOF

[OK] Created CANN 9.1 A3 torch/torch_npu venv:
  VENV_DIR=${VENV_DIR}
  torch=${TORCH_VERSION}
  torch_npu=${TORCH_NPU_VERSION}

To use it for this repo:
  cd ${REPO_ROOT}
  source ${VENV_DIR}/bin/activate
  source ${CANN91_ENV}
  export TORCH_EXTENSIONS_DIR=${REPO_ROOT}/.cache/torch_extensions_cann91_a3_torch271
  USE_HCCL_MASTER_9_1_A3=1 bash run_mode1_local_baseline_random_batch_threshold_multistep.sh

To leave the environment:
  deactivate
EOF
