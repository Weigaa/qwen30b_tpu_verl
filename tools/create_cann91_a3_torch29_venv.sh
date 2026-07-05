#!/usr/bin/env bash
set -euo pipefail

# Experimental CANN 9.1 A3 environment using torch 2.9 + stable torch_npu 2.9.
# It is intentionally separate from .venv_cann91_a3_torch271.

REPO_ROOT="${REPO_ROOT:-/workspace/cann-recipes-train/llm_rl/qwen3_true_mode5_shrinkaware_staged}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv_cann91_a3_torch29}"
WHEEL_CACHE="${WHEEL_CACHE:-/workspace/cann-recipes-train/llm_rl/qwen3/torch_npu_wheel_probe}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PYPI_INDEX="${PYPI_INDEX:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
ASCEND_PYPI="${ASCEND_PYPI:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.9.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.0}"
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
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel -i "${PYPI_INDEX}" --trusted-host mirrors.huaweicloud.com

python -m pip install --upgrade --no-cache-dir \
  "torch==${TORCH_VERSION}" \
  -i "${PYPI_INDEX}" --trusted-host mirrors.huaweicloud.com

# Keep numpy below 2.0 for the current vllm_ascend metadata and numba compatibility.
python -m pip install --upgrade --no-cache-dir \
  "numpy==1.26.4" "fsspec==2025.3.0" PyYAML psutil decorator attrs cloudpickle \
  -i "${PYPI_INDEX}" --trusted-host mirrors.huaweicloud.com

python -m pip install --upgrade --no-cache-dir \
  "torch-npu==${TORCH_NPU_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  -i "${PYPI_INDEX}" --trusted-host mirrors.huaweicloud.com

# shellcheck disable=SC1090
source "${CANN91_ENV}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_ROOT}/.cache/torch_extensions_cann91_a3_torch29}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

python - <<'PY'
import os
import ray
import numpy
import numba
import torch
import torch_npu
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
print("python", os.sys.executable)
print("ray", ray.__version__, ray.__file__)
print("numpy", numpy.__version__, numpy.__file__)
print("numba", numba.__version__, numba.__file__)
print("torch", torch.__version__, torch.__file__)
print("torch_npu", getattr(torch_npu, "__version__", None), torch_npu.__file__)
print("ngram_proposer_import=OK", NgramProposer)
print("ASCEND_HOME_PATH", os.environ.get("ASCEND_HOME_PATH"))
print("ASCEND_OPP_PATH", os.environ.get("ASCEND_OPP_PATH"))
print("TORCH_EXTENSIONS_DIR", os.environ.get("TORCH_EXTENSIONS_DIR"))
schema = str(torch.ops.npu.npu_moe_distribute_dispatch_v2.default._schema)
print("dispatch_v2_schema", schema)
missing = [name for name in ("elastic_info", "comm_alg") if name not in schema]
if missing:
    raise SystemExit(f"torch_npu schema is missing required fields: {missing}")
print("schema_check=OK")
PY

cat <<EOF2

[OK] Created experimental CANN 9.1 A3 torch/torch_npu venv:
  VENV_DIR=${VENV_DIR}
  torch=${TORCH_VERSION}
  torch_npu=${TORCH_NPU_VERSION}
  torchvision=${TORCHVISION_VERSION}

To use it:
  cd ${REPO_ROOT}
  source ${VENV_DIR}/bin/activate
  source ${CANN91_ENV}
  export TORCH_EXTENSIONS_DIR=${REPO_ROOT}/.cache/torch_extensions_cann91_a3_torch29
  USE_HCCL_MASTER_9_1_A3=1 bash run_mode1_local_baseline_random_batch_threshold_multistep.sh
EOF2
