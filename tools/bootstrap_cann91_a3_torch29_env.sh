#!/usr/bin/env bash
set -euo pipefail

# Bootstrap an isolated CANN 9.1 A3/910C + torch/torch_npu 2.9 runtime for this repo.
#
# Scope:
#   - Installs or binds a CANN 9.1 A3 runtime under an isolated directory.
#   - Binds matching NNAL/ATB 9.1 if provided.
#   - Creates a Python 3.11 venv with torch 2.9.0 + torch_npu 2.9.0 + torchvision 0.24.0.
#   - Generates a source-able env file for later training runs.
#
# It intentionally does NOT manage the repo versions of verl/vllm/vllm_ascend/megatron/mindspeed.
# Keep those from the checked-out training repository.
#
# Typical usage with local installers/wheels:
#   cd /workspace/cann-recipes-train/llm_rl/qwen3_true_mode5_shrinkaware_staged
#   CANN_TOOLKIT_RUN=/path/Ascend-cann-toolkit_9.1.0_linux-aarch64.run \
#   CANN_A3_OPS_RUN=/path/Ascend-cann-A3-ops_9.1.0_linux-aarch64.run \
#   NNAL_ROOT=/path/Ascend_nnal_9.1.0_beta3/nnal \
#   bash tools/bootstrap_cann91_a3_torch29_env.sh
#
# Typical usage if CANN/NNAL are already installed somewhere:
#   CANN_ROOT=/path/to/cann-9.1.0/cann NNAL_ROOT=/path/to/nnal bash tools/bootstrap_cann91_a3_torch29_env.sh
#
# After success:
#   source .venv_cann91_a3_torch29/bin/activate
#   source ./use_cann91_a3_torch29.sh

log() { printf '[bootstrap-cann91-a3-torch29] %s\n' "$*"; }
die() { printf '[bootstrap-cann91-a3-torch29] ERROR: %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# qwen3 root is the sibling helper/data directory used by this project; override if your layout differs.
QWEN3_ROOT="${QWEN3_ROOT:-$(cd "${REPO_ROOT}/../qwen3" 2>/dev/null && pwd || true)}"
[[ -n "${QWEN3_ROOT}" && -d "${QWEN3_ROOT}" ]] || QWEN3_ROOT="${REPO_ROOT}/.cann91_assets"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv_cann91_a3_torch29}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_ROOT}/.cache/torch_extensions_cann91_a3_torch29}"
WHEELHOUSE="${WHEELHOUSE:-${QWEN3_ROOT}/wheelhouse_cann91_torch29}"

PYPI_INDEX="${PYPI_INDEX:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.huaweicloud.com}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.9.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.0}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
FSSPEC_VERSION="${FSSPEC_VERSION:-2025.3.0}"

# CANN inputs. If CANN_ROOT is set, installers are skipped.
INSTALL_ROOT="${INSTALL_ROOT:-${QWEN3_ROOT}/cann91_a3_runtime}"
CANN_ROOT="${CANN_ROOT:-}"
CANN_TOOLKIT_RUN="${CANN_TOOLKIT_RUN:-${QWEN3_ROOT}/cann_master_downloads/Ascend-cann-toolkit_9.1.0_linux-aarch64.run}"
CANN_A3_OPS_RUN="${CANN_A3_OPS_RUN:-${QWEN3_ROOT}/cann_master_downloads/Ascend-cann-A3-ops_9.1.0_linux-aarch64.run}"

# NNAL/ATB 9.1 root. It should contain atb/set_env.sh. ASDSIP is intentionally not sourced.
NNAL_ROOT="${NNAL_ROOT:-${QWEN3_ROOT}/Ascend_nnal_9.1.0_beta3/nnal}"
ATB_SET_ENV="${ATB_SET_ENV:-${NNAL_ROOT}/atb/set_env.sh}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/use_cann91_a3_torch29.sh}"
FORCE_RECREATE_VENV="${FORCE_RECREATE_VENV:-0}"
SKIP_CANN_INSTALL="${SKIP_CANN_INSTALL:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

mkdir -p "${QWEN3_ROOT}" "${WHEELHOUSE}" "${TORCH_EXTENSIONS_DIR}"
cd "${REPO_ROOT}"

log "repo root: ${REPO_ROOT}"
log "asset root: ${QWEN3_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import platform, sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Python 3.11 is required for torch_npu 2.9 cp311 wheels; got {sys.version}")
if platform.machine() not in {"aarch64", "arm64"}:
    raise SystemExit(f"AArch64 machine is expected for Ascend torch_npu wheels; got {platform.machine()}")
PY

install_cann_if_needed() {
    if [[ -n "${CANN_ROOT}" ]]; then
        [[ -f "${CANN_ROOT}/set_env.sh" ]] || die "CANN_ROOT does not contain set_env.sh: ${CANN_ROOT}"
        return 0
    fi

    # Reuse an already installed isolated CANN root if present.
    for candidate in \
        "${INSTALL_ROOT}/cann" \
        "${INSTALL_ROOT}/cann-9.1.0/cann" \
        "${QWEN3_ROOT}/cann_master_9.1_a3/cann" \
        "${QWEN3_ROOT}/cann_master_9.1_a3/cann-9.1.0/cann"; do
        if [[ -f "${candidate}/set_env.sh" ]]; then
            CANN_ROOT="$(cd "${candidate}" && pwd)"
            log "reuse CANN root: ${CANN_ROOT}"
            return 0
        fi
    done

    [[ "${SKIP_CANN_INSTALL}" == "0" ]] || die "CANN_ROOT not set/found and SKIP_CANN_INSTALL=1"
    [[ -f "${CANN_TOOLKIT_RUN}" ]] || die "missing CANN toolkit installer: ${CANN_TOOLKIT_RUN}"
    [[ -f "${CANN_A3_OPS_RUN}" ]] || die "missing CANN A3 ops installer: ${CANN_A3_OPS_RUN}"

    mkdir -p "${INSTALL_ROOT}"
    log "install CANN toolkit to ${INSTALL_ROOT}"
    bash "${CANN_TOOLKIT_RUN}" --install --quiet --nox11 --install-path="${INSTALL_ROOT}"
    log "install CANN A3 ops to ${INSTALL_ROOT}"
    bash "${CANN_A3_OPS_RUN}" --install --quiet --nox11 --install-path="${INSTALL_ROOT}"

    for candidate in "${INSTALL_ROOT}/cann" "${INSTALL_ROOT}/cann-9.1.0/cann"; do
        if [[ -f "${candidate}/set_env.sh" ]]; then
            CANN_ROOT="$(cd "${candidate}" && pwd)"
            log "installed CANN root: ${CANN_ROOT}"
            return 0
        fi
    done
    die "CANN install finished but set_env.sh was not found under ${INSTALL_ROOT}"
}

install_cann_if_needed
[[ -f "${ATB_SET_ENV}" ]] || die "missing NNAL/ATB 9.1 set_env.sh: ${ATB_SET_ENV}; set NNAL_ROOT=/path/to/nnal or ATB_SET_ENV=/path/to/atb/set_env.sh"

log "write env file: ${ENV_FILE}"
cat > "${ENV_FILE}" <<EOF
#!/usr/bin/env bash
# Source this file after activating ${VENV_DIR} and before launching verl/vllm training.
# Generated by tools/bootstrap_cann91_a3_torch29_env.sh on $(date -Iseconds).
_cann91_torch29_fail() { echo "[cann91-a3-torch29] ERROR: \$*" >&2; return 1 2>/dev/null || exit 1; }
_CANN91_ROOT="${CANN_ROOT}"
_ATB91_SET_ENV="${ATB_SET_ENV}"
[[ -f "\${_CANN91_ROOT}/set_env.sh" ]] || _cann91_torch29_fail "missing \${_CANN91_ROOT}/set_env.sh"
[[ -f "\${_ATB91_SET_ENV}" ]] || _cann91_torch29_fail "missing \${_ATB91_SET_ENV}"
_cann91_had_nounset=0
case \$- in *u*) _cann91_had_nounset=1; set +u ;; esac
source "\${_CANN91_ROOT}/set_env.sh"
source "\${_ATB91_SET_ENV}" --cxx_abi=1
if [[ \${_cann91_had_nounset} -eq 1 ]]; then set -u; fi
unset _cann91_had_nounset
export ASCEND_HOME_PATH="\${_CANN91_ROOT}"
export ASCEND_TOOLKIT_HOME="\${_CANN91_ROOT}"
export ASCEND_OPP_PATH="\${_CANN91_ROOT}/opp"
export ASCEND_AICPU_PATH="\${_CANN91_ROOT}"
export LD_LIBRARY_PATH="\${_CANN91_ROOT}/lib64:\${_CANN91_ROOT}/aarch64-linux/lib64:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="\${_CANN91_ROOT}/python/site-packages:\${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}"
export USE_CANN91_A3_WITH_NNAL91=1
export CANN91_A3_TORCH29_ENV=1
export NNAL91_ROOT="$(cd "$(dirname "${ATB_SET_ENV}")/.." && pwd)"
export ATB91_HOME_PATH="\${ATB_HOME_PATH:-}"
echo "[cann91-a3-torch29] ASCEND_HOME_PATH=\${ASCEND_HOME_PATH}"
echo "[cann91-a3-torch29] ASCEND_OPP_PATH=\${ASCEND_OPP_PATH}"
echo "[cann91-a3-torch29] ATB_HOME_PATH=\${ATB_HOME_PATH:-}"
echo "[cann91-a3-torch29] TORCH_EXTENSIONS_DIR=\${TORCH_EXTENSIONS_DIR}"
unset _CANN91_ROOT _ATB91_SET_ENV
unset -f _cann91_torch29_fail 2>/dev/null || true
EOF
chmod +x "${ENV_FILE}"

if [[ "${FORCE_RECREATE_VENV}" == "1" && -d "${VENV_DIR}" ]]; then
    log "remove existing venv: ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    log "create venv with system site packages: ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

PIP_ARGS=(-i "${PYPI_INDEX}" --trusted-host "${PIP_TRUSTED_HOST}")
log "upgrade pip tooling"
python -m pip install --upgrade pip setuptools wheel "${PIP_ARGS[@]}"

log "install torch stack: torch=${TORCH_VERSION}, torch_npu=${TORCH_NPU_VERSION}, torchvision=${TORCHVISION_VERSION}"
python -m pip install --upgrade --no-cache-dir \
    "torch==${TORCH_VERSION}" \
    "torch-npu==${TORCH_NPU_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "numpy==${NUMPY_VERSION}" \
    "fsspec==${FSSPEC_VERSION}" \
    PyYAML psutil decorator attrs cloudpickle \
    "${PIP_ARGS[@]}"

# Always source the generated env before validation so torch_npu resolves against CANN 9.1 A3/ATB 9.1.
# shellcheck disable=SC1090
source "${ENV_FILE}"

if [[ "${SKIP_VERIFY}" != "1" ]]; then
    log "verify imports and NPU op schema"
    python - <<'PY'
import os, sys
print('python', sys.executable)
import torch
import torch_npu
import torchvision
print('torch', torch.__version__, torch.__file__)
print('torch_npu', getattr(torch_npu, '__version__', None), torch_npu.__file__)
print('torchvision', torchvision.__version__, torchvision.__file__)
print('ASCEND_HOME_PATH', os.environ.get('ASCEND_HOME_PATH'))
print('ASCEND_OPP_PATH', os.environ.get('ASCEND_OPP_PATH'))
print('ATB_HOME_PATH', os.environ.get('ATB_HOME_PATH'))
print('TORCH_EXTENSIONS_DIR', os.environ.get('TORCH_EXTENSIONS_DIR'))
schema = str(torch.ops.npu.npu_moe_distribute_dispatch_v2.default._schema)
print('dispatch_v2_schema', schema)
missing = [name for name in ('elastic_info', 'comm_alg') if name not in schema]
if missing:
    raise SystemExit(f'torch_npu schema is missing required fields: {missing}')
for name in ('vllm', 'vllm_ascend'):
    try:
        mod = __import__(name)
        print(name, getattr(mod, '__version__', '?'), getattr(mod, '__file__', '?'))
    except Exception as exc:
        print(name, 'WARN_IMPORT_FAILED', repr(exc))
print('verify=OK')
PY
fi

cat <<EOF

[OK] CANN 9.1 A3 + torch/torch_npu 2.9 environment is ready.

Use it like this:
  cd ${REPO_ROOT}
  source ${VENV_DIR}/bin/activate
  source ${ENV_FILE}
  # keep using this repo's verl/vllm/vllm_ascend code
  KV_TOKENS=240000 bash run_mode1_floor4_kv280576_cann91_test.sh

Important paths:
  CANN_ROOT=${CANN_ROOT}
  ATB_SET_ENV=${ATB_SET_ENV}
  VENV_DIR=${VENV_DIR}
  ENV_FILE=${ENV_FILE}
  TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}
EOF
