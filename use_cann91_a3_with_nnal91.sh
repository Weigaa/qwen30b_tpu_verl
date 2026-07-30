#!/usr/bin/env bash
# Source this after activating the torch2.9 venv and before launching training.
# It switches CANN runtime/HCCL to the local 9.1 A3 build and ATB to matching NNAL 9.1 beta3.
# For large-model ATB usage we intentionally do NOT source ASDSIP; the NNAL installer warns
# that sourcing ATB and ASDSIP together can have unexpected consequences.
#
# Keep this file source-safe: do not leave `set -euo pipefail` enabled in the caller shell.

_cann91_nnal91_fail() {
  echo "[cann91+nnal91] ERROR: $*" >&2
  return 1 2>/dev/null || exit 1
}

_QWEN3_ENV_ROOT="/workspace/cann-recipes-train/llm_rl/qwen3"
_CANN91_ENV="${_QWEN3_ENV_ROOT}/use_hccl_master_9_1_a3.sh"
_NNAL91_ROOT="${_QWEN3_ENV_ROOT}/Ascend_nnal_9.1.0_beta3/nnal"
_ATB91_ENV="${_NNAL91_ROOT}/atb/set_env.sh"

[[ -f "${_CANN91_ENV}" ]] || _cann91_nnal91_fail "missing ${_CANN91_ENV}"
[[ -f "${_ATB91_ENV}" ]] || _cann91_nnal91_fail "missing ${_ATB91_ENV}"

# Third-party set_env scripts are not nounset-safe. Preserve caller shell nounset state.
_cann91_had_nounset=0
case $- in *u*) _cann91_had_nounset=1; set +u ;; esac

# First select CANN/HCCL 9.1 A3.
if ! source "${_CANN91_ENV}"; then
  if [[ ${_cann91_had_nounset} -eq 1 ]]; then set -u; fi
  _cann91_nnal91_fail "failed to source ${_CANN91_ENV}"
fi

# Then select matching ATB 9.1. This prepends NNAL 9.1 ATB paths before any old 8.3 entries.
if ! source "${_ATB91_ENV}" --cxx_abi=1; then
  if [[ ${_cann91_had_nounset} -eq 1 ]]; then set -u; fi
  _cann91_nnal91_fail "failed to source ${_ATB91_ENV} --cxx_abi=1"
fi

if [[ ${_cann91_had_nounset} -eq 1 ]]; then set -u; fi
unset _cann91_had_nounset

export USE_CANN91_A3_WITH_NNAL91=1
export NNAL91_ROOT="${_NNAL91_ROOT}"
export ATB91_HOME_PATH="${ATB_HOME_PATH:-}"

echo "[cann91+nnal91] ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-}"
echo "[cann91+nnal91] ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-}"
echo "[cann91+nnal91] ATB_HOME_PATH=${ATB_HOME_PATH:-}"
echo "[cann91+nnal91] ASDSIP_HOME_PATH=${ASDSIP_HOME_PATH:-<unset>}"

# Diagnostic only. Never fail the caller because python is missing/misconfigured here.
if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY' || echo "[cann91+nnal91] WARN: failed to print LD_LIBRARY_PATH diagnostic" >&2
import os
print('[cann91+nnal91] LD_LIBRARY_PATH_HEAD=' + ':'.join(os.environ.get('LD_LIBRARY_PATH', '').split(':')[:12]))
PY
else
  echo "[cann91+nnal91] WARN: python3 not found; skip LD_LIBRARY_PATH diagnostic" >&2
fi

unset _QWEN3_ENV_ROOT _CANN91_ENV _NNAL91_ROOT _ATB91_ENV
unset -f _cann91_nnal91_fail 2>/dev/null || true
