#!/usr/bin/env bash
# Source this file from qwen3_true_mode5_shrinkaware_staged launchers to use
# the validated CANN 9.1 A3 + torch/torch_npu 2.9 runtime by default.

_cann91_runtime_tools_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_cann91_runtime_repo_root=$(cd "${_cann91_runtime_tools_dir}/.." && pwd)

: "${USE_CANN91_A3_WITH_NNAL91:=1}"
export USE_CANN91_A3_WITH_NNAL91

if [[ "${USE_CANN91_A3_WITH_NNAL91}" == "1" && "${CANN91_A3_TORCH29_ENV:-0}" != "1" ]]; then
    : "${CANN91_VENV_PATH:=${_cann91_runtime_repo_root}/.venv_cann91_a3_torch29}"
    : "${CANN91_ENV_SCRIPT:=${_cann91_runtime_repo_root}/use_cann91_a3_torch29.sh}"

    if [[ ! -f "${CANN91_ENV_SCRIPT}" ]]; then
        CANN91_ENV_SCRIPT="${CANN91_ENV_SCRIPT_FALLBACK:-/workspace/cann-recipes-train/llm_rl/qwen3/use_cann91_a3_with_nnal91.sh}"
    fi

    if [[ ! -f "${CANN91_VENV_PATH}/bin/activate" ]]; then
        echo "missing CANN 9.1 A3 torch2.9 venv: ${CANN91_VENV_PATH}" >&2
        echo "Run tools/bootstrap_cann91_a3_torch29_env.sh, or set USE_CANN91_A3_WITH_NNAL91=0 to use the caller environment." >&2
        return 2 2>/dev/null || exit 2
    fi
    if [[ ! -f "${CANN91_ENV_SCRIPT}" ]]; then
        echo "missing CANN 9.1 A3 environment script: ${CANN91_ENV_SCRIPT}" >&2
        echo "Run tools/bootstrap_cann91_a3_torch29_env.sh, or set CANN91_ENV_SCRIPT to the correct use_*.sh path." >&2
        return 2 2>/dev/null || exit 2
    fi

    _cann91_had_nounset=0
    case "$-" in
        *u*) _cann91_had_nounset=1; set +u ;;
    esac
    # Third-party set_env.sh files are not always nounset-safe.
    # Restore the caller's nounset state immediately after sourcing.
    source "${CANN91_VENV_PATH}/bin/activate"
    source "${CANN91_ENV_SCRIPT}"
    if [[ "${_cann91_had_nounset}" == "1" ]]; then
        set -u
    fi

    export USE_CANN91_A3_WITH_NNAL91=1
    export CANN91_A3_TORCH29_ENV=1
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${_cann91_runtime_repo_root}/.cache/torch_extensions_cann91_a3_torch29}"
    mkdir -p "${TORCH_EXTENSIONS_DIR}"

    case ":${PYTHONPATH:-}:" in
        *":${_cann91_runtime_repo_root}:"*) ;;
        *) export PYTHONPATH="${_cann91_runtime_repo_root}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac

    echo "[cann91 default] enabled python=$(command -v python) env_script=${CANN91_ENV_SCRIPT}"
elif [[ "${USE_CANN91_A3_WITH_NNAL91}" != "1" ]]; then
    export USE_CANN91_A3_WITH_NNAL91=0
fi

unset _cann91_runtime_tools_dir
unset _cann91_runtime_repo_root
unset _cann91_had_nounset
