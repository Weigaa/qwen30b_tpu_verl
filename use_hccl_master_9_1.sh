#!/usr/bin/env bash
# Source this file to run with locally built HCCL master (9.1.0) installed into
# an isolated CANN 9.1 toolkit tree. This does not overwrite /usr/local/Ascend.
# Usage:
#   source /workspace/cann-recipes-train/llm_rl/qwen3/use_hccl_master_9_1.sh

_HCCL_MASTER_CANN_ROOT="/workspace/cann-recipes-train/llm_rl/qwen3/cann_master_9.1/cann"
# Be safe when sourced from scripts using `set -u`. CANN's set_env.sh reads
# these variables through indirect expansion.
export PATH="${PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
if [[ ! -f "${_HCCL_MASTER_CANN_ROOT}/set_env.sh" ]]; then
  echo "missing CANN 9.1 set_env.sh: ${_HCCL_MASTER_CANN_ROOT}/set_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi
source "${_HCCL_MASTER_CANN_ROOT}/set_env.sh"

# Ensure the built HCCL master libs win over any pre-existing /usr/local/Ascend entries.
export LD_LIBRARY_PATH="${_HCCL_MASTER_CANN_ROOT}/lib64:${_HCCL_MASTER_CANN_ROOT}/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${_HCCL_MASTER_CANN_ROOT}/python/site-packages:${PYTHONPATH:-}"
export ASCEND_HOME_PATH="${_HCCL_MASTER_CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${_HCCL_MASTER_CANN_ROOT}"
export ASCEND_OPP_PATH="${_HCCL_MASTER_CANN_ROOT}/opp"
export ASCEND_AICPU_PATH="${_HCCL_MASTER_CANN_ROOT}"

# Marker for logs.
export HCCL_MASTER_9_1_ENABLED=1
