#!/usr/bin/env bash
# Source this file to run with CANN 9.1 A3/910C ops plus locally built HCCL master.
# Does not overwrite /usr/local/Ascend.
_HCCL_MASTER_CANN_ROOT="/workspace/cann-recipes-train/llm_rl/qwen3/cann_master_9.1_a3/cann"
export PATH="${PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
if [[ ! -f "${_HCCL_MASTER_CANN_ROOT}/set_env.sh" ]]; then
  echo "missing CANN 9.1 A3 set_env.sh: ${_HCCL_MASTER_CANN_ROOT}/set_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi
source "${_HCCL_MASTER_CANN_ROOT}/set_env.sh"
export LD_LIBRARY_PATH="${_HCCL_MASTER_CANN_ROOT}/lib64:${_HCCL_MASTER_CANN_ROOT}/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${_HCCL_MASTER_CANN_ROOT}/python/site-packages:${PYTHONPATH:-}"
export ASCEND_HOME_PATH="${_HCCL_MASTER_CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${_HCCL_MASTER_CANN_ROOT}"
export ASCEND_OPP_PATH="${_HCCL_MASTER_CANN_ROOT}/opp"
export ASCEND_AICPU_PATH="${_HCCL_MASTER_CANN_ROOT}"
export HCCL_MASTER_9_1_A3_ENABLED=1
