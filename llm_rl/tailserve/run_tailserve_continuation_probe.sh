#!/usr/bin/env bash
# Separate short inference diagnostic, not the DeepScaleR production workload.
set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export VLLM_USE_V1=1
export VLLM_DP_SIZE=16
export VLLM_ENABLE_EXPERT_PARALLEL=1
export VLLM_ENABLE_MC2=1
export VLLM_ENABLE_EPLB=0
export VLLM_ENABLE_GRAPH_MODE=0
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=0
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=0
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1
export VLLM_ASCEND_USE_NATIVE_QWEN3_MOE=0
export VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC=1
export VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=1
export VLLM_ASCEND_MC2_TOKENS_CAPACITY=512
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=0
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=0
export VLLM_ASCEND_USE_LEGACY_FUSED_MOE=0
export VLLM_ASCEND_USE_LEGACY_ATTENTION=0
export VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_IF_BASE_PORT=48301
export HCCL_BUFFSIZE=2048
export ALL_TO_ALL_RESHARD=1
export USE_ALLTOALL_OVERLAP=1
export TAILSERVE_ENABLE=1
exec torchrun --standalone --nproc-per-node=16 tools/probe_tailserve_continuation.py "$@"
