#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/workspace/cann-recipes-train/llm_rl/qwen3
PATCH_TREE=/workspace/cann-recipes-train/llm_rl/qwen3_true_mode5_a3cfdc2
LAUNCHER="$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"

# Launch from the patch tree and prepend it to PYTHONPATH so `python -m`
# resolves verl/vllm/vllm_ascend from the patched code, while HOME still
# points at qwen3 so elastic logs land in the user's working directory.
cd "$PATCH_TREE"

export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE:-4}
export VLLM_ASCEND_MODE3_TIMING_LOG=${VLLM_ASCEND_MODE3_TIMING_LOG:-1}
export VLLM_ASCEND_MODE3_TIMING_SYNC=${VLLM_ASCEND_MODE3_TIMING_SYNC:-1}
export VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=${VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE:-1}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-20000}
export MASTER_PORT=${MASTER_PORT:-12000}
export VERL_HCCL_IF_BASE_PORT_START=${VERL_HCCL_IF_BASE_PORT_START:-${HCCL_IF_BASE_PORT}}
export VERL_MASTER_PORT_START=${VERL_MASTER_PORT_START:-${MASTER_PORT}}

# Force launcher outputs into qwen3 rather than the patch tree.
export HOME="$REPO_ROOT"
export CONFIG_DIR="$PATCH_TREE/verl/trainer/config"
export PYTHONPATH="$PATCH_TREE${PYTHONPATH:+:$PYTHONPATH}"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
tee_log="$REPO_ROOT/mode4_bulk_request_test_${stamp}.log"

echo "[mode4 bulk request test] cwd=$REPO_ROOT launcher=$LAUNCHER"
echo "[mode4 bulk request test] tee_log=$tee_log"
echo "[mode4 bulk request test] elastic_mode=$VLLM_ASCEND_ELASTIC_EXECUTION_MODE HCCL_IF_BASE_PORT=$HCCL_IF_BASE_PORT MASTER_PORT=$MASTER_PORT VERL_HCCL_IF_BASE_PORT_START=$VERL_HCCL_IF_BASE_PORT_START"
echo "[mode4 bulk request test] runtime_cwd=$PATCH_TREE"
echo "[mode4 bulk request test] PYTHONPATH=$PYTHONPATH"

bash "$LAUNCHER" 2>&1 | tee "$tee_log"
