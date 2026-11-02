#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
export ZSH_VERSION=${ZSH_VERSION:-}
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
EP_SIZE=${DEEPSEEK_SEMANTIC_SMOKE_EP_SIZE:-1}
if [[ "$EP_SIZE" != 1 && "$EP_SIZE" != 16 ]]; then
    echo "DeepSeek semantic smoke supports EP size 1 or 16" >&2
    exit 2
fi
if [[ "$EP_SIZE" == 16 ]]; then
    default_devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
else
    default_devices=0
fi
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$default_devices}
export VLLM_USE_V1=1
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1
export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=0
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=0
if (( EP_SIZE > 1 )); then
    export VLLM_ENABLE_EXPERT_PARALLEL=1
else
    export VLLM_ENABLE_EXPERT_PARALLEL=0
fi
export VLLM_DP_SIZE=$EP_SIZE
export VLLM_ENABLE_MC2=0
export VLLM_MOE_PATTERN_STATS=0
export VLLM_MOE_STATS=0
export VLLM_MOE_STATS_TIMING=0
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

MODEL_PATH=${MODEL_PATH:-/data/DeepSeek-V2-Lite-Chat}
OUTPUT=${DEEPSEEK_SEMANTIC_SMOKE_OUTPUT:-/data/adafloor_shared_state/deepseek_v2_lite/semantic_smoke_chat.json}

exec python3 "$SCRIPT_DIR/tools/smoke_deepseek_v2_lite_vllm.py" \
    --model "$MODEL_PATH" \
    --max-tokens "${DEEPSEEK_SEMANTIC_SMOKE_MAX_TOKENS:-1024}" \
    --max-model-len "${DEEPSEEK_SEMANTIC_SMOKE_MAX_MODEL_LEN:-2048}" \
    --gpu-memory-utilization "${DEEPSEEK_SEMANTIC_SMOKE_GPU_MEMORY_UTILIZATION:-0.80}" \
    --expert-parallel-size "$EP_SIZE" \
    --requests-per-rank "${DEEPSEEK_SEMANTIC_SMOKE_REQUESTS_PER_RANK:-2}" \
    --chat-only \
    --require-stop \
    --require-answer \
    --reject-dialogue-continuation \
    --output "$OUTPUT" \
    "$@"
