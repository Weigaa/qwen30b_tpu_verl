#!/usr/bin/env bash
set -euo pipefail

# Sweep PanguProMoE-72B-A16B..A23B sidecar layouts on the 8 released NPUs.
# Each run launches the normal training+sidecar script once. Logs are separated
# by VERL_SIDECAR_RUN_TAG / state dir so results do not overwrite each other.
#
# Usage examples:
#   bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager_sidepangu_sweep.sh
#   START_INDEX=4 bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager_sidepangu_sweep.sh
#   ONLY_INDEX=6 bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager_sidepangu_sweep.sh
#
# Optional overrides:
#   VERL_SIDECAR_MODEL_PATH=/home/sharedata/models/pangu-pro-moe-model
#   VERL_SIDECAR_MAX_MODEL_LEN=16384
#   VERL_SIDECAR_MAX_TOKENS=1024

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="${SCRIPT_DIR}/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager_sidepangu.sh"

# Default to the 16k Pangu setting discussed for these experiments.
export VERL_SIDECAR_MAX_MODEL_LEN=${VERL_SIDECAR_MAX_MODEL_LEN:-16384}
export VERL_SIDECAR_MAX_TOKENS=${VERL_SIDECAR_MAX_TOKENS:-1024}
export VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=${VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS:-32768}
export VERL_SIDECAR_PROMPTS_FILE=${VERL_SIDECAR_PROMPTS_FILE:-"/home/qiuzy/verl_dev/data/gsm8k"}

# Keep the per-replica batch conservative for the larger MoE. Override per run
# from the shell if you want to push utilization after the layout is stable.
export VERL_SIDECAR_MAX_NUM_SEQS=${VERL_SIDECAR_MAX_NUM_SEQS:-4}
export VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA=${VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA:-4}
export VERL_SIDECAR_GENERATE_CHUNK_SIZE=${VERL_SIDECAR_GENERATE_CHUNK_SIZE:-4}

# Format: model_label data_parallel_size tensor_parallel_size ep_on
# DP here is vLLM internal data_parallel_size, not independent sidecar replicas.
CASES=(
  "PanguProMoE-72B-A16B 8 1 0"
  "PanguProMoE-72B-A17B 4 2 0"
  "PanguProMoE-72B-A18B 2 4 0"
  "PanguProMoE-72B-A19B 1 8 0"
  "PanguProMoE-72B-A20B 8 1 1"
  "PanguProMoE-72B-A21B 4 2 1"
  "PanguProMoE-72B-A22B 2 4 1"
  "PanguProMoE-72B-A23B 1 8 1"
)

START_INDEX=${START_INDEX:-0}
ONLY_INDEX=${ONLY_INDEX:-}

for index in "${!CASES[@]}"; do
    if [[ -n "${ONLY_INDEX}" && "${index}" != "${ONLY_INDEX}" ]]; then
        continue
    fi
    if [[ -z "${ONLY_INDEX}" && "${index}" -lt "${START_INDEX}" ]]; then
        continue
    fi

    read -r model_label dp_size tp_size ep_on <<< "${CASES[$index]}"
    run_tag=$(printf "%02d_%s_dp%s_tp%s_ep%s" \
        "${index}" "${model_label}" "${dp_size}" "${tp_size}" "${ep_on}")

    export VERL_SIDECAR_RUN_TAG="${run_tag}"
    export VERL_SIDECAR_MODEL_PATH=${VERL_SIDECAR_MODEL_PATH:-"/home/sharedata/models/pangu-pro-moe-model"}
    export VERL_SIDECAR_REPLICA_COUNT=1
    export VERL_SIDECAR_DATA_PARALLEL_SIZE="${dp_size}"
    export VERL_SIDECAR_TENSOR_PARALLEL_SIZE="${tp_size}"
    export VERL_SIDECAR_ENABLE_EXPERT_PARALLEL="${ep_on}"
    export VERL_SIDECAR_MODEL_IS_MOE=1
    export VERL_SIDECAR_STATE_DIR=${VERL_SIDECAR_STATE_DIR:-"sidecar_runs/state/${run_tag}_gsm8k_train"}

    echo "[pangu sweep] index=${index} tag=${run_tag} model=${VERL_SIDECAR_MODEL_PATH} dp=${dp_size} tp=${tp_size} ep=${ep_on} max_len=${VERL_SIDECAR_MAX_MODEL_LEN} max_tokens=${VERL_SIDECAR_MAX_TOKENS}"
    bash "${BASE_SCRIPT}"

    # Reset overridable per-case envs so the next case can choose its own path/tag.
    unset VERL_SIDECAR_MODEL_PATH
    unset VERL_SIDECAR_STATE_DIR
    unset VERL_SIDECAR_DATA_PARALLEL_SIZE
    unset VERL_SIDECAR_LOG_DIR
    unset VERL_SIDECAR_LEASE_LOG
    unset VERL_SIDECAR_LOG_FILE
    unset VERL_SIDECAR_OUTPUT_FILE
    unset VERL_SIDECAR_MONITOR_LOG
    unset VERL_SIDECAR_TRAIN_LOG
    unset VERL_SIDECAR_RUN_TAG

done
