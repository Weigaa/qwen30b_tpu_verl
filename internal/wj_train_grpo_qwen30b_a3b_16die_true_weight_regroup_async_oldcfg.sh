#!/usr/bin/env bash
# Backward-compatible entrypoint kept for older notes/commands.
#
# Use the explicit scripts for new runs:
#   - eager: internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_eager_fast.sh
#   - graph: internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_graph_fast.sh
set -ex

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "${VLLM_ROLLOUT_MODE:-}" in
    graph)
        exec bash "${SCRIPT_DIR}/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_graph_fast.sh" "$@"
        ;;
    eager|"")
        if [ "${VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER:-0}" = "1" ] || \
           [ "${VLLM_ENABLE_GRAPH_MODE:-0}" = "1" ]; then
            exec bash "${SCRIPT_DIR}/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_graph_fast.sh" "$@"
        fi
        exec bash "${SCRIPT_DIR}/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_eager_fast.sh" "$@"
        ;;
    *)
        echo "Invalid VLLM_ROLLOUT_MODE=${VLLM_ROLLOUT_MODE}; expected eager or graph" >&2
        exit 2
        ;;
esac
