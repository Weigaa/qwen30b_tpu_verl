#!/usr/bin/env bash
set -euo pipefail

# One-step v0.14 Natural-F2 correctness check:
# eager + native sleep(level=2) + KV-first + no ACLGraph.
# It reuses the public common epoch0 checkpoint and does not run graph capture.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-dry-run}
case "$ACTION" in
    dry-run|run) ;;
    -h|--help)
        echo "Usage: $0 {dry-run|run}"
        exit 0
        ;;
    *) echo "Usage: $0 {dry-run|run}" >&2; exit 2 ;;
esac

RESULT_ROOT=${EAGER_NATURAL_F2_ROOT:-/workspace/adafloor_graph_results/qwen3_v014_eager_sleep2_natural_f2_correctness}
RUN_NAME=${EAGER_NATURAL_F2_RUN_NAME:-qwen3_eager_sleep2_natural_f2_seed0}
COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}

export COMMON_EPOCH0_ROOT
export FAIR_OUTPUT_ROOT="$RESULT_ROOT"
export FAIR_RUN_NAME="$RUN_NAME"
export DYNAMIC_RUN_NAME="$RUN_NAME"
export FAIR_START_EPOCH=1
export FAIR_TOTAL_EPOCHS=2
export FAIR_TRAIN_STEPS=1
export FAIR_TRAIN_BATCH_SIZE=32
export FAIR_ROLLOUT_N=16
export FAIR_MAX_RESPONSE_LENGTH=16384
export FAIR_PROMPTS_PER_EPOCH=160
export FAIR_FREEZE_ACTOR=0
export FAIR_KEEP_COMPLETED_CHECKPOINTS=0
export VERL_PAIRED_REQUEST_SAMPLING_SEEDS=0

# Eager correctness path.  The graph lifecycle flag remains enabled so this
# exercises the same sleep2 boundary, but model_runner must not touch ACLGraph
# pools when enforce_eager=True.
export ROLLOUT_ENFORCE_EAGER=True
export VLLM_ROLLOUT_TASK_QUEUE_ENABLE=${VLLM_ROLLOUT_TASK_QUEUE_ENABLE:-2}
export TASK_QUEUE_ENABLE="$VLLM_ROLLOUT_TASK_QUEUE_ENABLE"
# Match the validated v0.14 eager baseline's CANN norm path. The filtered
# custom-op bundle does not provide a usable AddRmsNormBias binary here.
export VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1
export VLLM_ENABLE_GRAPH_MODE=0
export VLLM_ASCEND_ELASTIC_ACLGRAPH=0
export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=0
export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=0
export VLLM_ROLLOUT_NATIVE_SLEEP_MODE=1
export VLLM_ROLLOUT_SLEEP_LEVEL=2
export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE=1
export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_RETAIN=0
export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_KV_FIRST=1
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true
export VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true
export VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=1
export VLLM_QWEN3_MOE_REDUCE_RESULTS=1
export VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1
export VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH="$ROOT/vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend"

if [[ "$ACTION" == dry-run ]]; then
    cat <<EOF
mode=eager
sleep_level=$VLLM_ROLLOUT_SLEEP_LEVEL lifecycle=$VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE kv_first=$VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_KV_FIRST
graph=disabled tq=$TASK_QUEUE_ENABLE
common_epoch0=$COMMON_EPOCH0_ROOT
output=$RESULT_ROOT/$RUN_NAME
workload=batch32_n16_prompts160_steps1
EOF
    exit 0
fi

mkdir -p "$RESULT_ROOT"
exec "$ROOT/run_paper_fair_epoch1_2_from_common_epoch0.sh" adafloor_n_f2
