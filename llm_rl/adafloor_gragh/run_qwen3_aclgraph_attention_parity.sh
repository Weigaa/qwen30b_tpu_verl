#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE="${1:-}"
if [[ "$MODE" != "eager" && "$MODE" != "graph" ]]; then
    echo "usage: $0 eager|graph" >&2
    exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/adafloor_shared_state/qwen3_aclgraph_attention_piecewise_parity_20260812}"
RUN_ROOT="$OUTPUT_ROOT/$MODE"
if [[ -e "$RUN_ROOT" ]]; then
    echo "refusing to reuse parity output: $RUN_ROOT" >&2
    exit 2
fi

export OUTPUT_ROOT="$RUN_ROOT"
export OUTPUT_SUBDIR=run
export BATCH_SIZES=16
export ROLLOUT_N=1
export DECODE_TOKENS="${DECODE_TOKENS:-64}"
export MAX_PROMPT_LENGTH=512
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=1024
export VERL_ROLLOUT_BENCH_WARMUP_STEPS="${VERL_ROLLOUT_BENCH_WARMUP_STEPS:-0}"
export VERL_ROLLOUT_BENCH_MEASURE_STEPS="${VERL_ROLLOUT_BENCH_MEASURE_STEPS:-1}"
export DATASET_FRACTION=1.0
export DATA_SHUFFLE=False
export TASK_QUEUE_ENABLE=1
export VLLM_ENABLE_GRAPH_MODE=0
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-12000}"
export VERL_MASTER_PORT_START="${VERL_MASTER_PORT_START:-28416}"

extra_args=(
    actor_rollout_ref.rollout.temperature=0.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.ignore_eos=True
    actor_rollout_ref.rollout.seed=101
    actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
)

if [[ "$MODE" == "graph" ]]; then
    extension="${VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION:-/workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so}"
    if [[ ! -f "$extension" ]]; then
        echo "missing PyTorch-compatible vllm_ascend extension: $extension" >&2
        exit 2
    fi
    export ROLLOUT_ENFORCE_EAGER=False
    export VLLM_ASCEND_ELASTIC_ACLGRAPH=1
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION="${VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION:-1}"
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=0
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_ALLOW_TASK_QUEUE_2=0
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION="$(readlink -f "$extension")"
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256="$(sha256sum "$extension" | awk '{print $1}')"
    extra_args+=("actor_rollout_ref.rollout.cudagraph_capture_sizes=[1]")
else
    export ROLLOUT_ENFORCE_EAGER=True
    export VLLM_ASCEND_ELASTIC_ACLGRAPH=0
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=0
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=0
fi

mkdir -p "$RUN_ROOT"
{
    echo "schema_version=1"
    echo "mode=$MODE"
    echo "stack=vllm-0.11.0_vllm-ascend-0.11.0rc0"
    echo "task_queue_enable=1"
    echo "batch_size=16"
    echo "rollout_n=1"
    echo "decode_tokens=${DECODE_TOKENS}"
    echo "temperature=0.0"
    echo "seed=101"
    echo "warmup_steps=$VERL_ROLLOUT_BENCH_WARMUP_STEPS"
    echo "measure_steps=$VERL_ROLLOUT_BENCH_MEASURE_STEPS"
    if [[ "$MODE" == "graph" && "$VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION" == "1" ]]; then
        echo "attention_graph=true"
    else
        echo "attention_graph=false"
    fi
    echo "moe_graph=false"
    echo "cudagraph_mode=$([[ "$MODE" == graph ]] && echo PIECEWISE || echo NONE)"
    echo "cudagraph_copy_inputs=$([[ "$MODE" == graph ]] && echo true || echo false)"
} > "$RUN_ROOT/protocol.env"

contract_files=(
    "$SCRIPT_DIR/run_qwen3_aclgraph_attention_parity.sh"
    "$SCRIPT_DIR/run_rollout_decode_batchsize_benchmark.sh"
    "$SCRIPT_DIR/verl/trainer/ppo/ray_trainer.py"
    "$SCRIPT_DIR/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    "$SCRIPT_DIR/vllm_ascend/attention/attention_v1.py"
    "$SCRIPT_DIR/vllm_ascend/compilation/acl_graph.py"
    "$SCRIPT_DIR/vllm_ascend/platform.py"
    "$SCRIPT_DIR/vllm_ascend/worker/model_runner_v1.py"
)
sha256sum "${contract_files[@]}" > "$RUN_ROOT/code_sha256.txt"

"$SCRIPT_DIR/run_rollout_decode_batchsize_benchmark.sh" "${extra_args[@]}"
summary="$RUN_ROOT/run/summary_batch_16.json"
if [[ ! -s "$summary" ]]; then
    echo "parity arm did not produce its required summary: $summary" >&2
    exit 1
fi
