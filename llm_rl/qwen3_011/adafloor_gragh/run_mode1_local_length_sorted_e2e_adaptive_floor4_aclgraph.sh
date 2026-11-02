#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

GRAPH_MODE="${ADAFLOOR_GRAPH_MODE:-elastic_aclgraph}"
ACLGRAPH_MODE="${ADAFLOOR_ACLGRAPH_MODE:-FULL_DECODE_ONLY}"
CAPTURE_PROFILE="${ADAFLOOR_GRAPH_CAPTURE_PROFILE:-balanced}"
BASE_RUNNER="${ADAFLOOR_GRAPH_BASE_RUNNER:-$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh}"
ASCEND_EXTENSION="${VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION:-/workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so}"

if [[ -n "${ADAFLOOR_GRAPH_CAPTURE_SIZES:-}" ]]; then
    CAPTURE_SIZES="$ADAFLOOR_GRAPH_CAPTURE_SIZES"
    CAPTURE_PROFILE=custom
else
    case "$CAPTURE_PROFILE" in
        memory_saver)
            CAPTURE_SIZES='[1,2,4,8]'
            ;;
        balanced)
            CAPTURE_SIZES='[1,2,4,8,16,32]'
            ;;
        full_coverage)
            CAPTURE_SIZES='[1,2,4,8,16,32,64]'
            ;;
        *)
            echo "unsupported ADAFLOOR_GRAPH_CAPTURE_PROFILE=$CAPTURE_PROFILE; expected memory_saver, balanced, or full_coverage" >&2
            exit 2
            ;;
    esac
fi

if [[ "$GRAPH_MODE" != "elastic_aclgraph" ]]; then
    echo "unsupported ADAFLOOR_GRAPH_MODE=$GRAPH_MODE; expected elastic_aclgraph" >&2
    exit 2
fi
if [[ "$ACLGRAPH_MODE" != "FULL_DECODE_ONLY" \
        && "$ACLGRAPH_MODE" != "PIECEWISE" ]]; then
    echo "unsupported ADAFLOOR_ACLGRAPH_MODE=$ACLGRAPH_MODE; expected FULL_DECODE_ONLY or PIECEWISE" >&2
    exit 2
fi

if [[ "$BASE_RUNNER" != /* ]]; then
    BASE_RUNNER="$SCRIPT_DIR/$BASE_RUNNER"
fi
if [[ ! -f "$BASE_RUNNER" ]]; then
    echo "missing AdaFloor base runner: $BASE_RUNNER" >&2
    exit 2
fi
if [[ ! -f "$ASCEND_EXTENSION" ]]; then
    echo "missing PyTorch-compatible vLLM Ascend extension: $ASCEND_EXTENSION" >&2
    echo "set VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION to a compatible vllm_ascend_C shared library" >&2
    exit 2
fi

if [[ ! "$CAPTURE_SIZES" =~ ^\[[[:space:]]*[1-9][0-9]*([[:space:]]*,[[:space:]]*[1-9][0-9]*)*[[:space:]]*\]$ ]]; then
    echo "ADAFLOOR_GRAPH_CAPTURE_SIZES must be a non-empty list of positive integers: $CAPTURE_SIZES" >&2
    exit 2
fi

export VLLM_ASCEND_ELASTIC_ACLGRAPH="${VLLM_ASCEND_ELASTIC_ACLGRAPH:-1}"
export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION="${VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION:-1}"
export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE="${VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE:-1}"
export VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION="$(readlink -f "$ASCEND_EXTENSION")"
export VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256="$(sha256sum "$VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION" | awk '{print $1}')"
export VLLM_ENABLE_GRAPH_MODE="${VLLM_ENABLE_GRAPH_MODE:-0}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export VLLM_ASCEND_ELASTIC_ACLGRAPH_ALLOW_TASK_QUEUE_2="${VLLM_ASCEND_ELASTIC_ACLGRAPH_ALLOW_TASK_QUEUE_2:-0}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"
export VERL_SIDECAR_ENABLE="${VERL_SIDECAR_ENABLE:-0}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-12000}"
export VERL_MASTER_PORT_START="${VERL_MASTER_PORT_START:-28416}"

if [[ "$VLLM_ASCEND_ELASTIC_ACLGRAPH" != "1" ]]; then
    echo "elastic_aclgraph requires VLLM_ASCEND_ELASTIC_ACLGRAPH=1" >&2
    exit 2
fi
if [[ "$VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION" != "0" \
        && "$VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION" != "1" ]]; then
    echo "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION must be 0 or 1" >&2
    exit 2
fi
if [[ "$VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION" == "1" \
        && "$ACLGRAPH_MODE" != "FULL_DECODE_ONLY" ]]; then
    echo "Attention capture requires ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY" >&2
    exit 2
fi
if [[ "$VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE" != "0" \
        && "$VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE" != "1" ]]; then
    echo "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE must be 0 or 1" >&2
    exit 2
fi
if [[ "$VLLM_ENABLE_GRAPH_MODE" != "0" ]]; then
    echo "elastic_aclgraph requires VLLM_ENABLE_GRAPH_MODE=0 (TorchAir disabled)" >&2
    exit 2
fi
if [[ "$TASK_QUEUE_ENABLE" != "1" \
        && ! ( "$TASK_QUEUE_ENABLE" == "2" \
            && "$VLLM_ASCEND_ELASTIC_ACLGRAPH_ALLOW_TASK_QUEUE_2" == "1" ) ]]; then
    echo "elastic_aclgraph requires TASK_QUEUE_ENABLE=1; TQ2 is allowed only with VLLM_ASCEND_ELASTIC_ACLGRAPH_ALLOW_TASK_QUEUE_2=1" >&2
    exit 2
fi
if [[ "${ROLLOUT_ENFORCE_EAGER,,}" != "false" ]]; then
    echo "elastic_aclgraph requires ROLLOUT_ENFORCE_EAGER=False" >&2
    exit 2
fi
if [[ "$VERL_SIDECAR_ENABLE" != "0" ]]; then
    echo "elastic_aclgraph does not support VERL_SIDECAR_ENABLE=1" >&2
    exit 2
fi

for cache_var in \
    XDG_CACHE_HOME HF_HOME TRITON_CACHE_DIR TORCHAIR_CACHE_HOME \
    ASCEND_CACHE_PATH ASCEND_WORK_PATH; do
    cache_dir="${!cache_var:-}"
    if [[ -z "$cache_dir" ]]; then
        continue
    fi
    if ! mkdir -p "$cache_dir" || [[ ! -d "$cache_dir" || ! -w "$cache_dir" ]]; then
        echo "$cache_var must name a writable cache directory: $cache_dir" >&2
        exit 2
    fi
done

echo "[AdaFloor ACLGraph] mode=$GRAPH_MODE cudagraph_mode=$ACLGRAPH_MODE capture_profile=$CAPTURE_PROFILE capture_sizes=$CAPTURE_SIZES attention_capture=$VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION moe_capture=$VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE task_queue=$TASK_QUEUE_ENABLE tq2_diagnostic=$VLLM_ASCEND_ELASTIC_ACLGRAPH_ALLOW_TASK_QUEUE_2 runner=$BASE_RUNNER"
echo "[AdaFloor ACLGraph] ascend_extension=$VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION sha256=$VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256"

exec bash "$BASE_RUNNER" \
    "actor_rollout_ref.rollout.cudagraph_mode=$ACLGRAPH_MODE" \
    "actor_rollout_ref.rollout.cudagraph_capture_sizes=$CAPTURE_SIZES" \
    actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
    "$@"
