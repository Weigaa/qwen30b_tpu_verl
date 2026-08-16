#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POLICY=${1:-}
if [[ "$POLICY" != "planned" && "$POLICY" != "natural" ]]; then
    echo "usage: $0 planned|natural [extra hydra args...]" >&2
    exit 2
fi
shift

EXECUTION_MODE=${ADAFLOOR_DYNAMIC_GATE_EXECUTION_MODE:-graph}
if [[ "$EXECUTION_MODE" != "graph" && "$EXECUTION_MODE" != "eager" ]]; then
    echo "ADAFLOOR_DYNAMIC_GATE_EXECUTION_MODE must be graph or eager" >&2
    exit 2
fi

GATE_PLAN_STEPS=${ADAFLOOR_DYNAMIC_GATE_PLAN_STEPS:-2}
GATE_EXECUTED_STEPS=${ADAFLOOR_DYNAMIC_GATE_EXECUTED_STEPS:-$GATE_PLAN_STEPS}
if [[ "$GATE_PLAN_STEPS" != "1" && "$GATE_PLAN_STEPS" != "2" ]]; then
    echo "ADAFLOOR_DYNAMIC_GATE_PLAN_STEPS must be 1 or 2" >&2
    exit 2
fi
if [[ "$GATE_EXECUTED_STEPS" != "1" && "$GATE_EXECUTED_STEPS" != "2" ]]; then
    echo "ADAFLOOR_DYNAMIC_GATE_EXECUTED_STEPS must be 1 or 2" >&2
    exit 2
fi
if (( GATE_EXECUTED_STEPS > GATE_PLAN_STEPS )); then
    echo "ADAFLOOR_DYNAMIC_GATE_EXECUTED_STEPS cannot exceed the plan steps" >&2
    exit 2
fi

GATE_FLOOR=${ADAFLOOR_DYNAMIC_GATE_FLOOR:-4}
if [[ "$GATE_FLOOR" != "2" && "$GATE_FLOOR" != "4" ]]; then
    echo "ADAFLOOR_DYNAMIC_GATE_FLOOR must be 2 or 4" >&2
    exit 2
fi
if [[ "$GATE_FLOOR" == "2" ]]; then
    GRAPH_BASE_RUNNER="$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2.sh"
    # The paper eager F2 cap is 131072 tokens. Reserve graph headroom by
    # reducing the diagnostic graph F2 physical cap to 98304 tokens. The
    # short gate's work remains far below either cap and the real 640-token
    # scheduler limit is unchanged.
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS:-380800}"
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2:-98304}"
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4:-280576}"
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8:-315648}"
    # Match the proven eager Natural-F2 bootstrap contract. Profiling 2048
    # tokens selects AllToAll on A3, so the large fixed MC2 allocation is
    # deferred until actor offload, step KV resize, and graph capture.
    export VLLM_ASCEND_MODE1_PROFILE_MAX_TOKENS="${VLLM_ASCEND_MODE1_PROFILE_MAX_TOKENS:-2048}"
    export VLLM_ASCEND_MODE1_PROFILE_AVOID_MC2_BOUNDARY="${VLLM_ASCEND_MODE1_PROFILE_AVOID_MC2_BOUNDARY:-1}"
    export VLLM_ASCEND_MODE1_PROFILE_EXPECT_MOE_COMM="${VLLM_ASCEND_MODE1_PROFILE_EXPECT_MOE_COMM:-alltoall}"
    export VLLM_ASCEND_MODE1_PROFILE_EXTRA_MC2_DUMMY="${VLLM_ASCEND_MODE1_PROFILE_EXTRA_MC2_DUMMY:-0}"
    export VLLM_ASCEND_MODE1_PREWARM_FULLWORLD_ALLTOALLV="${VLLM_ASCEND_MODE1_PREWARM_FULLWORLD_ALLTOALLV:-1}"
    export VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC="${VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC:-1}"
    export VLLM_ASCEND_MODE1_COLD_INIT_KV_TOKENS="${VLLM_ASCEND_MODE1_COLD_INIT_KV_TOKENS:-2048}"
    export VLLM_ASCEND_MODE1_USE_COLD_INIT_KV_CAP="${VLLM_ASCEND_MODE1_USE_COLD_INIT_KV_CAP:-1}"
    export VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES="${VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES:-3221225472}"
else
    GRAPH_BASE_RUNNER="$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"
fi

STAMP=${ADAFLOOR_DYNAMIC_GATE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
ROOT=${ADAFLOOR_DYNAMIC_GATE_ROOT:-/data/adafloor_shared_state/qwen3_adafloor_full_decode_dynamic_gate_${STAMP}_${POLICY}}
BASELINE=${ADAFLOOR_DYNAMIC_GATE_BASELINE:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe}
EXTENSION=${VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION:-/workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so}
PLAN_ONLY=${ADAFLOOR_DYNAMIC_GATE_PLAN_ONLY:-0}

required_paths=("$BASELINE/offline_planning_history.json")
if [[ "$EXECUTION_MODE" == "graph" ]]; then
    required_paths+=("$EXTENSION")
fi
for path in "${required_paths[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "missing dynamic-gate input: $path" >&2
        exit 2
    fi
done
if [[ -e "$ROOT" ]]; then
    echo "refusing to reuse dynamic-gate root: $ROOT" >&2
    exit 2
fi
if [[ "$PLAN_ONLY" != "0" && "$PLAN_ONLY" != "1" ]]; then
    echo "ADAFLOOR_DYNAMIC_GATE_PLAN_ONLY must be 0 or 1" >&2
    exit 2
fi
if [[ "$PLAN_ONLY" == "0" ]] \
        && pgrep -f '[p]ython3 -m verl\.trainer\.main_ppo|[r]ay::TaskRunner\.run' >/dev/null; then
    echo "another VERL training process is active" >&2
    exit 3
fi

mkdir -p "$ROOT" "$ROOT/cache" "$ROOT/work"
export OUTPUT_ROOT="$ROOT"
export OUTPUT_SUBDIR=run
export PLAN_DIR="$ROOT/oracle"
export BASELINE_DIRS="$BASELINE"
export TRAIN_BATCH_SIZE=32
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=128
export MAX_RESPONSE_LEN=128
export ROLLOUT_N=1
if [[ "$GATE_FLOOR" == "2" ]]; then
    # Match the paper eager F2 scheduler contract. profile_run independently
    # caps the synthetic forward at 2048 tokens, so this does not increase the
    # profile workload or move it onto MC2.
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS=17408
    export ROLLOUT_MAX_MODEL_LEN=17408
else
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS=640
    export ROLLOUT_MAX_MODEL_LEN=640
fi
export ROLLOUT_MAX_NUM_SEQS=32
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.9
export TRAINER_TOTAL_EPOCHS=1
export PLAN_STEPS="$GATE_PLAN_STEPS"
export DATASET_FRACTION_FOR_ORACLE=0.005
if [[ "$GATE_PLAN_STEPS" == "1" ]]; then
    export FORCE_SELECTED_FLOORS="$GATE_FLOOR"
else
    export FORCE_SELECTED_FLOORS="$GATE_FLOOR,16"
fi
export IGNORE_TAIL_TIES_AT_RESPONSE_CAP=1
export EXPECT_NO_RESPONSE_CAPS=1
export VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE=0
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY="$POLICY"
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=8,16,32,64,64
export VERL_SIDECAR_ENABLE=0
export SAVE_CKPT_ENABLE=0
export MODE1_PLAN_ONLY="$PLAN_ONLY"
export ADAFLOOR_GRAPH_BASE_RUNNER="$GRAPH_BASE_RUNNER"

export VLLM_ENABLE_GRAPH_MODE=0
if [[ "$EXECUTION_MODE" == "graph" ]]; then
    GATE_CAPTURE_SIZES=${ADAFLOOR_DYNAMIC_GATE_CAPTURE_SIZES:-'[2]'}
    if [[ ! "$GATE_CAPTURE_SIZES" =~ ^\[[0-9]+(,[0-9]+)*\]$ ]]; then
        echo "ADAFLOOR_DYNAMIC_GATE_CAPTURE_SIZES must be a compact integer list" >&2
        exit 2
    fi
    export ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY
    export ADAFLOOR_GRAPH_CAPTURE_SIZES="$GATE_CAPTURE_SIZES"
    export VLLM_ASCEND_ELASTIC_ACLGRAPH=1
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1
    export ROLLOUT_ENFORCE_EAGER=False
    export TASK_QUEUE_ENABLE=1
    EXECUTION_RUNNER="$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh"
else
    export VLLM_ASCEND_ELASTIC_ACLGRAPH=0
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=0
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=0
    export ROLLOUT_ENFORCE_EAGER=True
    export TASK_QUEUE_ENABLE=${ADAFLOOR_DYNAMIC_GATE_EAGER_TASK_QUEUE_ENABLE:-2}
    if [[ "$TASK_QUEUE_ENABLE" != "1" && "$TASK_QUEUE_ENABLE" != "2" ]]; then
        echo "ADAFLOOR_DYNAMIC_GATE_EAGER_TASK_QUEUE_ENABLE must be 1 or 2" >&2
        exit 2
    fi
    EXECUTION_RUNNER="$GRAPH_BASE_RUNNER"
fi
export VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG=1
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp23s0f3}
export VERL_HCCL_IF_BASE_PORT_START=${VERL_HCCL_IF_BASE_PORT_START:-12000}
export VERL_MASTER_PORT_START=${VERL_MASTER_PORT_START:-28416}
export VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION="$(readlink -f "$EXTENSION")"
export VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256="$(sha256sum "$EXTENSION" | awk '{print $1}')"

export XDG_CACHE_HOME="$ROOT/cache/xdg"
export HF_HOME="$ROOT/cache/huggingface"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
export TORCHAIR_CACHE_HOME="$ROOT/cache/torchair"
export ASCEND_CACHE_PATH="$ROOT/cache/ascend"
export ASCEND_WORK_PATH="$ROOT/work/ascend"
mkdir -p \
    "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" \
    "$TORCHAIR_CACHE_HOME" "$ASCEND_CACHE_PATH" "$ASCEND_WORK_PATH"

{
    echo "schema_version=1"
    echo "experiment=qwen3_adafloor_full_decode_dynamic_gate"
    echo "policy=$POLICY"
    echo "execution_mode=$EXECUTION_MODE"
    echo "stack=vllm-0.11.0_vllm-ascend-0.11.0rc0_cann-8.5.0"
    echo "cudagraph_mode=$([[ "$EXECUTION_MODE" == graph ]] && echo FULL_DECODE_ONLY || echo NONE)"
    echo "capture_sizes=$([[ "$EXECUTION_MODE" == graph ]] && echo "${GATE_CAPTURE_SIZES:1:${#GATE_CAPTURE_SIZES}-2}" || echo none)"
    echo "attention_graph=$([[ "$EXECUTION_MODE" == graph ]] && echo true || echo false)"
    echo "moe_graph=$([[ "$EXECUTION_MODE" == graph ]] && echo true || echo false)"
    echo "task_queue_enable=$TASK_QUEUE_ENABLE"
    echo "plan_steps=$GATE_PLAN_STEPS"
    echo "executed_steps=$GATE_EXECUTED_STEPS"
    echo "forced_floors=$FORCE_SELECTED_FLOORS"
    echo "profile_max_tokens=${VLLM_ASCEND_MODE1_PROFILE_MAX_TOKENS:-unset}"
    echo "profile_avoid_mc2_boundary=${VLLM_ASCEND_MODE1_PROFILE_AVOID_MC2_BOUNDARY:-unset}"
    echo "profile_expected_moe_comm=${VLLM_ASCEND_MODE1_PROFILE_EXPECT_MOE_COMM:-unset}"
    echo "profile_extra_mc2_dummy=${VLLM_ASCEND_MODE1_PROFILE_EXTRA_MC2_DUMMY:-unset}"
    echo "preweight_fullworld_alltoallv=${VLLM_ASCEND_MODE1_PREWARM_FULLWORLD_ALLTOALLV:-unset}"
    echo "cpu_dp_metadata_sync=${VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC:-unset}"
    echo "cold_init_kv_tokens=${VLLM_ASCEND_MODE1_COLD_INIT_KV_TOKENS:-unset}"
    echo "use_cold_init_kv_cap=${VLLM_ASCEND_MODE1_USE_COLD_INIT_KV_CAP:-unset}"
    echo "max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
    echo "max_model_len=$ROLLOUT_MAX_MODEL_LEN"
    echo "mc2_materialization_phase=post_actor_offload_after_step_kv_resize"
    echo "profile_memory_diagnostics=true"
    echo "kv_tokens_floor2=${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2:-unset}"
    echo "kv_tokens_floor4=${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4:-unset}"
    echo "kv_tokens_floor8=${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8:-unset}"
    echo "kv_tokens_floor16=${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS:-unset}"
    echo "tail_validation_tokens=8,16,32,64,64"
    echo "attention_workspace_bucket_is_not_max_model_len=true"
    echo "baseline=$BASELINE"
    echo "plan_only=$PLAN_ONLY"
} > "$ROOT/protocol.env"

contract_files=(
    "$SCRIPT_DIR/run_qwen3_adafloor_full_decode_dynamic_gate.sh"
    "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh"
    "$GRAPH_BASE_RUNNER"
    "$SCRIPT_DIR/vllm/compilation/decorators.py"
    "$SCRIPT_DIR/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    "$SCRIPT_DIR/vllm_ascend/attention/attention_v1.py"
    "$SCRIPT_DIR/vllm_ascend/compilation/acl_graph.py"
    "$SCRIPT_DIR/vllm_ascend/platform.py"
    "$SCRIPT_DIR/vllm_ascend/worker/model_runner_v1.py"
    "$SCRIPT_DIR/vllm_ascend/worker/worker_v1.py"
    "$SCRIPT_DIR/tools/verify_adafloor_aclgraph_smoke.py"
)
sha256sum "${contract_files[@]}" > "$ROOT/code_sha256.txt"

if [[ "$PLAN_ONLY" == "0" ]]; then
    ray stop --force >/dev/null 2>&1 || true
fi

"$EXECUTION_RUNNER" \
    trainer.total_training_steps="$GATE_EXECUTED_STEPS" \
    "trainer.rollout_data_dir=$ROOT/rollout_data" \
    "trainer.rollout_length_dir=$ROOT/rollout_length" \
    'trainer.logger=["console"]' \
    actor_rollout_ref.actor.optim.lr=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.ignore_eos=True \
    actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.seed=101 \
    "$@"

if [[ "$PLAN_ONLY" == "1" ]]; then
    echo "[dynamic gate] plan-only complete mode=$EXECUTION_MODE root=$ROOT"
    exit 0
fi

if [[ "$EXECUTION_MODE" == "graph" ]]; then
    python3 "$SCRIPT_DIR/tools/verify_adafloor_aclgraph_smoke.py" --root "$ROOT"
    echo "[dynamic FULL_DECODE_ONLY gate] PASS policy=$POLICY root=$ROOT"
else
    echo "[dynamic eager alignment gate] run complete policy=$POLICY root=$ROOT"
fi
