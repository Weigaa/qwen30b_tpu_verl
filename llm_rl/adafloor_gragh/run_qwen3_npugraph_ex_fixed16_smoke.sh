#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_TREE="$SCRIPT_DIR/npugraph_ex_runtime"
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
TASK_QUEUE_MODE=${ADAFLOOR_NPUGRAPH_TASK_QUEUE_ENABLE:-1}
CAPTURE_SIZES=${ADAFLOOR_NPUGRAPH_CAPTURE_SIZES:-'[1,2,4,8,16,32]'}
GPU_MEMORY_UTILIZATION=${ADAFLOOR_NPUGRAPH_GPU_MEMORY_UTILIZATION:-0.9}
MAX_NUM_BATCHED_TOKENS=${ADAFLOOR_NPUGRAPH_MAX_NUM_BATCHED_TOKENS:-17408}
ROOT=${ADAFLOOR_NPUGRAPH_SMOKE_ROOT:-/data/adafloor_shared_state/qwen3_npugraph_ex_fixed16_tq${TASK_QUEUE_MODE}_smoke_${STAMP}}
RUN_NAME=npugraph_ex_fixed16_tq${TASK_QUEUE_MODE}_seed101_source_step1
EPOCH_DIR="$ROOT/$RUN_NAME/epoch_001_mode1_planned"
CACHE_ROOT="$ROOT/cache"
RAY_TMP=${ADAFLOOR_NPUGRAPH_RAY_TMP:-/tmp/ag_npx_f16_${STAMP}}
COMMON_EPOCH0=/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe
COMMON_CKPT="$COMMON_EPOCH0/checkpoints/qwen3moe_for_eagle3/global_step_5"

if [[ "$TASK_QUEUE_MODE" == 2 ]]; then
    echo "TASK_QUEUE_ENABLE=2 is unsupported by torch-npu NPUGraph capture; use 1" >&2
    exit 2
fi
if [[ "$TASK_QUEUE_MODE" != 1 ]]; then
    echo "ADAFLOOR_NPUGRAPH_TASK_QUEUE_ENABLE must be 1, got: $TASK_QUEUE_MODE" >&2
    exit 2
fi

if [[ -e "$ROOT" ]]; then
    echo "smoke root already exists; use a fresh ADAFLOOR_NPUGRAPH_SMOKE_ROOT: $ROOT" >&2
    exit 2
fi
for path in "$RUNTIME_TREE" "$COMMON_EPOCH0" "$COMMON_CKPT" /data/deepscaler/train.parquet; do
    [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done

mkdir -p "$EPOCH_DIR/oracle" "$CACHE_ROOT"/{xdg,hf,triton,torchair,ascend,work} "$RAY_TMP"

export ADAFLOOR_RUNTIME_TREE="$RUNTIME_TREE"
export LOCAL_TEST_LAUNCHER="$RUNTIME_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"
export OUTPUT_ROOT="$ROOT"
export OUTPUT_SUBDIR="$RUN_NAME/epoch_001_mode1_planned"
export PLAN_DIR="$EPOCH_DIR/oracle"
export BASELINE_DIRS="$COMMON_EPOCH0"
export PLAN_STEPS=5
export FORCE_SELECTED_FLOORS=16,16,16,16,16
export FAST_STEP_SUBSET=1
export FAST_STEP_SUBSET_STEPS=1
export TRAINER_TOTAL_EPOCHS=1
export TRAIN_BATCH_SIZE=32
export ROLLOUT_N=16
export MAX_RESPONSE_LENGTH=16384
export MAX_RESPONSE_LEN=16384
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=planned
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=380800
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=280576
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=377344
export VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS=0
export VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS=0

export VLLM_ENABLE_GRAPH_MODE=1
export NPUGRAPH_EX_ENABLE_STATIC_KERNEL=False
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1
export VLLM_ASCEND_NPUGRAPH_EX_MOE_CUSTOM_OP_BOUNDARY=0
export ROLLOUT_ENFORCE_EAGER=False
export TASK_QUEUE_ENABLE="$TASK_QUEUE_MODE"
export VLLM_ROLLOUT_TASK_QUEUE_ENABLE="$TASK_QUEUE_MODE"
export ROLLOUT_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS"
# Sixteen ranks compile the same full model concurrently. Bound each rank's
# compiler fan-out so graph compatibility is not confounded by host pipe and
# inotify exhaustion during this smoke.
export MAX_JOBS=${MAX_JOBS:-1}
export TE_PARALLEL_COMPILER=${TE_PARALLEL_COMPILER:-1}
export MAX_COMPILE_CORE_NUMBER=${MAX_COMPILE_CORE_NUMBER:-1}
export VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD=1
export VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD=1
export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
export VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
export VERL_SIDECAR_ENABLE=0
export VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1
export VERL_HCCL_IF_BASE_PORT_START=12000
export VERL_MASTER_PORT_START=28416

export RAY_TMPDIR="$RAY_TMP"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export HF_HOME="$CACHE_ROOT/hf"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHAIR_CACHE_HOME="$CACHE_ROOT/torchair"
export ASCEND_CACHE_PATH="$CACHE_ROOT/ascend"
export ASCEND_WORK_PATH="$CACHE_ROOT/work"

sha256sum \
    "$SCRIPT_DIR/run_qwen3_npugraph_ex_fixed16_smoke.sh" \
    "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh" \
    "$LOCAL_TEST_LAUNCHER" \
    "$RUNTIME_TREE/verl/trainer/config/rollout/rollout.yaml" \
    "$RUNTIME_TREE/verl/trainer/constants_ppo.py" \
    "$RUNTIME_TREE/verl/single_controller/ray/base.py" \
    "$RUNTIME_TREE/verl/workers/config/rollout.py" \
    "$RUNTIME_TREE/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py" \
    "$RUNTIME_TREE/vllm_ascend/shrink_aware/assignment.py" \
    "$RUNTIME_TREE/vllm_ascend/shrink_aware/planner.py" \
    "$RUNTIME_TREE/vllm_ascend/shrink_aware/trigger.py" \
    "$RUNTIME_TREE/patches/verl/utils/hybrid_data_parallel/hdp.py" \
    "$RUNTIME_TREE/patches/verl/features/rollout_optimize/__init__.py" \
    "$RUNTIME_TREE/patches/vllm_ascend/spec_decode/sam_proposer.py" \
    "$RUNTIME_TREE/vllm_ascend/compilation/acl_graph.py" \
    "$RUNTIME_TREE/vllm_ascend/compilation/compiler_interface.py" \
    "$RUNTIME_TREE/vllm_ascend/compilation/npu_graph_ex_pass_manager.py" \
    "$RUNTIME_TREE/vllm_ascend/ops/fused_moe_legacy.py" \
    "$RUNTIME_TREE/vllm_ascend/ops/common_fused_moe.py" \
    "$RUNTIME_TREE/vllm_ascend/ops/fused_moe/token_dispatcher.py" \
    "$RUNTIME_TREE/vllm/model_executor/models/qwen3_moe.py" \
    "$RUNTIME_TREE/vllm_ascend/ops/layernorm.py" \
    "$RUNTIME_TREE/vllm_ascend/ops/fused_moe/moe_mlp.py" \
    "$RUNTIME_TREE/vllm_ascend/ops/moe/token_dispatcher.py" \
    "$RUNTIME_TREE/vllm_ascend/ops/moe/moe_mlp.py" \
    "$RUNTIME_TREE/vllm_ascend/worker/worker_v1.py" > "$ROOT/code_sha256.txt"

printf '%s\n' \
    'schema_version=1' \
    'experiment=qwen3_npugraph_ex_fixed16_smoke' \
    "root=$ROOT" \
    "runtime_tree=$RUNTIME_TREE" \
    'vllm_version=0.14.1' \
    'vllm_ascend_version=0.14.0rc1' \
    "task_queue_enable=$TASK_QUEUE_MODE" \
    'npugraph_ex=true' \
    'enable_nz=false' \
    'force_torch_npu_add_rms_norm=true' \
    'moe_custom_op_boundary=false' \
    'moe_group_list_materialization=npu_dtype_cast_int64' \
    "capture_sizes=$CAPTURE_SIZES" \
    "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
    "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" \
    "max_jobs=$MAX_JOBS" \
    "te_parallel_compiler=$TE_PARALLEL_COMPILER" \
    "max_compile_core_number=$MAX_COMPILE_CORE_NUMBER" \
    'seed=101' \
    'planner_prompts=160' \
    'planner_steps=5' \
    'executed_source_step=1' \
    'executed_prompts=32' \
    'rollout_n=16' \
    'forced_floor=16' \
    'tail_guard=planner_default' \
    'actor_frozen=true' \
    "common_checkpoint=$COMMON_CKPT" > "$ROOT/protocol.env"

exec "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh" \
    actor_rollout_ref.rollout.cudagraph_capture_sizes="$CAPTURE_SIZES" \
    actor_rollout_ref.rollout.seed=101 \
    actor_rollout_ref.actor.optim.lr=0.0 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
    trainer.total_training_steps=1 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$COMMON_CKPT"
