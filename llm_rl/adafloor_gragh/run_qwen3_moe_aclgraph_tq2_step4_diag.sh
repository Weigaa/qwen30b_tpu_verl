#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
ROOT=${ADAFLOOR_TQ2_GRAPH_DIAG_ROOT:-/data/adafloor_shared_state/qwen3_moe_aclgraph_tq2_step4_diag_${STAMP}}
RUN_NAME=planned_moe_aclgraph_tq2_seed101_source_step4
EPOCH_DIR="$ROOT/$RUN_NAME/epoch_001_mode1_planned"
CACHE_ROOT="$ROOT/cache"
RAY_TMP="${ADAFLOOR_TQ2_GRAPH_RAY_TMP:-/tmp/ada_tq2_s4_${STAMP}}"
COMMON_EPOCH0=/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe
COMMON_CKPT="$COMMON_EPOCH0/checkpoints/qwen3moe_for_eagle3/global_step_5"

if [[ -e "$ROOT" ]]; then
    echo "diagnostic root already exists; use a fresh ADAFLOOR_TQ2_GRAPH_DIAG_ROOT: $ROOT" >&2
    exit 2
fi
for path in "$COMMON_EPOCH0" "$COMMON_CKPT" /data/deepscaler/train.parquet; do
    if [[ ! -e "$path" ]]; then
        echo "missing required input: $path" >&2
        exit 2
    fi
done

mkdir -p "$EPOCH_DIR/oracle" "$CACHE_ROOT"/{xdg,hf,triton,torchair,ascend,work} "$RAY_TMP"

export OUTPUT_ROOT="$ROOT"
export OUTPUT_SUBDIR="$RUN_NAME/epoch_001_mode1_planned"
export PLAN_DIR="$EPOCH_DIR/oracle"
export BASELINE_DIRS="$COMMON_EPOCH0"
export PLAN_STEPS=5
export FAST_STEP_SUBSET=1
export FAST_STEP_SUBSET_STEPS=4
export TRAINER_TOTAL_EPOCHS=1
export TRAIN_BATCH_SIZE=32
export ROLLOUT_N=16
export MAX_RESPONSE_LENGTH=16384
export MAX_RESPONSE_LEN=16384
export VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE=1
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=planned
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=380800
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=280576
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=377344
export VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS=1
export VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS=1
export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS=147456
export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR4=147456
export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8=114688
export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR16=0

export ADAFLOOR_GRAPH_BASE_RUNNER="$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"
export ADAFLOOR_GRAPH_CAPTURE_PROFILE=balanced
export ADAFLOOR_GRAPH_CAPTURE_SIZES='[1,2,4,8,16,32]'
export VLLM_ASCEND_ELASTIC_ACLGRAPH=1
export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=0
export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1
export VLLM_ASCEND_ELASTIC_ACLGRAPH_ALLOW_TASK_QUEUE_2=1
export VLLM_ENABLE_GRAPH_MODE=0
export ROLLOUT_ENFORCE_EAGER=False
export TASK_QUEUE_ENABLE=2
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

printf '%s\n' \
    "experiment=qwen3_moe_aclgraph_tq2_step4_diag" \
    "root=$ROOT" \
    "task_queue_enable=2" \
    "task_queue_2_diagnostic_gate=1" \
    "seed=101" \
    "planner_prompts=160" \
    "planner_steps=5" \
    "executed_source_step=4" \
    "executed_prompts=32" \
    "rollout_n=16" \
    "tail_guard=true" \
    "graph_attention=false" \
    "graph_moe=true" \
    "reset_trainer_progress_after_resume=true" \
    "ray_tmp=$RAY_TMP" \
    "common_checkpoint=$COMMON_CKPT" > "$ROOT/protocol.env"

exec "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh" \
    actor_rollout_ref.rollout.seed=101 \
    actor_rollout_ref.actor.optim.lr=0.0 \
    actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
    trainer.total_training_steps=1 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$COMMON_CKPT"
