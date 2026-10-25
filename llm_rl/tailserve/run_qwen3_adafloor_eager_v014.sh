#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$ROOT/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"
cd "$ROOT"
ACTION=${1:-dry-run}
if [[ $# -gt 0 ]]; then
    shift
fi

case "$ACTION" in
    dry-run|run) ;;
    *)
        echo "usage: $0 [dry-run|run] [hydra overrides ...]" >&2
        exit 2
        ;;
esac

export VLLM_ENABLE_GRAPH_MODE=0
export ROLLOUT_ENFORCE_EAGER=True
export VLLM_ROLLOUT_TASK_QUEUE_ENABLE=${VLLM_ROLLOUT_TASK_QUEUE_ENABLE:-2}
export VLLM_ROLLOUT_ASYNC_SCHEDULING=false
export VLLM_ROLLOUT_DATA_REBALANCE=0
export VLLM_ROLLOUT_LENGTH_BALANCE=0
export VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS=0

export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=1
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MOE_MODE=lossless
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=${ADAFLOOR_MIN_FLOOR:-4}
export VLLM_ASCEND_SHRINK_AWARE_ENABLE=1
export VLLM_ASCEND_SHRINK_AWARE_MODE=staged
export VLLM_ASCEND_SHRINK_AWARE_STAGES=${ADAFLOOR_SHRINK_STAGES:-8,4}
export VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY=manual
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=${ADAFLOOR_TARGET_POLICY:-planned}
export VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=${ADAFLOOR_INTERMEDIATE_RANKS:-8,9,10,11,12,13,14,15}
export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=${ADAFLOOR_FINAL_RANKS:-12,13,14,15}
export VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS=${ADAFLOOR_STAGE_RANKS:-'[[8,9,10,11,12,13,14,15],[12,13,14,15]]'}
export VLLM_ASCEND_SHRINK_AWARE_MIN_WINDOW_SECONDS=${ADAFLOOR_MIN_WINDOW_SECONDS:-1.0}
export VLLM_ASCEND_SHRINK_AWARE_LOGGING=${ADAFLOOR_LOGGING:-1}

# TailGuard remains opt-in so shrink/restore can first be compared at identical
# generated-work settings. Optimized plans may supply a per-step cap.
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=0
export VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE=${ADAFLOOR_TAIL_GUARD:-0}
export VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_TOKENS=${ADAFLOOR_TAIL_GUARD_CAP:-4096}
export VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_FLOORS=${ADAFLOOR_TAIL_GUARD_FLOORS:-4}
TAIL_GUARD_LABEL=off
if [[ "${VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE,,}" =~ ^(1|true|yes|on)$ ]]; then
    TAIL_GUARD_LABEL=on
fi
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=${VLLM_EPOCH_LENGTH_REGROUP_ENABLE:-1}
export VLLM_EAGER_BASELINE_NO_RESAMPLE=0

export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-qwen3_adafloor_eager_v014}
LENGTH_SOURCE=${ADAFLOOR_LENGTH_SOURCE:-existing_regroup}
ASSIGNMENT_POLICY=${ADAFLOOR_ASSIGNMENT_POLICY:-manual_5_2_1}

HYDRA_ARGS=(
    actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_scheduling=true
    actor_rollout_ref.rollout.shrink_aware.shrink_aware_mode=staged
    "actor_rollout_ref.rollout.shrink_aware.shrink_stages=[${VLLM_ASCEND_SHRINK_AWARE_STAGES}]"
    actor_rollout_ref.rollout.shrink_aware.survivor_selection_policy=manual
    "actor_rollout_ref.rollout.shrink_aware.target_policy=${VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY}"
    "actor_rollout_ref.rollout.shrink_aware.intermediate_survivor_ranks=[${VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS}]"
    "actor_rollout_ref.rollout.shrink_aware.final_survivor_ranks=[${VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS}]"
    "actor_rollout_ref.rollout.shrink_aware.length_prediction_source=${LENGTH_SOURCE}"
    "actor_rollout_ref.rollout.shrink_aware.assignment_policy=${ASSIGNMENT_POLICY}"
    "actor_rollout_ref.rollout.shrink_aware.min_shrink_window_seconds=${VLLM_ASCEND_SHRINK_AWARE_MIN_WINDOW_SECONDS}"
    actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_logging=true
)

if [[ "$ASSIGNMENT_POLICY" == "optimized_rank_plan" ]]; then
    : "${ADAFLOOR_PLAN_PATH:?ADAFLOOR_PLAN_PATH is required for optimized_rank_plan}"
    [[ -f "$ADAFLOOR_PLAN_PATH" ]] || {
        echo "AdaFloor plan not found: $ADAFLOOR_PLAN_PATH" >&2
        exit 2
    }
    HYDRA_ARGS+=(
        "actor_rollout_ref.rollout.shrink_aware.optimized_rank_plan_path=${ADAFLOOR_PLAN_PATH}"
    )
fi

if [[ "$LENGTH_SOURCE" == "oracle_trace" ]]; then
    : "${ADAFLOOR_ORACLE_TRACE_PATH:?ADAFLOOR_ORACLE_TRACE_PATH is required for oracle_trace}"
    [[ -f "$ADAFLOOR_ORACLE_TRACE_PATH" ]] || {
        echo "AdaFloor oracle trace not found: $ADAFLOOR_ORACLE_TRACE_PATH" >&2
        exit 2
    }
    HYDRA_ARGS+=(
        "actor_rollout_ref.rollout.shrink_aware.oracle_trace_path=${ADAFLOOR_ORACLE_TRACE_PATH}"
    )
fi

if [[ "$ACTION" == "dry-run" ]]; then
    printf '%s\n' \
        "AdaFloor v0.14 eager migration dry-run" \
        "base_script=$BASE_SCRIPT" \
        "target_policy=$VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY" \
        "stages=$VLLM_ASCEND_SHRINK_AWARE_STAGES" \
        "stage_ranks=$VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS" \
        "floor=$VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE" \
        "length_source=$LENGTH_SOURCE" \
        "assignment_policy=$ASSIGNMENT_POLICY" \
        "enforce_eager=$ROLLOUT_ENFORCE_EAGER" \
        "async_scheduling=$VLLM_ROLLOUT_ASYNC_SCHEDULING" \
        "task_queue=$VLLM_ROLLOUT_TASK_QUEUE_ENABLE" \
        "tailguard=$TAIL_GUARD_LABEL"
    printf 'hydra_arg=%s\n' "${HYDRA_ARGS[@]}" "$@"
    exit 0
fi

exec bash "$BASE_SCRIPT" "${HYDRA_ARGS[@]}" "$@"
