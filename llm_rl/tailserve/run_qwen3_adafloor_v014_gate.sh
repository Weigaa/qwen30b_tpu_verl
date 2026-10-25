#!/usr/bin/env bash
# Minimal deterministic 0.14 AdaFloor execution gate.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$ROOT/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"
MODE=${1:-dry-run}
if [[ $# -gt 0 ]]; then
    shift
fi
case "$MODE" in
    eager|graph|dry-run) ;;
    *) echo "usage: $0 {eager|graph|dry-run} [Hydra overrides ...]" >&2; exit 2 ;;
esac

EXECUTION_MODE=${ADAFLOOR_V014_GATE_MODE:-eager}
if [[ "$MODE" != dry-run ]]; then
    EXECUTION_MODE=$MODE
fi
if [[ "$EXECUTION_MODE" != eager && "$EXECUTION_MODE" != graph ]]; then
    echo "ADAFLOOR_V014_GATE_MODE must be eager or graph" >&2
    exit 2
fi

TARGET_POLICY=${ADAFLOOR_V014_GATE_TARGET_POLICY:-planned}
if [[ "$TARGET_POLICY" != planned && "$TARGET_POLICY" != natural ]]; then
    echo "ADAFLOOR_V014_GATE_TARGET_POLICY must be planned or natural" >&2
    exit 2
fi

STAMP=${ADAFLOOR_V014_GATE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
GATE_ROOT=${ADAFLOOR_V014_GATE_ROOT:-$ROOT/validation_results/qwen3_adafloor_v014_gate_${STAMP}}
OUTPUT_SUBDIR=${ADAFLOOR_V014_GATE_RUN_NAME:-${EXECUTION_MODE}_tq$([[ "$EXECUTION_MODE" == graph ]] && echo 1 || echo 2)}
OUTPUT_DIR="$GATE_ROOT/$OUTPUT_SUBDIR"
if [[ "$MODE" != dry-run && -e "$OUTPUT_DIR" ]]; then
    echo "refusing to reuse output directory: $OUTPUT_DIR" >&2
    exit 2
fi

# Identical deterministic work in both arms: 32 prompts, one response each.
export OUTPUT_ROOT="$GATE_ROOT"
export OUTPUT_SUBDIR
export TRAIN_BATCH_SIZE=32
export ACTOR_PPO_MINI_BATCH_SIZE=32
export ROLLOUT_N=1
export ROLLOUT_MAX_NUM_SEQS=2
export MAX_RESPONSE_LENGTH=128
export DATASET_FRACTION=0.005
export TRAINER_TOTAL_EPOCHS=1
export TRAINER_TOTAL_TRAINING_STEPS=1
export SAVE_CKPT_ENABLE=0
export GIT_TRACK_RUN_LOGS=0
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=8,16,32,64,64

# AdaFloor staged path. TailGuard is disabled; the rank-local fixed lengths
# above exist only to make the functional gate deterministic. The target
# policy is selected below and is verified from runtime trigger evidence.
TARGET_FLOOR=${ADAFLOOR_V014_GATE_FLOOR:-4}
case "$TARGET_FLOOR" in
    4)
        SHRINK_STAGES=8,4
        STAGE_RANKS='[[8,9,10,11,12,13,14,15],[12,13,14,15]]'
        INTERMEDIATE_RANKS=8,9,10,11,12,13,14,15
        FINAL_RANKS=12,13,14,15
        ASSIGNMENT_POLICY=manual_5_2_1
        ;;
    2)
        SHRINK_STAGES=8,4,2
        STAGE_RANKS='[[8,9,10,11,12,13,14,15],[12,13,14,15],[14,15]]'
        INTERMEDIATE_RANKS=8,9,10,11,12,13,14,15
        FINAL_RANKS=14,15
        # The legacy 5/2/1 helper is intentionally specialized for the
        # 8-donor/4-wave2/4-final floor4 layout. The generic role-aware LPT
        # assignment handles floor2's 8/4/2/2 stage roles and still assigns
        # exactly two prompt occurrences to every rank in this 32-row gate.
        ASSIGNMENT_POLICY=default
        # Pin both graph and eager to the same cache capacity. FULL_DECODE
        # profiling naturally leaves 869 blocks (111,232 tokens), while eager
        # would otherwise consume that graph headroom as another 98 KV blocks
        # and leave no room for the first live weight onload/prefill.
        NUM_GPU_BLOCKS_OVERRIDE=869
        # Match the validated 0.11 mode1 floor2 memory contract. These broad
        # phase-specific reserves are mutually exclusive and are already
        # covered by the memory-utilization slack. Charging all of them on top
        # of floor2's preloaded expert storage leaves zero KV blocks in 0.14.
        # Keep the zero-redundancy expert-growth reserve, which is computed
        # separately from these tunable generic headrooms.
        export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_HEADROOM_BYTES=0
        export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_LOW_FLOOR_HEADROOM_BYTES=0
        export VLLM_ASCEND_POST_SHRINK_PREFILL_ALLTOALL_HEADROOM_BYTES=0
        export VLLM_ASCEND_POST_RESTORE_DP_HEADROOM_BYTES=0
        export VLLM_ASCEND_POST_RESTORE_EP_HEADROOM_BYTES=0
        export VLLM_ASCEND_POST_RESTORE_MOE_DISPATCH_HEADROOM_BYTES=0
        export VLLM_ASCEND_FIRST_LIVE_PREFILL_HEADROOM_BYTES=0
        export VLLM_ASCEND_FIRST_LIVE_PREFILL_LOW_FLOOR_HEADROOM_BYTES=0
        export VLLM_ASCEND_EXTRA_ELASTIC_SAFETY_HEADROOM_BYTES=0
        export VLLM_ASCEND_FLOOR_PREALLOC_HEADROOM_SAFETY_BYTES=0
        ;;
    *)
        echo "ADAFLOOR_V014_GATE_FLOOR must be 2 or 4" >&2
        exit 2
        ;;
esac
export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=1
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MOE_MODE=lossless
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=$TARGET_FLOOR
export VLLM_ASCEND_MC2_MIN_EP_SIZE=$TARGET_FLOOR
export VLLM_ASCEND_SHRINK_AWARE_ENABLE=1
export VLLM_ASCEND_SHRINK_AWARE_MODE=staged
export VLLM_ASCEND_SHRINK_AWARE_STAGES=$SHRINK_STAGES
export VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY=manual
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=$TARGET_POLICY
export VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=$INTERMEDIATE_RANKS
export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=$FINAL_RANKS
export VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS=$STAGE_RANKS
export VLLM_ASCEND_SHRINK_AWARE_MIN_WINDOW_SECONDS=0
export VLLM_ASCEND_SHRINK_AWARE_LOGGING=1
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=0
export VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE=0
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=0
export VLLM_ROLLOUT_DATA_REBALANCE=0
export VLLM_ROLLOUT_LENGTH_BALANCE=0
export VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS=0
export VLLM_ROLLOUT_ASYNC_SCHEDULING=false

CAPTURE_SIZES=${ADAFLOOR_V014_GATE_CAPTURE_SIZES:-[2]}
if [[ "$EXECUTION_MODE" == graph ]]; then
    export VLLM_ENABLE_GRAPH_MODE=1
    export VLLM_ASCEND_ELASTIC_ACLGRAPH=1
    export ROLLOUT_ENFORCE_EAGER=False
    export VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1
    export TASK_QUEUE_ENABLE=1
    export VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER=1
    export VLLM_ROLLOUT_UNSAFE_TRUE_GRAPH_WITH_RESAMPLER=1
    export VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE=0
    export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
    export VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
    export VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD=0
    export VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD=0
    export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
    export VLLM_ROLLOUT_FREE_CACHE_ENGINE=True
    export VLLM_ROLLOUT_SLEEP_LEVEL=1
    export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=0
    # Keep the same KV-cache budget as the eager arm. A lower 0.70 budget is
    # exhausted after the identical AdaFloor shrink/restore safety reserves
    # are applied, before any cache blocks can be allocated.
    export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=0.83
else
    export VLLM_ENABLE_GRAPH_MODE=0
    export VLLM_ASCEND_ELASTIC_ACLGRAPH=0
    export ROLLOUT_ENFORCE_EAGER=True
    export VLLM_ROLLOUT_TASK_QUEUE_ENABLE=2
    export TASK_QUEUE_ENABLE=2
    # Use the same native CaMem sleep/wake lifecycle as the graph arm.  The
    # 0.14 manual path rebuilds every rollout parameter with torch.empty and
    # exceeds floor2's nearly full-device lossless expert layout before the
    # first decode.  Native wake restores the already validated storage layout
    # while enforce_eager=True still keeps execution entirely out of ACLGraph.
    export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
    export VLLM_ROLLOUT_FREE_CACHE_ENGINE=True
    export VLLM_ROLLOUT_SLEEP_LEVEL=1
    export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=0
    export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=0.83
fi

HYDRA_ARGS=(
    actor_rollout_ref.actor.optim.lr=0.0
    actor_rollout_ref.actor.use_kl_loss=False
    algorithm.use_kl_in_reward=False
    actor_rollout_ref.rollout.seed=101
    actor_rollout_ref.rollout.temperature=0.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.ignore_eos=True
    actor_rollout_ref.rollout.max_model_len=640
    actor_rollout_ref.rollout.max_num_batched_tokens=640
    actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_scheduling=true
    actor_rollout_ref.rollout.shrink_aware.shrink_aware_mode=staged
    "actor_rollout_ref.rollout.shrink_aware.shrink_stages=[$SHRINK_STAGES]"
    actor_rollout_ref.rollout.shrink_aware.survivor_selection_policy=manual
    actor_rollout_ref.rollout.shrink_aware.target_policy=$TARGET_POLICY
    "actor_rollout_ref.rollout.shrink_aware.intermediate_survivor_ranks=[$INTERMEDIATE_RANKS]"
    "actor_rollout_ref.rollout.shrink_aware.final_survivor_ranks=[$FINAL_RANKS]"
    "actor_rollout_ref.rollout.shrink_aware.stage_survivor_ranks=$STAGE_RANKS"
    actor_rollout_ref.rollout.shrink_aware.length_prediction_source=existing_regroup
    actor_rollout_ref.rollout.shrink_aware.assignment_policy=$ASSIGNMENT_POLICY
    actor_rollout_ref.rollout.shrink_aware.min_shrink_window_seconds=0
    actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_logging=true
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
    'trainer.logger=["console"]'
)
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE:-}" ]]; then
    HYDRA_ARGS+=(
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=$NUM_GPU_BLOCKS_OVERRIDE"
    )
fi
if [[ "$EXECUTION_MODE" == graph ]]; then
    HYDRA_ARGS+=(
        actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY
        "actor_rollout_ref.rollout.cudagraph_capture_sizes=${CAPTURE_SIZES}"
    )
else
    HYDRA_ARGS+=(
        actor_rollout_ref.rollout.cudagraph_mode=null
        actor_rollout_ref.rollout.cudagraph_capture_sizes=null
    )
fi
HYDRA_ARGS+=("$@")

if [[ "$MODE" == dry-run ]]; then
    printf '%s\n' \
        "execution_mode=$EXECUTION_MODE" \
        "output_dir=$OUTPUT_DIR" \
        "task_queue_enable=$TASK_QUEUE_ENABLE" \
        "manual_free_cache_engine=$VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE" \
        "sleep_level=$VLLM_ROLLOUT_SLEEP_LEVEL" \
        "cudagraph_mode=$([[ "$EXECUTION_MODE" == graph ]] && echo FULL_DECODE_ONLY || echo NONE)" \
        "capture_sizes=$([[ "$EXECUTION_MODE" == graph ]] && echo "$CAPTURE_SIZES" || echo none)" \
        "workload=32_prompts_x_n1" \
        "target_policy=$TARGET_POLICY" \
        "target_floor=$TARGET_FLOOR" \
        "shrink_stages=16_to_${SHRINK_STAGES//,/_to_}" \
        "assignment_policy=$ASSIGNMENT_POLICY" \
        "stage_survivor_ranks=$STAGE_RANKS" \
        "num_gpu_blocks_override=${NUM_GPU_BLOCKS_OVERRIDE:-auto}" \
        "tailguard=off"
    printf 'hydra_arg=%s\n' "${HYDRA_ARGS[@]}"
    exit 0
fi

mkdir -p "$GATE_ROOT"
exec bash "$BASE_SCRIPT" "${HYDRA_ARGS[@]}"
