#!/usr/bin/env bash
# TailServe native v0.14 prototype. dry-run is host-only, run uses 16 NPUs.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
ACTION=${1:-dry-run}
case "$ACTION" in dry-run|run) shift || true ;; *) echo 'usage: bash run_tailserve.sh [dry-run|run] [Hydra overrides]' >&2; exit 2 ;; esac
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TAILSERVE_ENABLE=${TAILSERVE_ENABLE:-1}
export TAILSERVE_RUN_ID=${TAILSERVE_RUN_ID:-tailserve_$(date -u +%Y%m%dT%H%M%SZ)}
export TAILSERVE_SPOOL=${TAILSERVE_SPOOL:-/data/tailserve_spool/demo}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-17408}
export TAILSERVE_PER_RANK_TOKENS=${TAILSERVE_PER_RANK_TOKENS:-$ROLLOUT_MAX_NUM_BATCHED_TOKENS}
export TAILSERVE_MAX_AUX_RANKS=${TAILSERVE_MAX_AUX_RANKS:-16}
export TAILSERVE_GLOBAL_TOKENS=${TAILSERVE_GLOBAL_TOKENS:-$((TAILSERVE_MAX_AUX_RANKS * TAILSERVE_PER_RANK_TOKENS))}
export TAILSERVE_MAX_PROMPT_TOKENS=${TAILSERVE_MAX_PROMPT_TOKENS:-$MAX_PROMPT_LENGTH}
export TAILSERVE_MAX_NEW_TOKENS=${TAILSERVE_MAX_NEW_TOKENS:-$MAX_RESPONSE_LENGTH}
export TAILSERVE_AUX_INTERVAL_ROUNDS=${TAILSERVE_AUX_INTERVAL_ROUNDS:-1}
export TAILSERVE_INTERFERENCE_FEEDBACK=${TAILSERVE_INTERFERENCE_FEEDBACK:-0}
export TAILSERVE_MODE=${TAILSERVE_MODE:-graph}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/data/tailserve_results}
export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-$TAILSERVE_RUN_ID}
export RECORD_DIR="$OUTPUT_ROOT/$OUTPUT_SUBDIR"
export TRAINER_TOTAL_EPOCHS=1
export TRAINER_TOTAL_TRAINING_STEPS=${TAILSERVE_RL_STEPS:-1}
export DATA_SHUFFLE=False
export DATASET_FRACTION=${DATASET_FRACTION:-0.005}
export TRAIN_BATCH_SIZE=32
export ROLLOUT_N=16
export VLLM_ROLLOUT_ASYNC_SCHEDULING=false
export VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true
export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=0
export VLLM_ASCEND_SHRINK_AWARE_ENABLE=0
export VLLM_ASCEND_SHRINK_AWARE_MODE=off
export VLLM_ENABLE_EPLB=0
export VLLM_ENABLE_EXPERT_PARALLEL=1
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=0
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=0
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=0
unset VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP
export VERL_SIDECAR_ENABLE=0
export VLLM_ROLLOUT_NATIVE_SLEEP_MODE=1
export VLLM_ROLLOUT_SLEEP_LEVEL=2
export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE=1
export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_KV_FIRST=1
export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
export VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
export VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD=0
export VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD=0
export VLLM_ROLLOUT_DEFER_KV_CACHE_UNTIL_WEIGHT_LOAD=0
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=0
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=0
export VLLM_ASCEND_USE_LEGACY_FUSED_MOE=0
export VLLM_ASCEND_USE_LEGACY_ATTENTION=0
export VLLM_ASCEND_EAGER_COMPILE_PASS_FUSION=0
export VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1
export VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=1
export VLLM_ASCEND_MC2_TOKENS_CAPACITY=512
export VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC=1
export PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
graph_args=()
case "$TAILSERVE_MODE" in
 graph)
  export ROLLOUT_ENFORCE_EAGER=False
  export VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1
  export VLLM_ENABLE_GRAPH_MODE=0
  graph_args+=(++actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY
               'actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32]')
  ;;
 eager)
  export ROLLOUT_ENFORCE_EAGER=True
  export VLLM_ROLLOUT_TASK_QUEUE_ENABLE=2
  export VLLM_ENABLE_GRAPH_MODE=0
  ;;
 *) echo 'TAILSERVE_MODE must be graph or eager' >&2; exit 2 ;;
esac
ARGS=(actor_rollout_ref.rollout.seed=0 data.shuffle=False
      actor_rollout_ref.rollout.async_scheduling=false
      actor_rollout_ref.rollout.enable_chunked_prefill=true
      trainer.logger='[console]' "${graph_args[@]}")
COMMON=${TAILSERVE_COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}
if [[ -f "$COMMON/reuse.env" ]]; then
    # Trusted local experiment metadata. Never delete or write this checkpoint.
    source "$COMMON/reuse.env"
    ARGS+=(trainer.resume_mode=resume_path "trainer.resume_from_path=$BASELINE_INITIAL_RESUME_CKPT")
    export VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1
fi
if [[ "$ACTION" == dry-run ]]; then
    python -m tailserve_runtime.preflight
    printf 'Command:'
    printf ' %q' bash "$ROOT/run_mode0_no_shrink_baseline.sh" "${ARGS[@]}" "$@"
    printf '\n'
    exit 0
fi
[[ ! -e "$RECORD_DIR" ]] || { echo "refusing to overwrite $RECORD_DIR" >&2; exit 2; }
python -m tailserve_runtime.preflight
exec bash "$ROOT/run_mode0_no_shrink_baseline.sh" "${ARGS[@]}" "$@"
