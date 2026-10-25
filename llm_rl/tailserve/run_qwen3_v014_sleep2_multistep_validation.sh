#!/usr/bin/env bash
# Bounded real-RL validation, not the old fixed-token synthetic gate.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-eager}
ACTION=${2:-dry-run}
case "$MODE" in eager|graph) ;; *) echo 'mode must be eager or graph' >&2; exit 2;; esac
case "$ACTION" in dry-run|run) ;; *) echo 'action must be dry-run or run' >&2; exit 2;; esac
STEPS=${SLEEP2_VALIDATION_STEPS:-2}
[[ "$STEPS" =~ ^[1-5]$ ]] || { echo 'steps must be in [1,5]' >&2; exit 2; }
export COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}
export FAIR_OUTPUT_ROOT=${SLEEP2_VALIDATION_ROOT:-/data/adafloor_graph_results/qwen3_v014_sleep2_${MODE}_multistep_$(date -u +%Y%m%dT%H%M%SZ)}
export DYNAMIC_RUN_NAME=qwen3_${MODE}_sleep2_natural_f2_seed0
export FAIR_RUN_NAME=$DYNAMIC_RUN_NAME
export FAIR_START_EPOCH=1 FAIR_TOTAL_EPOCHS=2 FAIR_TRAIN_STEPS=$STEPS
export FAIR_TRAIN_BATCH_SIZE=32 FAIR_ROLLOUT_N=16 FAIR_MAX_RESPONSE_LENGTH=16384
export FAIR_PROMPTS_PER_EPOCH=160 FAIR_FREEZE_ACTOR=0 FAIR_KEEP_COMPLETED_CHECKPOINTS=0
export VERL_PAIRED_REQUEST_SAMPLING_SEEDS=0
export VLLM_ASCEND_MC2_MIN_EP_SIZE=2
# Bound transport scratch, not KV/model capacity. MC2 partitions its workspace
# using this global value; do NOT override only its pg_options buffer size.
# CANN's Qwen3 BF16 MC2 tiling reports >=609MiB at capacity512, H2048,
# topk8 and128 experts (invariant across EP16/8/4/2). Round up to640MiB.
export HCCL_BUFFSIZE=${SLEEP2_HCCL_BUFFSIZE:-640}
[[ "$HCCL_BUFFSIZE" =~ ^[0-9]+$ ]] && (( HCCL_BUFFSIZE >= 640 )) || {
    echo 'This MC2 validation requires HCCL_BUFFSIZE >=640 MiB' >&2; exit 2;
}
export VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1
export VLLM_ENABLE_GRAPH_MODE=0
export VLLM_ROLLOUT_NATIVE_SLEEP_MODE=1 VLLM_ROLLOUT_SLEEP_LEVEL=2
export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE=1 VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_RETAIN=0
export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_KV_FIRST=1 VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true
export VLLM_ROLLOUT_ASYNC_SCHEDULING=false
export VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=1 VLLM_QWEN3_MOE_REDUCE_RESULTS=1
export VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1
export VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH="$ROOT/vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend"
export VLLM_ROLLOUT_VERIFY_WEIGHT_COVERAGE=1 VLLM_ROLLOUT_WEIGHT_VALUE_AUDIT=1
export VLLM_ROLLOUT_PHASE_MEMORY_LOG=1 VLLM_ROLLOUT_STAGE_TIMING=1
export VLLM_ASCEND_NATIVE_SLEEP_PARK_IDLE=${VLLM_ASCEND_NATIVE_SLEEP_PARK_IDLE:-1}
export VLLM_ASCEND_RUNTIME_PROGRESS_INTERVAL=256 GIT_TRACK_RUN_LOGS=0
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE=1
export VLLM_ASCEND_MODE1_PREEMPTION_AUDIT=1
if [[ "$MODE" == eager ]]; then
    export ROLLOUT_ENFORCE_EAGER=True TASK_QUEUE_ENABLE=2 VLLM_ROLLOUT_TASK_QUEUE_ENABLE=2
    export VLLM_ASCEND_ELASTIC_ACLGRAPH=0
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=0 VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=0
else
    export ROLLOUT_ENFORCE_EAGER=False TASK_QUEUE_ENABLE=1 VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1
    export ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY ADAFLOOR_GRAPH_CAPTURE_SIZES='[16,32]'
    export VLLM_ASCEND_ELASTIC_ACLGRAPH=1
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1 VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_DYNAMIC_DISABLE=0
    export VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC=1 VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE=0
    export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
    export VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1 VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
    export VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM=0
    export FAIR_ADAFLOOR_TARGET="$ROOT/run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh"
    export ADAFLOOR_GRAPH_BASE_RUNNER="$ROOT/run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh"
fi
printf 'mode=%s tq=%s steps=%s MC2_min_EP=2 native_sleep=2 lifecycle=1 KV_first=1\n' "$MODE" "$TASK_QUEUE_ENABLE" "$STEPS"
printf 'common_epoch0=%s\nresult_root=%s\n' "$COMMON_EPOCH0_ROOT" "$FAIR_OUTPUT_ROOT"
printf 'hccl_buffer_mib=%s (same setting for eager and graph)\n' "$HCCL_BUFFSIZE"
printf 'startup_kv_blocks=1024 block_size=128 tokens=131072 (first F2 step only)\n'
printf 'adaptive_kv=1 per_step_tokens=131072,131072,280576,280576,380800 preemption_audit=1\n'
[[ "$ACTION" == dry-run ]] && exit 0
[[ ! -e "$FAIR_OUTPUT_ROOT/$DYNAMIC_RUN_NAME" ]] || { echo 'refusing to overwrite an existing run' >&2; exit 2; }
mkdir -p "$FAIR_OUTPUT_ROOT"
if [[ "$VLLM_ASCEND_NATIVE_SLEEP_PARK_IDLE" == 1 ]]; then
    python3 "$ROOT/tools/build_camem_lifecycle.py"
fi
exec bash "$ROOT/run_paper_fair_epoch1_2_from_common_epoch0.sh" adafloor_n_f2 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=1024
