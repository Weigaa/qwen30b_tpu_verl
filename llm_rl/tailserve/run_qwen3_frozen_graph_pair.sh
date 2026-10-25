#!/usr/bin/env bash
# Same immutable workload, two version arms; never regenerate either plan.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARM=${1:-both}
ACTION=${2:-dry-run}
case "$ARM" in both|v011|v014) ;; *) echo 'arm must be both,v011,v014' >&2; exit 2;; esac
case "$ACTION" in dry-run|run) ;; *) echo 'action must be dry-run or run' >&2; exit 2;; esac
export ADAFLOOR_FROZEN_INPUTS_DIR=${ADAFLOOR_FROZEN_INPUTS_DIR:-/data/adafloor_shared_state/frozen_v011_k8_seed0_epoch1_v1}
export ADAFLOOR_FROZEN_INPUT_TOOL="$HERE/tools/frozen_graph_inputs.py"
PAIR_ROOT=${FROZEN_PAIR_ROOT:-/data/adafloor_graph_results/qwen3_frozen_v011_v014_$(date -u +%Y%m%dT%H%M%SZ)}
python3 "$ADAFLOOR_FROZEN_INPUT_TOOL" verify "$ADAFLOOR_FROZEN_INPUTS_DIR"
if [[ "$ARM" == both ]]; then
    for arm in v014 v011; do
        finalize=0
        if [[ "$ACTION" == run ]] && grep -qE '\[run\] end_time=.* exit_code=0' \
                "$PAIR_ROOT/$arm/frozen_graph_seed0/epoch_001_mode1_natural/logs/"*.txt 2>/dev/null; then
            finalize=1
        fi
        FAIR_FINALIZE_EXISTING="$finalize" FROZEN_PAIR_ROOT="$PAIR_ROOT" bash "$0" "$arm" "$ACTION"
    done
    if [[ "$ACTION" == run ]]; then
        python3 "$HERE/tools/compare_frozen_graph_pair.py" "$PAIR_ROOT" --expected-steps 5
    fi
    exit 0
fi
if [[ "$ARM" == v011 ]]; then
    REPO=/workspace/cann-recipes-train/llm_rl/adafloor_gragh
    # Preserve the verified v0.11 initialization allocator. Forcing False to
    # match native sleep prevents reuse/compaction of the large model-load
    # transient and can starve the first HCCL workspace before any KV exists.
    # The v0.11 helper now actually applies the NPU switch, just like v0.14:
    # OFF before rollout allocations/capture and ON in trainer_mode. Native
    # sleep construction requires False in v0.14's initialization environment.
    # Existing live segments are not converted by a runtime setting change.
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    initial_kv_args=()
    export VLLM_ROLLOUT_NATIVE_SLEEP_MODE=0 VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=0
    export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE=0
    export VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
    export VLLM_ASCEND_MODE1_STAGE_WEIGHTS_BEFORE_KV_REBUILD=1
    export VLLM_ASCEND_MODE1_BOOTSTRAP_KV_BEFORE_WEIGHT_ONLOAD=1
    export VLLM_ASCEND_MODE1_BOOTSTRAP_KV_TOKENS=103680
else
    REPO="$HERE"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    initial_kv_args=(++actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=810)
    export VLLM_ROLLOUT_NATIVE_SLEEP_MODE=1 VLLM_ROLLOUT_SLEEP_LEVEL=2
    export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE=1 VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_RETAIN=0
    export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_KV_FIRST=1 VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
    export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0 VLLM_ASCEND_NATIVE_SLEEP_PARK_IDLE=1
    export VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
fi
export COMMON_EPOCH0_ROOT=/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent
export FAIR_OUTPUT_ROOT="$PAIR_ROOT/$ARM"
export FAIR_RUN_NAME=frozen_graph_seed0 DYNAMIC_RUN_NAME=frozen_graph_seed0
export FAIR_START_EPOCH=1 FAIR_TOTAL_EPOCHS=2 FAIR_TRAIN_STEPS=5
export FAIR_TRAIN_BATCH_SIZE=32 FAIR_ROLLOUT_N=16 FAIR_MAX_RESPONSE_LENGTH=16384
export FAIR_PROMPTS_PER_EPOCH=160 FAIR_FREEZE_ACTOR=0 FAIR_KEEP_COMPLETED_CHECKPOINTS=0
export VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED=0
export HCCL_BUFFSIZE=800 TASK_QUEUE_ENABLE=1 VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC=1 VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=0
export VLLM_ENABLE_GRAPH_MODE=0 ROLLOUT_ENFORCE_EAGER=False
export ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY ADAFLOOR_GRAPH_CAPTURE_SIZES='[32]'
export VLLM_ASCEND_ACLGRAPH_BUCKET_COUNT=8
export VLLM_ASCEND_ELASTIC_ACLGRAPH=1 VLLM_ASCEND_ELASTIC_ACLGRAPH_DYNAMIC_DISABLE=0
export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1 VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1
export VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE=0 VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC=1
export VLLM_ASCEND_MC2_MIN_EP_SIZE=2 VLLM_ASCEND_FORCE_ALLTOALL_MOE=0
export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true
export VLLM_ROLLOUT_ASYNC_SCHEDULING=false VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS=0
export VLLM_ROLLOUT_DATA_REBALANCE=0 VLLM_ROLLOUT_USE_TQDM=0 VERL_SIDECAR_ENABLE=0
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE=1 VLLM_ASCEND_MODE1_PREEMPTION_AUDIT=1
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2=103680
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=253184
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=288256
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=353408
export VLLM_ASCEND_MODE1_STEP_KV_CAP_INCLUDES_PLANNED_HEADROOM=1
export VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1
# v0.11 gates AscendSampler with the first flag; v0.14 constructs it
# unconditionally. Enable the old backend too, while keeping both versions
# on torch_npu's fused filter rather than a separate local custom kernel.
export VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1 VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM=0
export VLLM_ROLLOUT_VERIFY_WEIGHT_COVERAGE=0 VLLM_ROLLOUT_WEIGHT_VALUE_AUDIT=0
export VLLM_ROLLOUT_PHASE_MEMORY_LOG=0 VLLM_ROLLOUT_STAGE_TIMING=0 VLLM_ROLLOUT_MEMORY_OWNERSHIP_LOG=0
export VLLM_ASCEND_RUNTIME_PROGRESS_INTERVAL=256 GIT_TRACK_RUN_LOGS=0
export VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1
export VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH="$HERE/vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend"
[[ -d "$VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH" ]] || { echo 'missing shared filtered OPP bundle' >&2; exit 2; }
export FAIR_ADAFLOOR_TARGET="$REPO/run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh"
export ADAFLOOR_GRAPH_BASE_RUNNER="$REPO/run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh"
export PLANNER_TOKENIZER_PATH=/data/Qwen3-30B-A3B KV_ADMISSION_REQUIRE_PROMPT_TOKENS=1
printf 'frozen_arm=%s repo=%s output=%s\n' "$ARM" "$REPO" "$FAIR_OUTPUT_ROOT"
printf 'seed=0 request_seed=paired TQ=1 HCCL=800 metadata=CPU buckets=frozen_K8\n'
printf 'sampler=AscendSampler topk_topp_optimization=1 local_custom_topk=0\n'
printf 'allocator_init=%s initial_kv_override=%s\n' "$PYTORCH_NPU_ALLOC_CONF" "${initial_kv_args[*]:-none (v011 cold-init and per-step KV policy)}"
printf 'allocator_phase_policy=rollout:False,trainer:True actual_NPU_setter=enabled\n'
printf 'floors=2,4,4,4,16 KV=103680,253184,253184,253184,353408 steps=5\n'
[[ "$ACTION" == dry-run ]] && exit 0
[[ ! -d "$FAIR_OUTPUT_ROOT/frozen_graph_seed0" || "${FAIR_FINALIZE_EXISTING:-0}" == 1 ]] || { echo 'refusing existing run' >&2; exit 2; }
mkdir -p "$FAIR_OUTPUT_ROOT"
[[ "$ARM" != v014 ]] || python3 "$HERE/tools/build_camem_lifecycle.py"
cd "$REPO"
bash run_paper_fair_epoch1_2_from_common_epoch0.sh adafloor_n_f2 \
    actor_rollout_ref.rollout.seed=0 actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.megatron.seed=42 actor_rollout_ref.ref.megatron.seed=42 \
    ++data.seed=42 trainer.save_freq=1 "${initial_kv_args[@]}"
# Record the successful wrapper exit in a NEW log, not by rewriting the
# trainer's original log. Legacy v0.11 does not emit v0.14's end_time marker.
printf '[run] end_time=%s exit_code=0 frozen_arm=%s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$ARM" \
    | tee -a "$FAIR_OUTPUT_ROOT/frozen_graph_seed0/epoch_001_mode1_natural/logs/frozen_arm_completion.txt"
