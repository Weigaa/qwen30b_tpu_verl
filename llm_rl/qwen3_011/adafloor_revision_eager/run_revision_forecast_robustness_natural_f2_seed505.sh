#!/usr/bin/env bash
# One frozen five-step Natural-F2 forecast-sensitivity experiment.
# Normal x1.0 is reused from existing seed505 evidence and is not rerun.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAPER_REPO=${PAPER_REPO:-/workspace/cann-recipes-train/llm_rl/qwen3_011/qwen3_shrink_aware}
COMMON=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe}
ROOT=${REVISION_FORECAST_ROOT:-/workspace/adafloor_revision_results/forecast_robustness_natural_f2_seed505_v1}
VARIANT=${ADAFLOOR_FORECAST_VARIANT:-natural_f2}
ACTION=${1:-dry-run}
[[ "$ACTION" == dry-run || "$ACTION" == run ]] || exit 2
[[ "$VARIANT" == natural_f2 || "$VARIANT" == natural_f4 ]] || exit 2
COMMON_CKPT="$COMMON/checkpoints/qwen3moe_for_eagle3/global_step_5"
[[ -f "$COMMON/offline_planning_history.json" && -f "$COMMON_CKPT/.PRESERVE_COMMON_EPOCH0" ]]
[[ $(realpath -m "$ROOT") == /workspace/adafloor_revision_results/* ]]
printf 'variant=%s\nseed=505\nsteps=5\nstart_global_step=5\ntarget_global_step=10\nactor_frozen=true\nnormal_x1_reused=true\nconditions=under_0p7,over_1p3,mixed\neager=true\nTQ=2\ngraph=false\ncommon_epoch0=%s\noutput=%s\n' "$VARIANT" "$COMMON" "$ROOT"
[[ "$ACTION" != dry-run ]] || exit 0
[[ ! -e "$ROOT" ]] || { echo "refusing existing root:$ROOT" >&2; exit 2; }
[[ -z $(pgrep -f 'run_mode1_local_length_sorted_e2e_adaptive_floor2|verl.trainer.main_ppo|torch.distributed.run' || true) ]] || exit 3
mkdir -p "$ROOT"
python3 "$REPO/tools/create_revision_forecast_scenarios.py" \
    --history "$COMMON/offline_planning_history.json" \
    --output-root "$ROOT/scenarios" --mixed-seed 505 > "$ROOT/scenario_generation.json"
CHILD_SCRIPT="$PAPER_REPO/run_mode1_local_length_sorted_e2e_adaptive_floor2.sh"
if [[ "$VARIANT" == natural_f4 ]]; then CHILD_SCRIPT="$PAPER_REPO/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"; fi
sha256sum "$0" "$REPO/tools/create_revision_forecast_scenarios.py" \
    "$CHILD_SCRIPT" \
    "$PAPER_REPO/tools/build_mode1_length_sorted_e2e_plan.py" > "$ROOT/code_sha256.txt"
sha256sum "$COMMON/offline_planning_history.json" \
    "$COMMON_CKPT/.PRESERVE_COMMON_EPOCH0" > "$ROOT/input_sha256.txt"
printf 'status=RUNNING\ncommon_epoch0=%s\nnormal_result_reused=true\n' "$COMMON" > "$ROOT/protocol.env"
cd "$PAPER_REPO"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TASK_QUEUE_ENABLE=2 VLLM_ENABLE_GRAPH_MODE=0 ROLLOUT_ENFORCE_EAGER=True
export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=1 VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4 VLLM_ASCEND_SHRINK_AWARE_ENABLE=1
export VLLM_ASCEND_SHRINK_AWARE_MODE=staged
export VLLM_ASCEND_SHRINK_AWARE_STAGES=$([[ "$VARIANT" == natural_f4 ]] && echo 8,4 || echo 8,4,2)
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=natural VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY=manual
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=380800
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2=131072
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=280576
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=315648
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE=1
export ADAFLOOR_REVISION_HARD_KV_GUARD=1
export ADAFLOOR_REVISION_EFFECTIVE_CAPS='2:131072,4:280576,8:315648,16:380800'
export VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING=1
export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1
export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=1
export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB=28672
export VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1
export VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0
export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=0
export VLLM_ASCEND_MODE1_PARITY_FORCE_CLEANUP_AFTER_FLOOR_GROUP_RELEASE=1
export VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE=1
export VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC=1
export VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP=1
export VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP_KIND=mc2_dispatcher_only
export VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP=0
export VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD=1
export VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD=1
export VLLM_ASCEND_MODE1_PARITY_SYNC_BEFORE_DEVICE_PG_RETIRE=1
export VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE=1
export VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_RETIRE=1
export VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=0
export VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK=1
export VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS=0
export VLLM_ASCEND_MODE1_PARITY_FORCE_CLEANUP_AFTER_FLOOR_GROUP_RELEASE=1
export VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES=0
export VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_RESTORE_MOE_DISPATCH_HEADROOM_BYTES=0
export VLLM_ASCEND_FIRST_LIVE_PREFILL_HEADROOM_BYTES=0
export VLLM_ASCEND_FIRST_LIVE_PREFILL_LOW_FLOOR_HEADROOM_BYTES=0
export VLLM_ASCEND_EXTRA_ELASTIC_SAFETY_HEADROOM_BYTES=0
export VLLM_ASCEND_FLOOR_PREALLOC_HEADROOM_SAFETY_BYTES=0
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1 VLLM_ASCEND_USE_NATIVE_QWEN3_MOE=0
export VLLM_ENABLE_EXPERT_PARALLEL=1 VLLM_ENABLE_MC2=1 USE_ALLTOALL_OVERLAP=1 ALL_TO_ALL_RESHARD=1
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=800 RAY_DEDUP_LOGS=0 OMP_NUM_THREADS=4
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_EVENT_ENABLE=0 ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
export MODEL_PATH=/data/Qwen3-30B-A3B DISTCP_PATH=/data/Qwen3-30B-A3B_megatron
export TRAIN_FILE_ORIG=/data/deepscaler/train.parquet TEST_FILE=/data/deepscaler/test.parquet
export TRAINER_TOTAL_EPOCHS=1 PLAN_STEPS=5 TRAIN_BATCH_SIZE=32 ROLLOUT_N=16
export MIN_ADAPTIVE_FLOOR=$([[ "$VARIANT" == natural_f4 ]] && echo 4 || echo 2)
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=16384 MAX_RESPONSE_LEN=16384
export ROLLOUT_MAX_NUM_SEQS=32 ROLLOUT_MAX_NUM_BATCHED_TOKENS=17408
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 DATASET_FRACTION_FOR_ORACLE=0.005
export SAVE_CKPT_ENABLE=0 TRAINER_SAVE_FREQ=-1 DYNAMIC_ENABLE_CKPT_CHAIN=0
export VERL_SIDECAR_ENABLE=0 VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE=0
export CHECK_LOCAL_INPUTS=1
for scenario in under_0p7 mixed; do
    OUT="$ROOT/$scenario"
    mkdir -p "$OUT"
    export BASELINE_DIR="$ROOT/scenarios/$scenario"
    export BASELINE_DIRS="$BASELINE_DIR"
    export PLAN_DIR="$OUT/oracle"
    export OUTPUT_ROOT="$OUT"
    export OUTPUT_SUBDIR="run"
    export DYNAMIC_RUN_NAME="forecast_${scenario}_${VARIANT}_seed505"
    printf 'scenario=%s\nforecast_condition=%s\nseed=505\nsteps=5\nactor_frozen=true\n' "$scenario" "$scenario" > "$OUT/protocol.env"
    # The common epoch0 checkpoint is already at global_step=5.  A five-step
    # follow-up must target global_step=10.  Targeting 5 would be a no-op.
    set +e
    bash "$CHILD_SCRIPT" \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="$COMMON/checkpoints/qwen3moe_for_eagle3/global_step_5" \
        trainer.total_training_steps=10 \
        actor_rollout_ref.actor.optim.lr=0.0 \
        actor_rollout_ref.rollout.seed=505 \
        trainer.total_epochs=2 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 > "$OUT/run.log" 2>&1
    rc=$?
    set -e
    printf 'exit_code=%s\n' "$rc" >> "$OUT/protocol.env"
    if (( rc != 0 )); then
        echo "forecast arm failed:$scenario rc=$rc; evidence retained" >&2
        exit "$rc"
    fi
    rollout_count=$(find "$OUT/run/rollout_data" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
    length_count=$(find "$OUT/run/rollout_length" -maxdepth 1 -type f -name 'length_*.txt' | wc -l)
    if [[ "$rollout_count" -ne 5 || "$length_count" -ne 5 ]]; then
        echo "forecast arm produced incomplete five-step artifacts: scenario=$scenario rollout_data=$rollout_count rollout_length=$length_count" >&2
        printf 'artifact_validation=failed\nrollout_data_files=%s\nrollout_length_files=%s\n' "$rollout_count" "$length_count" >> "$OUT/protocol.env"
        exit 4
    fi
    printf 'artifact_validation=passed\nrollout_data_files=%s\nrollout_length_files=%s\n' "$rollout_count" "$length_count" >> "$OUT/protocol.env"
done
printf 'status=COMPLETE\n' >> "$ROOT/protocol.env"
echo "forecast robustness runs complete root=$ROOT"
