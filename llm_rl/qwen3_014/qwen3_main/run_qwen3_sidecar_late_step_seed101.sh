#!/usr/bin/env bash
# One-seed, late-step v0.14 FULL_DECODE_ONLY graph + Sidecar pair.
# The only arm difference is VERL_SIDECAR_ENABLE=0/1.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-run}
case "$ACTION" in smoke|run|verify) ;; *) echo "usage: $0 {smoke|run|verify}" >&2; exit 2 ;; esac

COMMON_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}
SOURCE_ROOT=${LATE_STEP_SOURCE_ROOT:-/data/adafloor_graph_results/mlsys_v014_graph_sidecar_gated_pairs_20261022T161500Z/seed_101/off/primary_planned_f4_noguard_seed101_off/epoch_001_mode1_planned}
SOURCE_PLAN_STEP=${LATE_STEP_SOURCE_PLAN_STEP:-4}
ROOT_OUT=${LATE_STEP_OUTPUT_ROOT:-/data/adafloor_graph_results/mlsys_v014_graph_sidecar_late_step_seed101_$(date -u +%Y%m%dT%H%M%SZ)}
ASSET_DIR="$ROOT_OUT/assets"
BASE_SCRIPT="$ROOT/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"
MODEL_PATH=${MODEL_PATH:-/data/Qwen3-30B-A3B}
SOURCE_TRAIN="$SOURCE_ROOT/oracle/length_sorted_train.parquet"
SOURCE_PLAN="$SOURCE_ROOT/oracle/length_sorted_rank_plan.json"
SOURCE_ORACLE="$SOURCE_ROOT/oracle/length_sorted_length_oracle.json"
CKPT="$COMMON_ROOT/epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5"

prepare_assets() {
    [[ -f "$COMMON_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]] || { echo "missing common epoch0 marker" >&2; exit 2; }
    [[ -d "$CKPT/actor" ]] || { echo "missing common epoch0 checkpoint: $CKPT" >&2; exit 2; }
    for f in "$SOURCE_TRAIN" "$SOURCE_PLAN" "$SOURCE_ORACLE"; do [[ -f "$f" ]] || { echo "missing source asset: $f" >&2; exit 2; }; done
    mkdir -p "$ASSET_DIR"
    python3 "$ROOT/tools/build_late_step_sidecar_assets.py" \
        --source-plan "$SOURCE_PLAN" --source-train "$SOURCE_TRAIN" \
        --source-oracle "$SOURCE_ORACLE" --source-step "$SOURCE_PLAN_STEP" \
        --output-dir "$ASSET_DIR" > "$ROOT_OUT/asset_build.json"
}

smoke() {
    python3 -m py_compile "$ROOT/tools/build_late_step_sidecar_assets.py"
    bash -n "$0" "$BASE_SCRIPT" "$ROOT/internal/watch_elastic_shrink_and_run_sidecar.sh"
    prepare_assets
    python3 - "$ASSET_DIR/asset_manifest.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['selected_floor']==8; assert x['source_plan_step']==4
print('SMOKE_PASS late_step=4 floor=8 released=0..7')
PY
}

run_arm() (
    set -euo pipefail
    local arm=$1
    local out="$ROOT_OUT/$arm"
    [[ ! -e "$out" ]] || { echo "refusing to overwrite $out" >&2; exit 2; }
    mkdir -p "$out/cache" "$out/logs" "$out/oracle"
    # The causal window gate resolves the plan beside the primary train log;
    # publish immutable copies there so it never falls back to
    # prediction_unavailable merely because the assets live under the shared
    # experiment directory.
    cp -- "$ASSET_DIR/late_step_plan.json" "$out/oracle/length_sorted_rank_plan.json"
    cp -- "$ASSET_DIR/late_step_oracle.json" "$out/oracle/length_sorted_length_oracle.json"
    export OUTPUT_ROOT="$ROOT_OUT" OUTPUT_SUBDIR="$arm"
    export TRAIN_FILE="$ASSET_DIR/late_step_train.parquet" TEST_FILE="$ASSET_DIR/late_step_train.parquet"
    export TRAIN_BATCH_SIZE=32 ACTOR_PPO_MINI_BATCH_SIZE=32 DATASET_FRACTION=1.0
    export MAX_RESPONSE_LENGTH=16384 ROLLOUT_N=16 ROLLOUT_MAX_NUM_SEQS=32
    export TRAINER_TOTAL_EPOCHS=1 TRAINER_TOTAL_TRAINING_STEPS=1 SAVE_CKPT_ENABLE=0 GIT_TRACK_RUN_LOGS=0
    export VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1
    export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=0 VLLM_ROLLOUT_DATA_REBALANCE=0 VLLM_ROLLOUT_LENGTH_BALANCE=0
    export VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS=0 VLLM_ROLLOUT_ASYNC_SCHEDULING=false
    export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=false VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true
    export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=0.90 VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
    export VLLM_ROLLOUT_FREE_CACHE_ENGINE=True VLLM_ROLLOUT_NATIVE_SLEEP_MODE=1 VLLM_ROLLOUT_SLEEP_LEVEL=2
    export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE=1 VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_RETAIN=0
    export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_KV_FIRST=1 VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
    export VLLM_ASCEND_NATIVE_SLEEP_PARK_IDLE=1
    export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1 VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
    export VLLM_ASCEND_ELASTIC_ACLGRAPH=1 VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1 VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_DYNAMIC_DISABLE=0 VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE=0
    export VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC=1 VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC=1 VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=0
    export VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE=1 VLLM_ASCEND_MODE1_PREEMPTION_AUDIT=1
    export VLLM_ASCEND_MODE1_STEP_KV_CAP_INCLUDES_PLANNED_HEADROOM=1
    # The planner's logical step-4 cap is 235264.  The validated v0.14
    # runtime promotes floor8 to the physical 349952-token cap and reserves
    # the planned group headroom; otherwise the startup validator sees only
    # eight resident local experts and rejects the required sixteen slots.
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=349952 VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=380800
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16=380800 VANILLA_KV_TOKENS_PER_RANK=380800
    export VLLM_ASCEND_MODE1_PARITY_PREFILL_PLANNED_EXPERT_SLOTS=1
    export VLLM_ASCEND_MODE1_PARITY_PREFILL_PLANNED_EXPERT_SLOTS_ON_RESIZE=1
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS=147456
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8=114688
    export VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE=1
    export VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP=1
    export VLLM_ASCEND_MODE1_PARITY_FORCE_DESTROY_PLANNED_PRUNE=1
    export VLLM_ASCEND_MODE1_PARITY_FORCE_CLEANUP_AFTER_FLOOR_GROUP_RELEASE=1
    export VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE=1
    export VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE=1
    export VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD=1
    export VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD=1
    export VLLM_ASCEND_MODE1_PARITY_SYNC_BEFORE_DEVICE_PG_RETIRE=1
    export VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE=1
    export VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_RETIRE=1
    export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=1
    export VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS=1
    # Planned floor8 needs 16 local slots per rank at Full16 startup.  The
    # formal v0.14 Planned envelope obtains those slots from 384 global
    # redundant experts (not the 128-expert default used by this first draft).
    export VLLM_ASCEND_INIT_REDUNDANCY_EXPERT=384
    export VLLM_ASCEND_FUSED_MOE_BACKEND=legacy VLLM_ASCEND_USE_LEGACY_FUSED_MOE=1
    export VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=1 VLLM_ASCEND_MC2_TOKENS_CAPACITY=512
    export VLLM_ASCEND_MC2_MIN_EP_SIZE=8 VLLM_ASCEND_FORCE_ALLTOALL_MOE=0 VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1
    export VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1 VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM=0
    export VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1 VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH="$ROOT/vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend"
    export VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION=/workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
    export VLLM_ENABLE_GRAPH_MODE=0 TASK_QUEUE_ENABLE=1 VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1 ROLLOUT_ENFORCE_EAGER=False
    export ADAFLOOR_GRAPH_SIDECAR_ENABLE=1 ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY ADAFLOOR_GRAPH_CAPTURE_SIZES='[5,8,12,17,19,23,27,32]'
    export VLLM_ASCEND_ACLGRAPH_BUCKET_COUNT=8
    export VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER=0 VLLM_ROLLOUT_UNSAFE_TRUE_GRAPH_WITH_RESAMPLER=1
    export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=false VLLM_ASCEND_RUNTIME_PROGRESS_INTERVAL=256
    export VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=1 VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=1
    export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=1 VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
    # Formal Planned-F4 keeps the minimum compute group at 4 even when this
    # selected late step stops at floor8; that is what provisions the startup
    # redundancy map used by the precreated floor group.
    export VLLM_ASCEND_ELASTIC_MOE_MODE=lossless VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4 VLLM_ASCEND_SHRINK_AWARE_ENABLE=1
    export VLLM_ASCEND_SHRINK_AWARE_MODE=staged VLLM_ASCEND_SHRINK_AWARE_STAGES=8 VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY=manual
    export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=planned VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=8,9,10,11,12,13,14,15
    export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=8,9,10,11,12,13,14,15 VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS='[[8,9,10,11,12,13,14,15]]'
    export VLLM_ASCEND_SHRINK_AWARE_MIN_WINDOW_SECONDS=0 VLLM_ASCEND_SHRINK_AWARE_LOGGING=1
    # The single-step replay carries the exact source role plan and lets the
    # runtime create its communicators at the shrink boundary.  Do not force
    # the standalone startup precreate validator here: that validator expects
    # the multi-stage epoch driver rather than this extracted floor8 window.
    export VLLM_ASCEND_MODE1_PRECREATE_PLANNED_FLOOR_GROUPS=0 VLLM_ASCEND_MODE1_PRECREATE_COMM_WARMUP=1
    export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1 VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
    export VLLM_ROLLOUT_PHASE_MEMORY_LOG=1 VLLM_ROLLOUT_STAGE_DEBUG=1 VERL_LOGGING_LEVEL=INFO
    export VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 VERL_SIDECAR_ENABLE=$([[ "$arm" == on ]] && echo 1 || echo 0)
    export VERL_SIDECAR_WINDOW_GATE_ENABLE=1 VERL_SIDECAR_WINDOW_GATE_MIN_SECONDS=30 VERL_SIDECAR_START_ONCE=1
    export VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE=1 VERL_SIDECAR_REQUIRE_SHRINK_QUORUM=1 VERL_SIDECAR_SHRINK_QUORUM_SIZE=16
    export VERL_SIDECAR_WORLD_SIZE=16 VERL_SIDECAR_EXPECTED_ACTIVE_RANKS=8 VERL_SIDECAR_MODEL_PATH=/data/Qwen2.5-1.5B-Instruct
    export VERL_SIDECAR_PROMPTS_FILE=/data/gsm8k VERL_SIDECAR_DATA_SPLIT=train VERL_SIDECAR_PARALLEL_MODE=dp
    export VERL_SIDECAR_TENSOR_PARALLEL_SIZE=1 VERL_SIDECAR_DATA_PARALLEL_SIZE=1 VERL_SIDECAR_REPLICA_COUNT=8 VERL_SIDECAR_ENABLE_EXPERT_PARALLEL=0
    export VERL_SIDECAR_N=1 VERL_SIDECAR_TEMPERATURE=0.0 VERL_SIDECAR_TOP_P=1.0 VERL_SIDECAR_GPU_MEMORY_UTILIZATION=0.80
    export VERL_SIDECAR_MAX_MODEL_LEN=6144 VERL_SIDECAR_MAX_TOKENS=4096 VERL_SIDECAR_MAX_NUM_SEQS=128 VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=65536
    export VERL_SIDECAR_MAX_PROMPTS=1024 VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA=128 VERL_SIDECAR_GENERATE_CHUNK_SIZE=128
    export VERL_SIDECAR_REPEAT_UNTIL_KILLED=1 VERL_SIDECAR_STREAM_CHECKPOINT=1 VERL_SIDECAR_RESET_OUTPUT_ON_START=0
    export VERL_SIDECAR_LOG_DIR="$out/sidecar" VERL_SIDECAR_LEASE_LOG="$out/sidecar/lease.log"
    export VERL_SIDECAR_SINGLETON_GROUP_REUSE=0 VERL_SIDECAR_ENGINE_MULTIPROCESSING=1
    # Keep the primary 16-rank HCCL allocator disjoint from stale Ray/HCCL
    # sessions.  This is the same allocator envelope used by the formal v0.14
    # paired experiment; without it HcclGetRootInfo can fail before rollout.
    local port_offset=0
    [[ "$arm" == on ]] && port_offset=1
    export HCCL_IF_BASE_PORT=12000 VERL_HCCL_IF_BASE_PORT_START=12000 VERL_HCCL_IF_BASE_PORT_BLOCK=16384
    export MASTER_PORT=$((26000 + port_offset)) VERL_MASTER_PORT_START=$((28416 + port_offset))
    export RAY_TMPDIR="/tmp/late_sc101_${arm}_$$" XDG_CACHE_HOME="$out/cache/xdg" TORCHAIR_CACHE_HOME="$out/cache/torchair" TRITON_CACHE_DIR="$out/cache/triton" ASCEND_CACHE_PATH="$out/cache/ascend" ASCEND_WORK_PATH="$out/cache/work"
    export PYTHONPATH="$ROOT"
    mkdir -p "$RAY_TMPDIR" "$XDG_CACHE_HOME" "$TORCHAIR_CACHE_HOME" "$TRITON_CACHE_DIR" "$ASCEND_CACHE_PATH" "$ASCEND_WORK_PATH"
    local hydra=(trainer.resume_mode=resume_path "trainer.resume_from_path=$CKPT" actor_rollout_ref.actor.optim.lr=0.0 actor_rollout_ref.actor.use_kl_loss=False algorithm.use_kl_in_reward=False
      actor_rollout_ref.rollout.seed=101 actor_rollout_ref.rollout.temperature=0.9 actor_rollout_ref.rollout.top_p=0.9 actor_rollout_ref.rollout.top_k=50 actor_rollout_ref.rollout.max_model_len=17408
      actor_rollout_ref.rollout.max_num_batched_tokens=17408 actor_rollout_ref.rollout.max_num_seqs=32 actor_rollout_ref.rollout.n=16 actor_rollout_ref.rollout.gpu_memory_utilization=0.90
      +actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=2734 +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=128
      actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_scheduling=true actor_rollout_ref.rollout.shrink_aware.target_policy=planned actor_rollout_ref.rollout.shrink_aware.shrink_aware_mode=staged
      actor_rollout_ref.rollout.shrink_aware.shrink_stages='[8]' actor_rollout_ref.rollout.shrink_aware.survivor_selection_policy=manual
      actor_rollout_ref.rollout.shrink_aware.intermediate_survivor_ranks='[8,9,10,11,12,13,14,15]' actor_rollout_ref.rollout.shrink_aware.final_survivor_ranks='[8,9,10,11,12,13,14,15]'
      actor_rollout_ref.rollout.shrink_aware.length_prediction_source=oracle_trace "actor_rollout_ref.rollout.shrink_aware.oracle_trace_path=$ASSET_DIR/late_step_oracle.json"
      actor_rollout_ref.rollout.shrink_aware.assignment_policy=optimized_rank_plan "actor_rollout_ref.rollout.shrink_aware.optimized_rank_plan_path=$ASSET_DIR/late_step_plan.json"
      +actor_rollout_ref.rollout.shrink_aware.aclgraph_bucket_plan_path=$ASSET_DIR/late_step_plan.json +actor_rollout_ref.rollout.shrink_aware.aclgraph_bucket_count=8 actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_logging=true
      trainer.total_epochs=1 trainer.total_training_steps=1 +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY actor_rollout_ref.rollout.cudagraph_capture_sizes='[5,8,12,17,19,23,27,32]' actor_rollout_ref.rollout.async_scheduling=false 'trainer.logger=["console"]')
    echo "[late sidecar] arm=$arm source_step=$SOURCE_PLAN_STEP gate=60s output=$out"
    bash "$BASE_SCRIPT" "${hydra[@]}"
)

verify() {
    python3 - "$ROOT_OUT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);
for arm in ('off','on'):
 p=root/arm
 logs=list((p/'logs').glob('*.txt'))
 if not logs: raise SystemExit(f'missing log {p}')
 text=max(logs,key=lambda x:x.stat().st_mtime).read_text(errors='replace')
 if 'Training Progress: 100%' not in text: raise SystemExit(f'incomplete {arm}')
 if 'response/aborted_ratio:0.0' not in text: raise SystemExit(f'aborted responses {arm}')
 print(arm,'PASS',p)
lease=root/'on'/'sidecar'/'lease.log'
if lease.exists(): print('sidecar_lease=',lease)
PY
}

if [[ "$ACTION" == smoke ]]; then smoke; exit 0; fi
prepare_assets
if [[ "$ACTION" == verify ]]; then verify; exit 0; fi
mkdir -p "$ROOT_OUT"
run_arm off
run_arm on
verify
