#!/usr/bin/env bash
set -euo pipefail

# Keep each long child epoch on the exact launcher revision with which it
# started, including its post-training artifact summarization.
if [[ "${ADAFLOOR_FLOOR2_CHILD_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    floor2_child_source=$(realpath "${BASH_SOURCE[0]}")
    floor2_child_snapshot=$(mktemp "${floor2_child_source}.run-snapshot.XXXXXX")
    cp -- "$floor2_child_source" "$floor2_child_snapshot"
    chmod 700 "$floor2_child_snapshot"
    set +e
    ADAFLOOR_FLOOR2_CHILD_SNAPSHOT_ACTIVE=1 \
        "$floor2_child_snapshot" "$@"
    floor2_child_rc=$?
    set -e
    rm -f -- "$floor2_child_snapshot"
    exit "$floor2_child_rc"
fi

usage() {
    cat <<'EOF'
Usage:
  ./run_mode1_local_length_sorted_e2e_adaptive_floor2.sh [extra hydra args...]

Mode=1 adaptive-floor length-sorted end-to-end rollout-time experiment:
  1. Reads baseline rollout responses from mode1_baseline_random_batch_floor4.
  2. Sorts prompts by baseline max response length and chunks every 32 prompts
     into one rollout step.
  3. Assigns each 32-prompt step to 16 ranks with KV-aware pair matching.
  4. Selects each step's lowest KV-feasible floor from 2/4/8/16.
     Floor=2 is executed as 16->8->4->2, never as a direct 8->2 jump.
  5. Runs mode=1 with per-step survivor plans.
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
PATCH_TREE="${PATCH_TREE:-$REPO_ROOT}"
LAUNCHER="${LOCAL_TEST_LAUNCHER:-$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh}"
HIERARCHICAL_DP2_EP8="${HIERARCHICAL_DP2_EP8:-0}"
DP2_EP8_GROUPING="${DP2_EP8_GROUPING:-length_sorted}"
DP2_EP8_SEED="${DP2_EP8_SEED:-20270731}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

MODEL_PATH="${MODEL_PATH:-/data/Qwen3-30B-A3B}"
DISTCP_PATH="${DISTCP_PATH:-/data/Qwen3-30B-A3B_megatron}"
TRAIN_FILE_ORIG="${TRAIN_FILE_ORIG:-/data/deepscaler/train.parquet}"
TEST_FILE="${TEST_FILE:-/data/deepscaler/test.parquet}"
BASELINE_DIR="${BASELINE_DIR:-$REPO_ROOT/mode1_baseline_random_batch_floor4}"
BASELINE_DIRS="${BASELINE_DIRS:-$BASELINE_DIR}"
BASELINE_DIRS="${BASELINE_DIRS//,/:}"
IFS=':' read -r -a BASELINE_DIR_ARRAY <<< "$BASELINE_DIRS"

if [[ "${CHECK_LOCAL_INPUTS:-1}" == "1" ]]; then
    for required_path in "$MODEL_PATH" "$DISTCP_PATH" "$TRAIN_FILE_ORIG" "$TEST_FILE"; do
        if [[ ! -e "$required_path" ]]; then
            echo "missing local input path: $required_path" >&2
            exit 2
        fi
    done
    for required_path in "${BASELINE_DIR_ARRAY[@]}"; do
        if [[ -n "$required_path" && ! -e "$required_path" ]]; then
            echo "missing baseline input path: $required_path" >&2
            exit 2
        fi
    done
fi

export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=1
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2
export VLLM_ASCEND_SHRINK_AWARE_ENABLE=1
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS="${VLLM_ASCEND_REGISTER_CUSTOM_MODELS:-1}"
export VLLM_ASCEND_USE_NATIVE_QWEN3_MOE=0
export VLLM_ASCEND_SHRINK_AWARE_MODE=staged
if [[ "$HIERARCHICAL_DP2_EP8" == "1" ]]; then
    export VLLM_DP_SIZE=8
    export VLLM_ASCEND_SHRINK_AWARE_STAGES=4,2
    export VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=4,5,6,7
    export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=6,7
else
    export VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4,2
    export VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=8,9,10,11,12,13,14,15
    export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=14,15
fi
export VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY=manual
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY="${VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY:-natural}"
export VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE="${VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE:-$([[ "${VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY}" =~ ^(planned|fixed|plan)$ ]] && echo 1 || echo 0)}"
export VLLM_EAGER_BASELINE_NO_RESAMPLE=0
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=0
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
export VLLM_ROLLOUT_EARLY_STOP_ENABLE="${VLLM_ROLLOUT_EARLY_STOP_ENABLE:-0}"
export VERL_SIDECAR_ENABLE="${VERL_SIDECAR_ENABLE:-0}"
export VLLM_MOE_PATTERN_STATS=0
export VLLM_MOE_STATS=0
export VLLM_MOE_STATS_TIMING=0
export VLLM_ASCEND_NATIVE_MOE_TOPK_DEBUG=0

export VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES="${VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES:-0}"
export VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES="${VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES:-0}"
export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_HEADROOM_BYTES="${VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_HEADROOM_BYTES:-0}"
export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_LOW_FLOOR_HEADROOM_BYTES="${VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_LOW_FLOOR_HEADROOM_BYTES:-0}"
export VLLM_ASCEND_POST_SHRINK_PREFILL_ALLTOALL_HEADROOM_BYTES="${VLLM_ASCEND_POST_SHRINK_PREFILL_ALLTOALL_HEADROOM_BYTES:-0}"
export VLLM_ASCEND_POST_RESTORE_DP_HEADROOM_BYTES="${VLLM_ASCEND_POST_RESTORE_DP_HEADROOM_BYTES:-0}"
export VLLM_ASCEND_POST_RESTORE_EP_HEADROOM_BYTES="${VLLM_ASCEND_POST_RESTORE_EP_HEADROOM_BYTES:-0}"
export VLLM_ASCEND_POST_RESTORE_MOE_DISPATCH_HEADROOM_BYTES="${VLLM_ASCEND_POST_RESTORE_MOE_DISPATCH_HEADROOM_BYTES:-0}"
export VLLM_ASCEND_FIRST_LIVE_PREFILL_HEADROOM_BYTES="${VLLM_ASCEND_FIRST_LIVE_PREFILL_HEADROOM_BYTES:-0}"
export VLLM_ASCEND_FIRST_LIVE_PREFILL_LOW_FLOOR_HEADROOM_BYTES="${VLLM_ASCEND_FIRST_LIVE_PREFILL_LOW_FLOOR_HEADROOM_BYTES:-0}"
export VLLM_ASCEND_EXTRA_ELASTIC_SAFETY_HEADROOM_BYTES="${VLLM_ASCEND_EXTRA_ELASTIC_SAFETY_HEADROOM_BYTES:-0}"
export VLLM_ASCEND_FLOOR_PREALLOC_HEADROOM_SAFETY_BYTES="${VLLM_ASCEND_FLOOR_PREALLOC_HEADROOM_SAFETY_BYTES:-0}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS:-377344}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2:-147456}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4:-280576}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8:-377344}"
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE="${VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE:-1}"
export VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE="${VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE:-0}"
export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE="${VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE:-1}"
export VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE="${VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE:-0}"
export VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK="${VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK:-0}"
export VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS="${VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS:-0}"
export VLLM_ASCEND_MODE1_PARITY_FORCE_CLEANUP_AFTER_FLOOR_GROUP_RELEASE="${VLLM_ASCEND_MODE1_PARITY_FORCE_CLEANUP_AFTER_FLOOR_GROUP_RELEASE:-1}"
export VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP="${VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP:-0}"
export VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE="${VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE:-0}"
export VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE="${VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE:-1}"
export VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC="${VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC:-1}"
export VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE="${VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE:-1}"
export VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE="${VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE:-1}"
export VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP="${VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP:-1}"
export VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP_KIND="${VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP_KIND:-mc2_dispatcher_only}"
export VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP="${VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP:-0}"
export VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG="${VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG:-0}"
export VLLM_ASCEND_MODE1_FULLWORLD_RESTORE_DIAG="${VLLM_ASCEND_MODE1_FULLWORLD_RESTORE_DIAG:-0}"
export VLLM_ASCEND_MODE1_FULLWORLD_RESTORE_DIAG_DISPATCH_BUDGET="${VLLM_ASCEND_MODE1_FULLWORLD_RESTORE_DIAG_DISPATCH_BUDGET:-4}"
export VLLM_ASCEND_MODE1_STEP_TIMELINE_LOG="${VLLM_ASCEND_MODE1_STEP_TIMELINE_LOG:-0}"
export VLLM_ASCEND_MODE1_COMM_CACHE_STATE_DETAILED="${VLLM_ASCEND_MODE1_COMM_CACHE_STATE_DETAILED:-0}"
export VLLM_ASCEND_MODE1_SKIP_SAME_BLOCK_KV_RESIZE="${VLLM_ASCEND_MODE1_SKIP_SAME_BLOCK_KV_RESIZE:-1}"
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE="${VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE:-0}"
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS="${VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS:-dp,ep}"
export VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD="${VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD:-1}"
export VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD="${VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD:-1}"
export VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY="${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY:-0}"
export VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE="${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE:-0}"
export VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD="${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD:-0}"
export VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH="${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH:-0}"
export VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES="${VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES:-1,2,4,8}"
export VLLM_ASCEND_MODE1_PARITY_SYNC_BEFORE_DEVICE_PG_RETIRE="${VLLM_ASCEND_MODE1_PARITY_SYNC_BEFORE_DEVICE_PG_RETIRE:-1}"
export VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE="${VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE:-1}"
export VLLM_ASCEND_MODE1_PARITY_PRESERVE_DEVICE_PG_ON_FULL_RESTORE="${VLLM_ASCEND_MODE1_PARITY_PRESERVE_DEVICE_PG_ON_FULL_RESTORE:-0}"
export VLLM_ASCEND_MODE1_PARITY_PRESERVE_GROUP_REFS_ON_FULL_RESTORE="${VLLM_ASCEND_MODE1_PARITY_PRESERVE_GROUP_REFS_ON_FULL_RESTORE:-0}"
export VLLM_ASCEND_MODE1_PARITY_SKIP_CLEANUP_ON_PRESERVED_FULL_RESTORE="${VLLM_ASCEND_MODE1_PARITY_SKIP_CLEANUP_ON_PRESERVED_FULL_RESTORE:-0}"
export VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_RETIRE="${VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_RETIRE:-1}"
export VLLM_ASCEND_CUSTOM_MODE1_DEBUG=0
export VLLM_ASCEND_CUSTOM_MODE1_TIMING_EVENTS=0
export VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG=0
export VLLM_ASCEND_CUSTOM_MODE1_ROLLOUT_RELOAD_DIAG="${VLLM_ASCEND_CUSTOM_MODE1_ROLLOUT_RELOAD_DIAG:-0}"
export VLLM_ASCEND_CUSTOM_MODE1_GLOBAL_TENSOR_SCAN="${VLLM_ASCEND_CUSTOM_MODE1_GLOBAL_TENSOR_SCAN:-0}"
export VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT="${VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT:-1}"
export VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT="${VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT:-0}"
export VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT="${VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT:-0}"
export VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS="${VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS:-8}"
export VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC="${VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC:-0}"

export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-47241}"
export MASTER_PORT="${MASTER_PORT:-26240}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-47241}"

export HOME="$REPO_ROOT"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_length_sorted_e2e_adaptive_floor2}"
export CONFIG_DIR="$PATCH_TREE/verl/trainer/config"
export PYTHONPATH="$PATCH_TREE${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH DISTCP_PATH TEST_FILE

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
ROLLOUT_N="${ROLLOUT_N:-16}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.9}"
TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
PLAN_STEPS="${PLAN_STEPS:-5}"
DATASET_FRACTION_FOR_ORACLE="${DATASET_FRACTION_FOR_ORACLE:-0.005}"
SHRINK_AWARE_LOGGING="${SHRINK_AWARE_LOGGING:-false}"
MAX_RANK_PEAK_TOKENS="${MAX_RANK_PEAK_TOKENS:-$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS}"
FLOOR_KV_CAPS="${FLOOR_KV_CAPS:-2:$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2,4:$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4,8:$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8,16:$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS}"
if [[ "$HIERARCHICAL_DP2_EP8" == "1" ]]; then
    FLOOR_KV_CAPS="${DP2_EP8_FLOOR_KV_CAPS:-2:$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2,4:$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4,8:$VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8}"
fi
MIN_ADAPTIVE_FLOOR="${MIN_ADAPTIVE_FLOOR:-2}"
ACTIVE_PEAK_SAFETY_FACTOR="${ACTIVE_PEAK_SAFETY_FACTOR:-1.16}"
LENGTH_EMA_DECAY="${LENGTH_EMA_DECAY:-0.3}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-16384}"
TAIL_GUARD_RATIO_QUANTILE="${TAIL_GUARD_RATIO_QUANTILE:-0.95}"
TAIL_GUARD_RATIO_WINDOW="${TAIL_GUARD_RATIO_WINDOW:-3}"
TAIL_GUARD_DEFAULT_RATIO="${TAIL_GUARD_DEFAULT_RATIO:-1.20}"
TAIL_GUARD_MIN_CAP="${TAIL_GUARD_MIN_CAP:-4096}"
TAIL_GUARD_ROUND_TO="${TAIL_GUARD_ROUND_TO:-512}"
MAX_CROSS_STEP_REPAIR_SWAPS="${MAX_CROSS_STEP_REPAIR_SWAPS:-8}"
REPAIR_CANDIDATE_LIMIT="${REPAIR_CANDIDATE_LIMIT:-8}"
RANK_MATCHING_POLICY="${RANK_MATCHING_POLICY:-release_area}"
KV_SAFE_FIXED_FLOOR="${KV_SAFE_FIXED_FLOOR:-0}"
FORCE_SELECTED_FLOORS="${FORCE_SELECTED_FLOORS:-}"
if [[ -z "$FORCE_SELECTED_FLOORS" ]]; then
    # Full adaptive floor2 validation should only lower the minimum allowed
    # floor; it must not force every step to floor2.  Probe wrappers that need a
    # fixed sequence still set FORCE_SELECTED_FLOOR or FORCE_SELECTED_FLOORS.
    FORCE_SELECTED_FLOOR="${FORCE_SELECTED_FLOOR:-}"
else
    FORCE_SELECTED_FLOOR="${FORCE_SELECTED_FLOOR:-}"
fi
IGNORE_TAIL_TIES_AT_RESPONSE_CAP="${IGNORE_TAIL_TIES_AT_RESPONSE_CAP:-1}"

PLAN_DIR="${PLAN_DIR:-$REPO_ROOT/mode1_length_sorted_e2e_adaptive_floor2/oracle}"
OPT_TRAIN_FILE="${OPT_TRAIN_FILE:-$PLAN_DIR/length_sorted_train.parquet}"
OPT_RANK_PLAN="${OPT_RANK_PLAN:-$PLAN_DIR/length_sorted_rank_plan.json}"
OPT_SUMMARY="${OPT_SUMMARY:-$PLAN_DIR/length_sorted_rank_plan_summary.json}"
OPT_ORACLE="${OPT_ORACLE:-$PLAN_DIR/length_sorted_length_oracle.json}"
mkdir -p "$PLAN_DIR"

baseline_args=()
for baseline_dir_item in "${BASELINE_DIR_ARRAY[@]}"; do
    if [[ -n "$baseline_dir_item" ]]; then
        baseline_args+=(--baseline-dir "$baseline_dir_item")
    fi
done

PLAN_EXTRA_ARGS=()
if [[ -n "${FORCE_SELECTED_FLOOR:-}" && "${FORCE_SELECTED_FLOOR:-0}" != "0" ]]; then
    PLAN_EXTRA_ARGS+=(--force-selected-floor "$FORCE_SELECTED_FLOOR")
fi
if [[ -n "${FORCE_SELECTED_FLOORS:-}" ]]; then
    PLAN_EXTRA_ARGS+=(--force-selected-floors "$FORCE_SELECTED_FLOORS")
fi
if [[ "${IGNORE_TAIL_TIES_AT_RESPONSE_CAP,,}" =~ ^(1|true|yes|on)$ ]]; then
    PLAN_EXTRA_ARGS+=(--ignore-tail-ties-at-response-cap)
fi
PLAN_EXTRA_ARGS+=(--rank-matching-policy "$RANK_MATCHING_POLICY")
if [[ "$KV_SAFE_FIXED_FLOOR" != "0" ]]; then
    PLAN_EXTRA_ARGS+=(--kv-safe-fixed-floor "$KV_SAFE_FIXED_FLOOR")
fi
if [[ "${REQUIRE_COMPACT_HISTORY:-0}" == "1" ]]; then
    PLAN_EXTRA_ARGS+=(--require-compact-history)
fi
if [[ "${REPEAT_PROMPT_SET_TO_FILL:-0}" == "1" ]]; then
    PLAN_EXTRA_ARGS+=(--repeat-prompt-set-to-fill)
fi

if [[ "$HIERARCHICAL_DP2_EP8" == "1" ]]; then
    DP2_PLAN_EXTRA_ARGS=()
    if [[ "${REQUIRE_COMPACT_HISTORY:-0}" == "1" ]]; then
        DP2_PLAN_EXTRA_ARGS+=(--require-compact-history)
    fi
    if [[ "${DYNAMIC_DISABLE_TAIL_GUARD:-0}" == "1" ]]; then
        DP2_PLAN_EXTRA_ARGS+=(--disable-tail-guard)
    fi
    python3 -u "$PATCH_TREE/tools/build_dp2_ep8_adafloor_plan.py" \
        "${baseline_args[@]}" \
        --length-ema-decay "$LENGTH_EMA_DECAY" \
        --train-file "$TRAIN_FILE_ORIG" \
        --output-train "$OPT_TRAIN_FILE" \
        --output-plan "$OPT_RANK_PLAN" \
        --output-summary "$OPT_SUMMARY" \
        --output-oracle "$OPT_ORACLE" \
        --grouping "$DP2_EP8_GROUPING" \
        --seed "$DP2_EP8_SEED" \
        --steps "$PLAN_STEPS" \
        --responses-per-prompt "$ROLLOUT_N" \
        --dataset-fraction "$DATASET_FRACTION_FOR_ORACLE" \
        --floor-kv-caps "$FLOOR_KV_CAPS" \
        --active-peak-safety-factor "$ACTIVE_PEAK_SAFETY_FACTOR" \
        --max-response-len "$MAX_RESPONSE_LEN" \
        --tail-guard-ratio-quantile "$TAIL_GUARD_RATIO_QUANTILE" \
        --tail-guard-ratio-window "$TAIL_GUARD_RATIO_WINDOW" \
        --tail-guard-default-ratio "$TAIL_GUARD_DEFAULT_RATIO" \
        --tail-guard-min-cap "$TAIL_GUARD_MIN_CAP" \
        --tail-guard-round-to "$TAIL_GUARD_ROUND_TO" \
        --max-cross-step-repair-swaps "$MAX_CROSS_STEP_REPAIR_SWAPS" \
        --repair-candidate-limit "$REPAIR_CANDIDATE_LIMIT" \
        "${DP2_PLAN_EXTRA_ARGS[@]}"

    python3 - "$OPT_SUMMARY" "$PLAN_STEPS" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_steps = int(sys.argv[2])
steps = payload.get("steps", [])
floor_caps = {
    int(floor): float(cap)
    for floor, cap in payload.get("floor_kv_caps", {}).items()
}
failures = []
if payload.get("topology") != "external_dp2_ep8":
    failures.append(f"topology={payload.get('topology')!r}")
if len(steps) != expected_steps:
    failures.append(f"steps={len(steps)} expected={expected_steps}")
for step in steps:
    workers = step.get("workers", [])
    if len(workers) != 2:
        failures.append(f"step={step.get('step')} workers={len(workers)}")
    for worker in workers:
        floor = int(worker.get("selected_floor", -1))
        if floor not in (2, 4, 8):
            failures.append(
                f"step={step.get('step')} worker={worker.get('worker_id')} "
                f"invalid_floor={floor}")
        peaks = [
            float(value)
            for value in worker.get("rank_adjusted_peak_loads", {}).values()
        ]
        cap = float(worker.get("kv_cap", -1))
        expected_cap = floor_caps.get(floor)
        if expected_cap is None or cap != expected_cap:
            failures.append(
                f"step={step.get('step')} worker={worker.get('worker_id')} "
                f"floor={floor} kv_cap={cap} expected_cap={expected_cap}")
        if len(peaks) != 8 or max(peaks, default=float("inf")) > cap:
            failures.append(
                f"step={step.get('step')} worker={worker.get('worker_id')} "
                f"KV unsafe peaks={peaks} cap={cap}")
if failures:
    raise SystemExit("DP2 EP8 plan validation failed:\n  " + "\n  ".join(failures))
print(
    "[dp2-ep8] strict plan validation passed "
    f"grouping={payload['grouping']} "
    f"floors={[[w['selected_floor'] for w in s['workers']] for s in steps]}"
)
PY

    DP2_EP8_MIN_SELECTED_FLOOR=$(python3 - "$OPT_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
floors = [
    int(worker["selected_floor"])
    for step in payload["steps"]
    for worker in step["workers"]
]
if not floors:
    raise SystemExit("DP2 EP8 plan has no worker floor decisions")
print(min(floors))
PY
    )
    export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="$DP2_EP8_MIN_SELECTED_FLOOR"
    case "$DP2_EP8_MIN_SELECTED_FLOOR" in
        2)
            export VLLM_ASCEND_SHRINK_AWARE_STAGES=4,2
            export VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=4,5,6,7
            export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=6,7
            ;;
        4)
            export VLLM_ASCEND_SHRINK_AWARE_STAGES=4
            export VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=4,5,6,7
            export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=4,5,6,7
            ;;
        8)
            export VLLM_ASCEND_SHRINK_AWARE_STAGES=8
            export VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=0,1,2,3,4,5,6,7
            export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=0,1,2,3,4,5,6,7
            ;;
        *)
            echo "unsupported DP2 EP8 minimum selected floor: $DP2_EP8_MIN_SELECTED_FLOOR" >&2
            exit 2
            ;;
    esac
    echo "[dp2-ep8] runtime floor ladder minimum=$DP2_EP8_MIN_SELECTED_FLOOR stages=$VLLM_ASCEND_SHRINK_AWARE_STAGES policy=$VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY"
else
python3 -u "$PATCH_TREE/tools/build_mode1_length_sorted_e2e_plan.py" \
    "${baseline_args[@]}" \
    --length-ema-decay "$LENGTH_EMA_DECAY" \
    --train-file "$TRAIN_FILE_ORIG" \
    --output-train "$OPT_TRAIN_FILE" \
    --output-plan "$OPT_RANK_PLAN" \
    --output-summary "$OPT_SUMMARY" \
    --output-oracle "$OPT_ORACLE" \
    --steps "$PLAN_STEPS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --responses-per-prompt "$ROLLOUT_N" \
    --dataset-fraction "$DATASET_FRACTION_FOR_ORACLE" \
    --max-rank-peak-tokens "$MAX_RANK_PEAK_TOKENS" \
    --adaptive-floor \
    --min-adaptive-floor "$MIN_ADAPTIVE_FLOOR" \
    --floor-kv-caps "$FLOOR_KV_CAPS" \
    --active-peak-safety-factor "$ACTIVE_PEAK_SAFETY_FACTOR" \
    --max-response-len "$MAX_RESPONSE_LEN" \
    --tail-guard-ratio-quantile "$TAIL_GUARD_RATIO_QUANTILE" \
    --tail-guard-ratio-window "$TAIL_GUARD_RATIO_WINDOW" \
    --tail-guard-default-ratio "$TAIL_GUARD_DEFAULT_RATIO" \
    --tail-guard-min-cap "$TAIL_GUARD_MIN_CAP" \
    --tail-guard-round-to "$TAIL_GUARD_ROUND_TO" \
    --max-cross-step-repair-swaps "$MAX_CROSS_STEP_REPAIR_SWAPS" \
    --repair-candidate-limit "$REPAIR_CANDIDATE_LIMIT" \
    "${PLAN_EXTRA_ARGS[@]}"

if [[ -n "${PHYSICAL_FLOOR_KV_CAPS:-}" ]]; then
    python3 "$PATCH_TREE/tools/promote_kv_admission_caps.py" \
        --plan "$OPT_RANK_PLAN" \
        --summary "$OPT_SUMMARY" \
        --physical-caps "$PHYSICAL_FLOOR_KV_CAPS"
fi

python3 - "$OPT_SUMMARY" "$RANK_MATCHING_POLICY" "$KV_SAFE_FIXED_FLOOR" "${FORCE_SELECTED_FLOOR:-0}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_matching = sys.argv[2]
fixed_floor = int(sys.argv[3])
forced_floor = int(sys.argv[4] or 0)
steps = json.loads(summary_path.read_text(encoding="utf-8"))
failures = []
for step in steps:
    step_index = int(step.get("step", -1))
    matching = step.get("rank_matching_policy")
    selected_floor = int(step.get("selected_floor", -1))
    if matching != expected_matching:
        failures.append(
            f"step={step_index} matching={matching!r} expected={expected_matching!r}")
    if fixed_floor and selected_floor not in (fixed_floor, 16):
        failures.append(
            f"step={step_index} selected_floor={selected_floor} "
            f"expected one of ({fixed_floor}, 16)")
    if forced_floor and selected_floor != forced_floor:
        failures.append(
            f"step={step_index} selected_floor={selected_floor} "
            f"expected forced floor {forced_floor}")
if failures:
    raise SystemExit("baseline policy validation failed:\n  " + "\n  ".join(failures))
print(
    "[mode1 length-sorted e2e] baseline policy validation passed: "
    f"matching={expected_matching} fixed_floor={fixed_floor or 'adaptive'} "
    f"forced_floor={forced_floor or 'none'} "
    f"selected_floors={[int(step['selected_floor']) for step in steps]}"
)
PY
fi

if [[ "${EXPECT_NO_RESPONSE_CAPS:-0}" == "1" ]]; then
    if [[ "${VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE:-0}" != "0" ]]; then
        echo "no-response-cap validation failed: short-step cap is still enabled" >&2
        exit 5
    fi
    python3 - "$OPT_SUMMARY" "$MAX_RESPONSE_LEN" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
max_response_len = int(sys.argv[2])
payload = json.loads(summary_path.read_text(encoding="utf-8"))
steps = payload.get("steps", payload) if isinstance(payload, dict) else payload
failures = []
for step_index, step in enumerate(steps, start=1):
    enabled = bool(step.get("tail_guard_enabled", False))
    cap = int(step.get("tail_guard_response_cap", -1))
    if enabled or cap != max_response_len:
        failures.append(
            f"step={step_index} tail_guard_enabled={enabled} "
            f"tail_guard_response_cap={cap} expected={max_response_len}"
        )
if failures:
    raise SystemExit(
        "no-response-cap validation failed:\n  " + "\n  ".join(failures)
    )
print(
    f"[mode1 length-sorted e2e floor2] no-response-cap validation passed: "
    f"steps={len(steps)} cap={max_response_len}"
)
PY
fi

if [[ "${FAST_STEP_SUBSET:-0}" == "1" ]]; then
    FAST_STEP_SUBSET_STEPS="${FAST_STEP_SUBSET_STEPS:-1,5}"
    fast_slug="${FAST_STEP_SUBSET_STEPS//,/}"
    FAST_TRAIN_FILE="${FAST_TRAIN_FILE:-$PLAN_DIR/length_sorted_fast_steps_${fast_slug}_train.parquet}"
    FAST_RANK_PLAN="${FAST_RANK_PLAN:-$PLAN_DIR/length_sorted_fast_steps_${fast_slug}_rank_plan.json}"
    FAST_ORACLE="${FAST_ORACLE:-$PLAN_DIR/length_sorted_fast_steps_${fast_slug}_length_oracle.json}"
    python3 "$PATCH_TREE/tools/build_mode1_fast_step_subset.py" \
        --input-train "$OPT_TRAIN_FILE" \
        --input-plan "$OPT_RANK_PLAN" \
        --input-oracle "$OPT_ORACLE" \
        --output-train "$FAST_TRAIN_FILE" \
        --output-plan "$FAST_RANK_PLAN" \
        --output-oracle "$FAST_ORACLE" \
        --steps "$FAST_STEP_SUBSET_STEPS"
    OPT_TRAIN_FILE="$FAST_TRAIN_FILE"
    OPT_RANK_PLAN="$FAST_RANK_PLAN"
    OPT_ORACLE="$FAST_ORACLE"
fi

if [[ "$HIERARCHICAL_DP2_EP8" == "1" \
      && "${DP2_EP8_PLAN_ONLY:-0}" == "1" ]]; then
    echo "[dp2-ep8] plan-only complete train=$OPT_TRAIN_FILE plan=$OPT_RANK_PLAN summary=$OPT_SUMMARY"
    exit 0
fi

TRAIN_FILE="$OPT_TRAIN_FILE"
export TRAIN_FILE

if [[ "$HIERARCHICAL_DP2_EP8" == "1" ]]; then
    case "${DP2_EP8_MIN_SELECTED_FLOOR:-2}" in
        2)
            SHRINK_STAGES_ARG='[4,2]'
            INTERMEDIATE_RANKS_ARG='[4,5,6,7]'
            FINAL_RANKS_ARG='[6,7]'
            ;;
        4)
            SHRINK_STAGES_ARG='[4]'
            INTERMEDIATE_RANKS_ARG='[4,5,6,7]'
            FINAL_RANKS_ARG='[4,5,6,7]'
            ;;
        8)
            SHRINK_STAGES_ARG='[8]'
            INTERMEDIATE_RANKS_ARG='[0,1,2,3,4,5,6,7]'
            FINAL_RANKS_ARG='[0,1,2,3,4,5,6,7]'
            ;;
    esac
else
    SHRINK_STAGES_ARG="[${VLLM_ASCEND_SHRINK_AWARE_STAGES}]"
    INTERMEDIATE_RANKS_ARG="[${VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS}]"
    FINAL_RANKS_ARG="[${VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS}]"
fi

cd "$PATCH_TREE"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
tee_log="$REPO_ROOT/mode1_local_length_sorted_e2e_adaptive_floor2_${stamp}.log"

printf '[mode1 length-sorted e2e] runtime_cwd=%s\n' "$PATCH_TREE"
printf '[mode1 length-sorted e2e] launcher=%s tee_log=%s\n' "$LAUNCHER" "$tee_log"
printf '[mode1 length-sorted e2e] train=%s plan=%s summary=%s oracle=%s baseline=%s\n' \
    "$TRAIN_FILE" "$OPT_RANK_PLAN" "$OPT_SUMMARY" "$OPT_ORACLE" "$BASELINE_DIRS"
printf '[mode1 length-sorted e2e] model=%s distcp=%s test=%s\n' \
    "$MODEL_PATH" "$DISTCP_PATH" "$TEST_FILE"
printf '[mode1 length-sorted e2e adaptive] mode=1 global_floor=2 util=%s kv_caps=%s total_epochs=%s\n' \
    "$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    "$FLOOR_KV_CAPS" \
    "$TRAINER_TOTAL_EPOCHS"
printf '[mode1 length-sorted e2e adaptive] target_policy=%s fixed_topology_reuse=%s cache_floor_groups=%s keep_mc2_group_cache=%s\n' \
    "$VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY" \
    "$VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE" \
    "$VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS" \
    "$VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE"
printf '[mode1 length-sorted e2e adaptive] max_rank_peak_tokens=%s min_adaptive_floor=%s active_peak_safety_factor=%s max_response_len=%s shrink_not_required=true\n' \
    "$MAX_RANK_PEAK_TOKENS" "$MIN_ADAPTIVE_FLOOR" "$ACTIVE_PEAK_SAFETY_FACTOR" "$MAX_RESPONSE_LEN"
printf '[mode1 length-sorted e2e adaptive] force_selected_floor=%s ignore_tail_ties_at_response_cap=%s\n' \
    "$FORCE_SELECTED_FLOOR" "$IGNORE_TAIL_TIES_AT_RESPONSE_CAP"
printf '[mode1 length-sorted e2e adaptive] force_selected_floors=%s plan_steps=%s history_dirs=%s\n' \
    "${FORCE_SELECTED_FLOORS:-}" "$PLAN_STEPS" "${#BASELINE_DIR_ARRAY[@]}"
printf '[mode1 length-sorted e2e adaptive] rank_matching_policy=%s kv_safe_fixed_floor=%s\n' \
    "$RANK_MATCHING_POLICY" "$KV_SAFE_FIXED_FLOOR"
printf '[mode1 length-sorted e2e] max_cross_step_repair_swaps=%s repair_candidate_limit=%s\n' \
    "$MAX_CROSS_STEP_REPAIR_SWAPS" "$REPAIR_CANDIDATE_LIMIT"
printf '[mode1 length-sorted e2e] hierarchical_dp2_ep8=%s grouping=%s vllm_dp_size=%s\n' \
    "$HIERARCHICAL_DP2_EP8" "$DP2_EP8_GROUPING" "${VLLM_DP_SIZE:-16}"

if [[ "${MODE1_PLAN_ONLY:-0}" == "1" ]]; then
    echo "[mode1 length-sorted e2e adaptive floor2] plan-only complete"
    exit 0
fi

bash "$LAUNCHER" \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.shuffle=False \
    data.dataset_fraction=1.0 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="$DISTCP_PATH" \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="$DISTCP_PATH" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_scheduling=true \
    actor_rollout_ref.rollout.shrink_aware.shrink_aware_mode=staged \
    actor_rollout_ref.rollout.shrink_aware.shrink_stages="$SHRINK_STAGES_ARG" \
    actor_rollout_ref.rollout.shrink_aware.survivor_selection_policy=manual \
    actor_rollout_ref.rollout.shrink_aware.intermediate_survivor_ranks="$INTERMEDIATE_RANKS_ARG" \
    actor_rollout_ref.rollout.shrink_aware.final_survivor_ranks="$FINAL_RANKS_ARG" \
    actor_rollout_ref.rollout.shrink_aware.length_prediction_source=oracle_trace \
    actor_rollout_ref.rollout.shrink_aware.oracle_trace_path="$OPT_ORACLE" \
    actor_rollout_ref.rollout.shrink_aware.assignment_policy=optimized_rank_plan \
    actor_rollout_ref.rollout.shrink_aware.optimized_rank_plan_path="$OPT_RANK_PLAN" \
    actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_logging="$SHRINK_AWARE_LOGGING" \
    trainer.total_epochs="$TRAINER_TOTAL_EPOCHS" \
    "$@" 2>&1 | tee "$tee_log"
