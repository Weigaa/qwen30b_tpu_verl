#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

usage() {
    cat <<'EOF'
Usage: ./run_deepseek_v2_lite_kv_probe.sh LIFECYCLE FLOOR

LIFECYCLE is natural_f4, natural_f2, planned_f4, or planned_f2. A floor4
lifecycle accepts floors 4, 8, and 16. A floor2 lifecycle also accepts floor 2.

The probe runs one short step with native KV token caps disabled. Validated
short-response length profiles are bound to the exact one-step planner subset
so every requested lifecycle floor is reachable. The runtime still resumes
the common epoch0 checkpoint and uses the formal prompt, response, scheduler,
and memory configuration. The probe records vLLM's measured GPU KV cache size
and does not edit the verified cap file.
EOF
}

if [[ $# == 1 && ("$1" == -h || "$1" == --help) ]]; then
    usage
    exit 0
fi
if [[ $# != 2 ]]; then
    usage >&2
    exit 2
fi

lifecycle=$1
floor=$2
shift 2

WORKLOAD_PROFILE_PATH=${DEEPSEEK_WORKLOAD_PROFILE_PATH:-}
WORKLOAD_PROFILE_ID=unspecified
WORKLOAD_PROFILE_SHA256=unspecified
if [[ -n "$WORKLOAD_PROFILE_PATH" ]]; then
    if [[ ! -f "$WORKLOAD_PROFILE_PATH" ]]; then
        echo "missing DeepSeek workload profile: $WORKLOAD_PROFILE_PATH" >&2
        exit 2
    fi
    WORKLOAD_PROFILE_PATH=$(realpath "$WORKLOAD_PROFILE_PATH")
    # shellcheck disable=SC1090
    source "$WORKLOAD_PROFILE_PATH"
    WORKLOAD_PROFILE_ID=${DEEPSEEK_WORKLOAD_PROFILE_ID:-}
    measured_profile_sha256=$(sha256sum "$WORKLOAD_PROFILE_PATH")
    measured_profile_sha256=${measured_profile_sha256%% *}
    if [[ -z "$WORKLOAD_PROFILE_ID" ]]; then
        echo "DeepSeek workload profile has no DEEPSEEK_WORKLOAD_PROFILE_ID" >&2
        exit 2
    fi
    if [[ -n "${DEEPSEEK_WORKLOAD_PROFILE_SHA256:-}" \
          && "$DEEPSEEK_WORKLOAD_PROFILE_SHA256" != "$measured_profile_sha256" ]]; then
        echo "DeepSeek workload profile SHA256 mismatch" >&2
        exit 2
    fi
    WORKLOAD_PROFILE_SHA256=$measured_profile_sha256
    export DEEPSEEK_WORKLOAD_PROFILE_SHA256=$WORKLOAD_PROFILE_SHA256
fi

case "$lifecycle:$floor" in
    natural_f2:2|natural_f2:4|natural_f2:8|natural_f2:16|\
    planned_f2:2|planned_f2:4|planned_f2:8|planned_f2:16|\
    natural_f4:4|natural_f4:8|natural_f4:16|\
    planned_f4:4|planned_f4:8|planned_f4:16)
        target=$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh
        ;;
    *)
        echo "unsupported lifecycle/floor pair: $lifecycle/$floor" >&2
        usage >&2
        exit 2
        ;;
esac

DEEPSEEK_PROBE_TAIL_GUARD_MIN_CAP=${DEEPSEEK_PROBE_TAIL_GUARD_MIN_CAP:-4096}
DEEPSEEK_PROBE_TAIL_GUARD_ROUND_TO=${DEEPSEEK_PROBE_TAIL_GUARD_ROUND_TO:-512}
require_positive_integer() {
    local name=$1
    local value=$2
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got: $value" >&2
        exit 2
    fi
}
require_positive_integer \
    DEEPSEEK_PROBE_TAIL_GUARD_MIN_CAP \
    "$DEEPSEEK_PROBE_TAIL_GUARD_MIN_CAP"
require_positive_integer \
    DEEPSEEK_PROBE_TAIL_GUARD_ROUND_TO \
    "$DEEPSEEK_PROBE_TAIL_GUARD_ROUND_TO"

COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/common_epoch0_deepseek_v2_lite_tq1_no_overlap_gpu09}
REUSE_ENV=$COMMON_EPOCH0_ROOT/reuse.env
if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
      || ! -f "$REUSE_ENV" ]]; then
    echo "missing completed DeepSeek common epoch0: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$REUSE_ENV"
COMMON_EPOCH0_HISTORY=$DYNAMIC_INITIAL_BASELINE_DIR
COMMON_EPOCH0_METADATA_ENV=$COMMON_EPOCH0_ROOT/common_epoch0_metadata.env
if [[ ! -f "$COMMON_EPOCH0_METADATA_ENV" ]]; then
    echo "missing DeepSeek common epoch0 metadata: $COMMON_EPOCH0_METADATA_ENV" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$COMMON_EPOCH0_METADATA_ENV"

require_common_value() {
    local name=$1
    local expected=$2
    if [[ "${!name:-}" != "$expected" ]]; then
        echo "DeepSeek common epoch0 protocol mismatch for $name" >&2
        echo "recorded=${!name:-<missing>} expected=$expected" >&2
        exit 2
    fi
}

require_common_path() {
    local name=$1
    local expected=$2
    if [[ -z "${!name:-}" \
          || "$(realpath -m "${!name}")" != "$(realpath -m "$expected")" ]]; then
        echo "DeepSeek common epoch0 protocol path mismatch for $name" >&2
        echo "recorded=${!name:-<missing>} expected=$expected" >&2
        exit 2
    fi
}

export REPO_ROOT=$SCRIPT_DIR
export PATCH_TREE=$SCRIPT_DIR
export MODEL_PATH=/data/DeepSeek-V2-Lite-Chat
export MODEL_REVISION=85864749cd611b4353ce1decdb286193298f64c7
export DISTCP_PATH=/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4
export LOCAL_TEST_LAUNCHER=$SCRIPT_DIR/internal/wj_train_grpo_deepseek_v2_lite_16die_adafloor.sh
export BASELINE_LAUNCHER=$LOCAL_TEST_LAUNCHER
export DYNAMIC_MODE1_CHILD_SCRIPT=$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh
export CHECKPOINT_MODEL_DIR_NAME=deepseek_v2_lite_chat
export TRAIN_LOG_PREFIX=deepseek-v2-lite-chat-adafloor
export PLANNER_TOKENIZER_PATH=$MODEL_PATH
export TRAIN_FILE_ORIG=/data/deepscaler/train.parquet
export TEST_FILE=/data/deepscaler/test.parquet
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM=${DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM:-False}
export DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP=${DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP:-False}
export DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS=${DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS:-False}
export DEEPSEEK_ACTOR_RECOMPUTE_METHOD=${DEEPSEEK_ACTOR_RECOMPUTE_METHOD:-uniform}
export DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS=${DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS:-1}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-800}
expected_execution_profile="deepseek-v2-lite_tq${TASK_QUEUE_ENABLE}_a2a${DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM}_shared${DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP}_dealloc${DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS}_recompute${DEEPSEEK_ACTOR_RECOMPUTE_METHOD}x${DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS}_hccl${HCCL_BUFFSIZE}"
case "$lifecycle" in
    natural_f4)
        runtime_profile_path=$SCRIPT_DIR/internal/deepseek_v2_lite_natural_f4_runtime_profile.sh
        runtime_policy=natural
        profile_id_name=DEEPSEEK_N_F4_RUNTIME_PROFILE_ID
        profile_files_name=DEEPSEEK_N_F4_RUNTIME_PROFILE_FILES
        ;;
    natural_f2)
        runtime_profile_path=$SCRIPT_DIR/internal/deepseek_v2_lite_natural_f2_runtime_profile.sh
        runtime_policy=natural
        profile_id_name=DEEPSEEK_N_F2_RUNTIME_PROFILE_ID
        profile_files_name=DEEPSEEK_N_F2_RUNTIME_PROFILE_FILES
        ;;
    planned_f4)
        runtime_profile_path=$SCRIPT_DIR/internal/deepseek_v2_lite_planned_f4_runtime_profile.sh
        runtime_policy=planned
        profile_id_name=DEEPSEEK_P_F4_RUNTIME_PROFILE_ID
        profile_files_name=DEEPSEEK_P_F4_RUNTIME_PROFILE_FILES
        ;;
    planned_f2)
        runtime_profile_path=$SCRIPT_DIR/internal/deepseek_v2_lite_planned_f2_runtime_profile.sh
        runtime_policy=planned
        profile_id_name=DEEPSEEK_P_F2_RUNTIME_PROFILE_ID
        profile_files_name=DEEPSEEK_P_F2_RUNTIME_PROFILE_FILES
        ;;
esac
if [[ ! -f "$runtime_profile_path" ]]; then
    echo "missing DeepSeek runtime profile for $lifecycle: $runtime_profile_path" >&2
    exit 2
fi
# shellcheck disable=SC1091
source "$runtime_profile_path"
runtime_profile=${!profile_id_name:-}
runtime_profile_files=${!profile_files_name:-}
if [[ -z "$runtime_profile" || -z "$runtime_profile_files" ]]; then
    echo "$runtime_profile_path did not define $profile_id_name and $profile_files_name" >&2
    exit 2
fi
profile_hash_args=()
IFS=, read -r -a runtime_profile_file_array <<< "$runtime_profile_files"
for runtime_profile_file in "${runtime_profile_file_array[@]}"; do
    profile_hash_args+=(--profile "$runtime_profile_file")
done
runtime_profile_sha256=$(python3 \
    "$SCRIPT_DIR/tools/hash_deepseek_runtime_profile.py" \
    --root "$SCRIPT_DIR" "${profile_hash_args[@]}")
execution_code_sha256=$(python3 \
    "$SCRIPT_DIR/tools/hash_deepseek_execution_code.py" \
    --root "$SCRIPT_DIR")
if [[ "${COMMON_EPOCH0_EXECUTION_PROFILE_USED:-}" != "$expected_execution_profile" ]]; then
    echo "DeepSeek common epoch0 execution profile mismatch" >&2
    echo "recorded=${COMMON_EPOCH0_EXECUTION_PROFILE_USED:-<missing>} expected=$expected_execution_profile" >&2
    exit 2
fi
if [[ -z "${COMMON_EPOCH0_MODEL_PATH:-}" \
      || "$(realpath -m "$COMMON_EPOCH0_MODEL_PATH")" != "$(realpath -m "$MODEL_PATH")" ]]; then
    echo "DeepSeek common epoch0 model mismatch" >&2
    exit 2
fi
if [[ "${COMMON_EPOCH0_MODEL_REVISION:-}" != "$MODEL_REVISION" ]]; then
    echo "DeepSeek common epoch0 model revision mismatch" >&2
    exit 2
fi
if [[ -z "${COMMON_EPOCH0_DISTCP_PATH:-}" \
      || "$(realpath -m "$COMMON_EPOCH0_DISTCP_PATH")" != "$(realpath -m "$DISTCP_PATH")" ]]; then
    echo "DeepSeek common epoch0 distributed checkpoint mismatch" >&2
    exit 2
fi
if [[ "${COMMON_EPOCH0_CHECKPOINT_MODEL_DIR_NAME:-}" != "$CHECKPOINT_MODEL_DIR_NAME" ]]; then
    echo "DeepSeek common epoch0 checkpoint namespace mismatch" >&2
    exit 2
fi
require_common_path COMMON_EPOCH0_TRAIN_FILE_USED /data/deepscaler/train.parquet
require_common_path COMMON_EPOCH0_TEST_FILE_USED /data/deepscaler/test.parquet
WORKLOAD_DATASET_FRACTION=${COMMON_EPOCH0_DATASET_FRACTION:-0.005}
WORKLOAD_TRAIN_BATCH_SIZE=${COMMON_EPOCH0_TRAIN_BATCH_SIZE:-32}
WORKLOAD_ROLLOUT_N=${COMMON_EPOCH0_ROLLOUT_N:-16}
WORKLOAD_MAX_NUM_SEQS=${COMMON_EPOCH0_MAX_NUM_SEQS:-32}
WORKLOAD_TRAIN_STEPS=${COMMON_EPOCH0_TRAIN_STEPS:-5}
WORKLOAD_PROMPTS_TOTAL=${COMMON_EPOCH0_PROMPTS_TOTAL:-$((WORKLOAD_TRAIN_STEPS * WORKLOAD_TRAIN_BATCH_SIZE))}
WORKLOAD_EXPECTED_RESPONSES_PER_STEP=${COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP:-$((WORKLOAD_TRAIN_BATCH_SIZE * WORKLOAD_ROLLOUT_N))}
WORKLOAD_MAX_PROMPT_LENGTH=${COMMON_EPOCH0_MAX_PROMPT_LENGTH:-1024}
WORKLOAD_MAX_RESPONSE_LENGTH=${COMMON_EPOCH0_MAX_RESPONSE_LENGTH:-16384}
WORKLOAD_MAX_NUM_BATCHED_TOKENS=${COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS:-17408}
require_common_value COMMON_EPOCH0_DATASET_FRACTION_USED "$WORKLOAD_DATASET_FRACTION"
require_common_value COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED "$WORKLOAD_TRAIN_BATCH_SIZE"
require_common_value COMMON_EPOCH0_ROLLOUT_N_USED "$WORKLOAD_ROLLOUT_N"
require_common_value COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED "$WORKLOAD_MAX_PROMPT_LENGTH"
require_common_value COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED "$WORKLOAD_MAX_RESPONSE_LENGTH"
require_common_value COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED "$WORKLOAD_MAX_NUM_BATCHED_TOKENS"
require_common_value COMMON_EPOCH0_MAX_NUM_SEQS_USED "$WORKLOAD_MAX_NUM_SEQS"
require_common_value COMMON_EPOCH0_GPU_MEMORY_UTILIZATION_USED 0.9
require_common_value COMMON_EPOCH0_KV_BLOCK_SIZE_USED 128
require_common_value COMMON_EPOCH0_TRAIN_STEPS_USED "$WORKLOAD_TRAIN_STEPS"
if [[ -n "$WORKLOAD_PROFILE_PATH" ]]; then
    require_common_value COMMON_EPOCH0_PROMPTS_TOTAL_USED "$WORKLOAD_PROMPTS_TOTAL"
    require_common_value COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED \
        "$WORKLOAD_EXPECTED_RESPONSES_PER_STEP"
    require_common_value COMMON_EPOCH0_WORKLOAD_PROFILE_ID "$WORKLOAD_PROFILE_ID"
    require_common_value COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256 "$WORKLOAD_PROFILE_SHA256"
    require_common_value COMMON_EPOCH0_PREEMPTION_POLICY_USED \
        "${COMMON_EPOCH0_PREEMPTION_POLICY:-record}"
fi
if [[ ! -d "${DYNAMIC_INITIAL_RESUME_CKPT:-}/actor" \
      || ! -f "${DYNAMIC_INITIAL_RESUME_CKPT:-}/.PRESERVE_COMMON_EPOCH0" ]]; then
    echo "DeepSeek common epoch0 resume checkpoint is incomplete" >&2
    exit 2
fi
PROBE_PLANNING_HISTORY_ROOT=${DEEPSEEK_KV_PROBE_HISTORY_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/kv_probe_positive_release_trigger_v2}
PROBE_PLANNING_HISTORY_SOURCE_ROOT=${DEEPSEEK_KV_PROBE_HISTORY_SOURCE_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/threshold_actor_2step_n16_tq1_no_overlap_20260804T031350Z/epoch_000_mode0_probe}
PROBE_TRAIN_FILE=$TRAIN_FILE_ORIG
PROBE_DATASET_FRACTION=${DEEPSEEK_KV_PROBE_DATASET_FRACTION:-0.0009}
PROBE_PROMPT_COUNT=${DEEPSEEK_KV_PROBE_PROMPT_COUNT:-$WORKLOAD_TRAIN_BATCH_SIZE}
PROBE_RESPONSES_PER_PROMPT=${DEEPSEEK_KV_PROBE_RESPONSES_PER_PROMPT:-$WORKLOAD_ROLLOUT_N}
PROBE_MAX_RESPONSE=${DEEPSEEK_KV_PROBE_TRIGGER_MAX_RESPONSE:-64}
if [[ -n "${DEEPSEEK_KV_PROBE_SOURCE_STEPS:-}" ]]; then
    PROBE_SOURCE_STEPS=$DEEPSEEK_KV_PROBE_SOURCE_STEPS
elif (( PROBE_PROMPT_COUNT == 64 )); then
    PROBE_SOURCE_STEPS=1,2
else
    PROBE_SOURCE_STEPS=2
fi
PROBE_PLANNING_HISTORY_MANIFEST=$PROBE_PLANNING_HISTORY_ROOT/kv_probe_trigger_manifest.json
PROBE_PLANNING_HISTORY_FILE=$PROBE_PLANNING_HISTORY_ROOT/offline_planning_history.json
if [[ ! -f "$PROBE_PLANNING_HISTORY_MANIFEST" ]]; then
    if [[ -e "$PROBE_PLANNING_HISTORY_ROOT" ]]; then
        echo "incomplete DeepSeek KV probe trigger exists: $PROBE_PLANNING_HISTORY_ROOT" >&2
        exit 2
    fi
    python3 "$SCRIPT_DIR/tools/prepare_deepseek_kv_probe_trigger.py" build \
        --source-root "$PROBE_PLANNING_HISTORY_SOURCE_ROOT" \
        --output-root "$PROBE_PLANNING_HISTORY_ROOT" \
        --train-file "$PROBE_TRAIN_FILE" \
        --dataset-fraction "$PROBE_DATASET_FRACTION" \
        --tokenizer-path "$PLANNER_TOKENIZER_PATH" \
        --prompt-count "$PROBE_PROMPT_COUNT" \
        --responses-per-prompt "$PROBE_RESPONSES_PER_PROMPT" \
        --max-response "$PROBE_MAX_RESPONSE" \
        --source-steps "$PROBE_SOURCE_STEPS"
fi
python3 "$SCRIPT_DIR/tools/prepare_deepseek_kv_probe_trigger.py" verify \
    --output-root "$PROBE_PLANNING_HISTORY_ROOT" \
    --train-file "$PROBE_TRAIN_FILE" \
    --dataset-fraction "$PROBE_DATASET_FRACTION" \
    --tokenizer-path "$PLANNER_TOKENIZER_PATH"
probe_history_digest=$(sha256sum "$PROBE_PLANNING_HISTORY_FILE")
PROBE_PLANNING_HISTORY_SHA256=${probe_history_digest%% *}
probe_manifest_digest=$(sha256sum "$PROBE_PLANNING_HISTORY_MANIFEST")
PROBE_PLANNING_HISTORY_MANIFEST_SHA256=${probe_manifest_digest%% *}
PROBE_TRIGGER_SUBSET=$PROBE_PLANNING_HISTORY_ROOT/rollout_data/1.jsonl
probe_trigger_digest=$(sha256sum "$PROBE_TRIGGER_SUBSET")
PROBE_TRIGGER_SUBSET_SHA256=${probe_trigger_digest%% *}
export DYNAMIC_OUTPUT_ROOT=${DEEPSEEK_KV_PROBE_OUTPUT_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/kv_probes}
export DYNAMIC_RUN_NAME=${DYNAMIC_RUN_NAME:-${lifecycle}_floor${floor}_auto_kv}
export DYNAMIC_SKIP_MODE0_PROBE=1
export DYNAMIC_INITIAL_BASELINE_DIR=$PROBE_PLANNING_HISTORY_ROOT
export DYNAMIC_START_EPOCH=1
export DYNAMIC_TOTAL_EPOCHS=2
export DYNAMIC_PLAN_STEPS=1
export DYNAMIC_TRAIN_STEPS=1
export DYNAMIC_DATASET_FRACTION=$PROBE_DATASET_FRACTION
export DYNAMIC_LENGTH_EMA_DECAY=0.3
export DYNAMIC_ENABLE_CKPT_CHAIN=0
export DYNAMIC_BUILD_OFFLINE_PLANNING_HISTORY=0
export DYNAMIC_ENABLE_THRESHOLD_CONTROL=0
export DYNAMIC_SHRINK_POLICY=$runtime_policy
unset DYNAMIC_FORCE_SELECTED_FLOOR FORCE_SELECTED_FLOOR FORCE_SELECTED_FLOORS
export DYNAMIC_FORCE_SELECTED_FLOORS=$floor
export DYNAMIC_SHORT_STEP_CAP_ENABLE=0
export DYNAMIC_FULL_MAX_PROMPT_LENGTH=$WORKLOAD_MAX_PROMPT_LENGTH
export DYNAMIC_FULL_MAX_RESPONSE_LENGTH=$WORKLOAD_MAX_RESPONSE_LENGTH
export DYNAMIC_FULL_MAX_RESPONSE_LEN=$WORKLOAD_MAX_RESPONSE_LENGTH
export DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS=$WORKLOAD_MAX_NUM_BATCHED_TOKENS
export MAX_PROMPT_LENGTH=$WORKLOAD_MAX_PROMPT_LENGTH
export MAX_RESPONSE_LENGTH=$WORKLOAD_MAX_RESPONSE_LENGTH
export MAX_RESPONSE_LEN=$WORKLOAD_MAX_RESPONSE_LENGTH
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=$WORKLOAD_MAX_NUM_BATCHED_TOKENS
unset DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS
export DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP=4,16,32,64,64
export DYNAMIC_TAIL_GUARD_RATIO_QUANTILE=0.95
export DYNAMIC_TAIL_GUARD_RATIO_WINDOW=3
export DYNAMIC_TAIL_GUARD_DEFAULT_RATIO=1.20
export DYNAMIC_TAIL_GUARD_MIN_CAP=$DEEPSEEK_PROBE_TAIL_GUARD_MIN_CAP
export DYNAMIC_TAIL_GUARD_ROUND_TO=$DEEPSEEK_PROBE_TAIL_GUARD_ROUND_TO
export DYNAMIC_DISABLE_TAIL_GUARD=0
export ROLLOUT_MAX_NUM_SEQS=$WORKLOAD_MAX_NUM_SEQS
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.9
export ROLLOUT_ENFORCE_EAGER=True
export ACTIVE_PEAK_SAFETY_FACTOR=1.16
if [[ "$lifecycle" == *_f2 ]]; then
    export MIN_ADAPTIVE_FLOOR=2
    export VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4,2
    export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2
else
    export MIN_ADAPTIVE_FLOOR=4
    export VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4
    export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4
fi
export RANK_MATCHING_POLICY=release_area
export VLLM_KV_BLOCK_SIZE=128
export TRAIN_BATCH_SIZE=$WORKLOAD_TRAIN_BATCH_SIZE
export ROLLOUT_N=$WORKLOAD_ROLLOUT_N
export SAVE_CKPT_ENABLE=0
export TRAINER_SAVE_FREQ=-1

# Disable runtime token caps. FLOOR_KV_CAPS is only a feasibility hint for the
# forced planner decision and is deliberately larger than any physical cache.
export VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=0
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=0
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2=0
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=0
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=0
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16=0
export FLOOR_KV_CAPS=2:1000000000,4:1000000000,8:1000000000,16:1000000000
export MAX_RANK_PEAK_TOKENS=1000000000
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_FAIL_ON_UNMET_TARGET=0
unset PHYSICAL_FLOOR_KV_CAPS
unset MODE1_PLAN_ONLY

validate_plan() {
    local summary=$1
    python3 - "$summary" "$floor" "$PROBE_PROMPT_COUNT" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
floor = int(sys.argv[2])
prompt_count = int(sys.argv[3])
payload = json.loads(path.read_text(encoding="utf-8"))
steps = payload.get("steps", payload) if isinstance(payload, dict) else payload
if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], dict):
    raise SystemExit(f"expected exactly one probe plan in {path}")
plan = steps[0]
if plan.get("feasible") is not True or int(plan.get("selected_floor", -1)) != floor:
    raise SystemExit(
        f"invalid forced floor plan in {path}: "
        f"feasible={plan.get('feasible')!r} floor={plan.get('selected_floor')!r}"
    )
release = float(plan.get("release_area", -1))
thresholds = [float(value) for value in plan.get("schedule_thresholds", [])]
predicted_exit = float(plan.get("predicted_step_exit", -1))
if not math.isfinite(release) or release < 0:
    raise SystemExit(f"invalid release area in {path}: {release}")
if floor < 16 and (
    release <= 0
    or not thresholds
    or not math.isfinite(predicted_exit)
    or any(not math.isfinite(value) for value in thresholds)
    or max(thresholds) >= predicted_exit
):
    raise SystemExit(
        f"floor{floor} has no positive pre-exit release window in {path}: "
        f"release={release} thresholds={thresholds} exit={predicted_exit}"
    )
detail_path = path.with_name("length_sorted_rank_plan.json")
details = json.loads(detail_path.read_text(encoding="utf-8"))
if not isinstance(details, list) or len(details) != 1:
    raise SystemExit(f"expected exactly one detailed probe plan in {detail_path}")
rank_sources = details[0].get("rank_to_source_idx", {})
source_indices = sorted(
    int(index)
    for indices in rank_sources.values()
    for index in indices
)
if source_indices != list(range(prompt_count)):
    raise SystemExit(
        f"probe trigger mapped to unexpected source rows in {detail_path}: "
        f"{source_indices}"
    )
print(
    f"validated floor{floor} plan release_area={release} "
    f"thresholds={thresholds} predicted_exit={predicted_exit} "
    f"tail_guard_cap={plan.get('tail_guard_response_cap')}"
)
PY
}

run_plan_preflight() {
    local preflight_root=$DYNAMIC_OUTPUT_ROOT/$DYNAMIC_RUN_NAME/plan_preflight
    local preflight_plan_dir=$preflight_root/oracle
    local preflight_summary=$preflight_plan_dir/length_sorted_rank_plan_summary.json
    if [[ ! -f "$preflight_summary" ]]; then
        if [[ -e "$preflight_root" ]]; then
            echo "incomplete DeepSeek probe preflight already exists: $preflight_root" >&2
            exit 2
        fi
        echo "[DeepSeek KV probe] plan-only preflight floor=$floor"
        OUTPUT_ROOT="$DYNAMIC_OUTPUT_ROOT" \
        OUTPUT_SUBDIR="$DYNAMIC_RUN_NAME/plan_preflight" \
        BASELINE_DIRS="$PROBE_PLANNING_HISTORY_ROOT" \
        PLAN_DIR="$preflight_plan_dir" \
        TRAIN_FILE_ORIG="$TRAIN_FILE_ORIG" \
        TEST_FILE="$TEST_FILE" \
        MODEL_PATH="$MODEL_PATH" \
        DISTCP_PATH="$DISTCP_PATH" \
        TRAINER_TOTAL_EPOCHS=1 \
        DATASET_FRACTION_FOR_ORACLE="$DYNAMIC_DATASET_FRACTION" \
        LENGTH_EMA_DECAY=0.3 \
        IGNORE_TAIL_TIES_AT_RESPONSE_CAP=0 \
        TAIL_GUARD_RATIO_QUANTILE="$DYNAMIC_TAIL_GUARD_RATIO_QUANTILE" \
        TAIL_GUARD_RATIO_WINDOW="$DYNAMIC_TAIL_GUARD_RATIO_WINDOW" \
        TAIL_GUARD_DEFAULT_RATIO="$DYNAMIC_TAIL_GUARD_DEFAULT_RATIO" \
        TAIL_GUARD_MIN_CAP="$DYNAMIC_TAIL_GUARD_MIN_CAP" \
        TAIL_GUARD_ROUND_TO="$DYNAMIC_TAIL_GUARD_ROUND_TO" \
        PLAN_STEPS=1 \
        FORCE_SELECTED_FLOORS="$floor" \
        REQUIRE_COMPACT_HISTORY=1 \
        MODE1_PLAN_ONLY=1 \
        "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"
    fi
    validate_plan "$preflight_summary"
}

echo "[DeepSeek KV probe] lifecycle=$lifecycle floor=$floor output=$DYNAMIC_OUTPUT_ROOT/$DYNAMIC_RUN_NAME"
echo "[DeepSeek KV probe] task_queue=$TASK_QUEUE_ENABLE moe_overlap=$DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM/$DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP deallocate_pipeline_outputs=$DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS"
echo "[DeepSeek KV probe] execution_profile=$expected_execution_profile"
echo "[DeepSeek KV probe] runtime_profile=$runtime_profile"
echo "[DeepSeek KV probe] runtime_profile_sha256=$runtime_profile_sha256"
echo "[DeepSeek KV probe] execution_code_sha256=$execution_code_sha256"
echo "[DeepSeek KV probe] workload_profile=$WORKLOAD_PROFILE_ID sha256=$WORKLOAD_PROFILE_SHA256"
echo "[DeepSeek KV probe] common_epoch0_history=$COMMON_EPOCH0_HISTORY"
echo "[DeepSeek KV probe] planning_history=$PROBE_PLANNING_HISTORY_ROOT"
echo "[DeepSeek KV probe] planning_history_sha256=$PROBE_PLANNING_HISTORY_SHA256"
echo "[DeepSeek KV probe] planning_manifest_sha256=$PROBE_PLANNING_HISTORY_MANIFEST_SHA256"
echo "[DeepSeek KV probe] trigger_subset_sha256=$PROBE_TRIGGER_SUBSET_SHA256"
echo "[DeepSeek KV probe] trigger_prompts=$PROBE_PROMPT_COUNT responses_per_prompt=$PROBE_RESPONSES_PER_PROMPT source_steps=$PROBE_SOURCE_STEPS"
echo "[DeepSeek KV probe] workload_profile=prompt${DYNAMIC_FULL_MAX_PROMPT_LENGTH}_response${DYNAMIC_FULL_MAX_RESPONSE_LENGTH}_batched${DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS}_maxseqs${ROLLOUT_MAX_NUM_SEQS} dataset_fraction=$DYNAMIC_DATASET_FRACTION"
echo "[DeepSeek KV probe] tail_guard_min_cap=$DYNAMIC_TAIL_GUARD_MIN_CAP tail_guard_round_to=$DYNAMIC_TAIL_GUARD_ROUND_TO"
echo "[DeepSeek KV probe] runtime_cap=disabled planned_headroom=${DEEPSEEK_PROBE_PLANNED_HEADROOM_TOKENS:-n/a} planned_training_min_free_mib=${DEEPSEEK_PROBE_PLANNED_MIN_FREE_MIB:-n/a}"
if [[ "${DEEPSEEK_PROBE_DRY_RUN:-0}" == 1 ]]; then
    echo "[DeepSeek KV probe] target=$target"
    echo "[DeepSeek KV probe] dry run only"
    exit 0
fi

run_plan_preflight
if [[ "${DEEPSEEK_PROBE_PLAN_ONLY:-0}" == 1 ]]; then
    echo "[DeepSeek KV probe] plan-only preflight complete"
    exit 0
fi

actual_epoch_root=$DYNAMIC_OUTPUT_ROOT/$DYNAMIC_RUN_NAME/epoch_001_mode1_${runtime_policy}
if [[ -e "$actual_epoch_root" ]]; then
    echo "refusing to overwrite DeepSeek KV probe: $actual_epoch_root" >&2
    exit 2
fi

"$target" "$@" \
    trainer.resume_mode=resume_path \
    "trainer.resume_from_path=$DYNAMIC_INITIAL_RESUME_CKPT"

latest_log=$(find "$actual_epoch_root" -type f -path '*/logs/*.txt' \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -z "$latest_log" || ! -f "$latest_log" ]]; then
    echo "probe completed without a discoverable log" >&2
    exit 3
fi
python3 "$SCRIPT_DIR/tools/summarize_deepseek_kv_probe.py" \
    --lifecycle "$lifecycle" \
    --floor "$floor" \
    --world-size 16 \
    --run-root "$actual_epoch_root" \
    --log "$latest_log" \
    --model-revision "$MODEL_REVISION" \
    --execution-profile "$expected_execution_profile" \
    --runtime-profile "$runtime_profile" \
    --runtime-profile-sha256 "$runtime_profile_sha256" \
    --execution-code-sha256 "$execution_code_sha256" \
    --max-prompt-length "$DYNAMIC_FULL_MAX_PROMPT_LENGTH" \
    --max-response-length "$DYNAMIC_FULL_MAX_RESPONSE_LENGTH" \
    --max-num-batched-tokens "$DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$ROLLOUT_MAX_NUM_SEQS" \
    --tail-guard-min-cap "$DYNAMIC_TAIL_GUARD_MIN_CAP" \
    --tail-guard-round-to "$DYNAMIC_TAIL_GUARD_ROUND_TO" \
    --gpu-memory-utilization "$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    --enforce-eager "$ROLLOUT_ENFORCE_EAGER" \
    --block-size "$VLLM_KV_BLOCK_SIZE" \
    --common-epoch0-root "$COMMON_EPOCH0_ROOT" \
    --planning-history-root "$PROBE_PLANNING_HISTORY_ROOT" \
    --planning-history-sha256 "$PROBE_PLANNING_HISTORY_SHA256" \
    --planning-history-manifest-sha256 "$PROBE_PLANNING_HISTORY_MANIFEST_SHA256" \
    --planning-trigger-subset-sha256 "$PROBE_TRIGGER_SUBSET_SHA256" \
    --output "$DYNAMIC_OUTPUT_ROOT/$DYNAMIC_RUN_NAME/kv_probe_summary.json"
