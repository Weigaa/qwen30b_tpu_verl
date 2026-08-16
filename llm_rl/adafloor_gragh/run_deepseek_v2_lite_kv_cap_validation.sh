#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

usage() {
    cat <<'EOF'
Usage: ./run_deepseek_v2_lite_kv_cap_validation.sh [LIFECYCLE]

LIFECYCLE defaults to natural_f4 and accepts natural_f4, natural_f2,
planned_f4, or planned_f2. Each floor is authorized independently.
EOF
}

if (( $# > 1 )) || [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    [[ "${1:-}" == -h || "${1:-}" == --help ]] && exit 0
    exit 2
fi
lifecycle=${1:-${DEEPSEEK_KV_CAP_LIFECYCLE:-natural_f4}}

case "$lifecycle" in
    natural_f4)
        prefix=DEEPSEEK_N_F4
        label="Natural floor4"
        profile_path=$SCRIPT_DIR/internal/deepseek_v2_lite_natural_f4_runtime_profile.sh
        profile_id_name=DEEPSEEK_N_F4_RUNTIME_PROFILE_ID
        profile_files_name=DEEPSEEK_N_F4_RUNTIME_PROFILE_FILES
        floors=(16 8 4)
        min_floor=4
        policy=natural
        ;;
    natural_f2)
        prefix=DEEPSEEK_N_F2
        label="Natural floor2"
        profile_path=$SCRIPT_DIR/internal/deepseek_v2_lite_natural_f2_runtime_profile.sh
        profile_id_name=DEEPSEEK_N_F2_RUNTIME_PROFILE_ID
        profile_files_name=DEEPSEEK_N_F2_RUNTIME_PROFILE_FILES
        floors=(16 8 4 2)
        min_floor=2
        policy=natural
        ;;
    planned_f4)
        prefix=DEEPSEEK_P_F4
        label="Planned floor4"
        profile_path=$SCRIPT_DIR/internal/deepseek_v2_lite_planned_f4_runtime_profile.sh
        profile_id_name=DEEPSEEK_P_F4_RUNTIME_PROFILE_ID
        profile_files_name=DEEPSEEK_P_F4_RUNTIME_PROFILE_FILES
        floors=(16 8 4)
        min_floor=4
        policy=planned
        ;;
    planned_f2)
        prefix=DEEPSEEK_P_F2
        label="Planned floor2"
        profile_path=$SCRIPT_DIR/internal/deepseek_v2_lite_planned_f2_runtime_profile.sh
        profile_id_name=DEEPSEEK_P_F2_RUNTIME_PROFILE_ID
        profile_files_name=DEEPSEEK_P_F2_RUNTIME_PROFILE_FILES
        floors=(16 8 4 2)
        min_floor=2
        policy=planned
        ;;
    *)
        echo "unsupported DeepSeek KV authorization lifecycle: $lifecycle" >&2
        usage >&2
        exit 2
        ;;
esac

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
WORKLOAD_PROBE_DATASET_FRACTION=${DEEPSEEK_KV_PROBE_DATASET_FRACTION:-0.0009}

CAP_ENV=${DEEPSEEK_KV_CAP_ENV:-$SCRIPT_DIR/deepseek_v2_lite_kv_caps.env}
COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/common_epoch0_deepseek_v2_lite_tq1_no_overlap_gpu09}
TRIGGER_ROOT=${DEEPSEEK_KV_PROBE_HISTORY_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/kv_probe_positive_release_trigger_v2}
REUSE_ENV=$COMMON_EPOCH0_ROOT/reuse.env
METADATA_ENV=$COMMON_EPOCH0_ROOT/common_epoch0_metadata.env
TARGET=$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh
MODE1_CHILD=$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh
LOCK_PATH=${DEEPSEEK_EXCLUSIVE_NPU_LOCK:-/data/adafloor_shared_state/deepseek_v2_lite/.adafloor_npu_exclusive.lock}

for path in "$CAP_ENV" "$REUSE_ENV" "$METADATA_ENV" "$profile_path"; do
    if [[ ! -f "$path" ]]; then
        echo "missing DeepSeek $label authorization input: $path" >&2
        exit 2
    fi
done
if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]]; then
    echo "common DeepSeek epoch0 is not complete: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "$CAP_ENV"
# shellcheck disable=SC1090
source "$REUSE_ENV"
# shellcheck disable=SC1090
source "$METADATA_ENV"
# shellcheck disable=SC1090
source "$profile_path"

require_value() {
    local name=$1
    local expected=$2
    if [[ "${!name:-}" != "$expected" ]]; then
        echo "DeepSeek $label authorization mismatch for $name" >&2
        echo "recorded=${!name:-<missing>} expected=$expected" >&2
        exit 2
    fi
}

require_path() {
    local name=$1
    local expected=$2
    if [[ -z "${!name:-}" \
          || "$(realpath -m "${!name}")" != "$(realpath -m "$expected")" ]]; then
        echo "DeepSeek $label authorization path mismatch for $name" >&2
        echo "recorded=${!name:-<missing>} expected=$expected" >&2
        exit 2
    fi
}

require_cap() {
    local name=$1
    local value=${!name:-}
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]] || (( value % 128 != 0 )); then
        echo "$name must be a positive multiple of 128, got ${value:-<unset>}" >&2
        exit 2
    fi
}

require_value "${prefix}_KV_CAPS_VERIFIED" 0
require_value DEEPSEEK_KV_CAP_TARGET_RATIO 1.0
require_value DEEPSEEK_KV_CAP_BLOCK_SIZE 128
require_value DEEPSEEK_KV_CAP_TRAIN_BATCH_SIZE "$WORKLOAD_TRAIN_BATCH_SIZE"
require_value DEEPSEEK_KV_CAP_ROLLOUT_N "$WORKLOAD_ROLLOUT_N"
require_value DEEPSEEK_KV_CAP_COMMON_STEPS "$WORKLOAD_TRAIN_STEPS"
require_value DEEPSEEK_KV_CAP_PROMPTS_TOTAL "$WORKLOAD_PROMPTS_TOTAL"
require_value DEEPSEEK_KV_CAP_EXPECTED_RESPONSES_PER_STEP \
    "$WORKLOAD_EXPECTED_RESPONSES_PER_STEP"
require_value DEEPSEEK_KV_CAP_DATASET_FRACTION "$WORKLOAD_DATASET_FRACTION"
require_value DEEPSEEK_KV_CAP_MAX_PROMPT_LENGTH "$WORKLOAD_MAX_PROMPT_LENGTH"
require_value DEEPSEEK_KV_CAP_MAX_RESPONSE_LENGTH "$WORKLOAD_MAX_RESPONSE_LENGTH"
require_value DEEPSEEK_KV_CAP_MAX_NUM_BATCHED_TOKENS "$WORKLOAD_MAX_NUM_BATCHED_TOKENS"
require_value DEEPSEEK_KV_CAP_MAX_NUM_SEQS "$WORKLOAD_MAX_NUM_SEQS"
require_value DEEPSEEK_KV_CAP_GPU_MEMORY_UTILIZATION 0.9
require_value DEEPSEEK_KV_CAP_ENFORCE_EAGER True
require_path DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT "$COMMON_EPOCH0_ROOT"
require_path DEEPSEEK_KV_CAP_PROBE_HISTORY_ROOT "$TRIGGER_ROOT"
require_path COMMON_EPOCH0_MODEL_PATH /data/DeepSeek-V2-Lite-Chat
require_path COMMON_EPOCH0_DISTCP_PATH /data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4
require_value COMMON_EPOCH0_MODEL_REVISION 85864749cd611b4353ce1decdb286193298f64c7
require_value COMMON_EPOCH0_CHECKPOINT_MODEL_DIR_NAME deepseek_v2_lite_chat
require_value COMMON_EPOCH0_EXECUTION_PROFILE_USED deepseek-v2-lite_tq1_a2aFalse_sharedFalse_deallocFalse_recomputeuniformx1_hccl800
require_path COMMON_EPOCH0_TRAIN_FILE_USED /data/deepscaler/train.parquet
require_path COMMON_EPOCH0_TEST_FILE_USED /data/deepscaler/test.parquet
require_value COMMON_EPOCH0_DATASET_FRACTION_USED "$WORKLOAD_DATASET_FRACTION"
require_value COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED "$WORKLOAD_TRAIN_BATCH_SIZE"
require_value COMMON_EPOCH0_ROLLOUT_N_USED "$WORKLOAD_ROLLOUT_N"
require_value COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED "$WORKLOAD_MAX_PROMPT_LENGTH"
require_value COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED "$WORKLOAD_MAX_RESPONSE_LENGTH"
require_value COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED "$WORKLOAD_MAX_NUM_BATCHED_TOKENS"
require_value COMMON_EPOCH0_MAX_NUM_SEQS_USED "$WORKLOAD_MAX_NUM_SEQS"
require_value COMMON_EPOCH0_GPU_MEMORY_UTILIZATION_USED 0.9
require_value COMMON_EPOCH0_KV_BLOCK_SIZE_USED 128
require_value COMMON_EPOCH0_TRAIN_STEPS_USED "$WORKLOAD_TRAIN_STEPS"
if [[ -n "$WORKLOAD_PROFILE_PATH" ]]; then
    require_value DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID "$WORKLOAD_PROFILE_ID"
    require_value DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256 "$WORKLOAD_PROFILE_SHA256"
    require_value DEEPSEEK_KV_CAP_COMMON_PREEMPTION_POLICY \
        "${COMMON_EPOCH0_PREEMPTION_POLICY:-record}"
    require_value COMMON_EPOCH0_PROMPTS_TOTAL_USED "$WORKLOAD_PROMPTS_TOTAL"
    require_value COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED \
        "$WORKLOAD_EXPECTED_RESPONSES_PER_STEP"
    require_value COMMON_EPOCH0_WORKLOAD_PROFILE_ID "$WORKLOAD_PROFILE_ID"
    require_value COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256 "$WORKLOAD_PROFILE_SHA256"
    require_value COMMON_EPOCH0_PREEMPTION_POLICY_USED \
        "${COMMON_EPOCH0_PREEMPTION_POLICY:-record}"
fi

if [[ ! -d "${DYNAMIC_INITIAL_RESUME_CKPT:-}/actor" \
      || ! -f "${DYNAMIC_INITIAL_RESUME_CKPT:-}/.PRESERVE_COMMON_EPOCH0" ]]; then
    echo "common DeepSeek epoch0 checkpoint is incomplete" >&2
    exit 2
fi
python3 "$SCRIPT_DIR/tools/prepare_deepseek_kv_probe_trigger.py" verify \
    --output-root "$TRIGGER_ROOT"

profile_hash_args=()
runtime_profile_files=${!profile_files_name:-}
IFS=, read -r -a runtime_profile_file_array <<< "$runtime_profile_files"
for runtime_profile_file in "${runtime_profile_file_array[@]}"; do
    profile_hash_args+=(--profile "$runtime_profile_file")
done
runtime_profile_sha256=$(python3 \
    "$SCRIPT_DIR/tools/hash_deepseek_runtime_profile.py" \
    --root "$SCRIPT_DIR" "${profile_hash_args[@]}")
execution_code_sha256=$(python3 \
    "$SCRIPT_DIR/tools/hash_deepseek_execution_code.py" --root "$SCRIPT_DIR")
recorded_profile_name=${prefix}_RUNTIME_PROFILE
recorded_profile_sha_name=${prefix}_RUNTIME_PROFILE_SHA256
require_value "$recorded_profile_name" "${!profile_id_name}"
require_value "$recorded_profile_sha_name" "$runtime_profile_sha256"
require_value DEEPSEEK_EXECUTION_CODE_SHA256 "$execution_code_sha256"
require_value DEEPSEEK_KV_CAP_MODEL_REVISION "$COMMON_EPOCH0_MODEL_REVISION"
require_value DEEPSEEK_KV_CAP_EXECUTION_PROFILE "$COMMON_EPOCH0_EXECUTION_PROFILE_USED"

for floor in "${floors[@]}"; do
    admission_name=${prefix}_KV_ADMISSION_FLOOR${floor}
    physical_name=${prefix}_KV_PHYSICAL_FLOOR${floor}
    require_cap "$admission_name"
    require_cap "$physical_name"
    if (( ${!admission_name} >= ${!physical_name} )); then
        echo "$label floor${floor} admission cap must be smaller than physical cap" >&2
        exit 2
    fi
done
if [[ "$policy" == planned ]]; then
    for floor in "${floors[@]}"; do
        headroom_name=${prefix}_HEADROOM_FLOOR${floor}
        headroom=${!headroom_name:-}
        if ! [[ "$headroom" =~ ^[0-9]+$ ]] || (( headroom % 128 != 0 )); then
            echo "$headroom_name must be a nonnegative multiple of 128" >&2
            exit 2
        fi
    done
    training_name=${prefix}_TRAINING_MIN_FREE_MIB
    if ! [[ "${!training_name:-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "$training_name must be a measured positive integer" >&2
        exit 2
    fi
fi

default_root=/data/adafloor_shared_state/deepseek_v2_lite/kv_cap_authorization_${lifecycle}_${execution_code_sha256:0:12}
RUN_ROOT=${DEEPSEEK_KV_CAP_VALIDATION_OUTPUT_ROOT:-$default_root}
SUMMARY_PATH=$RUN_ROOT/KV_CAP_AUTHORIZATION_SUMMARY.json
floor_csv=$(IFS=,; printf '%s' "${floors[*]}")
RESUME_AUDIT=${DEEPSEEK_KV_CAP_VALIDATION_RESUME_AUDIT:-0}
if [[ "$RESUME_AUDIT" != 0 && "$RESUME_AUDIT" != 1 ]]; then
    echo "DEEPSEEK_KV_CAP_VALIDATION_RESUME_AUDIT must be 0 or 1" >&2
    exit 2
fi

echo "[DeepSeek KV authorization] lifecycle=$lifecycle candidate=$CAP_ENV"
echo "[DeepSeek KV authorization] common_epoch0=$COMMON_EPOCH0_ROOT"
echo "[DeepSeek KV authorization] trigger=$TRIGGER_ROOT"
echo "[DeepSeek KV authorization] workload_profile=$WORKLOAD_PROFILE_ID sha256=$WORKLOAD_PROFILE_SHA256"
echo "[DeepSeek KV authorization] output=$RUN_ROOT floors=$floor_csv"

if [[ "${DEEPSEEK_KV_CAP_VALIDATION_DRY_RUN:-0}" == 1 ]]; then
    for floor in "${floors[@]}"; do
        admission_name=${prefix}_KV_ADMISSION_FLOOR${floor}
        physical_name=${prefix}_KV_PHYSICAL_FLOOR${floor}
        echo "[DeepSeek KV authorization] dry_run floor=$floor admission=${!admission_name} physical=${!physical_name}"
    done
    exit 0
fi

mkdir -p "$(dirname "$LOCK_PATH")"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo "another DeepSeek calibration or authorization holds $LOCK_PATH" >&2
    exit 2
fi
npu_processes=$(npu-smi info | awk '
    /Process id/ { in_process_table=1; next }
    in_process_table && /^\|/ && !/No running processes found/ { print }
')
if [[ -n "$npu_processes" ]]; then
    echo "DeepSeek $label authorization requires exclusive idle NPUs" >&2
    printf '%s\n' "$npu_processes" >&2
    exit 2
fi
if [[ "$RESUME_AUDIT" == 1 ]]; then
    if [[ ! -d "$RUN_ROOT" || ! -f "$RUN_ROOT/INCOMPLETE" \
          || -f "$RUN_ROOT/COMPLETE" ]]; then
        echo "resume-audit requires one incomplete authorization root: $RUN_ROOT" >&2
        exit 2
    fi
    for floor in "${floors[@]}"; do
        if [[ ! -d "$RUN_ROOT/floor${floor}/epoch_001_mode1_${policy}" ]]; then
            echo "resume-audit is missing completed floor${floor} runtime artifacts" >&2
            exit 2
        fi
    done
else
    if [[ -e "$RUN_ROOT" ]]; then
        echo "refusing to overwrite DeepSeek authorization root: $RUN_ROOT" >&2
        exit 2
    fi
    mkdir -p "$RUN_ROOT"
    printf '%s\n' \
        "INCOMPLETE DeepSeek $label KV cap authorization" \
        "candidate=$CAP_ENV" \
        "execution_code_sha256=$execution_code_sha256" \
        "floors=$floor_csv" > "$RUN_ROOT/INCOMPLETE"
fi
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN_ROOT/EXIT_CODE"' EXIT

floor_caps() {
    local kind=$1
    local entries=()
    local candidate_floor
    local name
    for candidate_floor in "${floors[@]}"; do
        name=${prefix}_KV_${kind}_FLOOR${candidate_floor}
        entries+=("${candidate_floor}:${!name}")
    done
    local joined
    joined=$(IFS=,; printf '%s' "${entries[*]}")
    printf '%s' "$joined"
}

run_floor() (
    local floor=$1
    local candidate_floor
    local source_name
    local target_name

    export REPO_ROOT=$SCRIPT_DIR
    export PATCH_TREE=$SCRIPT_DIR
    export MODEL_PATH=/data/DeepSeek-V2-Lite-Chat
    export MODEL_REVISION=85864749cd611b4353ce1decdb286193298f64c7
    export DISTCP_PATH=/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4
    export LOCAL_TEST_LAUNCHER=$SCRIPT_DIR/internal/wj_train_grpo_deepseek_v2_lite_16die_adafloor.sh
    export BASELINE_LAUNCHER=$LOCAL_TEST_LAUNCHER
    export DYNAMIC_MODE1_CHILD_SCRIPT=$MODE1_CHILD
    export CHECKPOINT_MODEL_DIR_NAME=deepseek_v2_lite
    export TRAIN_LOG_PREFIX=deepseek-v2-lite-${lifecycle}-cap-authorization-f${floor}
    export PLANNER_TOKENIZER_PATH=$MODEL_PATH
    export TRAIN_FILE_ORIG=/data/deepscaler/train.parquet
    export TEST_FILE=/data/deepscaler/test.parquet
    export TASK_QUEUE_ENABLE=1
    export DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM=False
    export DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP=False
    export DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS=False
    export DEEPSEEK_ACTOR_RECOMPUTE_METHOD=uniform
    export DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS=1
    export HCCL_BUFFSIZE=800

    export DYNAMIC_OUTPUT_ROOT=$RUN_ROOT
    export DYNAMIC_RUN_NAME=floor${floor}
    export DYNAMIC_SKIP_MODE0_PROBE=1
    export DYNAMIC_INITIAL_BASELINE_DIR=$TRIGGER_ROOT
    export DYNAMIC_START_EPOCH=1
    export DYNAMIC_TOTAL_EPOCHS=2
    export DYNAMIC_PLAN_STEPS=1
    export DYNAMIC_TRAIN_STEPS=1
    export DYNAMIC_DATASET_FRACTION=$WORKLOAD_PROBE_DATASET_FRACTION
    export DYNAMIC_LENGTH_EMA_DECAY=0.3
    export DYNAMIC_ENABLE_CKPT_CHAIN=0
    export DYNAMIC_BUILD_OFFLINE_PLANNING_HISTORY=0
    export DYNAMIC_ENABLE_THRESHOLD_CONTROL=0
    export DYNAMIC_SHRINK_POLICY=$policy
    export DYNAMIC_FORCE_SELECTED_FLOORS=$floor
    unset DYNAMIC_FORCE_SELECTED_FLOOR FORCE_SELECTED_FLOOR FORCE_SELECTED_FLOORS
    # The plan TailGuard cap is part of the authorization protocol. Without
    # this gate, capped tail ranks can all finish together and skip the exact
    # Natural quorum, making a nominal floor run exercise no transition.
    export DYNAMIC_SHORT_STEP_CAP_ENABLE=1
    export DYNAMIC_FULL_MAX_PROMPT_LENGTH=$WORKLOAD_MAX_PROMPT_LENGTH
    export DYNAMIC_FULL_MAX_RESPONSE_LENGTH=$WORKLOAD_MAX_RESPONSE_LENGTH
    export DYNAMIC_FULL_MAX_RESPONSE_LEN=$WORKLOAD_MAX_RESPONSE_LENGTH
    export DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS=$WORKLOAD_MAX_NUM_BATCHED_TOKENS
    unset DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS
    unset DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP
    unset VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS
    unset VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP
    export DYNAMIC_TAIL_GUARD_RATIO_QUANTILE=0.95
    export DYNAMIC_TAIL_GUARD_RATIO_WINDOW=3
    export DYNAMIC_TAIL_GUARD_DEFAULT_RATIO=1.20
    export DYNAMIC_TAIL_GUARD_MIN_CAP=4096
    export DYNAMIC_TAIL_GUARD_ROUND_TO=512
    export DYNAMIC_DISABLE_TAIL_GUARD=0
    export DYNAMIC_EXPECT_NO_RESPONSE_CAPS=0

    export TRAIN_BATCH_SIZE=$WORKLOAD_TRAIN_BATCH_SIZE
    export ROLLOUT_N=$WORKLOAD_ROLLOUT_N
    export ROLLOUT_MAX_NUM_SEQS=$WORKLOAD_MAX_NUM_SEQS
    export ROLLOUT_GPU_MEMORY_UTILIZATION=0.9
    export ROLLOUT_ENFORCE_EAGER=True
    export ACTIVE_PEAK_SAFETY_FACTOR=1.16
    export VLLM_KV_BLOCK_SIZE=128
    export SAVE_CKPT_ENABLE=0
    export TRAINER_SAVE_FREQ=-1
    export BASELINE_ALLOW_INFEASIBLE_PLAN=0
    unset ALLOW_INFEASIBLE_PLAN
    unset MODE1_PLAN_ONLY DEEPSEEK_PROBE_PLAN_ONLY

    export MIN_ADAPTIVE_FLOOR=$min_floor
    export RANK_MATCHING_POLICY=release_area
    export VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1
    export VLLM_ASCEND_MODE1_ADAPTIVE_KV_FAIL_ON_UNMET_TARGET=1
    export VLLM_ASCEND_MODE1_ADAPTIVE_KV_MIN_TARGET_RATIO=1.0
    for candidate_floor in "${floors[@]}"; do
        source_name=${prefix}_KV_PHYSICAL_FLOOR${candidate_floor}
        target_name=VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR${candidate_floor}
        printf -v "$target_name" '%s' "${!source_name}"
        export "$target_name"
    done
    source_name=${prefix}_KV_PHYSICAL_FLOOR16
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=${!source_name}
    export FLOOR_KV_CAPS
    FLOOR_KV_CAPS=$(floor_caps ADMISSION)
    export PHYSICAL_FLOOR_KV_CAPS
    PHYSICAL_FLOOR_KV_CAPS=$(floor_caps PHYSICAL)
    source_name=${prefix}_KV_ADMISSION_FLOOR16
    export MAX_RANK_PEAK_TOKENS=${!source_name}

    echo "[DeepSeek KV authorization] start lifecycle=$lifecycle floor=$floor"
    "$TARGET" \
        trainer.resume_mode=resume_path \
        "trainer.resume_from_path=$DYNAMIC_INITIAL_RESUME_CKPT"
)

if [[ "$RESUME_AUDIT" == 0 ]]; then
    for floor in "${floors[@]}"; do
        run_floor "$floor"
    done
fi

verification_migration_args=()
if [[ -n "${DEEPSEEK_KV_CAP_EXPECTED_RUNTIME_EXECUTION_SHA256:-}" \
      || -n "${DEEPSEEK_KV_CAP_EXPECTED_VERIFICATION_CODE_SHA256:-}" ]]; then
    if [[ -z "${DEEPSEEK_KV_CAP_EXPECTED_RUNTIME_EXECUTION_SHA256:-}" \
          || -z "${DEEPSEEK_KV_CAP_EXPECTED_VERIFICATION_CODE_SHA256:-}" ]]; then
        echo "authorization code migration requires both expected SHA256 values" >&2
        exit 2
    fi
    verification_migration_args+=(
        --expected-runtime-execution-sha256
        "$DEEPSEEK_KV_CAP_EXPECTED_RUNTIME_EXECUTION_SHA256"
        --expected-verification-code-sha256
        "$DEEPSEEK_KV_CAP_EXPECTED_VERIFICATION_CODE_SHA256"
    )
fi

python3 "$SCRIPT_DIR/tools/verify_deepseek_kv_cap_run.py" \
    --lifecycle "$lifecycle" \
    --cap-env "$CAP_ENV" \
    --run-root "$RUN_ROOT" \
    --common-epoch0-root "$COMMON_EPOCH0_ROOT" \
    --trigger-root "$TRIGGER_ROOT" \
    "${verification_migration_args[@]}" \
    --output "$SUMMARY_PATH"

rm -f "$RUN_ROOT/INCOMPLETE"
printf '%s\n' \
    "COMPLETE DeepSeek $label KV cap authorization" \
    "floors=$floor_csv" \
    "summary=$SUMMARY_PATH" > "$RUN_ROOT/COMPLETE"
echo 0 > "$RUN_ROOT/EXIT_CODE"
trap - EXIT
echo "[DeepSeek KV authorization] complete lifecycle=$lifecycle caps promoted"
