#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAP_ENV=${DEEPSEEK_KV_CAP_ENV:-$SCRIPT_DIR/deepseek_v2_lite_kv_caps.env}
WORKLOAD_PROFILE_PATH=${DEEPSEEK_WORKLOAD_PROFILE_PATH:-}
N_F4_RUNTIME_PROFILE_PATH=$SCRIPT_DIR/internal/deepseek_v2_lite_natural_f4_runtime_profile.sh
N_F2_RUNTIME_PROFILE_PATH=$SCRIPT_DIR/internal/deepseek_v2_lite_natural_f2_runtime_profile.sh
P_F4_RUNTIME_PROFILE_PATH=$SCRIPT_DIR/internal/deepseek_v2_lite_planned_f4_runtime_profile.sh
P_F2_RUNTIME_PROFILE_PATH=$SCRIPT_DIR/internal/deepseek_v2_lite_planned_f2_runtime_profile.sh
GENERALIZED_MODE1_CHILD_PATH=$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh

WORKLOAD_PROFILE_SHA256=
if [[ -n "$WORKLOAD_PROFILE_PATH" ]]; then
    if [[ ! -f "$WORKLOAD_PROFILE_PATH" ]]; then
        echo "missing DeepSeek workload profile: $WORKLOAD_PROFILE_PATH" >&2
        exit 2
    fi
    WORKLOAD_PROFILE_PATH=$(realpath "$WORKLOAD_PROFILE_PATH")
    WORKLOAD_PROFILE_SHA256=$(sha256sum "$WORKLOAD_PROFILE_PATH" | awk '{print $1}')
    # shellcheck disable=SC1090
    source "$WORKLOAD_PROFILE_PATH"
    if [[ -z "${DEEPSEEK_WORKLOAD_PROFILE_ID:-}" ]]; then
        echo "DeepSeek workload profile does not define DEEPSEEK_WORKLOAD_PROFILE_ID" >&2
        exit 2
    fi
fi

usage() {
    cat <<'EOF'
Usage: ./run_deepseek_v2_lite_fair_compare.sh VARIANT

Variants are vanilla, lengthsort, lengthsort_guard, fixed4, minskew,
adafloor_n_f4, adafloor_n_f2, adafloor_p_f4, adafloor_p_f2,
adafloor_n_f2_noguard, and all.

Set DEEPSEEK_FAIR_DRY_RUN=1 to validate paths and print each delegated command
without starting Ray, vLLM, conversion, or training.
EOF
}

if [[ $# -lt 1 || "$1" == -h || "$1" == --help ]]; then
    usage
    exit 0
fi
variant=$1
shift
if (( $# != 0 )); then
    echo "DeepSeek fair comparisons do not accept Hydra overrides" >&2
    exit 2
fi

if [[ ! -f "$CAP_ENV" ]]; then
    echo "missing verified DeepSeek KV cap file: $CAP_ENV" >&2
    echo "Start from deepseek_v2_lite_kv_caps.env.example after running the probes." >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$CAP_ENV"
cap_validation_mode=${DEEPSEEK_KV_CAP_VALIDATION_MODE:-0}
if [[ "$cap_validation_mode" == 1 ]]; then
    case "$variant" in
        adafloor_n_f4|adafloor_n_f2|adafloor_p_f4|adafloor_p_f2) ;;
        *)
            echo "DeepSeek KV cap validation requires one AdaFloor lifecycle" >&2
            exit 2
            ;;
    esac
fi
if [[ "$cap_validation_mode" == 1 ]]; then
    if [[ -z "${FAIR_START_EPOCH+x}" ]]; then
        export FAIR_START_EPOCH=1
    fi
    if [[ -z "${FAIR_TOTAL_EPOCHS+x}" ]]; then
        export FAIR_TOTAL_EPOCHS=2
    fi
else
    if [[ -z "${FAIR_START_EPOCH+x}" ]]; then
        export FAIR_START_EPOCH=1
    fi
    if [[ -z "${FAIR_TOTAL_EPOCHS+x}" ]]; then
        export FAIR_TOTAL_EPOCHS=3
    fi
fi
if [[ -z "${FAIR_FREEZE_ACTOR+x}" ]]; then
    export FAIR_FREEZE_ACTOR=0
fi
fair_dataset_fraction=${DEEPSEEK_FAIR_DATASET_FRACTION:-}
if [[ -z "${DYNAMIC_DATASET_FRACTION+x}" ]]; then
    if [[ -n "$fair_dataset_fraction" ]]; then
        export DYNAMIC_DATASET_FRACTION=$fair_dataset_fraction
    elif [[ -n "${COMMON_EPOCH0_DATASET_FRACTION:-}" ]]; then
        export DYNAMIC_DATASET_FRACTION=$COMMON_EPOCH0_DATASET_FRACTION
    else
        export DYNAMIC_DATASET_FRACTION=0.005
    fi
fi
if [[ -n "$fair_dataset_fraction" \
      && "$DYNAMIC_DATASET_FRACTION" != "$fair_dataset_fraction" ]]; then
    echo "DEEPSEEK_FAIR_DATASET_FRACTION differs from DYNAMIC_DATASET_FRACTION" >&2
    exit 2
fi
export FAIR_DATASET_FRACTION=$DYNAMIC_DATASET_FRACTION
export TRAIN_FILE_ORIG=/data/deepscaler/train.parquet
export TEST_FILE=/data/deepscaler/test.parquet
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1
export VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1
export MAX_CROSS_STEP_REPAIR_SWAPS=8
export REPAIR_CANDIDATE_LIMIT=8

require_cap() {
    local name=$1
    local value=${!name:-0}
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]] || (( value % 128 != 0 )); then
        echo "$name must be a verified positive multiple of 128, got $value" >&2
        exit 2
    fi
}

require_nonnegative_block_tokens() {
    local name=$1
    local value=${!name:-}
    if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value % 128 != 0 )); then
        echo "$name must be a nonnegative multiple of 128, got ${value:-<unset>}" >&2
        exit 2
    fi
}

require_lifecycle_verified() {
    local name=$1
    local lifecycle=$2
    if [[ "${!name:-0}" != 1 ]]; then
        echo "$lifecycle KV caps are not verified in $CAP_ENV" >&2
        exit 2
    fi
}

verify_runtime_provenance() (
    local profile_path=$1
    local profile_id_name=$2
    local profile_files_name=$3
    local recorded_profile_name=$4
    local recorded_sha_name=$5
    local label=$6
    local expected_profile
    local profile_files
    local current_profile_sha256
    local current_execution_code_sha256
    local profile_hash_args=()
    local runtime_profile_file
    local runtime_profile_file_array=()

    # shellcheck disable=SC1090
    source "$profile_path"
    expected_profile=${!profile_id_name:-}
    profile_files=${!profile_files_name:-}
    IFS=, read -r -a runtime_profile_file_array <<< "$profile_files"
    for runtime_profile_file in "${runtime_profile_file_array[@]}"; do
        profile_hash_args+=(--profile "$runtime_profile_file")
    done
    current_profile_sha256=$(python3 \
        "$SCRIPT_DIR/tools/hash_deepseek_runtime_profile.py" \
        --root "$SCRIPT_DIR" "${profile_hash_args[@]}")
    if [[ "${!recorded_profile_name:-}" != "$expected_profile" ]]; then
        echo "$label runtime profile does not match its KV probes" >&2
        exit 2
    fi
    if [[ "${!recorded_sha_name:-}" != "$current_profile_sha256" ]]; then
        echo "$label runtime profile SHA256 does not match its KV probes" >&2
        exit 2
    fi
    current_execution_code_sha256=$(python3 \
        "$SCRIPT_DIR/tools/hash_deepseek_execution_code.py" \
        --root "$SCRIPT_DIR")
    if [[ "${DEEPSEEK_EXECUTION_CODE_SHA256:-}" \
          != "$current_execution_code_sha256" ]]; then
        echo "DeepSeek execution code SHA256 does not match its KV probes" >&2
        exit 2
    fi
)

verify_n_f4_runtime_provenance() {
    verify_runtime_provenance \
        "$N_F4_RUNTIME_PROFILE_PATH" \
        DEEPSEEK_N_F4_RUNTIME_PROFILE_ID \
        DEEPSEEK_N_F4_RUNTIME_PROFILE_FILES \
        DEEPSEEK_N_F4_RUNTIME_PROFILE \
        DEEPSEEK_N_F4_RUNTIME_PROFILE_SHA256 \
        "Natural floor4"
}

verify_n_f2_runtime_provenance() {
    verify_runtime_provenance \
        "$N_F2_RUNTIME_PROFILE_PATH" \
        DEEPSEEK_N_F2_RUNTIME_PROFILE_ID \
        DEEPSEEK_N_F2_RUNTIME_PROFILE_FILES \
        DEEPSEEK_N_F2_RUNTIME_PROFILE \
        DEEPSEEK_N_F2_RUNTIME_PROFILE_SHA256 \
        "Natural floor2"
}

verify_p_f4_runtime_provenance() {
    verify_runtime_provenance \
        "$P_F4_RUNTIME_PROFILE_PATH" \
        DEEPSEEK_P_F4_RUNTIME_PROFILE_ID \
        DEEPSEEK_P_F4_RUNTIME_PROFILE_FILES \
        DEEPSEEK_P_F4_RUNTIME_PROFILE \
        DEEPSEEK_P_F4_RUNTIME_PROFILE_SHA256 \
        "Planned floor4"
}

verify_p_f2_runtime_provenance() {
    verify_runtime_provenance \
        "$P_F2_RUNTIME_PROFILE_PATH" \
        DEEPSEEK_P_F2_RUNTIME_PROFILE_ID \
        DEEPSEEK_P_F2_RUNTIME_PROFILE_FILES \
        DEEPSEEK_P_F2_RUNTIME_PROFILE \
        DEEPSEEK_P_F2_RUNTIME_PROFILE_SHA256 \
        "Planned floor2"
}

export MODEL_PATH=${MODEL_PATH:-/data/DeepSeek-V2-Lite-Chat}
export MODEL_REVISION=${MODEL_REVISION:-85864749cd611b4353ce1decdb286193298f64c7}
export DISTCP_PATH=${DISTCP_PATH:-/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4}
export COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/common_epoch0_deepseek_v2_lite_tq1_no_overlap_gpu09}
if [[ -z "${DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT:-}" \
      || "$(realpath -m "$DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT")" \
         != "$(realpath -m "$COMMON_EPOCH0_ROOT")" ]]; then
    echo "DeepSeek KV cap common epoch0 provenance mismatch" >&2
    exit 2
fi
export FAIR_OUTPUT_ROOT=${FAIR_OUTPUT_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/fair_comparisons}
export LOCAL_TEST_LAUNCHER=$SCRIPT_DIR/internal/wj_train_grpo_deepseek_v2_lite_16die_adafloor.sh
export BASELINE_LAUNCHER=$LOCAL_TEST_LAUNCHER
export CHECKPOINT_MODEL_DIR_NAME=${CHECKPOINT_MODEL_DIR_NAME:-deepseek_v2_lite_chat}
export TRAIN_LOG_PREFIX=${TRAIN_LOG_PREFIX:-deepseek-v2-lite-chat-adafloor}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM=${DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM:-False}
export DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP=${DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP:-False}
export DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS=${DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS:-False}
export DEEPSEEK_ACTOR_RECOMPUTE_METHOD=${DEEPSEEK_ACTOR_RECOMPUTE_METHOD:-uniform}
export DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS=${DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS:-1}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-800}
export EXPECTED_COMMON_EPOCH0_EXECUTION_PROFILE="deepseek-v2-lite_tq${TASK_QUEUE_ENABLE}_a2a${DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM}_shared${DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP}_dealloc${DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS}_recompute${DEEPSEEK_ACTOR_RECOMPUTE_METHOD}x${DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS}_hccl${HCCL_BUFFSIZE}"
export PLANNER_TOKENIZER_PATH=${PLANNER_TOKENIZER_PATH:-$MODEL_PATH}
export BASELINE_ALLOW_INFEASIBLE_PLAN=0
export FAIR_TRAIN_BATCH_SIZE=${FAIR_TRAIN_BATCH_SIZE:-${COMMON_EPOCH0_TRAIN_BATCH_SIZE:-32}}
export FAIR_ROLLOUT_N=${FAIR_ROLLOUT_N:-${COMMON_EPOCH0_ROLLOUT_N:-16}}
export FAIR_MAX_RESPONSE_LENGTH=${FAIR_MAX_RESPONSE_LENGTH:-${COMMON_EPOCH0_MAX_RESPONSE_LENGTH:-16384}}
export FAIR_PROMPTS_PER_EPOCH=${FAIR_PROMPTS_PER_EPOCH:-${COMMON_EPOCH0_PROMPTS_TOTAL:-160}}
export VLLM_KV_BLOCK_SIZE=128
export DYNAMIC_ENABLE_THRESHOLD_CONTROL=0
export DYNAMIC_FULL_MAX_PROMPT_LENGTH=${DYNAMIC_FULL_MAX_PROMPT_LENGTH:-${COMMON_EPOCH0_MAX_PROMPT_LENGTH:-1024}}
export DYNAMIC_FULL_MAX_RESPONSE_LENGTH=${DYNAMIC_FULL_MAX_RESPONSE_LENGTH:-$FAIR_MAX_RESPONSE_LENGTH}
export DYNAMIC_FULL_MAX_RESPONSE_LEN=${DYNAMIC_FULL_MAX_RESPONSE_LEN:-$FAIR_MAX_RESPONSE_LENGTH}
export DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS=${DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS:-${COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS:-17408}}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-${COMMON_EPOCH0_MAX_NUM_SEQS:-32}}
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.9
export ROLLOUT_ENFORCE_EAGER=True
export ACTIVE_PEAK_SAFETY_FACTOR=1.16
export DYNAMIC_TAIL_GUARD_RATIO_QUANTILE=0.95
export DYNAMIC_TAIL_GUARD_RATIO_WINDOW=3
export DYNAMIC_TAIL_GUARD_DEFAULT_RATIO=1.20
export DYNAMIC_TAIL_GUARD_MIN_CAP=4096
export DYNAMIC_TAIL_GUARD_ROUND_TO=512
export DYNAMIC_DISABLE_TAIL_GUARD=0
export DYNAMIC_EXPECT_NO_RESPONSE_CAPS=0
export DYNAMIC_IGNORE_TAIL_TIES_AT_RESPONSE_CAP=0
export DYNAMIC_SHORT_STEP_CAP_ENABLE=${DYNAMIC_SHORT_STEP_CAP_ENABLE:-1}
export DYNAMIC_SHORT_STEP_EXIT_THRESHOLD=4096
export DYNAMIC_SHORT_STEP_CAP_TOKENS=4096
export DYNAMIC_SHORT_STEP_CAP_FLOORS=4
export VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_FAIL_ON_UNMET_TARGET=1
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_MIN_TARGET_RATIO=1.0
if [[ "${DEEPSEEK_KV_CAP_MODEL_REVISION:-}" != "$MODEL_REVISION" ]]; then
    echo "DeepSeek KV cap model revision mismatch" >&2
    exit 2
fi
if [[ "${DEEPSEEK_KV_CAP_EXECUTION_PROFILE:-}" != "$EXPECTED_COMMON_EPOCH0_EXECUTION_PROFILE" ]]; then
    echo "DeepSeek KV cap execution profile mismatch" >&2
    exit 2
fi
if ! [[ "$FAIR_TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ \
        && "$FAIR_ROLLOUT_N" =~ ^[1-9][0-9]*$ \
        && "$FAIR_PROMPTS_PER_EPOCH" =~ ^[1-9][0-9]*$ \
        && "$ROLLOUT_MAX_NUM_SEQS" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid DeepSeek fair workload dimensions" >&2
    exit 2
fi
if (( FAIR_PROMPTS_PER_EPOCH % FAIR_TRAIN_BATCH_SIZE != 0 )); then
    echo "FAIR_PROMPTS_PER_EPOCH must be divisible by FAIR_TRAIN_BATCH_SIZE" >&2
    exit 2
fi
computed_fair_expected_steps=$((FAIR_PROMPTS_PER_EPOCH / FAIR_TRAIN_BATCH_SIZE))
FAIR_EXPECTED_STEPS=${FAIR_EXPECTED_STEPS:-$computed_fair_expected_steps}
if ! [[ "$FAIR_EXPECTED_STEPS" =~ ^[1-9][0-9]*$ ]] \
   || (( FAIR_EXPECTED_STEPS != computed_fair_expected_steps )); then
    echo "FAIR_EXPECTED_STEPS must equal prompts_per_epoch / train_batch_size" >&2
    exit 2
fi
export FAIR_EXPECTED_STEPS
fair_force_selected_floor=${DEEPSEEK_FAIR_FORCE_SELECTED_FLOOR:-}
if [[ -n "$fair_force_selected_floor" ]]; then
    if [[ ! "$fair_force_selected_floor" =~ ^(2|4|8|16)$ ]]; then
        echo "DEEPSEEK_FAIR_FORCE_SELECTED_FLOOR must be one of 2, 4, 8, or 16" >&2
        exit 2
    fi
    if [[ "$variant" != adafloor_n_f2 || "$FAIR_EXPECTED_STEPS" != 1 \
          || "$cap_validation_mode" == 1 ]]; then
        echo "DEEPSEEK_FAIR_FORCE_SELECTED_FLOOR is restricted to a one-step Natural floor2 gate" >&2
        exit 2
    fi
fi
if [[ "${DEEPSEEK_KV_CAP_TRAIN_BATCH_SIZE:-32}" != "$FAIR_TRAIN_BATCH_SIZE" \
      || "${DEEPSEEK_KV_CAP_ROLLOUT_N:-}" != "$FAIR_ROLLOUT_N" \
      || "${DEEPSEEK_KV_CAP_EXPECTED_RESPONSES_PER_STEP:-$((FAIR_TRAIN_BATCH_SIZE * FAIR_ROLLOUT_N))}" \
         != "$((FAIR_TRAIN_BATCH_SIZE * FAIR_ROLLOUT_N))" \
      || "${DEEPSEEK_KV_CAP_MAX_PROMPT_LENGTH:-}" != "$DYNAMIC_FULL_MAX_PROMPT_LENGTH" \
      || "${DEEPSEEK_KV_CAP_MAX_RESPONSE_LENGTH:-}" != "$FAIR_MAX_RESPONSE_LENGTH" \
      || "${DEEPSEEK_KV_CAP_MAX_NUM_BATCHED_TOKENS:-}" != "$DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS" \
      || "${DEEPSEEK_KV_CAP_MAX_NUM_SEQS:-}" != "$ROLLOUT_MAX_NUM_SEQS" \
      || "${DEEPSEEK_KV_CAP_GPU_MEMORY_UTILIZATION:-}" != 0.9 \
      || "${DEEPSEEK_KV_CAP_ENFORCE_EAGER:-}" != True ]]; then
    echo "DeepSeek KV cap workload profile mismatch" >&2
    exit 2
fi
if [[ "${DEEPSEEK_KV_CAP_BLOCK_SIZE:-}" != 128 \
      || "${DEEPSEEK_KV_CAP_ROLLOUT_N:-}" != 16 \
      || "${DEEPSEEK_KV_CAP_TARGET_RATIO:-}" != 1.0 ]]; then
    echo "DeepSeek KV cap safety policy mismatch" >&2
    exit 2
fi
if [[ -n "$WORKLOAD_PROFILE_PATH" ]]; then
    if ! [[ "${COMMON_EPOCH0_TRAIN_STEPS:-}" =~ ^[1-9][0-9]*$ \
            && "${COMMON_EPOCH0_TRAIN_BATCH_SIZE:-}" =~ ^[1-9][0-9]*$ \
            && "${COMMON_EPOCH0_ROLLOUT_N:-}" =~ ^[1-9][0-9]*$ \
            && "${COMMON_EPOCH0_PROMPTS_TOTAL:-}" =~ ^[1-9][0-9]*$ \
            && "${COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP:-}" =~ ^[1-9][0-9]*$ \
            && "${COMMON_EPOCH0_MAX_NUM_SEQS:-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "invalid DeepSeek workload profile dimensions" >&2
        exit 2
    fi
    if (( COMMON_EPOCH0_TRAIN_BATCH_SIZE % 16 != 0 \
          || COMMON_EPOCH0_PROMPTS_TOTAL \
             != COMMON_EPOCH0_TRAIN_BATCH_SIZE * COMMON_EPOCH0_TRAIN_STEPS \
          || COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP \
             != COMMON_EPOCH0_TRAIN_BATCH_SIZE * COMMON_EPOCH0_ROLLOUT_N )); then
        echo "inconsistent DeepSeek workload profile dimensions" >&2
        exit 2
    fi
    if [[ "${DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID:-}" \
          != "$DEEPSEEK_WORKLOAD_PROFILE_ID" \
          || "${DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256:-}" \
             != "$WORKLOAD_PROFILE_SHA256" ]]; then
        echo "DeepSeek KV cap workload profile provenance mismatch" >&2
        exit 2
    fi
    if [[ "${DEEPSEEK_KV_CAP_COMMON_STEPS:-}" \
          != "${COMMON_EPOCH0_TRAIN_STEPS:-}" \
          || "${DEEPSEEK_KV_CAP_PROMPTS_TOTAL:-}" \
             != "${COMMON_EPOCH0_PROMPTS_TOTAL:-}" \
          || "${DEEPSEEK_KV_CAP_PROMPTS_PER_RANK:-}" \
             != "$((COMMON_EPOCH0_TRAIN_BATCH_SIZE / 16))" \
          || "${DEEPSEEK_KV_CAP_DATASET_FRACTION:-}" \
             != "${COMMON_EPOCH0_DATASET_FRACTION:-}" \
          || "${DEEPSEEK_KV_CAP_COMMON_PREEMPTION_POLICY:-}" \
             != "${COMMON_EPOCH0_PREEMPTION_POLICY:-}" ]]; then
        echo "DeepSeek KV cap common workload provenance mismatch" >&2
        exit 2
    fi
    if (( FAIR_EXPECTED_STEPS == 1 )); then
        expected_fair_dataset_fraction=${DEEPSEEK_KV_PROBE_DATASET_FRACTION:-}
    else
        expected_fair_dataset_fraction=$COMMON_EPOCH0_DATASET_FRACTION
    fi
    if [[ -z "$expected_fair_dataset_fraction" \
          || "$DYNAMIC_DATASET_FRACTION" != "$expected_fair_dataset_fraction" \
          || "$FAIR_DATASET_FRACTION" != "$expected_fair_dataset_fraction" ]]; then
        echo "DeepSeek fair dataset fraction does not match its workload profile" >&2
        exit 2
    fi
    if (( FAIR_EXPECTED_STEPS != 1 \
          && FAIR_EXPECTED_STEPS != COMMON_EPOCH0_TRAIN_STEPS )); then
        echo "DeepSeek fair run must use either one gate step or the full workload profile" >&2
        exit 2
    fi
fi
export FAIR_EXPECTED_DISTCP_SHARDS=${DEEPSEEK_EXPECTED_DISTCP_SHARDS:-auto}
export EXPECTED_COMMON_EPOCH0_MODEL_PATH=$MODEL_PATH
export EXPECTED_COMMON_EPOCH0_MODEL_REVISION=$MODEL_REVISION
export EXPECTED_COMMON_EPOCH0_DISTCP_PATH=$DISTCP_PATH
export EXPECTED_COMMON_EPOCH0_CHECKPOINT_MODEL_DIR_NAME=$CHECKPOINT_MODEL_DIR_NAME
export FAIR_DRY_RUN=${DEEPSEEK_FAIR_DRY_RUN:-${FAIR_DRY_RUN:-0}}
preflight_all_variants() {
    verify_n_f4_runtime_provenance
    verify_n_f2_runtime_provenance
    verify_p_f4_runtime_provenance
    verify_p_f2_runtime_provenance
    require_cap DEEPSEEK_VANILLA_KV_PHYSICAL_TOKENS
    require_lifecycle_verified DEEPSEEK_N_F4_KV_CAPS_VERIFIED "Natural floor4"
    require_lifecycle_verified DEEPSEEK_N_F2_KV_CAPS_VERIFIED "Natural floor2"
    require_lifecycle_verified DEEPSEEK_P_F4_KV_CAPS_VERIFIED "Planned floor4"
    require_lifecycle_verified DEEPSEEK_P_F2_KV_CAPS_VERIFIED "Planned floor2"
    local name
    for name in \
        DEEPSEEK_N_F4_KV_ADMISSION_FLOOR4 \
        DEEPSEEK_N_F4_KV_ADMISSION_FLOOR8 \
        DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR4 \
        DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR8 \
        DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR16 \
        DEEPSEEK_N_F2_KV_ADMISSION_FLOOR2 \
        DEEPSEEK_N_F2_KV_ADMISSION_FLOOR4 \
        DEEPSEEK_N_F2_KV_ADMISSION_FLOOR8 \
        DEEPSEEK_N_F2_KV_ADMISSION_FLOOR16 \
        DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR2 \
        DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR4 \
        DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR8 \
        DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR16 \
        DEEPSEEK_P_F4_KV_ADMISSION_FLOOR4 \
        DEEPSEEK_P_F4_KV_ADMISSION_FLOOR8 \
        DEEPSEEK_P_F4_KV_ADMISSION_FLOOR16 \
        DEEPSEEK_P_F4_KV_PHYSICAL_FLOOR4 \
        DEEPSEEK_P_F4_KV_PHYSICAL_FLOOR8 \
        DEEPSEEK_P_F4_KV_PHYSICAL_FLOOR16 \
        DEEPSEEK_P_F2_KV_ADMISSION_FLOOR2 \
        DEEPSEEK_P_F2_KV_ADMISSION_FLOOR4 \
        DEEPSEEK_P_F2_KV_ADMISSION_FLOOR8 \
        DEEPSEEK_P_F2_KV_ADMISSION_FLOOR16 \
        DEEPSEEK_P_F2_KV_PHYSICAL_FLOOR2 \
        DEEPSEEK_P_F2_KV_PHYSICAL_FLOOR4 \
        DEEPSEEK_P_F2_KV_PHYSICAL_FLOOR8 \
        DEEPSEEK_P_F2_KV_PHYSICAL_FLOOR16; do
        require_cap "$name"
    done
    for name in \
        DEEPSEEK_P_F4_HEADROOM_FLOOR4 \
        DEEPSEEK_P_F4_HEADROOM_FLOOR8 \
        DEEPSEEK_P_F4_HEADROOM_FLOOR16 \
        DEEPSEEK_P_F2_HEADROOM_FLOOR2 \
        DEEPSEEK_P_F2_HEADROOM_FLOOR4 \
        DEEPSEEK_P_F2_HEADROOM_FLOOR8 \
        DEEPSEEK_P_F2_HEADROOM_FLOOR16; do
        require_nonnegative_block_tokens "$name"
    done
    if ! [[ "${DEEPSEEK_P_F4_TRAINING_MIN_FREE_MIB:-0}" =~ ^[1-9][0-9]*$ ]]; then
        echo "DEEPSEEK_P_F4_TRAINING_MIN_FREE_MIB must be measured and positive" >&2
        exit 2
    fi
    if ! [[ "${DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB:-0}" =~ ^[1-9][0-9]*$ ]]; then
        echo "DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB must be measured and positive" >&2
        exit 2
    fi
}

if [[ "$variant" == all ]]; then
    preflight_all_variants
fi

if [[ "${DEEPSEEK_VALIDATE_ASSETS_ON_LAUNCH:-1}" == 1 ]]; then
    python3 "$SCRIPT_DIR/tools/validate_deepseek_v2_lite_assets.py" \
        --model-path "$MODEL_PATH" \
        --distcp-path "$DISTCP_PATH" \
        --expected-revision "$MODEL_REVISION" \
        --expected-pp-size 4 \
        --expected-ep-size 4
fi

run_one() (
    local item=$1
    shift
    unset FAIR_RUN_NAME DYNAMIC_RUN_NAME FLOOR_KV_CAPS PHYSICAL_FLOOR_KV_CAPS
    unset MAX_RANK_PEAK_TOKENS
    unset VANILLA_KV_TOKENS_PER_RANK
    unset VANILLA_KV_ADMISSION_TOKENS_PER_RANK
    unset VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS
    unset VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2
    unset VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4
    unset VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8
    unset VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16
    unset VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR4
    unset VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR2
    unset VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8
    unset VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR16
    unset VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS
    unset VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB
    unset VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB_FLOOR
    unset FAIR_PLANNED_MIN_FREE_MIB_FLOOR
    unset MIN_ADAPTIVE_FLOOR VLLM_ASCEND_SHRINK_AWARE_STAGES
    unset VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE
    unset RANK_MATCHING_POLICY FORCE_SELECTED_FLOOR FORCE_SELECTED_FLOORS
    unset DYNAMIC_FORCE_SELECTED_FLOOR DYNAMIC_FORCE_SELECTED_FLOORS
    unset DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS
    unset DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP
    unset VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS
    unset VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP
    case "$item" in
        vanilla|lengthsort|lengthsort_guard)
            require_cap DEEPSEEK_VANILLA_KV_PHYSICAL_TOKENS
            require_cap DEEPSEEK_VANILLA_KV_ADMISSION_TOKENS
            if (( DEEPSEEK_VANILLA_KV_ADMISSION_TOKENS \
                  >= DEEPSEEK_VANILLA_KV_PHYSICAL_TOKENS )); then
                echo "Vanilla admission capacity must be smaller than physical capacity" >&2
                exit 2
            fi
            export VANILLA_KV_TOKENS_PER_RANK=$DEEPSEEK_VANILLA_KV_PHYSICAL_TOKENS
            export VANILLA_KV_ADMISSION_TOKENS_PER_RANK=$DEEPSEEK_VANILLA_KV_ADMISSION_TOKENS
            ;;
        fixed4|adafloor_n_f4)
            # shellcheck disable=SC1091
            source "$N_F4_RUNTIME_PROFILE_PATH"
            verify_n_f4_runtime_provenance
            if [[ "$cap_validation_mode" == 1 ]]; then
                if [[ "$item" != adafloor_n_f4 ]]; then
                    echo "Natural floor4 cap validation cannot run $item" >&2
                    exit 2
                fi
                echo "[DeepSeek fair compare] validating unverified Natural floor4 KV caps"
            else
                require_lifecycle_verified \
                    DEEPSEEK_N_F4_KV_CAPS_VERIFIED "Natural floor4"
            fi
            for name in \
                DEEPSEEK_N_F4_KV_ADMISSION_FLOOR4 \
                DEEPSEEK_N_F4_KV_ADMISSION_FLOOR8 \
                DEEPSEEK_N_F4_KV_ADMISSION_FLOOR16 \
                DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR4 \
                DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR8 \
                DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR16; do
                require_cap "$name"
            done
            export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=$DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR4
            export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=$DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR8
            export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16=$DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR16
            export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=$DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR16
            export FLOOR_KV_CAPS="4:$DEEPSEEK_N_F4_KV_ADMISSION_FLOOR4,8:$DEEPSEEK_N_F4_KV_ADMISSION_FLOOR8,16:$DEEPSEEK_N_F4_KV_ADMISSION_FLOOR16"
            export PHYSICAL_FLOOR_KV_CAPS="4:$DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR4,8:$DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR8,16:$DEEPSEEK_N_F4_KV_PHYSICAL_FLOOR16"
            export MAX_RANK_PEAK_TOKENS=$DEEPSEEK_N_F4_KV_ADMISSION_FLOOR16
            export MIN_ADAPTIVE_FLOOR=4
            export VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4
            export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4
            export RANK_MATCHING_POLICY=release_area
            export DYNAMIC_SHRINK_POLICY=natural
            export FAIR_ADAFLOOR_TARGET=$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh
            if [[ "$cap_validation_mode" == 1 ]]; then
                validation_floors=${DEEPSEEK_KV_CAP_VALIDATION_FORCE_FLOORS:-}
                if [[ "$validation_floors" != 4,8,16,16,16 ]]; then
                    echo "DeepSeek KV cap validation floor coverage mismatch" >&2
                    exit 2
                fi
                export DYNAMIC_FORCE_SELECTED_FLOORS=$validation_floors
            fi
            ;;
        minskew|adafloor_n_f2|adafloor_n_f2_noguard)
            # shellcheck disable=SC1091
            source "$N_F2_RUNTIME_PROFILE_PATH"
            verify_n_f2_runtime_provenance
            if [[ "$cap_validation_mode" == 1 ]]; then
                if [[ "$item" != adafloor_n_f2 ]]; then
                    echo "Natural floor2 cap validation cannot run $item" >&2
                    exit 2
                fi
                echo "[DeepSeek fair compare] validating unverified Natural floor2 KV caps"
            else
                require_lifecycle_verified \
                    DEEPSEEK_N_F2_KV_CAPS_VERIFIED "Natural floor2"
            fi
            for name in \
                DEEPSEEK_N_F2_KV_ADMISSION_FLOOR2 \
                DEEPSEEK_N_F2_KV_ADMISSION_FLOOR4 \
                DEEPSEEK_N_F2_KV_ADMISSION_FLOOR8 \
                DEEPSEEK_N_F2_KV_ADMISSION_FLOOR16 \
                DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR2 \
                DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR4 \
                DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR8 \
                DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR16; do
                require_cap "$name"
            done
            export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2=$DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR2
            export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=$DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR4
            export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=$DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR8
            export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16=$DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR16
            export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=$DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR16
            export FLOOR_KV_CAPS="2:$DEEPSEEK_N_F2_KV_ADMISSION_FLOOR2,4:$DEEPSEEK_N_F2_KV_ADMISSION_FLOOR4,8:$DEEPSEEK_N_F2_KV_ADMISSION_FLOOR8,16:$DEEPSEEK_N_F2_KV_ADMISSION_FLOOR16"
            export PHYSICAL_FLOOR_KV_CAPS="2:$DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR2,4:$DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR4,8:$DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR8,16:$DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR16"
            export MAX_RANK_PEAK_TOKENS=$DEEPSEEK_N_F2_KV_ADMISSION_FLOOR16
            export MIN_ADAPTIVE_FLOOR=2
            export VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4,2
            export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2
            export VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=8,9,10,11,12,13,14,15
            export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=14,15
            export DYNAMIC_SHORT_STEP_CAP_FLOORS=2,4
            export DYNAMIC_MODE1_CHILD_SCRIPT=$GENERALIZED_MODE1_CHILD_PATH
            export DYNAMIC_SHRINK_POLICY=natural
            export FAIR_ADAFLOOR_TARGET=$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh
            if [[ "$item" == minskew ]]; then
                export RANK_MATCHING_POLICY=min_skew
            else
                export RANK_MATCHING_POLICY=release_area
            fi
            if [[ "$cap_validation_mode" == 1 ]]; then
                validation_floors=${DEEPSEEK_KV_CAP_VALIDATION_FORCE_FLOORS:-}
                if [[ "$validation_floors" != 2,4,8,16,16 ]]; then
                    echo "DeepSeek KV cap validation floor coverage mismatch" >&2
                    exit 2
                fi
                export DYNAMIC_FORCE_SELECTED_FLOORS=$validation_floors
            elif [[ -n "$fair_force_selected_floor" ]]; then
                export DYNAMIC_FORCE_SELECTED_FLOOR=$fair_force_selected_floor
            fi
            echo "[DeepSeek fair compare] Natural floor2 runtime_profile=$DEEPSEEK_N_F2_RUNTIME_PROFILE_ID child=$DYNAMIC_MODE1_CHILD_SCRIPT stages=$VLLM_ASCEND_SHRINK_AWARE_STAGES final_ranks=$VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS"
            ;;
        adafloor_p_f4|adafloor_p_f2)
            local planned_prefix
            local planned_label
            local planned_profile_path
            local planned_profile_id_name
            local planned_floors=()
            local planned_entries=()
            local physical_entries=()
            local floor
            local admission_name
            local physical_name
            local headroom_name
            local runtime_cap_name
            local training_name
            if [[ "$item" == adafloor_p_f2 ]]; then
                planned_prefix=DEEPSEEK_P_F2
                planned_label="Planned floor2"
                planned_profile_path=$P_F2_RUNTIME_PROFILE_PATH
                planned_profile_id_name=DEEPSEEK_P_F2_RUNTIME_PROFILE_ID
                planned_floors=(2 4 8 16)
                verify_p_f2_runtime_provenance
                export MIN_ADAPTIVE_FLOOR=2
                export DYNAMIC_SHORT_STEP_CAP_FLOORS=2,4
            else
                planned_prefix=DEEPSEEK_P_F4
                planned_label="Planned floor4"
                planned_profile_path=$P_F4_RUNTIME_PROFILE_PATH
                planned_profile_id_name=DEEPSEEK_P_F4_RUNTIME_PROFILE_ID
                planned_floors=(4 8 16)
                verify_p_f4_runtime_provenance
                export MIN_ADAPTIVE_FLOOR=4
            fi
            # shellcheck disable=SC1090
            source "$planned_profile_path"
            if [[ "$cap_validation_mode" == 1 ]]; then
                echo "[DeepSeek fair compare] validating unverified $planned_label KV caps"
            else
                require_lifecycle_verified \
                    "${planned_prefix}_KV_CAPS_VERIFIED" "$planned_label"
            fi
            for floor in "${planned_floors[@]}"; do
                admission_name=${planned_prefix}_KV_ADMISSION_FLOOR${floor}
                physical_name=${planned_prefix}_KV_PHYSICAL_FLOOR${floor}
                headroom_name=${planned_prefix}_HEADROOM_FLOOR${floor}
                require_cap "$admission_name"
                require_cap "$physical_name"
                require_nonnegative_block_tokens "$headroom_name"
                runtime_cap_name=VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR${floor}
                printf -v "$runtime_cap_name" '%s' "${!physical_name}"
                export "$runtime_cap_name"
                runtime_cap_name=VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR${floor}
                printf -v "$runtime_cap_name" '%s' "${!headroom_name}"
                export "$runtime_cap_name"
                planned_entries+=("${floor}:${!admission_name}")
                physical_entries+=("${floor}:${!physical_name}")
            done
            physical_name=${planned_prefix}_KV_PHYSICAL_FLOOR16
            export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=${!physical_name}
            FLOOR_KV_CAPS=$(IFS=,; printf '%s' "${planned_entries[*]}")
            PHYSICAL_FLOOR_KV_CAPS=$(IFS=,; printf '%s' "${physical_entries[*]}")
            export FLOOR_KV_CAPS PHYSICAL_FLOOR_KV_CAPS
            admission_name=${planned_prefix}_KV_ADMISSION_FLOOR16
            export MAX_RANK_PEAK_TOKENS=${!admission_name}
            headroom_name=${planned_prefix}_HEADROOM_FLOOR${MIN_ADAPTIVE_FLOOR}
            export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS=${!headroom_name}
            training_name=${planned_prefix}_TRAINING_MIN_FREE_MIB
            training_min_free_mib=${!training_name:-0}
            if ! [[ "$training_min_free_mib" =~ ^[1-9][0-9]*$ ]]; then
                echo "$training_name must be measured and positive" >&2
                exit 2
            fi
            export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB=$training_min_free_mib
            export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB_FLOOR=$training_min_free_mib
            export FAIR_PLANNED_MIN_FREE_MIB_FLOOR=$training_min_free_mib
            export DYNAMIC_MODE1_CHILD_SCRIPT=$GENERALIZED_MODE1_CHILD_PATH
            export DYNAMIC_SHRINK_POLICY=planned
            export FAIR_ADAFLOOR_TARGET=$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh
            export RANK_MATCHING_POLICY=release_area
            if [[ "$cap_validation_mode" == 1 ]]; then
                validation_floors=${DEEPSEEK_KV_CAP_VALIDATION_FORCE_FLOORS:-}
                if [[ "$MIN_ADAPTIVE_FLOOR" == 2 ]]; then
                    expected_validation_floors=2,4,8,16,16
                else
                    expected_validation_floors=4,8,16,16,16
                fi
                if [[ "$validation_floors" != "$expected_validation_floors" ]]; then
                    echo "DeepSeek KV cap validation floor coverage mismatch" >&2
                    exit 2
                fi
                export DYNAMIC_FORCE_SELECTED_FLOORS=$validation_floors
            fi
            echo "[DeepSeek fair compare] $planned_label profile=${!planned_profile_id_name} child=$DYNAMIC_MODE1_CHILD_SCRIPT stages=$VLLM_ASCEND_SHRINK_AWARE_STAGES"
            ;;
        *)
            echo "unknown DeepSeek comparison variant: $item" >&2
            exit 2
            ;;
    esac

    if [[ "$item" == vanilla ]]; then
        export FAIR_RUN_NAME=deepseek_v2_lite_${item}_common_epoch0_epoch1_2
    else
        export DYNAMIC_RUN_NAME=deepseek_v2_lite_${item}_common_epoch0_epoch1_2
    fi
    "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" "$item" "$@"
    if [[ "$item" == adafloor_n_f4 \
          || "$item" == adafloor_n_f2 \
          || "$item" == adafloor_p_f4 \
          || "$item" == adafloor_p_f2 ]]; then
        if [[ "$cap_validation_mode" == 1 || "$FAIR_DRY_RUN" == 1 ]]; then
            return
        fi
        local run_root=$FAIR_OUTPUT_ROOT/$DYNAMIC_RUN_NAME
        local expected_epochs
        local epoch
        local audit_lifecycle
        expected_epochs=
        for (( epoch = FAIR_START_EPOCH; epoch < FAIR_TOTAL_EPOCHS; epoch++ )); do
            if [[ -n "$expected_epochs" ]]; then
                expected_epochs+=,
            fi
            expected_epochs+=$epoch
        done
        case "$item" in
            adafloor_n_f4) audit_lifecycle=natural_f4 ;;
            adafloor_n_f2) audit_lifecycle=natural_f2 ;;
            adafloor_p_f4) audit_lifecycle=planned_f4 ;;
            adafloor_p_f2) audit_lifecycle=planned_f2 ;;
        esac
        python3 "$SCRIPT_DIR/tools/audit_deepseek_n_f4_formal_run.py" \
            --lifecycle "$audit_lifecycle" \
            --run-root "$run_root" \
            --cap-env "$CAP_ENV" \
            --expected-epochs "$expected_epochs" \
            --expected-steps "$FAIR_EXPECTED_STEPS" \
            --output "$run_root/DEEPSEEK_PLAN_RUNTIME_AUDIT.json"
    fi
)

if [[ "$variant" == all ]]; then
    variants=(
        vanilla lengthsort lengthsort_guard fixed4 minskew
        adafloor_n_f4 adafloor_n_f2 adafloor_p_f4 adafloor_p_f2
        adafloor_n_f2_noguard
    )
    for item in "${variants[@]}"; do
        run_one "$item" "$@"
    done
else
    run_one "$variant" "$@"
fi
