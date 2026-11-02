#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAP_ENV=${DEEPSEEK_KV_CAP_ENV:-$SCRIPT_DIR/deepseek_v2_lite_kv_caps.env}
COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/common_epoch0_deepseek_v2_lite_tq1_no_overlap_gpu09}
PRIMARY_MODEL=${MODEL_PATH:-/data/DeepSeek-V2-Lite-Chat}
PRIMARY_DISTCP=${DISTCP_PATH:-/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4}
SIDECAR_MODEL=${VERL_SIDECAR_MODEL_PATH:-/data/Qwen2.5-1.5B-Instruct}
SIDECAR_PROMPTS=${VERL_SIDECAR_PROMPTS_FILE:-/data/gsm8k}
TARGET=$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh
VERIFIER=$SCRIPT_DIR/tools/verify_deepseek_sidecar_run.py
RUN_MODE=print
LIFECYCLE=${DEEPSEEK_SIDECAR_LIFECYCLE:-natural_f4}

usage() {
    cat <<'EOF'
Usage: ./run_deepseek_v2_lite_sidecar_smoke.sh \
           [--lifecycle natural_f4|natural_f2|planned_f4|planned_f2] \
           [--execute]

Without --execute, perform the fail-closed preflight and print the exact launch
and verification paths without starting an NPU process. Use --execute to run
the one-step DeepSeek AdaFloor smoke and its real Qwen2.5-1.5B sidecar. The
default lifecycle is natural_f4.
EOF
}

while (( $# )); do
    case "$1" in
        --execute)
            RUN_MODE=execute
            shift
            ;;
        --lifecycle)
            if (( $# < 2 )); then
                usage >&2
                exit 2
            fi
            LIFECYCLE=$2
            shift 2
            ;;
        --lifecycle=*)
            LIFECYCLE=${1#*=}
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

case "$LIFECYCLE" in
    natural_f4)
        LIFECYCLE_LABEL="Natural floor4"
        RUN_SUFFIX=n_f4
        CAP_PREFIX=DEEPSEEK_N_F4
        RUNTIME_PROFILE=$SCRIPT_DIR/internal/deepseek_v2_lite_natural_f4_runtime_profile.sh
        SHRINK_POLICY=natural
        IS_PLANNED=0
        TARGET_FLOOR=4
        CAP_FLOORS=(4 8 16)
        SHRINK_STAGES=8,4
        MIN_COMPUTE_GROUP_SIZE=4
        FINAL_RANKS=12,13,14,15
        SHORT_STEP_CAP_FLOORS=4
        ;;
    natural_f2)
        LIFECYCLE_LABEL="Natural floor2"
        RUN_SUFFIX=n_f2
        CAP_PREFIX=DEEPSEEK_N_F2
        RUNTIME_PROFILE=$SCRIPT_DIR/internal/deepseek_v2_lite_natural_f2_runtime_profile.sh
        SHRINK_POLICY=natural
        IS_PLANNED=0
        TARGET_FLOOR=2
        CAP_FLOORS=(2 4 8 16)
        SHRINK_STAGES=8,4,2
        MIN_COMPUTE_GROUP_SIZE=2
        FINAL_RANKS=14,15
        SHORT_STEP_CAP_FLOORS=2,4
        ;;
    planned_f4)
        LIFECYCLE_LABEL="Planned floor4"
        RUN_SUFFIX=p_f4
        CAP_PREFIX=DEEPSEEK_P_F4
        RUNTIME_PROFILE=$SCRIPT_DIR/internal/deepseek_v2_lite_planned_f4_runtime_profile.sh
        SHRINK_POLICY=planned
        IS_PLANNED=1
        TARGET_FLOOR=4
        CAP_FLOORS=(4 8 16)
        SHRINK_STAGES=8,4
        MIN_COMPUTE_GROUP_SIZE=4
        FINAL_RANKS=12,13,14,15
        SHORT_STEP_CAP_FLOORS=4
        ;;
    planned_f2)
        LIFECYCLE_LABEL="Planned floor2"
        RUN_SUFFIX=p_f2
        CAP_PREFIX=DEEPSEEK_P_F2
        RUNTIME_PROFILE=$SCRIPT_DIR/internal/deepseek_v2_lite_planned_f2_runtime_profile.sh
        SHRINK_POLICY=planned
        IS_PLANNED=1
        TARGET_FLOOR=2
        CAP_FLOORS=(2 4 8 16)
        SHRINK_STAGES=8,4,2
        MIN_COMPUTE_GROUP_SIZE=2
        FINAL_RANKS=14,15
        SHORT_STEP_CAP_FLOORS=2,4
        ;;
    *)
        echo "unsupported DeepSeek sidecar lifecycle: $LIFECYCLE" >&2
        usage >&2
        exit 2
        ;;
esac

for path in "$CAP_ENV" "$COMMON_EPOCH0_ROOT/reuse.env" \
            "$COMMON_EPOCH0_ROOT/common_epoch0_metadata.env" \
            "$RUNTIME_PROFILE" "$TARGET" "$VERIFIER"; do
    if [[ ! -f "$path" ]]; then
        echo "missing DeepSeek sidecar smoke input: $path" >&2
        exit 2
    fi
done
if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]]; then
    echo "common DeepSeek epoch0 is incomplete: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi
if [[ ! -f "$PRIMARY_MODEL/config.json" || ! -d "$PRIMARY_DISTCP" ]]; then
    echo "missing DeepSeek-V2-Lite HF or Megatron assets" >&2
    exit 2
fi
if [[ ! -f "$SIDECAR_MODEL/config.json" ]]; then
    echo "missing real Qwen2.5-1.5B sidecar model: $SIDECAR_MODEL" >&2
    echo "prepare it with $SCRIPT_DIR/prepare_qwen2_5_1_5b_sidecar_assets.sh" >&2
    exit 2
fi
if ! find "$SIDECAR_MODEL" -maxdepth 1 -type f \
        \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit \
        | grep -q .; then
    echo "sidecar model has no local weight file: $SIDECAR_MODEL" >&2
    exit 2
fi
if [[ ! -e "$SIDECAR_PROMPTS" ]]; then
    echo "missing sidecar prompts: $SIDECAR_PROMPTS" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "$CAP_ENV"
# shellcheck disable=SC1090
source "$COMMON_EPOCH0_ROOT/reuse.env"
# shellcheck disable=SC1090
source "$COMMON_EPOCH0_ROOT/common_epoch0_metadata.env"
# shellcheck disable=SC1090
source "$RUNTIME_PROFILE"

require_value() {
    local name=$1
    local expected=$2
    if [[ "${!name:-}" != "$expected" ]]; then
        echo "DeepSeek sidecar smoke mismatch for $name" >&2
        echo "recorded=${!name:-<missing>} expected=$expected" >&2
        exit 2
    fi
}

require_path() {
    local name=$1
    local expected=$2
    if [[ -z "${!name:-}" \
          || "$(realpath -m "${!name}")" != "$(realpath -m "$expected")" ]]; then
        echo "DeepSeek sidecar smoke path mismatch for $name" >&2
        echo "recorded=${!name:-<missing>} expected=$expected" >&2
        exit 2
    fi
}

require_cap() {
    local name=$1
    local value=${!name:-}
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]] || (( value % 128 != 0 )); then
        echo "$name must be a VERIFIED positive multiple of 128" >&2
        exit 2
    fi
}

require_nonnegative_block_tokens() {
    local name=$1
    local value=${!name:-}
    if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value % 128 != 0 )); then
        echo "$name must be a VERIFIED nonnegative multiple of 128" >&2
        exit 2
    fi
}

verified_name=${CAP_PREFIX}_KV_CAPS_VERIFIED
require_value "$verified_name" 1
require_value VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY "$SHRINK_POLICY"
require_value VLLM_ASCEND_SHRINK_AWARE_STAGES "$SHRINK_STAGES"
require_value VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE "$MIN_COMPUTE_GROUP_SIZE"
require_value VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS "$FINAL_RANKS"
require_value DEEPSEEK_KV_CAP_TARGET_RATIO 1.0
require_value DEEPSEEK_KV_CAP_BLOCK_SIZE 128
require_value DEEPSEEK_KV_CAP_ROLLOUT_N 16
require_value DEEPSEEK_KV_CAP_MAX_PROMPT_LENGTH 1024
require_value DEEPSEEK_KV_CAP_MAX_RESPONSE_LENGTH 16384
require_value DEEPSEEK_KV_CAP_MAX_NUM_BATCHED_TOKENS 17408
require_value DEEPSEEK_KV_CAP_MAX_NUM_SEQS 32
require_value DEEPSEEK_KV_CAP_GPU_MEMORY_UTILIZATION 0.9
require_value DEEPSEEK_KV_CAP_ENFORCE_EAGER True
require_path DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT "$COMMON_EPOCH0_ROOT"
require_path COMMON_EPOCH0_MODEL_PATH "$PRIMARY_MODEL"
require_path COMMON_EPOCH0_DISTCP_PATH "$PRIMARY_DISTCP"
require_value COMMON_EPOCH0_MODEL_REVISION 85864749cd611b4353ce1decdb286193298f64c7
require_value COMMON_EPOCH0_EXECUTION_PROFILE_USED deepseek-v2-lite_tq1_a2aFalse_sharedFalse_deallocFalse_recomputeuniformx1_hccl800
require_value COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED 32
require_value COMMON_EPOCH0_ROLLOUT_N_USED 16
require_value COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED 1024
require_value COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED 16384
require_value COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED 17408
require_value COMMON_EPOCH0_MAX_NUM_SEQS_USED 32
require_value COMMON_EPOCH0_GPU_MEMORY_UTILIZATION_USED 0.9
require_value COMMON_EPOCH0_KV_BLOCK_SIZE_USED 128
if [[ -z "${DEEPSEEK_KV_CAP_PROBE_HISTORY_ROOT:-}" ]]; then
    echo "VERIFIED DeepSeek caps do not name a positive release history" >&2
    exit 2
fi
TRIGGER_ROOT=$(realpath -m "$DEEPSEEK_KV_CAP_PROBE_HISTORY_ROOT")
python3 "$SCRIPT_DIR/tools/prepare_deepseek_kv_probe_trigger.py" verify \
    --output-root "$TRIGGER_ROOT"

for floor in "${CAP_FLOORS[@]}"; do
    admission_name=${CAP_PREFIX}_KV_ADMISSION_FLOOR${floor}
    physical_name=${CAP_PREFIX}_KV_PHYSICAL_FLOOR${floor}
    require_cap "$admission_name"
    require_cap "$physical_name"
    if (( ${!admission_name} >= ${!physical_name} )); then
        echo "floor${floor} admission cap must be below its physical cap" >&2
        exit 2
    fi
done
if (( IS_PLANNED )); then
    for floor in "${CAP_FLOORS[@]}"; do
        require_nonnegative_block_tokens "${CAP_PREFIX}_HEADROOM_FLOOR${floor}"
    done
    training_name=${CAP_PREFIX}_TRAINING_MIN_FREE_MIB
    training_min_free_mib=${!training_name:-0}
    if ! [[ "$training_min_free_mib" =~ ^[1-9][0-9]*$ ]]; then
        echo "$training_name must be measured and positive" >&2
        exit 2
    fi
    require_value DYNAMIC_REQUIRE_EXPLICIT_PLANNED_MEMORY_GUARD 1
    require_value VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING 1
    require_value VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD 1
    require_value VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT 1
    require_value VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB "$training_min_free_mib"
    require_value VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB_FLOOR "$training_min_free_mib"
else
    unset DYNAMIC_REQUIRE_EXPLICIT_PLANNED_MEMORY_GUARD
fi

profile_id_name=${CAP_PREFIX}_RUNTIME_PROFILE_ID
profile_files_name=${CAP_PREFIX}_RUNTIME_PROFILE_FILES
recorded_profile_name=${CAP_PREFIX}_RUNTIME_PROFILE
recorded_profile_sha_name=${CAP_PREFIX}_RUNTIME_PROFILE_SHA256
profile_files=${!profile_files_name:-}
if [[ -z "$profile_files" ]]; then
    echo "$LIFECYCLE_LABEL runtime profile closure is empty" >&2
    exit 2
fi
profile_hash_args=()
IFS=, read -r -a runtime_profile_file_array <<< "$profile_files"
for runtime_profile_file in "${runtime_profile_file_array[@]}"; do
    profile_hash_args+=(--profile "$runtime_profile_file")
done
profile_sha256=$(python3 "$SCRIPT_DIR/tools/hash_deepseek_runtime_profile.py" \
    --root "$SCRIPT_DIR" "${profile_hash_args[@]}")
execution_sha256=$(python3 "$SCRIPT_DIR/tools/hash_deepseek_execution_code.py" --root "$SCRIPT_DIR")
require_value "$recorded_profile_name" "${!profile_id_name}"
require_value "$recorded_profile_sha_name" "$profile_sha256"
require_value DEEPSEEK_EXECUTION_CODE_SHA256 "$execution_sha256"
require_value DEEPSEEK_KV_CAP_MODEL_REVISION "$COMMON_EPOCH0_MODEL_REVISION"
require_value DEEPSEEK_KV_CAP_EXECUTION_PROFILE "$COMMON_EPOCH0_EXECUTION_PROFILE_USED"

if [[ ! -d "${DYNAMIC_INITIAL_RESUME_CKPT:-}/actor" \
      || ! -f "${DYNAMIC_INITIAL_RESUME_CKPT:-}/.PRESERVE_COMMON_EPOCH0" ]]; then
    echo "common DeepSeek epoch0 checkpoint is incomplete" >&2
    exit 2
fi
if [[ "${ALLOW_INFEASIBLE_PLAN:-0}" != 0 ]]; then
    echo "ALLOW_INFEASIBLE_PLAN is forbidden for the DeepSeek sidecar smoke" >&2
    exit 2
fi
unset ALLOW_INFEASIBLE_PLAN

RUN_TAG=${DEEPSEEK_SIDECAR_SMOKE_TAG:-"$(date -u +%Y%m%dT%H%M%SZ)"}
RUN_ROOT=${DEEPSEEK_SIDECAR_SMOKE_OUTPUT_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/sidecar_smoke_${RUN_SUFFIX}_${execution_sha256:0:12}_${RUN_TAG}}
RUN_ROOT=$(realpath -m "$RUN_ROOT")
RUN_PARENT=$(dirname "$RUN_ROOT")
RUN_NAME=$(basename "$RUN_ROOT")
EPOCH_DIR=$RUN_ROOT/epoch_001_mode1_${SHRINK_POLICY}
SUMMARY_PATH=$RUN_ROOT/SIDECAR_SMOKE_SUMMARY.json
VERIFY_COMMAND=(python3 "$VERIFIER" --run-root "$RUN_ROOT" --cap-env "$CAP_ENV" \
    --lifecycle "$LIFECYCLE" --expected-model-path "$SIDECAR_MODEL" \
    --summary "$SUMMARY_PATH")
EXECUTE_COMMAND=(env DEEPSEEK_KV_CAP_ENV="$CAP_ENV" COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    MODEL_PATH="$PRIMARY_MODEL" DISTCP_PATH="$PRIMARY_DISTCP" \
    VERL_SIDECAR_MODEL_PATH="$SIDECAR_MODEL" VERL_SIDECAR_PROMPTS_FILE="$SIDECAR_PROMPTS" \
    DEEPSEEK_SIDECAR_SMOKE_OUTPUT_ROOT="$RUN_ROOT" \
    "$SCRIPT_DIR/run_deepseek_v2_lite_sidecar_smoke.sh" \
    --lifecycle "$LIFECYCLE" --execute)

printf '[DeepSeek sidecar smoke] lifecycle=%s\n' "$LIFECYCLE"
printf '[DeepSeek sidecar smoke] run_root=%s\n' "$RUN_ROOT"
printf '[DeepSeek sidecar smoke] epoch_dir=%s\n' "$EPOCH_DIR"
printf '[DeepSeek sidecar smoke] common_epoch0_checkpoint=%s\n' "$DYNAMIC_INITIAL_RESUME_CKPT"
printf '[DeepSeek sidecar smoke] planning_history=%s\n' "$TRIGGER_ROOT"
printf '[DeepSeek sidecar smoke] verification_summary=%s\n' "$SUMMARY_PATH"
printf '[DeepSeek sidecar smoke] launch_command='
printf ' %q' "${EXECUTE_COMMAND[@]}"
printf '\n[DeepSeek sidecar smoke] verify_command='
printf ' %q' "${VERIFY_COMMAND[@]}"
printf '\n'
if [[ "$RUN_MODE" == print ]]; then
    exit 0
fi

if [[ -e "$RUN_ROOT" ]]; then
    echo "refusing to overwrite DeepSeek sidecar smoke root: $RUN_ROOT" >&2
    exit 2
fi
if command -v npu-smi >/dev/null 2>&1; then
    npu_processes=$(npu-smi info | awk '
        /Process id/ { in_process_table=1; next }
        in_process_table && /^\|/ && !/No running processes found/ { print }
    ')
    if [[ -n "$npu_processes" ]]; then
        echo "DeepSeek sidecar smoke requires exclusive idle NPUs" >&2
        printf '%s\n' "$npu_processes" >&2
        exit 2
    fi
fi
LOCK_DIR=/data/adafloor_shared_state/deepseek_v2_lite/.sidecar_smoke.lock
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "another DeepSeek sidecar smoke owns $LOCK_DIR" >&2
    exit 2
fi
cleanup_lock() {
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

mkdir -p "$RUN_ROOT"
printf '%s\n' \
    "INCOMPLETE DeepSeek-V2-Lite $LIFECYCLE_LABEL real sidecar smoke" \
    "lifecycle=$LIFECYCLE" \
    "cap_env=$CAP_ENV" \
    "execution_code_sha256=$execution_sha256" \
    "common_epoch0=$COMMON_EPOCH0_ROOT" \
    "sidecar_model=$SIDECAR_MODEL" > "$RUN_ROOT/INCOMPLETE"

export REPO_ROOT=$SCRIPT_DIR
export PATCH_TREE=$SCRIPT_DIR
export MODEL_PATH=$PRIMARY_MODEL
export MODEL_REVISION=85864749cd611b4353ce1decdb286193298f64c7
export DISTCP_PATH=$PRIMARY_DISTCP
export LOCAL_TEST_LAUNCHER=$SCRIPT_DIR/internal/wj_train_grpo_deepseek_v2_lite_16die_adafloor.sh
export BASELINE_LAUNCHER=$LOCAL_TEST_LAUNCHER
export DYNAMIC_MODE1_CHILD_SCRIPT=$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh
export CHECKPOINT_MODEL_DIR_NAME=deepseek_v2_lite
export TRAIN_LOG_PREFIX=deepseek-v2-lite-sidecar-smoke
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

export DYNAMIC_OUTPUT_ROOT=$RUN_PARENT
export DYNAMIC_RUN_NAME=$RUN_NAME
export DYNAMIC_SKIP_MODE0_PROBE=1
export DYNAMIC_INITIAL_BASELINE_DIR=$TRIGGER_ROOT
export DYNAMIC_START_EPOCH=1
export DYNAMIC_TOTAL_EPOCHS=2
export DYNAMIC_PLAN_STEPS=1
export DYNAMIC_TRAIN_STEPS=1
export DYNAMIC_DATASET_FRACTION=0.0009
export DYNAMIC_LENGTH_EMA_DECAY=0.3
export DYNAMIC_ENABLE_CKPT_CHAIN=0
export DYNAMIC_BUILD_OFFLINE_PLANNING_HISTORY=0
export DYNAMIC_ENABLE_THRESHOLD_CONTROL=0
export DYNAMIC_SHRINK_POLICY=$SHRINK_POLICY
export DYNAMIC_FORCE_SELECTED_FLOORS=$TARGET_FLOOR
unset DYNAMIC_FORCE_SELECTED_FLOOR FORCE_SELECTED_FLOOR FORCE_SELECTED_FLOORS
export DYNAMIC_SHORT_STEP_CAP_ENABLE=1
export DYNAMIC_SHORT_STEP_EXIT_THRESHOLD=4096
export DYNAMIC_SHORT_STEP_CAP_TOKENS=4096
export DYNAMIC_SHORT_STEP_CAP_FLOORS=$SHORT_STEP_CAP_FLOORS
export DYNAMIC_FULL_MAX_PROMPT_LENGTH=1024
export DYNAMIC_FULL_MAX_RESPONSE_LENGTH=16384
export DYNAMIC_FULL_MAX_RESPONSE_LEN=16384
export DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS=17408
export DYNAMIC_TAIL_GUARD_RATIO_QUANTILE=0.95
export DYNAMIC_TAIL_GUARD_RATIO_WINDOW=3
export DYNAMIC_TAIL_GUARD_DEFAULT_RATIO=1.20
export DYNAMIC_TAIL_GUARD_MIN_CAP=4096
export DYNAMIC_TAIL_GUARD_ROUND_TO=512
export DYNAMIC_DISABLE_TAIL_GUARD=0
export DYNAMIC_EXPECT_NO_RESPONSE_CAPS=0

export TRAIN_BATCH_SIZE=32
export ROLLOUT_N=16
export ROLLOUT_MAX_NUM_SEQS=32
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.9
export ROLLOUT_ENFORCE_EAGER=True
export ACTIVE_PEAK_SAFETY_FACTOR=1.16
export VLLM_KV_BLOCK_SIZE=128
export SAVE_CKPT_ENABLE=0
export TRAINER_SAVE_FREQ=-1
export BASELINE_ALLOW_INFEASIBLE_PLAN=0

export MIN_ADAPTIVE_FLOOR=$TARGET_FLOOR
export VLLM_ASCEND_SHRINK_AWARE_STAGES=$SHRINK_STAGES
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=$MIN_COMPUTE_GROUP_SIZE
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=$SHRINK_POLICY
export VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=8,9,10,11,12,13,14,15
export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=$FINAL_RANKS
export RANK_MATCHING_POLICY=release_area
export VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_FAIL_ON_UNMET_TARGET=1
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_MIN_TARGET_RATIO=1.0
floor_cap_entries=()
physical_cap_entries=()
for floor in "${CAP_FLOORS[@]}"; do
    admission_name=${CAP_PREFIX}_KV_ADMISSION_FLOOR${floor}
    physical_name=${CAP_PREFIX}_KV_PHYSICAL_FLOOR${floor}
    runtime_cap_name=VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR${floor}
    export "$runtime_cap_name=${!physical_name}"
    if (( IS_PLANNED )); then
        headroom_name=${CAP_PREFIX}_HEADROOM_FLOOR${floor}
        runtime_headroom_name=VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR${floor}
        export "$runtime_headroom_name=${!headroom_name}"
    fi
    floor_cap_entries+=("$floor:${!admission_name}")
    physical_cap_entries+=("$floor:${!physical_name}")
done
IFS=,
export FLOOR_KV_CAPS="${floor_cap_entries[*]}"
export PHYSICAL_FLOOR_KV_CAPS="${physical_cap_entries[*]}"
unset IFS
floor16_physical_name=${CAP_PREFIX}_KV_PHYSICAL_FLOOR16
floor16_admission_name=${CAP_PREFIX}_KV_ADMISSION_FLOOR16
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=${!floor16_physical_name}
export MAX_RANK_PEAK_TOKENS=${!floor16_admission_name}
if (( IS_PLANNED )); then
    target_headroom_name=${CAP_PREFIX}_HEADROOM_FLOOR${TARGET_FLOOR}
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS=${!target_headroom_name}
    training_name=${CAP_PREFIX}_TRAINING_MIN_FREE_MIB
    export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB=${!training_name}
    export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB_FLOOR=${!training_name}
fi

# Override the capacity profile's no-sidecar calibration setting only after
# all lifecycle variables have been imported from that exact profile.
export VERL_SIDECAR_ENABLE=1
export VERL_SIDECAR_MULTI_STAGE=0
export VERL_SIDECAR_MODEL_PATH=$SIDECAR_MODEL
export VERL_SIDECAR_PROMPTS_FILE=$SIDECAR_PROMPTS
export VERL_SIDECAR_DATA_SPLIT=train
export VERL_SIDECAR_USE_SHORT_DATA=0
export VERL_SIDECAR_PARALLEL_MODE=dp
export VERL_SIDECAR_TENSOR_PARALLEL_SIZE=1
export VERL_SIDECAR_DATA_PARALLEL_SIZE=1
export VERL_SIDECAR_REPLICA_COUNT=1
export VERL_SIDECAR_ENABLE_EXPERT_PARALLEL=0
export VERL_SIDECAR_EXPECTED_ACTIVE_RANKS=$TARGET_FLOOR
export VERL_SIDECAR_WORLD_SIZE=16
export VERL_SIDECAR_START_TRIGGER=shrink_done
export VERL_SIDECAR_START_ONCE=1
export VERL_SIDECAR_WATCH_POLL_INTERVAL=0.1
export VERL_SIDECAR_GRACEFUL_KILL_SECONDS=10
export VERL_SIDECAR_GPU_MEMORY_UTILIZATION=0.60
export VERL_SIDECAR_MAX_MODEL_LEN=4096
export VERL_SIDECAR_MAX_TOKENS=32
export VERL_SIDECAR_MAX_NUM_SEQS=1
export VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=4096
export VERL_SIDECAR_MAX_PROMPTS=1
export VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE=1
export VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA=1
export VERL_SIDECAR_GENERATE_CHUNK_SIZE=1
export VERL_SIDECAR_N=1
export VERL_SIDECAR_TEMPERATURE=0.0
export VERL_SIDECAR_TOP_P=1.0
export VERL_SIDECAR_REPEAT_UNTIL_KILLED=0
export VERL_SIDECAR_MAX_ITERATIONS=1
export VERL_SIDECAR_STREAM_CHECKPOINT=1
export VERL_SIDECAR_RESET_OUTPUT_ON_START=1
export VERL_SIDECAR_MAX_SECONDS=180
export VERL_SIDECAR_LOG_DIR=$EPOCH_DIR/sidecar
export VERL_SIDECAR_LEASE_LOG=$EPOCH_DIR/sidecar/lease.log
export VERL_SIDECAR_LOG_FILE=$EPOCH_DIR/sidecar/infer.log
export VERL_SIDECAR_OUTPUT_FILE=$EPOCH_DIR/sidecar/outputs.jsonl
export VERL_SIDECAR_STATE_DIR=$EPOCH_DIR/sidecar/state
export VERL_SIDECAR_MONITOR_LOG=$EPOCH_DIR/sidecar/monitor.log
export VERL_SIDECAR_SCRIPT=$SCRIPT_DIR/internal/run_elastic_sidecar_infer.sh
unset VERL_SIDECAR_NPU_DEVICES

"$TARGET" \
    trainer.resume_mode=resume_path \
    "trainer.resume_from_path=$DYNAMIC_INITIAL_RESUME_CKPT"

"${VERIFY_COMMAND[@]}"
rm -f "$RUN_ROOT/INCOMPLETE"
printf '%s\n' \
    "COMPLETE DeepSeek-V2-Lite $LIFECYCLE_LABEL real sidecar smoke" \
    "lifecycle=$LIFECYCLE" \
    "summary=$SUMMARY_PATH" > "$RUN_ROOT/COMPLETE"
echo "[DeepSeek sidecar smoke] COMPLETE summary=$SUMMARY_PATH"
