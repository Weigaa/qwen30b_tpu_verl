#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Build the single shared epoch0 state used by paper comparisons. This run is
# intentionally mode0/no-shrink and does not apply any response-length guard.
# Keep both its rollout history and global_step_5 checkpoint permanently.

OUTPUT_ROOT="${COMMON_EPOCH0_OUTPUT_ROOT:-$SCRIPT_DIR}"
RUN_NAME="${COMMON_EPOCH0_RUN_NAME:-common_epoch0_probe_gpu09_kv380800_permanent}"
RUN_ROOT="$OUTPUT_ROOT/$RUN_NAME"
EPOCH_DIR="$RUN_ROOT/epoch_000_mode0_probe"
CHECKPOINT_MODEL_DIR_NAME="${CHECKPOINT_MODEL_DIR_NAME:-qwen3moe_for_eagle3}"
if [[ ! "$CHECKPOINT_MODEL_DIR_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid CHECKPOINT_MODEL_DIR_NAME=$CHECKPOINT_MODEL_DIR_NAME" >&2
    exit 2
fi
METADATA_ENV="$RUN_ROOT/common_epoch0_metadata.env"
RUN_CONTRACT_ENV="$RUN_ROOT/common_epoch0_run_contract.env"
FINALIZE_EXISTING="${COMMON_EPOCH0_FINALIZE_EXISTING:-0}"
ORIGINAL_EXECUTION_CODE_SHA256="${COMMON_EPOCH0_ORIGINAL_EXECUTION_CODE_SHA256:-unspecified}"

MODEL_PATH="${MODEL_PATH:-/data/Qwen3-30B-A3B}"
MODEL_REVISION="${MODEL_REVISION:-unknown}"
DISTCP_PATH="${DISTCP_PATH:-/data/Qwen3-30B-A3B_megatron}"
TRAIN_FILE="${TRAIN_FILE:-/data/deepscaler/train.parquet}"
TEST_FILE="${TEST_FILE:-/data/deepscaler/test.parquet}"

TRAIN_STEPS="${COMMON_EPOCH0_TRAIN_STEPS:-5}"
DATASET_FRACTION="${COMMON_EPOCH0_DATASET_FRACTION:-0.005}"
TRAIN_BATCH_SIZE="${COMMON_EPOCH0_TRAIN_BATCH_SIZE:-32}"
ROLLOUT_N="${COMMON_EPOCH0_ROLLOUT_N:-16}"
MAX_NUM_SEQS="${COMMON_EPOCH0_MAX_NUM_SEQS:-32}"
MAX_PROMPT_LENGTH="${COMMON_EPOCH0_MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${COMMON_EPOCH0_MAX_RESPONSE_LENGTH:-16384}"
MAX_NUM_BATCHED_TOKENS="${COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS:-17408}"
GPU_MEMORY_UTILIZATION="${COMMON_EPOCH0_GPU_MEMORY_UTILIZATION:-0.9}"
PROMPTS_TOTAL="${COMMON_EPOCH0_PROMPTS_TOTAL:-}"
EXPECTED_RESPONSES_PER_STEP="${COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP:-}"
PREEMPTION_POLICY="${COMMON_EPOCH0_PREEMPTION_POLICY:-forbid}"
WORKLOAD_PROFILE_ID="${COMMON_EPOCH0_WORKLOAD_PROFILE_ID:-unspecified}"
WORKLOAD_PROFILE_SHA256="${COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256:-unspecified}"
KV_TOKENS_PER_RANK="${COMMON_EPOCH0_KV_TOKENS_PER_RANK:-380800}"
KV_BLOCK_SIZE="${VLLM_KV_BLOCK_SIZE:-128}"
MIN_FREE_GB="${COMMON_EPOCH0_MIN_FREE_GB:-80}"
WORLD_SIZE=16
KV_BLOCKS=""
KV_OVERRIDE_ARGS=()

if [[ "$FINALIZE_EXISTING" != 0 && "$FINALIZE_EXISTING" != 1 ]]; then
    echo "COMMON_EPOCH0_FINALIZE_EXISTING must be 0 or 1" >&2
    exit 2
fi
if [[ "$ORIGINAL_EXECUTION_CODE_SHA256" == unspecified ]]; then
    ORIGINAL_EXECUTION_CODE_SHA256=$(python3 \
        "$SCRIPT_DIR/tools/hash_deepseek_execution_code.py" --root "$SCRIPT_DIR")
fi
if [[ ! "$ORIGINAL_EXECUTION_CODE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "COMMON_EPOCH0_ORIGINAL_EXECUTION_CODE_SHA256 must be a SHA256" >&2
    exit 2
fi

for positive_name in \
    TRAIN_STEPS TRAIN_BATCH_SIZE ROLLOUT_N MAX_NUM_SEQS \
    MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_NUM_BATCHED_TOKENS; do
    positive_value=${!positive_name}
    if ! [[ "$positive_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$positive_name must be a positive integer, got $positive_value" >&2
        exit 2
    fi
done
if (( TRAIN_BATCH_SIZE % WORLD_SIZE != 0 )); then
    echo "TRAIN_BATCH_SIZE must be divisible by world size $WORLD_SIZE" >&2
    exit 2
fi
if [[ -z "$PROMPTS_TOTAL" ]]; then
    PROMPTS_TOTAL=$((TRAIN_STEPS * TRAIN_BATCH_SIZE))
fi
if [[ -z "$EXPECTED_RESPONSES_PER_STEP" ]]; then
    EXPECTED_RESPONSES_PER_STEP=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
fi
if ! [[ "$PROMPTS_TOTAL" =~ ^[1-9][0-9]*$ ]] \
   || (( PROMPTS_TOTAL != TRAIN_STEPS * TRAIN_BATCH_SIZE )); then
    echo "COMMON_EPOCH0_PROMPTS_TOTAL must equal steps times batch size" >&2
    exit 2
fi
if ! [[ "$EXPECTED_RESPONSES_PER_STEP" =~ ^[1-9][0-9]*$ ]] \
   || (( EXPECTED_RESPONSES_PER_STEP != TRAIN_BATCH_SIZE * ROLLOUT_N )); then
    echo "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP must equal batch size times rollout n" >&2
    exit 2
fi
case "$PREEMPTION_POLICY" in
    forbid|record) ;;
    *)
        echo "COMMON_EPOCH0_PREEMPTION_POLICY must be forbid or record" >&2
        exit 2
        ;;
esac
if [[ "$WORKLOAD_PROFILE_ID" == unspecified ]]; then
    if [[ "$WORKLOAD_PROFILE_SHA256" != unspecified ]]; then
        echo "workload profile SHA requires a workload profile ID" >&2
        exit 2
    fi
elif [[ ! "$WORKLOAD_PROFILE_ID" =~ ^[A-Za-z0-9._-]+$ ]] \
     || [[ ! "$WORKLOAD_PROFILE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid workload profile identity or SHA256" >&2
    exit 2
fi
CHECKPOINT_PATH="$EPOCH_DIR/checkpoints/$CHECKPOINT_MODEL_DIR_NAME/global_step_$TRAIN_STEPS"
if [[ "$KV_TOKENS_PER_RANK" == "auto" ]]; then
    if (( KV_BLOCK_SIZE <= 0 )); then
        echo "KV block size must be positive" >&2
        exit 2
    fi
    KV_BLOCKS="auto"
else
    if ! [[ "$KV_TOKENS_PER_RANK" =~ ^[1-9][0-9]*$ ]] \
       || (( KV_BLOCK_SIZE <= 0 || KV_TOKENS_PER_RANK % KV_BLOCK_SIZE != 0 )); then
        echo "KV token capacity must be 'auto' or a positive multiple of block size" >&2
        exit 2
    fi
    KV_BLOCKS=$((KV_TOKENS_PER_RANK / KV_BLOCK_SIZE))
    KV_OVERRIDE_ARGS=(
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=$KV_BLOCKS"
    )
fi

for required_path in "$MODEL_PATH" "$DISTCP_PATH" "$TRAIN_FILE" "$TEST_FILE"; do
    if [[ ! -e "$required_path" ]]; then
        echo "missing required input: $required_path" >&2
        exit 2
    fi
done

MODEL_PATH_REAL=$(realpath "$MODEL_PATH")
DISTCP_PATH_REAL=$(realpath "$DISTCP_PATH")
TRAIN_FILE_REAL=$(realpath "$TRAIN_FILE")
TEST_FILE_REAL=$(realpath "$TEST_FILE")
TRAIN_FILE_SHA256=$(sha256sum "$TRAIN_FILE_REAL")
TRAIN_FILE_SHA256=${TRAIN_FILE_SHA256%% *}
TEST_FILE_SHA256=$(sha256sum "$TEST_FILE_REAL")
TEST_FILE_SHA256=${TEST_FILE_SHA256%% *}
RUN_CONTRACT_SHA256=""

write_expected_run_contract() {
    local output_path=$1
    {
        printf 'export COMMON_EPOCH0_RUN_CONTRACT_SCHEMA_VERSION=1\n'
        printf 'export COMMON_EPOCH0_RUN_CONTRACT_MODEL_PATH=%q\n' "$MODEL_PATH_REAL"
        printf 'export COMMON_EPOCH0_RUN_CONTRACT_MODEL_REVISION=%q\n' "$MODEL_REVISION"
        printf 'export COMMON_EPOCH0_RUN_CONTRACT_DISTCP_PATH=%q\n' "$DISTCP_PATH_REAL"
        printf 'export COMMON_EPOCH0_RUN_CONTRACT_WORKLOAD_PROFILE_ID=%q\n' "$WORKLOAD_PROFILE_ID"
        printf 'export COMMON_EPOCH0_RUN_CONTRACT_WORKLOAD_PROFILE_SHA256=%q\n' "$WORKLOAD_PROFILE_SHA256"
        printf 'export COMMON_EPOCH0_RUN_CONTRACT_EXECUTION_CODE_SHA256=%q\n' "$ORIGINAL_EXECUTION_CODE_SHA256"
        printf 'export COMMON_EPOCH0_RUN_CONTRACT_TRAIN_FILE=%q\n' "$TRAIN_FILE_REAL"
        printf 'export COMMON_EPOCH0_RUN_CONTRACT_TRAIN_FILE_SHA256=%q\n' "$TRAIN_FILE_SHA256"
        printf 'export COMMON_EPOCH0_RUN_CONTRACT_TEST_FILE=%q\n' "$TEST_FILE_REAL"
        printf 'export COMMON_EPOCH0_RUN_CONTRACT_TEST_FILE_SHA256=%q\n' "$TEST_FILE_SHA256"
    } > "$output_path"
}

create_run_contract() {
    local contract_tmp
    contract_tmp=$(mktemp "$RUN_ROOT/.common_epoch0_run_contract.env.tmp.XXXXXX")
    write_expected_run_contract "$contract_tmp"
    chmod 0444 "$contract_tmp"
    mv -T "$contract_tmp" "$RUN_CONTRACT_ENV"
    RUN_CONTRACT_SHA256=$(sha256sum "$RUN_CONTRACT_ENV")
    RUN_CONTRACT_SHA256=${RUN_CONTRACT_SHA256%% *}
}

validate_existing_run_contract() {
    if [[ ! -f "$RUN_CONTRACT_ENV" ]]; then
        echo "existing common epoch0 lacks immutable run contract: $RUN_CONTRACT_ENV" >&2
        exit 2
    fi
    local expected_tmp
    expected_tmp=$(mktemp "${TMPDIR:-/tmp}/common_epoch0_run_contract.expected.XXXXXX")
    write_expected_run_contract "$expected_tmp"
    if ! cmp -s "$expected_tmp" "$RUN_CONTRACT_ENV"; then
        rm -f "$expected_tmp"
        echo "existing common epoch0 run contract does not match current expected inputs" >&2
        echo "contract=$RUN_CONTRACT_ENV" >&2
        exit 2
    fi
    rm -f "$expected_tmp"
    local contract_mode
    contract_mode=$(stat -c '%a' "$RUN_CONTRACT_ENV")
    if (( (8#$contract_mode & 0222) != 0 )); then
        echo "common epoch0 run contract must be read-only: mode=$contract_mode" >&2
        exit 2
    fi
    RUN_CONTRACT_SHA256=$(sha256sum "$RUN_CONTRACT_ENV")
    RUN_CONTRACT_SHA256=${RUN_CONTRACT_SHA256%% *}
}

if [[ -e "$RUN_ROOT" ]]; then
    if [[ "$FINALIZE_EXISTING" != 1 ]]; then
        echo "refusing to overwrite permanent common epoch0: $RUN_ROOT" >&2
        echo "Remove an incomplete directory explicitly or set COMMON_EPOCH0_RUN_NAME to a new name." >&2
        exit 2
    fi
    if [[ ! -d "$EPOCH_DIR" || ! -f "$RUN_ROOT/INCOMPLETE" ]]; then
        echo "existing common epoch0 is not an explicitly incomplete run" >&2
        exit 2
    fi
    if [[ -e "$RUN_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]]; then
        echo "refusing to finalize an already committed common epoch0" >&2
        exit 2
    fi
    validate_existing_run_contract
else
    if [[ "$FINALIZE_EXISTING" == 1 ]]; then
        echo "COMMON_EPOCH0_FINALIZE_EXISTING requires an existing run" >&2
        exit 2
    fi
    mkdir -p "$OUTPUT_ROOT"
    available_kb=$(df -Pk "$OUTPUT_ROOT" | awk 'NR == 2 {print $4}')
    required_kb=$((MIN_FREE_GB * 1024 * 1024))
    if (( available_kb < required_kb )); then
        echo "insufficient free space for permanent epoch0 checkpoint" >&2
        echo "available_kb=$available_kb required_kb=$required_kb output_root=$OUTPUT_ROOT" >&2
        exit 2
    fi

    mkdir -p "$EPOCH_DIR"
    printf '%s\n' \
        "INCOMPLETE common epoch0 run" \
        "Do not reuse this directory unless the permanent checkpoint marker exists." \
        > "$RUN_ROOT/INCOMPLETE"
    create_run_contract
fi

echo "[common epoch0] output=$EPOCH_DIR"
echo "[common epoch0] model=$MODEL_PATH train=$TRAIN_FILE dataset_fraction=$DATASET_FRACTION"
echo "[common epoch0] workload_profile=$WORKLOAD_PROFILE_ID batch=$TRAIN_BATCH_SIZE rollout_n=$ROLLOUT_N max_num_seqs=$MAX_NUM_SEQS prompts=$PROMPTS_TOTAL expected_rows=$EXPECTED_RESPONSES_PER_STEP"
echo "[common epoch0] steps=$TRAIN_STEPS gpu_memory_utilization=$GPU_MEMORY_UTILIZATION kv_tokens_per_rank=$KV_TOKENS_PER_RANK blocks=$KV_BLOCKS preemption_policy=$PREEMPTION_POLICY"
echo "[common epoch0] tail_guard=disabled checkpoint=$CHECKPOINT_PATH"

if [[ "$FINALIZE_EXISTING" == 1 ]]; then
    echo "[common epoch0] validating and finalizing existing training output"
    run_rc=0
else
    set +e
    env \
        -u VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS \
        -u VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP \
        OUTPUT_ROOT="$OUTPUT_ROOT" \
        OUTPUT_SUBDIR="$RUN_NAME/epoch_000_mode0_probe" \
        RECORD_DIR="$EPOCH_DIR" \
        MODEL_PATH="$MODEL_PATH" \
        DISTCP_PATH="$DISTCP_PATH" \
        TRAIN_FILE="$TRAIN_FILE" \
        TEST_FILE="$TEST_FILE" \
        TRAINER_TOTAL_EPOCHS=1 \
        DATASET_FRACTION="$DATASET_FRACTION" \
        DATA_SHUFFLE=False \
        TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
        MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
        MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
        MAX_RESPONSE_LEN="$MAX_RESPONSE_LENGTH" \
        ROLLOUT_MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
        ROLLOUT_MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        ROLLOUT_N="$ROLLOUT_N" \
        ROLLOUT_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
        MODE0_SAVE_ROLLOUT_ARTIFACTS=1 \
        SAVE_CKPT_ENABLE=1 \
        TRAINER_SAVE_FREQ="$TRAIN_STEPS" \
        MAX_ACTOR_CKPT_TO_KEEP=1 \
        MAX_CRITIC_CKPT_TO_KEEP=1 \
        VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1 \
        "$SCRIPT_DIR/run_mode0_no_shrink_baseline.sh" \
            trainer.total_training_steps="$TRAIN_STEPS" \
            trainer.resume_mode=disable \
            "${KV_OVERRIDE_ARGS[@]}" \
            "$@"
    run_rc=$?
    set -e
fi

if (( run_rc != 0 )); then
    echo "common epoch0 failed with exit_code=$run_rc; incomplete output retained at $RUN_ROOT" >&2
    exit "$run_rc"
fi

latest_log=$(find "$EPOCH_DIR/logs" -maxdepth 1 -type f -name '*.txt' \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -z "$latest_log" || ! -f "$latest_log" ]]; then
    echo "common epoch0 log validation failed: no training log" >&2
    exit 3
fi

if [[ "$KV_TOKENS_PER_RANK" == auto ]]; then
    if ! measured_kv_tokens=$(python3 \
        "$SCRIPT_DIR/tools/extract_vllm_kv_capacity.py" \
        --log "$latest_log" \
        --world-size "$WORLD_SIZE" \
        --block-size "$KV_BLOCK_SIZE"); then
        echo "automatic common epoch0 did not report one valid KV capacity per rank" >&2
        exit 3
    fi
    printf '%s\n' "$measured_kv_tokens" > "$RUN_ROOT/MEASURED_GPU_KV_CACHE_TOKENS_PER_RANK"
    echo "[common epoch0] measured_gpu_kv_cache_tokens_per_rank=$measured_kv_tokens"
    effective_kv_tokens=$measured_kv_tokens
else
    effective_kv_tokens=$KV_TOKENS_PER_RANK
fi

if [[ ! -d "$EPOCH_DIR/rollout_data" || ! -d "$EPOCH_DIR/rollout_length" ]]; then
    echo "common epoch0 rollout artifact directories are missing" >&2
    exit 3
fi
rollout_count=$(find "$EPOCH_DIR/rollout_data" -maxdepth 1 -type f -name '*.jsonl' 2>/dev/null | wc -l)
length_count=$(find "$EPOCH_DIR/rollout_length" -maxdepth 1 -type f -name 'length_*.txt' 2>/dev/null | wc -l)
if (( rollout_count != TRAIN_STEPS || length_count != TRAIN_STEPS )); then
    echo "common epoch0 artifact validation failed: rollout=$rollout_count length=$length_count" >&2
    exit 3
fi
for (( step = 1; step <= TRAIN_STEPS; step++ )); do
    rollout_file="$EPOCH_DIR/rollout_data/$step.jsonl"
    length_file="$EPOCH_DIR/rollout_length/length_$step.txt"
    if [[ ! -f "$rollout_file" || ! -f "$length_file" ]]; then
        echo "common epoch0 missing step=$step artifact pair" >&2
        exit 3
    fi
    rollout_rows=$(wc -l < "$rollout_file")
    length_rows=$(wc -l < "$length_file")
    if (( rollout_rows != EXPECTED_RESPONSES_PER_STEP || length_rows != EXPECTED_RESPONSES_PER_STEP )); then
        echo "common epoch0 row validation failed at step=$step: rollout=$rollout_rows length=$length_rows" >&2
        exit 3
    fi
done

preemption_count=$(grep -ciE \
    'preempting request|request preempted' "$latest_log" || true)
python3 - "$latest_log" "$TRAIN_STEPS" "$PREEMPTION_POLICY" <<'PY'
import math
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
steps = int(sys.argv[2])
preemption_policy = sys.argv[3]
log = path.read_text(encoding="utf-8", errors="replace")
expected = list(range(1, steps + 1))
observed = [int(value) for value in re.findall(r"training/global_step:([0-9]+)", log)]
if observed != expected:
    raise SystemExit(f"global step metrics mismatch: {observed} != {expected}")
rollout_times = [float(value) for value in re.findall(r"rollout_output_time_s:\s*([0-9.eE+-]+)", log)]
if len(rollout_times) != steps or any(not math.isfinite(value) or value <= 0 for value in rollout_times):
    raise SystemExit(f"invalid rollout timings: {rollout_times}")
aborted = [float(value) for value in re.findall(r"response/aborted_ratio:([0-9.eE+-]+)", log)]
if len(aborted) != steps or any(value != 0.0 for value in aborted):
    raise SystemExit(f"invalid aborted ratios: {aborted}")
if (
    preemption_policy == "forbid"
    and re.search(r"preempting request|request preempted", log, flags=re.IGNORECASE)
):
    raise SystemExit("request preemption evidence found")
for marker in ("Training Progress: 100%", "After trainer.fit"):
    if marker not in log:
        raise SystemExit(f"missing log marker: {marker}")
oom = re.findall(
    r"NPU out of memory|Memory_Allocation_Failure|"
    r"Failed to allocate[^\r\n]*NPU memory|OutOfMemoryError|"
    r"ACL_ERROR_RT_MEMORY_ALLOCATION",
    log,
    flags=re.IGNORECASE,
)
if oom:
    raise SystemExit(f"NPU OOM evidence found: {oom[:5]}")
PY

checkpoint_root="$EPOCH_DIR/checkpoints/$CHECKPOINT_MODEL_DIR_NAME"
checkpoint_tracker="$checkpoint_root/latest_checkpointed_iteration.txt"
if [[ ! -f "$checkpoint_tracker" || $(tr -d '[:space:]' < "$checkpoint_tracker") != "$TRAIN_STEPS" ]]; then
    echo "common epoch0 checkpoint tracker is missing or is not global_step_$TRAIN_STEPS" >&2
    exit 4
fi
if [[ ! -d "$CHECKPOINT_PATH/actor/dist_ckpt" ]]; then
    echo "common epoch0 actor dist checkpoint is missing" >&2
    exit 4
fi
distcp_count=$(find "$CHECKPOINT_PATH/actor/dist_ckpt" -maxdepth 1 -type f -name '*.distcp' -size +0c | wc -l)
if (( distcp_count <= 0 )); then
    echo "common epoch0 global_step_$TRAIN_STEPS contains no nonempty actor distcp shard" >&2
    exit 4
fi

history_overwrite_args=()
if [[ "$FINALIZE_EXISTING" == 1 ]]; then
    history_overwrite_args=(--force)
fi
python3 "$SCRIPT_DIR/tools/build_offline_planning_history.py" \
    --baseline-dir "$EPOCH_DIR" \
    --steps "$TRAIN_STEPS" \
    --responses-per-prompt "$ROLLOUT_N" \
    "${history_overwrite_args[@]}"

commit_marker_tmp="$RUN_ROOT/.DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT.tmp.$$"
reuse_env_tmp="$RUN_ROOT/.reuse.env.tmp.$$"
metadata_env_tmp="$RUN_ROOT/.common_epoch0_metadata.env.tmp.$$"
printf '%s\n' \
    "PERMANENT COMMON EPOCH0 CHECKPOINT" \
    "Keep this directory for all paper baseline and AdaFloor reruns." \
    "Rollout history: $EPOCH_DIR" \
    "Resume checkpoint: $CHECKPOINT_PATH" \
    > "$commit_marker_tmp"
touch "$CHECKPOINT_PATH/.PRESERVE_COMMON_EPOCH0"

{
    printf 'export DYNAMIC_INITIAL_BASELINE_DIR=%q\n' "$EPOCH_DIR"
    printf 'export BASELINE_INITIAL_RESUME_CKPT=%q\n' "$CHECKPOINT_PATH"
    printf 'export DYNAMIC_INITIAL_RESUME_CKPT=%q\n' "$CHECKPOINT_PATH"
} > "$reuse_env_tmp"

{
    printf 'export COMMON_EPOCH0_MODEL_PATH=%q\n' "$MODEL_PATH_REAL"
    printf 'export COMMON_EPOCH0_MODEL_REVISION=%q\n' "$MODEL_REVISION"
    printf 'export COMMON_EPOCH0_DISTCP_PATH=%q\n' "$DISTCP_PATH_REAL"
    printf 'export COMMON_EPOCH0_CHECKPOINT_MODEL_DIR_NAME=%q\n' "$CHECKPOINT_MODEL_DIR_NAME"
    printf 'export COMMON_EPOCH0_KV_TOKENS_PER_RANK_USED=%q\n' "$KV_TOKENS_PER_RANK"
    printf 'export COMMON_EPOCH0_EFFECTIVE_KV_TOKENS_PER_RANK=%q\n' "$effective_kv_tokens"
    printf 'export COMMON_EPOCH0_ORIGINAL_EXECUTION_CODE_SHA256=%q\n' "$ORIGINAL_EXECUTION_CODE_SHA256"
    printf 'export COMMON_EPOCH0_RUN_CONTRACT_SHA256=%q\n' "$RUN_CONTRACT_SHA256"
    printf 'export COMMON_EPOCH0_TRAIN_STEPS_USED=%q\n' "$TRAIN_STEPS"
    printf 'export COMMON_EPOCH0_PROMPTS_TOTAL_USED=%q\n' "$PROMPTS_TOTAL"
    printf 'export COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED=%q\n' "$EXPECTED_RESPONSES_PER_STEP"
    printf 'export COMMON_EPOCH0_PREEMPTION_POLICY_USED=%q\n' "$PREEMPTION_POLICY"
    printf 'export COMMON_EPOCH0_PREEMPTION_COUNT=%q\n' "$preemption_count"
    printf 'export COMMON_EPOCH0_WORKLOAD_PROFILE_ID=%q\n' "$WORKLOAD_PROFILE_ID"
    printf 'export COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256=%q\n' "$WORKLOAD_PROFILE_SHA256"
    printf 'export COMMON_EPOCH0_EXECUTION_PROFILE_USED=%q\n' "${COMMON_EPOCH0_EXECUTION_PROFILE:-unspecified}"
    printf 'export COMMON_EPOCH0_TRAIN_FILE_USED=%q\n' "$TRAIN_FILE_REAL"
    printf 'export COMMON_EPOCH0_TRAIN_FILE_SHA256=%q\n' "$TRAIN_FILE_SHA256"
    printf 'export COMMON_EPOCH0_TEST_FILE_USED=%q\n' "$TEST_FILE_REAL"
    printf 'export COMMON_EPOCH0_TEST_FILE_SHA256=%q\n' "$TEST_FILE_SHA256"
    printf 'export COMMON_EPOCH0_DATASET_FRACTION_USED=%q\n' "$DATASET_FRACTION"
    printf 'export COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED=%q\n' "$TRAIN_BATCH_SIZE"
    printf 'export COMMON_EPOCH0_ROLLOUT_N_USED=%q\n' "$ROLLOUT_N"
    printf 'export COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED=%q\n' "$MAX_PROMPT_LENGTH"
    printf 'export COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED=%q\n' "$MAX_RESPONSE_LENGTH"
    printf 'export COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED=%q\n' "$MAX_NUM_BATCHED_TOKENS"
    printf 'export COMMON_EPOCH0_MAX_NUM_SEQS_USED=%q\n' "$MAX_NUM_SEQS"
    printf 'export COMMON_EPOCH0_GPU_MEMORY_UTILIZATION_USED=%q\n' "$GPU_MEMORY_UTILIZATION"
    printf 'export COMMON_EPOCH0_KV_BLOCK_SIZE_USED=%q\n' "$KV_BLOCK_SIZE"
} > "$metadata_env_tmp"

mv "$reuse_env_tmp" "$RUN_ROOT/reuse.env"
mv "$metadata_env_tmp" "$METADATA_ENV"
mv "$commit_marker_tmp" "$RUN_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT"
rm -f "$RUN_ROOT/INCOMPLETE"

echo "[common epoch0] complete"
echo "[common epoch0] rollout_history=$EPOCH_DIR"
echo "[common epoch0] checkpoint=$CHECKPOINT_PATH"
echo "[common epoch0] reuse with: source $RUN_ROOT/reuse.env"
echo "[common epoch0] metadata=$METADATA_ENV"
