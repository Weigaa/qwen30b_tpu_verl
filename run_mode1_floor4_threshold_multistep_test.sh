#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_mode1_floor4_threshold_multistep_test.sh [floor]
  bash run_mode1_floor4_threshold_multistep_test.sh --floor 4

Purpose:
  Run the known-good qwen3_true_mode5_a3cfdc2 mode=1 path for multiple
  threshold-controlled training steps. This is intentionally kept close to
  run_mode1_perf_clean_test.sh so it can be used as a reference against the
  shrinkaware_staged repo.

Useful overrides:
  BASELINE_TOTAL_TRAINING_STEPS=5
  MAX_RESPONSE_LENGTH=896
  VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896
  DATASET_FRACTION=0.005
  VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=288000
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
PATCH_TREE="${PATCH_TREE:-$REPO_ROOT}"
LAUNCHER="$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"

floor="${MODE1_FLOOR:-4}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--floor)
            [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; usage >&2; exit 2; }
            floor="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                floor="$1"
                shift
            else
                echo "unknown argument: $1" >&2
                usage >&2
                exit 2
            fi
            ;;
    esac
done

case "$floor" in
    1|2|4|8|16) ;;
    *)
        echo "unsupported mode=1 floor: $floor; expected one of 1,2,4,8,16" >&2
        exit 2
        ;;
esac

cd "$PATCH_TREE"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_root="${RUN_ROOT:-$REPO_ROOT/mode1_floor${floor}_threshold_multistep_runs/$stamp}"
mkdir -p "$run_root"
tee_log="$run_root/launcher.log"

# Match the stable mode=1 settings used by run_mode1_perf_clean_test.sh.
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="$floor"
export VLLM_ASCEND_CUSTOM_MODE1_DEBUG=0
export VLLM_ASCEND_CUSTOM_MODE1_TIMING_EVENTS=0
export VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG=0
export VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT="${VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT:-1}"
export VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT="${VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT:-0}"
export VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT="${VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT:-0}"
export VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS="${VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS:-8}"
export VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC="${VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC:-1}"

# Use the effective floor=4 KV cache size from pure-highkv-mode1-4.txt as the
# reference budget unless explicitly overridden. If this is still too tight for
# 5-step validation, try VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=288000.
if [[ "$floor" == "4" ]]; then
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS:-288000}"
fi

# Threshold/multistep knobs.
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-896}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export BASELINE_TOTAL_TRAINING_STEPS="${BASELINE_TOTAL_TRAINING_STEPS:-5}"
export DATASET_FRACTION="${DATASET_FRACTION:-0.005}"
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-256,512,640,768,896}"

# Keep diagnostics quiet enough for perf, but preserve memory/shrink evidence.
export PRINT_MEMORY="${PRINT_MEMORY:-1}"
export VLLM_ASCEND_FULL_REDUNDANCY_EXPERIMENT_LOG="${VLLM_ASCEND_FULL_REDUNDANCY_EXPERIMENT_LOG:-1}"
export VLLM_ASCEND_MODE3_TRANSFER_LOG=0
export VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG=0
export VLLM_ASCEND_MODE3_TIMING_LOG=0
export VLLM_ASCEND_MODE3_TIMING_SYNC=0
export VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS=0
export VLLM_ASCEND_BUCKET_OP_PROFILE=0
export VLLM_ASCEND_BUCKET_OP_PROFILE_BY_STAGE=0
export VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS=""
export VLLM_ASCEND_DUMMY_WASTE_TIMING=0
export VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=0
export VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE=0
export VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS=0

export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-47241}"
export MASTER_PORT="${MASTER_PORT:-26240}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-47241}"
export HOME="$run_root"
export CONFIG_DIR="$PATCH_TREE/verl/trainer/config"
export PYTHONPATH="$PATCH_TREE${PYTHONPATH:+:$PYTHONPATH}"

printf '[mode1 threshold multistep ref] runtime_cwd=%s\n' "$PATCH_TREE"
printf '[mode1 threshold multistep ref] run_root=%s\n' "$run_root"
printf '[mode1 threshold multistep ref] launcher=%s\n' "$LAUNCHER"
printf '[mode1 threshold multistep ref] floor=%s total_steps=%s max_response=%s max_batched_tokens=%s dataset_fraction=%s\n' \
    "$floor" "$BASELINE_TOTAL_TRAINING_STEPS" "$MAX_RESPONSE_LENGTH" \
    "$ROLLOUT_MAX_NUM_BATCHED_TOKENS" "$DATASET_FRACTION"
printf '[mode1 threshold multistep ref] tail_validate=%s mode1_kv_cap=%s direct_npu=%s scalar_direct=%s cpu_dp_metadata_sync=%s\n' \
    "$VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS" \
    "${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS:-unset}" \
    "$VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT" \
    "$VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT" \
    "$VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC"
printf '[mode1 threshold multistep ref] tee_log=%s\n' "$tee_log"

bash "$LAUNCHER" \
    data.dataset_fraction="$DATASET_FRACTION" \
    trainer.total_training_steps="$BASELINE_TOTAL_TRAINING_STEPS" \
    2>&1 | tee "$tee_log"
