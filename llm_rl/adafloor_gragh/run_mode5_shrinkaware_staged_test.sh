#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
PATCH_TREE="${PATCH_TREE:-$REPO_ROOT}"
LAUNCHER="${SHRINK_AWARE_LAUNCHER:-$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh}"

export VLLM_ASCEND_SHRINK_AWARE_ENABLE="${VLLM_ASCEND_SHRINK_AWARE_ENABLE:-1}"
export VLLM_ASCEND_SHRINK_AWARE_MODE="${VLLM_ASCEND_SHRINK_AWARE_MODE:-staged}"
export VLLM_ASCEND_SHRINK_AWARE_STAGES="${VLLM_ASCEND_SHRINK_AWARE_STAGES:-8,4}"
export VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY="${VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY:-topology_aware}"
export VLLM_ASCEND_SHRINK_AWARE_LENGTH_SOURCE="${VLLM_ASCEND_SHRINK_AWARE_LENGTH_SOURCE:-existing_regroup}"
export VLLM_ASCEND_SHRINK_AWARE_LOGGING="${VLLM_ASCEND_SHRINK_AWARE_LOGGING:-1}"
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE="${VLLM_ASCEND_ELASTIC_EXECUTION_MODE:-1}"
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE:-4}"
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE="${VLLM_EPOCH_LENGTH_REGROUP_ENABLE:-1}"
export VLLM_EAGER_BASELINE_NO_RESAMPLE="${VLLM_EAGER_BASELINE_NO_RESAMPLE:-0}"
export VLLM_ASCEND_CUSTOM_MODE1_DEBUG=0
export VLLM_ASCEND_CUSTOM_MODE1_TIMING_EVENTS=0
export VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG=0
export VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT="${VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT:-1}"
export VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT="${VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT:-0}"
export VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT="${VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT:-0}"
export VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS="${VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS:-8}"
export VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC="${VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC:-1}"
export VLLM_ASCEND_MODE3_TRANSFER_LOG=0
export VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG=0
export VLLM_ASCEND_MODE3_TIMING_LOG=0
export VLLM_ASCEND_MODE3_TIMING_SYNC=0
export VLLM_ASCEND_MODE3_TIMING_EVERY=1000000
export VLLM_ASCEND_MODE3_TIMING_FIRST_N=0
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

export HOME="$REPO_ROOT"
export CONFIG_DIR="$PATCH_TREE/verl/trainer/config"
export PYTHONPATH="$PATCH_TREE${PYTHONPATH:+:$PYTHONPATH}"

floor="${SHRINK_AWARE_FLOOR:-${MODE1_FLOOR:-4}}"
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  floor="$1"
  shift
elif [[ $# -gt 1 && ( "$1" == "--floor" || "$1" == "-f" ) ]]; then
  floor="$2"
  shift 2
fi

case "$floor" in
  1|2|4|8|16) ;;
  *)
    echo "unsupported mode=1 floor: $floor; expected one of 1,2,4,8,16" >&2
    exit 2
    ;;
esac

export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="$floor"

cd "$PATCH_TREE"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
tee_log="$REPO_ROOT/mode1_shrinkaware_staged_floor${floor}_${stamp}.log"

printf '[mode1 shrink-aware staged] runtime_cwd=%s\n' "$PATCH_TREE"
printf '[mode1 shrink-aware staged] launcher=%s length_regroup=%s baseline_no_resample=%s\n' \
  "$LAUNCHER" \
  "$VLLM_EPOCH_LENGTH_REGROUP_ENABLE" \
  "$VLLM_EAGER_BASELINE_NO_RESAMPLE"
printf '[mode1 shrink-aware staged] tee_log=%s\n' "$tee_log"
printf '[mode1 shrink-aware staged] floor=%s batch_direct_npu=%s allow_scalar=%s batch_experts=%s cpu_dp_metadata_sync=%s stages=%s\n' \
  "$floor" \
  "$VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT" \
  "$VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT" \
  "$VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS" \
  "$VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC" \
  "$VLLM_ASCEND_SHRINK_AWARE_STAGES"

exec bash "$LAUNCHER" \
  actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_scheduling=true \
  actor_rollout_ref.rollout.shrink_aware.shrink_aware_mode=staged \
  actor_rollout_ref.rollout.shrink_aware.shrink_stages='[8,4]' \
  actor_rollout_ref.rollout.shrink_aware.survivor_selection_policy=topology_aware \
  actor_rollout_ref.rollout.shrink_aware.length_prediction_source=existing_regroup \
  actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_logging=true \
  "$@" 2>&1 | tee "$tee_log"
