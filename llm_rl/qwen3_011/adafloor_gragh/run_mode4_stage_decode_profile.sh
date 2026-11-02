#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
PATCH_TREE="${PATCH_TREE:-$REPO_ROOT}"
LAUNCHER="$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"

cd "$PATCH_TREE"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
profile_root="${PROFILE_ROOT:-$REPO_ROOT/stage_decode_profiles/$stamp}"
tee_log="$REPO_ROOT/mode4_stage_decode_profile_${stamp}.log"

export WJ_RECORDS_DIR="$profile_root"
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE="${VLLM_ASCEND_ELASTIC_EXECUTION_MODE:-4}"
export VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE="${VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE:-1}"

# Keep text logs small; the profiler markers carry the per-stage timing.
export VLLM_ASCEND_MODE3_TIMING_LOG="${VLLM_ASCEND_MODE3_TIMING_LOG:-0}"
export VLLM_ASCEND_MODE3_TIMING_SYNC="${VLLM_ASCEND_MODE3_TIMING_SYNC:-0}"
export VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS=1

# Reuse the existing torch_npu bucket profiler, but arm it by stage instead of
# by fixed decode-step buckets.
export VLLM_ASCEND_BUCKET_OP_PROFILE=1
export VLLM_ASCEND_BUCKET_OP_PROFILE_BY_STAGE=1
export VLLM_ASCEND_BUCKET_OP_PROFILE_STAGES="${VLLM_ASCEND_BUCKET_OP_PROFILE_STAGES:-8,4,2,1}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_STAGE_SAMPLES="${VLLM_ASCEND_BUCKET_OP_PROFILE_STAGE_SAMPLES:-5}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_DIR="${VLLM_ASCEND_BUCKET_OP_PROFILE_DIR:-$profile_root/op_profiles}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_RANKS="${VLLM_ASCEND_BUCKET_OP_PROFILE_RANKS:-8,12,14,15}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS="${VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS:-mstx}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_LEVEL="${VLLM_ASCEND_BUCKET_OP_PROFILE_LEVEL:-level_none}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_SYNC="${VLLM_ASCEND_BUCKET_OP_PROFILE_SYNC:-1}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_ANALYSIS="${VLLM_ASCEND_BUCKET_OP_PROFILE_ANALYSIS:-1}"

export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-46741}"
export MASTER_PORT="${MASTER_PORT:-25740}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-46741}"

export HOME="$REPO_ROOT"
export CONFIG_DIR="$PATCH_TREE/verl/trainer/config"
export PYTHONPATH="$PATCH_TREE${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$profile_root" "$VLLM_ASCEND_BUCKET_OP_PROFILE_DIR"

echo "[mode4 stage decode profile] runtime_cwd=$PATCH_TREE"
echo "[mode4 stage decode profile] tee_log=$tee_log"
echo "[mode4 stage decode profile] profile_root=$profile_root"
echo "[mode4 stage decode profile] op_profile_dir=$VLLM_ASCEND_BUCKET_OP_PROFILE_DIR"
echo "[mode4 stage decode profile] stages=$VLLM_ASCEND_BUCKET_OP_PROFILE_STAGES samples=$VLLM_ASCEND_BUCKET_OP_PROFILE_STAGE_SAMPLES ranks=$VLLM_ASCEND_BUCKET_OP_PROFILE_RANKS contents=$VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS"
echo "[mode4 stage decode profile] ports HCCL_IF_BASE_PORT=$HCCL_IF_BASE_PORT MASTER_PORT=$MASTER_PORT VERL_HCCL_IF_BASE_PORT_START=$VERL_HCCL_IF_BASE_PORT_START"

bash "$LAUNCHER" 2>&1 | tee "$tee_log"
