#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
PATCH_TREE="${PATCH_TREE:-$REPO_ROOT}"
LAUNCHER="$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"

cd "$PATCH_TREE"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
tee_log="$REPO_ROOT/mode4_perf_clean_${stamp}.log"

# Performance run: disable profiler, stage-decode MSTX markers, and per-layer
# timing sync/logging. This is the run to compare against the previous 240s mode4.
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE="${VLLM_ASCEND_ELASTIC_EXECUTION_MODE:-4}"
export VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE="${VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE:-1}"
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

export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-20000}"
export MASTER_PORT="${MASTER_PORT:-12000}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-${HCCL_IF_BASE_PORT}}"
export VERL_MASTER_PORT_START="${VERL_MASTER_PORT_START:-${MASTER_PORT}}"

export HOME="$REPO_ROOT"
export CONFIG_DIR="$PATCH_TREE/verl/trainer/config"
export PYTHONPATH="$PATCH_TREE${PYTHONPATH:+:$PYTHONPATH}"

printf '[mode4 perf clean] runtime_cwd=%s\n' "$PATCH_TREE"
printf '[mode4 perf clean] tee_log=%s\n' "$tee_log"
printf '[mode4 perf clean] profile=0 timing_log=%s timing_sync=%s markers=%s\n' "$VLLM_ASCEND_MODE3_TIMING_LOG" "$VLLM_ASCEND_MODE3_TIMING_SYNC" "$VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS"
printf '[mode4 perf clean] ports HCCL_IF_BASE_PORT=%s MASTER_PORT=%s VERL_HCCL_IF_BASE_PORT_START=%s\n' "$HCCL_IF_BASE_PORT" "$MASTER_PORT" "$VERL_HCCL_IF_BASE_PORT_START"

bash "$LAUNCHER" 2>&1 | tee "$tee_log"
