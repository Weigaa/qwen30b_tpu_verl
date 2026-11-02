#!/usr/bin/env bash
set -euo pipefail

# Heavy validation run: capture CPU/NPU operator traces plus MSTX markers.
# Use this when the marker-only overlap windows look suspicious and need
# operator-level evidence.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS="${VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS:-cpu,npu,mstx}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_LEVEL="${VLLM_ASCEND_BUCKET_OP_PROFILE_LEVEL:-level0}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_ANALYSIS="${VLLM_ASCEND_BUCKET_OP_PROFILE_ANALYSIS:-1}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_SYNC="${VLLM_ASCEND_BUCKET_OP_PROFILE_SYNC:-1}"

# Keep the operator validation bounded. Override from the shell if needed.
export VLLM_ASCEND_BUCKET_OP_PROFILE_STAGE_SAMPLES="${VLLM_ASCEND_BUCKET_OP_PROFILE_STAGE_SAMPLES:-2}"
export VLLM_ASCEND_BUCKET_OP_PROFILE_RANKS="${VLLM_ASCEND_BUCKET_OP_PROFILE_RANKS:-8,12,14,15}"

exec "$SCRIPT_DIR/run_mode4_stage_decode_profile.sh"
