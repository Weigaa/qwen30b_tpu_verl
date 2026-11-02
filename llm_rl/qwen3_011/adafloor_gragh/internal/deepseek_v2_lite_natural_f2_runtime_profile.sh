#!/usr/bin/env bash

# Natural floor2 shares the DeepSeek mode1 memory policy with Natural floor4,
# then enables the final 4-to-2 transition.
DEEPSEEK_N_F2_PROFILE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$DEEPSEEK_N_F2_PROFILE_DIR/deepseek_v2_lite_natural_f4_runtime_profile.sh"
unset DEEPSEEK_N_F2_PROFILE_DIR

export DEEPSEEK_N_F2_RUNTIME_PROFILE_ID=deepseek-v2-lite-natural-f2-mode1-v1
export DEEPSEEK_N_F2_RUNTIME_PROFILE_FILES=internal/deepseek_v2_lite_natural_f4_runtime_profile.sh,internal/deepseek_v2_lite_natural_f2_runtime_profile.sh
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2
export VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4,2
export VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=8,9,10,11,12,13,14,15
export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=14,15
