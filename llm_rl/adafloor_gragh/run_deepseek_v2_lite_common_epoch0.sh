#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAP_ENV=${DEEPSEEK_KV_CAP_ENV:-$SCRIPT_DIR/deepseek_v2_lite_kv_caps.env}
WORKLOAD_PROFILE_PATH=${DEEPSEEK_WORKLOAD_PROFILE_PATH:-}

if (( $# != 0 )); then
    echo "DeepSeek common epoch0 does not accept workload overrides" >&2
    exit 2
fi

workload_profile_id=${DEEPSEEK_WORKLOAD_PROFILE_ID:-unspecified}
workload_profile_sha256=${DEEPSEEK_WORKLOAD_PROFILE_SHA256:-unspecified}
if [[ -n "$WORKLOAD_PROFILE_PATH" ]]; then
    if [[ ! -f "$WORKLOAD_PROFILE_PATH" ]]; then
        echo "missing DeepSeek workload profile: $WORKLOAD_PROFILE_PATH" >&2
        exit 2
    fi
    WORKLOAD_PROFILE_PATH=$(realpath "$WORKLOAD_PROFILE_PATH")
    # shellcheck disable=SC1090
    source "$WORKLOAD_PROFILE_PATH"
    workload_profile_id=${DEEPSEEK_WORKLOAD_PROFILE_ID:-}
    measured_profile_sha256=$(sha256sum "$WORKLOAD_PROFILE_PATH")
    measured_profile_sha256=${measured_profile_sha256%% *}
    if [[ -z "$workload_profile_id" ]]; then
        echo "DeepSeek workload profile has no DEEPSEEK_WORKLOAD_PROFILE_ID" >&2
        exit 2
    fi
    if [[ "${DEEPSEEK_WORKLOAD_PROFILE_SHA256:-$measured_profile_sha256}" \
          != "$measured_profile_sha256" ]]; then
        echo "DeepSeek workload profile SHA256 mismatch" >&2
        exit 2
    fi
    workload_profile_sha256=$measured_profile_sha256
fi
export COMMON_EPOCH0_WORKLOAD_PROFILE_ID=$workload_profile_id
export COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256=$workload_profile_sha256
export DEEPSEEK_WORKLOAD_PROFILE_ID=$workload_profile_id
export DEEPSEEK_WORKLOAD_PROFILE_SHA256=$workload_profile_sha256

if [[ -f "$CAP_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$CAP_ENV"
fi

kv_tokens=auto
cap_matches_workload=1
if [[ "$workload_profile_id" != unspecified \
      && ( "${DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID:-}" != "$workload_profile_id" \
           || "${DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256:-}" \
              != "$workload_profile_sha256" ) ]]; then
    cap_matches_workload=0
fi
if [[ "${DEEPSEEK_KV_CAPS_VERIFIED:-0}" == 1 \
      && "${DEEPSEEK_VANILLA_KV_PHYSICAL_TOKENS:-0}" =~ ^[1-9][0-9]*$ \
      && "$cap_matches_workload" == 1 ]]; then
    kv_tokens=$DEEPSEEK_VANILLA_KV_PHYSICAL_TOKENS
fi

export MODEL_PATH=${MODEL_PATH:-/data/DeepSeek-V2-Lite-Chat}
export MODEL_REVISION=${MODEL_REVISION:-85864749cd611b4353ce1decdb286193298f64c7}
export DISTCP_PATH=${DISTCP_PATH:-/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4}
export BASELINE_LAUNCHER=${BASELINE_LAUNCHER:-$SCRIPT_DIR/internal/wj_train_grpo_deepseek_v2_lite_16die_adafloor.sh}
export LOCAL_TEST_LAUNCHER=${LOCAL_TEST_LAUNCHER:-$BASELINE_LAUNCHER}
export CHECKPOINT_MODEL_DIR_NAME=${CHECKPOINT_MODEL_DIR_NAME:-deepseek_v2_lite_chat}
export TRAIN_LOG_PREFIX=${TRAIN_LOG_PREFIX:-deepseek-v2-lite-chat-adafloor}
export TRAIN_FILE=/data/deepscaler/train.parquet
export TEST_FILE=/data/deepscaler/test.parquet
if [[ -z "${COMMON_EPOCH0_DATASET_FRACTION+x}" ]]; then
    export COMMON_EPOCH0_DATASET_FRACTION=0.005
fi
export VLLM_KV_BLOCK_SIZE=128
export ROLLOUT_ENFORCE_EAGER=True
export COMMON_EPOCH0_OUTPUT_ROOT=${COMMON_EPOCH0_OUTPUT_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite}
export COMMON_EPOCH0_RUN_NAME=${COMMON_EPOCH0_RUN_NAME:-common_epoch0_deepseek_v2_lite_tq1_no_overlap_gpu09}
export COMMON_EPOCH0_KV_TOKENS_PER_RANK=$kv_tokens
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM=${DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM:-False}
export DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP=${DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP:-False}
export DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS=${DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS:-False}
export DEEPSEEK_ACTOR_RECOMPUTE_METHOD=${DEEPSEEK_ACTOR_RECOMPUTE_METHOD:-uniform}
export DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS=${DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS:-1}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-800}
export COMMON_EPOCH0_EXECUTION_PROFILE="deepseek-v2-lite_tq${TASK_QUEUE_ENABLE}_a2a${DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM}_shared${DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP}_dealloc${DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS}_recompute${DEEPSEEK_ACTOR_RECOMPUTE_METHOD}x${DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS}_hccl${HCCL_BUFFSIZE}"

echo "[DeepSeek-V2-Lite epoch0] workload_profile=$workload_profile_id sha256=$workload_profile_sha256"
echo "[DeepSeek-V2-Lite epoch0] kv_tokens_per_rank=$kv_tokens"
echo "[DeepSeek-V2-Lite epoch0] task_queue=$TASK_QUEUE_ENABLE moe_overlap=$DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM/$DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP deallocate_pipeline_outputs=$DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS"
echo "[DeepSeek-V2-Lite epoch0] execution_profile=$COMMON_EPOCH0_EXECUTION_PROFILE"
if [[ "$kv_tokens" == auto ]]; then
    echo "[DeepSeek-V2-Lite epoch0] vLLM will profile full16 KV capacity. Record the reported GPU KV cache size before fair reruns."
fi

exec "$SCRIPT_DIR/run_common_epoch0_probe_gpu09_kv380800_permanent.sh" "$@"
