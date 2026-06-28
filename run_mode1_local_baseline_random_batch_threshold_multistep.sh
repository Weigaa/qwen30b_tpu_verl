#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_mode1_local_baseline_random_batch_threshold_multistep.sh [extra hydra args...]

Fast multi-step smoke test for mode=1 floor=4 baseline:
  - reuses the local baseline random-batch script
  - uses elastic tail validation level caps to create uneven rank tails
  - exercises 16->8->4 shrink/restore paths faster than a full rollout

Useful overrides:
  MAX_RESPONSE_LENGTH=640
  VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,640,640
  DATASET_FRACTION=0.005
  LOCAL_TEST_LAUNCHER=internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# This smoke test can A/B eager vs regroup launchers. Keep the base script
# unchanged and override only the launcher plus local input paths here.
export LOCAL_TEST_LAUNCHER="${LOCAL_TEST_LAUNCHER:-$SCRIPT_DIR/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh}"
export MODEL_PATH="${MODEL_PATH:-/home/data/Qwen3-30B-A3B}"
export DISTCP_PATH="${DISTCP_PATH:-/home/data/Qwen3-30B-A3B_megatron}"
export TRAIN_FILE="${TRAIN_FILE:-/workspace/data/deepscaler/train.parquet}"
export TEST_FILE="${TEST_FILE:-/workspace/data/deepscaler/test.parquet}"

# Mirror the known-good mode=1/floor=4 threshold run from
# qwen3_true_mode5_a3cfdc2. The base script still carries several experimental
# defaults that re-enable synthetic post-shrink warmup and old-floor group
# destruction; both are known to cause 100s/300s communication stalls on this
# machine.
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS:-288000}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4:-288000}"
export VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC="${VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC:-1}"
export VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE="${VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE:-0}"
export VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC="${VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC:-0}"
export VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE="${VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE:-0}"
export VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE="${VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE:-0}"
export VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP="${VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP:-0}"
export VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP="${VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP:-0}"
export VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD="${VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD:-0}"
export VLLM_ASCEND_MODE1_PARITY_FORCE_CLEANUP_AFTER_FLOOR_GROUP_RELEASE="${VLLM_ASCEND_MODE1_PARITY_FORCE_CLEANUP_AFTER_FLOOR_GROUP_RELEASE:-0}"
export VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG="${VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG:-0}"
# Regroup launcher parity mode: first prove that regroup can run the same
# no-resample baseline as eager. Length regroup/dynamic-bsz changes the tail
# distribution and made low-floor MC2 restore behavior harder to compare.
export VLLM_EAGER_BASELINE_NO_RESAMPLE="${VLLM_EAGER_BASELINE_NO_RESAMPLE:-1}"
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE="${VLLM_EPOCH_LENGTH_REGROUP_ENABLE:-0}"

# Keep full-restore group lifecycle aligned with the eager success path. The
# previous defer-on-full-restore attempt moved the 200s wait out of destroy_ms
# but did not remove it, so do not keep that workaround in the parity test.
export VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE="${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE:-0}"
export VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD="${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD:-0}"
export VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH="${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH:-0}"
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP="${VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP:-1}"
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP_TOKENS="${VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP_TOKENS:-32}"

: "${MAX_PROMPT_LENGTH:=1024}"
: "${MAX_RESPONSE_LENGTH:=640}"
export MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

export VLLM_ROLLOUT_EARLY_STOP_ENABLE=0
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-256,512,640,640,640}"

export DATASET_FRACTION="${DATASET_FRACTION:-0.005}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_baseline_random_batch_floor4_threshold_multistep}"

printf '[mode1 threshold multistep] max_prompt=%s max_response=%s max_batched_tokens=%s\n' \
    "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" "$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
printf '[mode1 threshold multistep] early_stop=%s tail_validate_level_tokens=%s dataset_fraction=%s total_epochs=%s launcher=%s no_resample=%s\n' \
    "$VLLM_ROLLOUT_EARLY_STOP_ENABLE" \
    "$VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS" \
    "$DATASET_FRACTION" \
    "$TRAINER_TOTAL_EPOCHS" \
    "$LOCAL_TEST_LAUNCHER" \
    "$VLLM_EAGER_BASELINE_NO_RESAMPLE"

REGROUP_EAGER_PARITY_ARGS=(
    data.sampler.class_path=null
    data.sampler.class_name=null
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
)

exec "$SCRIPT_DIR/run_mode1_local_baseline_random_batch.sh" \
    "${REGROUP_EAGER_PARITY_ARGS[@]}" "$@"
