#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)

# This is the first full-length gate after the threshold-controlled smoke.
# Keep the rollout and GRPO shape identical to the intended EP16 workload.
export DEEPSEEK_ACTOR_PROBE_RUN_NAME="${DEEPSEEK_ACTOR_PROBE_RUN_NAME:-full_actor_2step_n16_no_overlap_$timestamp}"
export DEEPSEEK_ACTOR_PROBE_TRAINING_STEPS=2
export DEEPSEEK_ACTOR_PROBE_TRAIN_BATCH_SIZE=32
export DEEPSEEK_ACTOR_PROBE_ROLLOUT_N=16
export DEEPSEEK_ACTOR_PROBE_MAX_PROMPT_LENGTH=1024
export DEEPSEEK_ACTOR_PROBE_MAX_RESPONSE_LENGTH=16384
export DEEPSEEK_ACTOR_PROBE_MAX_NUM_BATCHED_TOKENS=17408
export DEEPSEEK_ACTOR_PROBE_ACTOR_TOKEN_CAP=17408
export DEEPSEEK_ACTOR_PROBE_LOG_PROB_TOKEN_CAP=17408
export DEEPSEEK_ACTOR_PROBE_MAX_NUM_SEQS=32
export DEEPSEEK_ACTOR_PROBE_KV_TOKENS_PER_RANK=621056
export DEEPSEEK_ACTOR_PROBE_TASK_QUEUE_ENABLE=1
export DEEPSEEK_ACTOR_PROBE_RECOMPUTE_METHOD=uniform
export DEEPSEEK_ACTOR_PROBE_RECOMPUTE_NUM_LAYERS=1
export DEEPSEEK_ACTOR_PROBE_DEALLOCATE_PIPELINE_OUTPUTS=False

# The full-length MLA matmul needs a 6.53 GiB temporary workspace. Execute NPU
# work eagerly and disable the training-only MoE overlap buffers so queued
# operations and communication state do not extend the peak allocation window.
export DEEPSEEK_ACTOR_PROBE_MOE_ALLTOALL_OVERLAP_COMM=False
export DEEPSEEK_ACTOR_PROBE_MOE_SHARED_EXPERT_OVERLAP=False

# A full run must not inherit the cohort caps used by the short smoke.
unset DEEPSEEK_ACTOR_PROBE_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP
unset VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS
unset VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP

echo "[DeepSeek full actor gate] mode=0 steps=2 n=16 response=16384 threshold=disabled"
echo "[DeepSeek full actor gate] task_queue=1 deallocate_pipeline_outputs=False moe_overlap=False kv_tokens=621056"
exec "$SCRIPT_DIR/run_deepseek_v2_lite_actor_update_probe.sh" "$@"
