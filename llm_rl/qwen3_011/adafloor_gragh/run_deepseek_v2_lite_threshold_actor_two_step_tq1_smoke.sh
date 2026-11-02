#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)

# Exercise the same actor memory path as the full-length gate while keeping
# generation short. This catches configuration and lifecycle failures before
# spending time on two 16K-token rollouts.
export DEEPSEEK_ACTOR_PROBE_RUN_NAME="${DEEPSEEK_ACTOR_PROBE_RUN_NAME:-threshold_actor_2step_n16_tq1_no_overlap_$timestamp}"
export DEEPSEEK_ACTOR_PROBE_TASK_QUEUE_ENABLE=1
export DEEPSEEK_ACTOR_PROBE_RECOMPUTE_METHOD=uniform
export DEEPSEEK_ACTOR_PROBE_RECOMPUTE_NUM_LAYERS=1
export DEEPSEEK_ACTOR_PROBE_MOE_ALLTOALL_OVERLAP_COMM=False
export DEEPSEEK_ACTOR_PROBE_MOE_SHARED_EXPERT_OVERLAP=False
export DEEPSEEK_ACTOR_PROBE_DEALLOCATE_PIPELINE_OUTPUTS=False

echo "[DeepSeek threshold actor TQ1 smoke] task_queue=1 moe_overlap=False deallocate_pipeline_outputs=False"
exec "$SCRIPT_DIR/run_deepseek_v2_lite_threshold_actor_two_step_smoke.sh" "$@"
