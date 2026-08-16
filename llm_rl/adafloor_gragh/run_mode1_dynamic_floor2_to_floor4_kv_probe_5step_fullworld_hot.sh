#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

# Five-step floor2 -> floor4 -> full-world probe with an explicit full-world
# MC2 dispatcher warmup after floor16 groups are refreshed and before the new KV
# cache is allocated.  This targets the path that the short step1/step5 probe
# missed: step3/4 build a floor4 state from floor2 history, then step5 restores
# to full-world.
export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_floor2_to_floor4_kv_probe_5step_fullworld_hot_${RUN_STAMP}}"
export DYNAMIC_PLAN_STEPS="${DYNAMIC_PLAN_STEPS:-5}"
export DYNAMIC_TRAIN_STEPS="${DYNAMIC_TRAIN_STEPS:-5}"
export DYNAMIC_FORCE_SELECTED_FLOORS="${DYNAMIC_FORCE_SELECTED_FLOORS:-2,2,4,4,16}"

export DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP="${DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP:-8,16,32,64,64;8,16,32,64,64;8,16,32,64,64;8,16,32,64,64;8,16,32,64,64}"

# Refresh full-world MoE groups while old KV has already been released, then
# materialize the MC2 dispatcher path before the large floor16 KV allocation.
export VLLM_ASCEND_MODE1_REFRESH_GROUPS_ON_KV_RESIZE="${VLLM_ASCEND_MODE1_REFRESH_GROUPS_ON_KV_RESIZE:-1}"
export VLLM_ASCEND_MODE1_FULLWORLD_REFRESH_MC2_WARMUP="${VLLM_ASCEND_MODE1_FULLWORLD_REFRESH_MC2_WARMUP:-1}"
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP="${VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP:-1}"
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP_TOKENS="${VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP_TOKENS:-32}"
export VLLM_ASCEND_MODE1_PARITY_MC2_WARMUP_ROUTE="${VLLM_ASCEND_MODE1_PARITY_MC2_WARMUP_ROUTE:-global}"
export VLLM_ASCEND_MODE1_KV_PREEMPT_DIAG="${VLLM_ASCEND_MODE1_KV_PREEMPT_DIAG:-0}"
export VLLM_ASCEND_MODE1_KV_PREEMPT_DIAG_LIMIT="${VLLM_ASCEND_MODE1_KV_PREEMPT_DIAG_LIMIT:-40}"

echo "[floor2->floor4 kv probe 5step fullworld-hot] run_name=$DYNAMIC_RUN_NAME"
echo "[floor2->floor4 kv probe 5step fullworld-hot] floors=$DYNAMIC_FORCE_SELECTED_FLOORS steps=$DYNAMIC_TRAIN_STEPS"
echo "[floor2->floor4 kv probe 5step fullworld-hot] refresh_groups=$VLLM_ASCEND_MODE1_REFRESH_GROUPS_ON_KV_RESIZE mc2_warmup=$VLLM_ASCEND_MODE1_FULLWORLD_REFRESH_MC2_WARMUP route=$VLLM_ASCEND_MODE1_PARITY_MC2_WARMUP_ROUTE"
echo "[floor2->floor4 kv probe 5step fullworld-hot] kv_preempt_diag=$VLLM_ASCEND_MODE1_KV_PREEMPT_DIAG limit=$VLLM_ASCEND_MODE1_KV_PREEMPT_DIAG_LIMIT"

exec "$SCRIPT_DIR/run_mode1_dynamic_floor2_to_floor4_kv_probe.sh" "$@"
