#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash run_hccl_floor2_lifecycle_repro.sh

Purpose:
  Minimal torch/torch_npu/HCCL reproducer for repeated floor2 lifecycle cost.
  No verl, vllm, or vllm_ascend is imported.

Default lifecycle per step:
  [0-15] -> [8-15] -> [12-15] -> [14-15] -> [0-15]

Useful overrides:
  STEPS=4
  HCCL_FLOOR2_STAGES="0-15 8-15 12-15 14-15 0-15"
  OP=all_to_all              # all_reduce | all_to_all | both
  TENSOR_NUMEL=8192
  COLLECTIVE_ITERS=1
  MEMBERS_ONLY=1             # control: non-members do not call new_group
  NO_DESTROY=1               # control: leak groups intentionally
  WORLD_BARRIER_AFTER_DESTROY=1
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/hccl_floor2_lifecycle_runs/$stamp}"
mkdir -p "$RUN_ROOT"

# If the staged repo's CANN 9.1 runtime helper exists, use it by default so this
# reproducer runs in the same software stack as the current floor2 experiment.
DEFAULT_RUNTIME="$SCRIPT_DIR/../qwen3_true_mode5_shrinkaware_staged/tools/use_default_cann91_a3_torch29_runtime.sh"
if [[ "${USE_STAGED_CANN91_RUNTIME:-1}" == "1" && -f "$DEFAULT_RUNTIME" ]]; then
  # shellcheck disable=SC1090
  source "$DEFAULT_RUNTIME"
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export WORLD_SIZE="${WORLD_SIZE:-16}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$WORLD_SIZE}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29741}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-49741}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-900}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-900}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-1}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-0}"

STEPS="${STEPS:-4}"
HCCL_FLOOR2_STAGES="${HCCL_FLOOR2_STAGES:-0-15 8-15 12-15 14-15 0-15}"
OP="${OP:-all_to_all}"
TENSOR_NUMEL="${TENSOR_NUMEL:-8192}"
COLLECTIVE_ITERS="${COLLECTIVE_ITERS:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
MEMBERS_ONLY="${MEMBERS_ONLY:-0}"
CREATE_CPU_GROUPS="${CREATE_CPU_GROUPS:-0}"
DESTROY_CPU_GROUPS="${DESTROY_CPU_GROUPS:-0}"
NO_DESTROY="${NO_DESTROY:-0}"
PRE_DESTROY_SYNC="${PRE_DESTROY_SYNC:-1}"
POST_DESTROY_SYNC="${POST_DESTROY_SYNC:-1}"
WORLD_BARRIER_AFTER_DESTROY="${WORLD_BARRIER_AFTER_DESTROY:-0}"
SLEEP_BETWEEN_STAGES_MS="${SLEEP_BETWEEN_STAGES_MS:-0}"
SLEEP_BETWEEN_STEPS_MS="${SLEEP_BETWEEN_STEPS_MS:-0}"

args=(
  "--nproc_per_node=$NPROC_PER_NODE"
  "--master_addr=$MASTER_ADDR"
  "--master_port=$MASTER_PORT"
  "$SCRIPT_DIR/hccl_floor2_lifecycle_repro.py"
  "--stages"
)
read -r -a stage_args <<< "$HCCL_FLOOR2_STAGES"
args+=("${stage_args[@]}")
args+=(
  "--steps" "$STEPS"
  "--op" "$OP"
  "--collective-iters" "$COLLECTIVE_ITERS"
  "--tensor-numel" "$TENSOR_NUMEL"
  "--timeout-sec" "$TIMEOUT_SEC"
  "--sleep-between-stages-ms" "$SLEEP_BETWEEN_STAGES_MS"
  "--sleep-between-steps-ms" "$SLEEP_BETWEEN_STEPS_MS"
)

if [[ "$MEMBERS_ONLY" == "1" ]]; then args+=("--members-only"); fi
if [[ "$CREATE_CPU_GROUPS" == "1" ]]; then args+=("--create-cpu-groups"); fi
if [[ "$DESTROY_CPU_GROUPS" == "1" ]]; then args+=("--destroy-cpu-groups"); fi
if [[ "$NO_DESTROY" == "1" ]]; then args+=("--no-destroy"); fi
if [[ "$PRE_DESTROY_SYNC" != "1" ]]; then args+=("--no-pre-destroy-sync"); fi
if [[ "$POST_DESTROY_SYNC" != "1" ]]; then args+=("--no-post-destroy-sync"); fi
if [[ "$WORLD_BARRIER_AFTER_DESTROY" == "1" ]]; then args+=("--world-barrier-after-destroy"); fi

{
  echo "[launcher] run_root=$RUN_ROOT"
  echo "[launcher] start_time=$(date --iso-8601=seconds)"
  echo "[launcher] ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  echo "[launcher] WORLD_SIZE=$WORLD_SIZE NPROC_PER_NODE=$NPROC_PER_NODE"
  echo "[launcher] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT HCCL_IF_BASE_PORT=$HCCL_IF_BASE_PORT"
  echo "[launcher] STEPS=$STEPS HCCL_FLOOR2_STAGES=$HCCL_FLOOR2_STAGES OP=$OP TENSOR_NUMEL=$TENSOR_NUMEL COLLECTIVE_ITERS=$COLLECTIVE_ITERS"
  echo "[launcher] MEMBERS_ONLY=$MEMBERS_ONLY NO_DESTROY=$NO_DESTROY CREATE_CPU_GROUPS=$CREATE_CPU_GROUPS DESTROY_CPU_GROUPS=$DESTROY_CPU_GROUPS"
  echo "[launcher] PRE_DESTROY_SYNC=$PRE_DESTROY_SYNC POST_DESTROY_SYNC=$POST_DESTROY_SYNC WORLD_BARRIER_AFTER_DESTROY=$WORLD_BARRIER_AFTER_DESTROY"
  python - <<'PY'
import torch
try:
    import torch_npu
    print(f"[launcher] torch={torch.__version__} torch_npu={getattr(torch_npu, '__version__', 'unknown')}")
except Exception as exc:
    print(f"[launcher] torch_import_error={exc!r}")
PY
} | tee "$RUN_ROOT/launcher.log"

set +e
python -m torch.distributed.run "${args[@]}" 2>&1 | tee "$RUN_ROOT/repro.log"
rc=${PIPESTATUS[0]}
set -e

{
  echo "[launcher] end_time=$(date --iso-8601=seconds)"
  echo "[launcher] exit_code=$rc"
  echo "[launcher] repro_log=$RUN_ROOT/repro.log"
  echo "[launcher] quick_analysis_commands:"
  echo "[launcher]   rg 'lifecycle_step_done|stage_create_done|stage_destroy_done|collective_done|RuntimeError|ERROR' $RUN_ROOT/repro.log"
  echo "[launcher] HCCL plogs: /root/ascend/log/run/plog, match by PID/time."
} | tee -a "$RUN_ROOT/launcher.log"

exit "$rc"
