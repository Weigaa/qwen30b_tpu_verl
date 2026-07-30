#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash run_hccl_overlap_group_destroy_repro.sh

Purpose:
  Minimal torch.distributed(HCCL) reproducer for slow destroy of overlapping
  4-rank groups.  No verl, vllm, or vllm_ascend is used.

Default:
  world_size=16
  groups: [12,13,14,15] -> [10,11,12,13]
  op: all_to_all

Useful overrides:
  HCCL_TEST_GROUPS="12-15 10-13"
  ROUNDS=3
  OP=all_to_all              # all_reduce | all_to_all | both
  COLLECTIVE_ITERS=2
  TENSOR_NUMEL=4096
  MASTER_PORT=29641
  HCCL_IF_BASE_PORT=48641
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  NO_DESTROY=1               # control run: create/use but do not destroy subgroups
  QUIESCE_BEFORE_DESTROY=1    # drain subgroup before destroy
  QUIESCE_OP=all_reduce       # all_reduce | all_to_all | none
  QUIESCE_SLEEP_MS=0
  NO_CPU_GROUP=1              # skip companion Gloo subgroups
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/hccl_overlap_group_destroy_runs/$stamp}"
mkdir -p "$RUN_ROOT"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export WORLD_SIZE="${WORLD_SIZE:-16}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$WORLD_SIZE}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29641}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-48641}"

# Make HCCL plog useful for issue reports.  Existing environment values win.
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-1}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-360}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-360}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-0}"

# Do not use bash's reserved GROUPS array.  Under root it expands to "0",
# which would accidentally test only the singleton group [0].
HCCL_TEST_GROUPS="${HCCL_TEST_GROUPS:-12-15 10-13}"
ROUNDS="${ROUNDS:-3}"
OP="${OP:-all_to_all}"
COLLECTIVE_ITERS="${COLLECTIVE_ITERS:-2}"
TENSOR_NUMEL="${TENSOR_NUMEL:-4096}"
BETWEEN_GROUPS_SLEEP="${BETWEEN_GROUPS_SLEEP:-0}"
BETWEEN_ROUNDS_SLEEP="${BETWEEN_ROUNDS_SLEEP:-2}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"
DESTROY_WORLD_BARRIER="${DESTROY_WORLD_BARRIER:-0}"
NO_DESTROY="${NO_DESTROY:-0}"
QUIESCE_BEFORE_DESTROY="${QUIESCE_BEFORE_DESTROY:-0}"
QUIESCE_OP="${QUIESCE_OP:-all_reduce}"
QUIESCE_PRE_SYNC="${QUIESCE_PRE_SYNC:-1}"
QUIESCE_POST_SYNC="${QUIESCE_POST_SYNC:-1}"
QUIESCE_CPU_BARRIER="${QUIESCE_CPU_BARRIER:-1}"
QUIESCE_SLEEP_MS="${QUIESCE_SLEEP_MS:-0}"
NO_CPU_GROUP="${NO_CPU_GROUP:-0}"

args=(
  "--nproc_per_node=$NPROC_PER_NODE"
  "--master_addr=$MASTER_ADDR"
  "--master_port=$MASTER_PORT"
  "$SCRIPT_DIR/hccl_overlap_group_destroy_repro.py"
  "--groups"
)

read -r -a group_args <<< "$HCCL_TEST_GROUPS"
args+=("${group_args[@]}")

args+=(
  "--rounds" "$ROUNDS"
  "--op" "$OP"
  "--collective-iters" "$COLLECTIVE_ITERS"
  "--tensor-numel" "$TENSOR_NUMEL"
  "--between-groups-sleep" "$BETWEEN_GROUPS_SLEEP"
  "--between-rounds-sleep" "$BETWEEN_ROUNDS_SLEEP"
  "--timeout-sec" "$TIMEOUT_SEC"
)

if [[ "$DESTROY_WORLD_BARRIER" == "1" ]]; then
    args+=("--destroy-world-barrier")
fi
if [[ "$NO_DESTROY" == "1" ]]; then
    args+=("--no-destroy")
fi
if [[ "$QUIESCE_BEFORE_DESTROY" == "1" ]]; then
    args+=("--quiesce-before-destroy")
    args+=("--quiesce-op" "$QUIESCE_OP")
    args+=("--quiesce-sleep-ms" "$QUIESCE_SLEEP_MS")
    if [[ "$QUIESCE_PRE_SYNC" != "1" ]]; then
        args+=("--quiesce-no-pre-sync")
    fi
    if [[ "$QUIESCE_POST_SYNC" != "1" ]]; then
        args+=("--quiesce-no-post-sync")
    fi
    if [[ "$QUIESCE_CPU_BARRIER" != "1" ]]; then
        args+=("--quiesce-no-cpu-barrier")
    fi
fi
if [[ "$NO_CPU_GROUP" == "1" ]]; then
    args+=("--no-cpu-group")
fi

{
    echo "[launcher] run_root=$RUN_ROOT"
    echo "[launcher] ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
    echo "[launcher] WORLD_SIZE=$WORLD_SIZE NPROC_PER_NODE=$NPROC_PER_NODE"
    echo "[launcher] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT HCCL_IF_BASE_PORT=$HCCL_IF_BASE_PORT"
    echo "[launcher] HCCL_TEST_GROUPS=$HCCL_TEST_GROUPS ROUNDS=$ROUNDS OP=$OP COLLECTIVE_ITERS=$COLLECTIVE_ITERS TENSOR_NUMEL=$TENSOR_NUMEL"
    echo "[launcher] BETWEEN_GROUPS_SLEEP=$BETWEEN_GROUPS_SLEEP BETWEEN_ROUNDS_SLEEP=$BETWEEN_ROUNDS_SLEEP"
    echo "[launcher] DESTROY_WORLD_BARRIER=$DESTROY_WORLD_BARRIER NO_DESTROY=$NO_DESTROY"
    echo "[launcher] QUIESCE_BEFORE_DESTROY=$QUIESCE_BEFORE_DESTROY QUIESCE_OP=$QUIESCE_OP QUIESCE_PRE_SYNC=$QUIESCE_PRE_SYNC QUIESCE_POST_SYNC=$QUIESCE_POST_SYNC QUIESCE_CPU_BARRIER=$QUIESCE_CPU_BARRIER QUIESCE_SLEEP_MS=$QUIESCE_SLEEP_MS NO_CPU_GROUP=$NO_CPU_GROUP"
    echo "[launcher] start_time=$(date --iso-8601=seconds)"
} | tee "$RUN_ROOT/launcher.log"

set +e
python -m torch.distributed.run "${args[@]}" 2>&1 | tee "$RUN_ROOT/repro.log"
rc=${PIPESTATUS[0]}
set -e

{
    echo "[launcher] end_time=$(date --iso-8601=seconds)"
    echo "[launcher] exit_code=$rc"
    echo "[launcher] repro_log=$RUN_ROOT/repro.log"
    echo "[launcher] HCCL plogs are under /root/ascend/log/run/plog; match by PID/time in repro.log."
} | tee -a "$RUN_ROOT/launcher.log"

exit "$rc"
