#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./run_mode3_mode4_floor4_dispatch_mem_diag.sh [mode3|mode4|mode5|all] [floor]

Purpose:
  Capture matched mode=3/mode=4/mode=5 DispatchV2/HCCL memory diagnostics at
  the target shrink floor. This is intended to explain why mode=3 may need
  more runtime memory than mode=4/5 under the same KV-cache budget.

Defaults:
  target=all
  floor=4

Environment overrides:
  DIAG_RUN_ROOT=mode3_mode4_dispatch_mem_diag_runs
  PATCH_TREE=/path/to/code/tree
  HCCL_IF_BASE_PORT=49041
  MASTER_PORT=29040
  VERL_HCCL_IF_BASE_PORT_START=49041

Examples:
  ./run_mode3_mode4_floor4_dispatch_mem_diag.sh mode3 4
  ./run_mode3_mode4_floor4_dispatch_mem_diag.sh mode4 4
  ./run_mode3_mode4_floor4_dispatch_mem_diag.sh mode5 4
  ./run_mode3_mode4_floor4_dispatch_mem_diag.sh all 4
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH_TREE=${PATCH_TREE:-$SCRIPT_DIR}
RUN_ROOT=${DIAG_RUN_ROOT:-$PATCH_TREE/mode3_mode4_dispatch_mem_diag_runs}
target=${1:-all}
floor=${2:-4}

case "$target" in
    mode3|3) target=mode3 ;;
    mode4|4) target=mode4 ;;
    mode5|5) target=mode5 ;;
    all) ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unsupported target: $target" >&2; usage >&2; exit 2 ;;
esac

case "$floor" in
    1|2|4|8|16) ;;
    *) echo "unsupported floor: $floor" >&2; usage >&2; exit 2 ;;
esac

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$RUN_ROOT/${stamp}_floor${floor}_${target}"
mkdir -p "$run_dir"

run_one() {
    local mode=$1
    local wrapper
    local case_dir="$run_dir/floor${floor}_${mode}"
    mkdir -p "$case_dir"

    case "$mode" in
        mode3) wrapper="$PATCH_TREE/run_mode3_perf_clean_test.sh" ;;
        mode4) wrapper="$PATCH_TREE/run_mode4_perf_clean_test.sh" ;;
        mode5) wrapper="$PATCH_TREE/run_mode5_perf_clean_test.sh" ;;
        *) echo "bad mode: $mode" >&2; exit 2 ;;
    esac
    if [[ ! -x "$wrapper" ]]; then
        echo "wrapper not executable: $wrapper" >&2
        exit 2
    fi

    local mode_offset=0
    [[ "$mode" == "mode4" ]] && mode_offset=100
    [[ "$mode" == "mode5" ]] && mode_offset=200

    cat > "$case_dir/diag_env.sh" <<EOF_ENV
export VLLM_ASCEND_DISPATCH_V2_DIAG_LOG=1
export VLLM_ASCEND_DISPATCH_V2_DIAG_EP_WORLD_SIZE=$floor
export VLLM_ASCEND_DISPATCH_V2_DIAG_FIRST_N=12
export VLLM_ASCEND_DISPATCH_V2_DIAG_ALL=0
export VLLM_ASCEND_MODE3_DISPATCH_BIND_DIAG_LOG=1
export VLLM_ASCEND_MODE3_DISPATCH_BIND_DIAG_STAGE=$floor
export VLLM_ASCEND_MODE3_DISPATCH_BIND_DIAG_FIRST_N=12
export VLLM_ASCEND_MODE3_DISPATCH_BIND_DIAG_ALL=0
export VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG=1
export VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG_ALL=0
EOF_ENV

    echo "[dispatch-mem-diag] mode=$mode floor=$floor case_dir=$case_dir"
    (
        cd "$PATCH_TREE"
        source "$case_dir/diag_env.sh"
        export REPO_ROOT="$case_dir"
        export PATCH_TREE="$PATCH_TREE"
        export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-$((49041 + mode_offset))}"
        export MASTER_PORT="${MASTER_PORT:-$((29040 + mode_offset))}"
        export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-$((49041 + mode_offset))}"
        echo "[dispatch-mem-diag] wrapper=$wrapper"
        echo "[dispatch-mem-diag] HCCL_IF_BASE_PORT=$HCCL_IF_BASE_PORT MASTER_PORT=$MASTER_PORT VERL_HCCL_IF_BASE_PORT_START=$VERL_HCCL_IF_BASE_PORT_START"
        bash "$wrapper" "$floor"
    ) 2>&1 | tee "$case_dir/launcher.log"
}

overall_rc=0
case "$target" in
    mode3) run_one mode3 || overall_rc=$? ;;
    mode4) run_one mode4 || overall_rc=$? ;;
    mode5) run_one mode5 || overall_rc=$? ;;
    all)
        run_one mode3 || overall_rc=$?
        run_one mode4 || overall_rc=$?
        run_one mode5 || overall_rc=$?
        ;;
esac

cat <<EOF_DONE
[dispatch-mem-diag] done rc=$overall_rc
[dispatch-mem-diag] run_dir=$run_dir

After the runs, compare with:
  rg -n "GPU KV cache size|Mode[345].*dispatch bind diag|Mode5 hybrid refresh breakdown|MC2 DispatchV2 diag before|MC2 DispatchV2 diag after|MC2 DispatchV2 diag error|Memory_Allocation_Failure|HcclAllocComResourceByTiling|rollout_output_time" "$run_dir"
EOF_DONE
exit "$overall_rc"
