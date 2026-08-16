#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Sweep planned floor-group headroom without imposing a runtime KV-token cap.
# Runtime KV size is measured by vLLM from available memory after planned
# floor groups are precreated/warmed. Planner caps stay fixed only to preserve
# the short threshold-test prompt/rank assignment.

HEADROOM_CANDIDATES="${FLOOR2_HEADROOM_CANDIDATES:-0 16384 32768 49152 65536 81920 98304 114688 122880 131072 139264 147456}"
PLANNER_KV_CAP_FLOOR2="${PLANNER_KV_CAP_FLOOR2:-147456}"
PLANNER_KV_CAP_FLOOR4="${PLANNER_KV_CAP_FLOOR4:-280576}"
PLANNER_KV_CAP_FLOOR8="${PLANNER_KV_CAP_FLOOR8:-377344}"
PLANNER_KV_CAP_FLOOR16="${PLANNER_KV_CAP_FLOOR16:-377344}"
FLOOR4_HEADROOM="${VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR4:-147456}"
FLOOR8_HEADROOM="${VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8:-114688}"
FLOOR16_HEADROOM="${VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR16:-0}"
BASE_OUTPUT_SUBDIR="${OUTPUT_SUBDIR_BASE:-mode1_length_sorted_e2e_adaptive_floor2_planned_auto_kv_headroom_sweep}"

echo "[floor2 planned auto-kv headroom sweep] runtime KV caps disabled; vLLM will auto-size KV"
echo "[floor2 planned auto-kv headroom sweep] planner caps: 2:${PLANNER_KV_CAP_FLOOR2},4:${PLANNER_KV_CAP_FLOOR4},8:${PLANNER_KV_CAP_FLOOR8},16:${PLANNER_KV_CAP_FLOOR16}"
echo "[floor2 planned auto-kv headroom sweep] headroom candidates: ${HEADROOM_CANDIDATES}"

for headroom in ${HEADROOM_CANDIDATES}; do
    export VLLM_ASCEND_MODE1_AUTO_KV_SIZE_HEADROOM_ONLY=1
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=0
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2=0
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=0
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=0
    export PLANNER_KV_CAP_FLOOR2
    export PLANNER_KV_CAP_FLOOR4
    export PLANNER_KV_CAP_FLOOR8
    export PLANNER_KV_CAP_FLOOR16
    export FLOOR_KV_CAPS="2:${PLANNER_KV_CAP_FLOOR2},4:${PLANNER_KV_CAP_FLOOR4},8:${PLANNER_KV_CAP_FLOOR8},16:${PLANNER_KV_CAP_FLOOR16}"
    export MAX_RANK_PEAK_TOKENS="${MAX_RANK_PEAK_TOKENS:-$PLANNER_KV_CAP_FLOOR16}"
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS="$headroom"
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR2="$headroom"
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR4="$FLOOR4_HEADROOM"
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8="$FLOOR8_HEADROOM"
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR16="$FLOOR16_HEADROOM"
    export VLLM_ASCEND_MODE1_STEP_KV_CAP_INCLUDES_PLANNED_HEADROOM=0
    export OUTPUT_SUBDIR="${BASE_OUTPUT_SUBDIR}_headroom${headroom}"
    export PLAN_DIR="$SCRIPT_DIR/$OUTPUT_SUBDIR/oracle"
    export VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR=0
    export VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM=0

    echo "[floor2 planned auto-kv headroom sweep] trying headroom=${headroom}, output=${OUTPUT_SUBDIR}"
    run_rc=0
    "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2_planned_threshold.sh" "$@" || run_rc=$?

    latest_log=$(ls -1t "$SCRIPT_DIR/$OUTPUT_SUBDIR"/logs/*.txt 2>/dev/null | head -1 || true)
    if [[ -z "$latest_log" ]]; then
        echo "[floor2 planned auto-kv headroom sweep] FAIL headroom=${headroom}: no log found (run_rc=${run_rc})"
        continue
    fi

    if python3 - "$latest_log" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
text = log_path.read_text(errors="ignore")
fatal = re.search(
    r"RuntimeError|Memory_Allocation_Failure|Failed to allocate|"
    r"Preempting|preempting|child run failed|exit_code=1|ERR00100|"
    r"OutOfMemory|\bOOM\b",
    text,
)
steps = {int(step) for step in re.findall(r"training/global_step:(\d+)", text)}
kv_sizes = re.findall(r"(?:GPU|NPU)?\s*KV cache size:\s*([0-9,]+)\s*tokens", text)
times = [
    float(x)
    for x in re.findall(
        r"(?:rollout_output_time_s|timing_s/gen)[:=]\s*([0-9.]+)",
        text,
    )
]
summary = (
    f"log={log_path.name} completed_steps={sorted(steps)} "
    f"kv_sizes={kv_sizes[-4:]} times={','.join(f'{t:.2f}' for t in times[:5])}"
)
if len(steps) < 5:
    print(f"FAIL {summary}")
    sys.exit(1)
if fatal:
    print(f"FAIL fatal={fatal.group(0)} {summary}")
    sys.exit(1)
print(summary)
PY
    then
        echo "[floor2 planned auto-kv headroom sweep] PASS headroom=${headroom}"
        echo "[floor2 planned auto-kv headroom sweep] minimal passing headroom in this grid: ${headroom}"
        exit 0
    fi

    echo "[floor2 planned auto-kv headroom sweep] FAIL headroom=${headroom} (run_rc=${run_rc})"
done

echo "[floor2 planned auto-kv headroom sweep] no candidate headroom passed: ${HEADROOM_CANDIDATES}" >&2
exit 1
