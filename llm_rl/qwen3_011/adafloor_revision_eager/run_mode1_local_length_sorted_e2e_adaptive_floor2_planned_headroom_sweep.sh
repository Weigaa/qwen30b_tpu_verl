#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Sweep planned floor-group KV headroom while keeping the planner floor KV
# caps fixed.  Headroom is applied only by the runtime KV sizing/resizing path;
# it must not change prompt-to-step or prompt-to-rank planning.

HEADROOM_CANDIDATES="${FLOOR2_HEADROOM_CANDIDATES:-0 16384 32768 49152 65536 81920 98304 114688 122880 131072 139264}"
FLOOR2_NOMINAL_CAP="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2:-147456}"
FLOOR4_NOMINAL_CAP="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4:-280576}"
FLOOR8_NOMINAL_CAP="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8:-377344}"
FLOOR16_NOMINAL_CAP="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS:-377344}"
FLOOR4_HEADROOM="${VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR4:-147456}"
FLOOR8_HEADROOM="${VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8:-114688}"
FLOOR16_HEADROOM="${VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR16:-0}"
BASE_OUTPUT_SUBDIR="${OUTPUT_SUBDIR_BASE:-mode1_length_sorted_e2e_adaptive_floor2_planned_headroom_sweep}"

echo "[floor2 planned headroom sweep] nominal_floor2_cap=${FLOOR2_NOMINAL_CAP}"
echo "[floor2 planned headroom sweep] headroom candidates: ${HEADROOM_CANDIDATES}"
echo "[floor2 planned headroom sweep] pass: complete 5 forced-floor2 threshold steps, no fatal errors/preemptions"

for headroom in ${HEADROOM_CANDIDATES}; do
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="$FLOOR16_NOMINAL_CAP"
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="$FLOOR2_NOMINAL_CAP"
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="$FLOOR4_NOMINAL_CAP"
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8="$FLOOR8_NOMINAL_CAP"
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS="$headroom"
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR2="$headroom"
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR4="$FLOOR4_HEADROOM"
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8="$FLOOR8_HEADROOM"
    export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR16="$FLOOR16_HEADROOM"
    export VLLM_ASCEND_MODE1_STEP_KV_CAP_INCLUDES_PLANNED_HEADROOM=0
    export FLOOR_KV_CAPS="2:${FLOOR2_NOMINAL_CAP},4:${FLOOR4_NOMINAL_CAP},8:${FLOOR8_NOMINAL_CAP},16:${FLOOR16_NOMINAL_CAP}"
    export OUTPUT_SUBDIR="${BASE_OUTPUT_SUBDIR}_headroom${headroom}"
    export PLAN_DIR="$SCRIPT_DIR/$OUTPUT_SUBDIR/oracle"
    export VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR=0
    export VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM=0

    echo "[floor2 planned headroom sweep] trying headroom=${headroom}, planner_floor_caps=${FLOOR_KV_CAPS}, output=${OUTPUT_SUBDIR}"
    run_rc=0
    "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2_planned_threshold.sh" "$@" || run_rc=$?

    latest_log=$(ls -1t "$SCRIPT_DIR/$OUTPUT_SUBDIR"/logs/*.txt 2>/dev/null | head -1 || true)
    if [[ -z "$latest_log" ]]; then
        echo "[floor2 planned headroom sweep] FAIL headroom=${headroom}: no log found (run_rc=${run_rc})"
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
steps = {
    int(step)
    for step in re.findall(r"training/global_step:(\d+)", text)
}
times = [
    float(x)
    for x in re.findall(
        r"(?:rollout_output_time_s|timing_s/gen)[:=]\s*([0-9.]+)",
        text,
    )
]
if len(steps) < 5:
    print(
        f"FAIL log={log_path.name} completed_steps={sorted(steps)} "
        f"times={','.join(f'{t:.2f}' for t in times[:5])}"
    )
    sys.exit(1)
if fatal:
    print(
        f"FAIL log={log_path.name} fatal={fatal.group(0)} "
        f"steps={sorted(steps)} times={','.join(f'{t:.2f}' for t in times[:5])}"
    )
    sys.exit(1)

print(
    f"log={log_path.name} completed_steps={sorted(steps)[:5]} "
    f"times={','.join(f'{t:.2f}' for t in times[:5])}"
)
PY
    then
        echo "[floor2 planned headroom sweep] PASS headroom=${headroom}"
        echo "[floor2 planned headroom sweep] minimal passing headroom in this grid: ${headroom}"
        exit 0
    fi

    echo "[floor2 planned headroom sweep] FAIL headroom=${headroom} (run_rc=${run_rc})"
done

echo "[floor2 planned headroom sweep] no candidate headroom passed: ${HEADROOM_CANDIDATES}" >&2
exit 1
