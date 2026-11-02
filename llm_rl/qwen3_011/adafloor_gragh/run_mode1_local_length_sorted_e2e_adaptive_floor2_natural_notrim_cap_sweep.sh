#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Sweep floor=2 natural-policy KV caps with reload allocator trimming disabled.
#
# A cap is considered usable only if it both completes the 5-step threshold
# workload and avoids the step2+ reload/post-process slowdown that trim caused.

CAP_CANDIDATES="${FLOOR2_CAP_CANDIDATES:-147456 139264 131072 122880 114688 106496 98304 90112 81920 73728 65536 57344 50000}"
BASE_OUTPUT_SUBDIR="${OUTPUT_SUBDIR_BASE:-mode1_length_sorted_e2e_adaptive_floor2_natural_notrim_cap_sweep}"
MAX_STEP2_PLUS_S="${MAX_STEP2_PLUS_S:-180}"
MAX_STEP2_PLUS_OVER_STEP1_S="${MAX_STEP2_PLUS_OVER_STEP1_S:-45}"

echo "[floor2 natural notrim cap sweep] candidates: ${CAP_CANDIDATES}"
echo "[floor2 natural notrim cap sweep] pass: complete 5 steps, no fatal errors, step2+ <= max(${MAX_STEP2_PLUS_S}s, step1+${MAX_STEP2_PLUS_OVER_STEP1_S}s)"

for cap in ${CAP_CANDIDATES}; do
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="$cap"
    export FLOOR_KV_CAPS="2:${cap},4:280576,8:377344,16:377344"
    export OUTPUT_SUBDIR="${BASE_OUTPUT_SUBDIR}_cap${cap}"
    export PLAN_DIR="$SCRIPT_DIR/$OUTPUT_SUBDIR/oracle"
    export VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR=0
    export VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM=0

    echo "[floor2 natural notrim cap sweep] trying cap=${cap}, output=${OUTPUT_SUBDIR}"
    run_rc=0
    "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2_natural_forcefloor2_cap50000_notrim.sh" "$@" || run_rc=$?

    latest_log=$(ls -1t "$SCRIPT_DIR/$OUTPUT_SUBDIR"/logs/*.txt 2>/dev/null | head -1 || true)
    if [[ -z "$latest_log" ]]; then
        echo "[floor2 natural notrim cap sweep] FAIL cap=${cap}: no log found (run_rc=${run_rc})"
        continue
    fi

    if python3 - "$latest_log" "$MAX_STEP2_PLUS_S" "$MAX_STEP2_PLUS_OVER_STEP1_S" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
max_step2_plus = float(sys.argv[2])
max_over_step1 = float(sys.argv[3])
text = log_path.read_text(errors="ignore")

fatal = re.search(r"Traceback|RuntimeError|OOM|Memory_Allocation|Preempting|ERR\d+|Exception", text)
times = [float(x) for x in re.findall(r"rollout_output_time_s[:=]\s*([0-9.]+)", text)]
if len(times) < 5:
    print(f"FAIL log={log_path.name} completed_steps={len(times)} times={times}")
    sys.exit(1)
if fatal:
    print(f"FAIL log={log_path.name} fatal={fatal.group(0)} times={times[:5]}")
    sys.exit(1)

limit = max(max_step2_plus, times[0] + max_over_step1)
slow = max(times[1:5]) > limit
print(
    f"log={log_path.name} times={','.join(f'{t:.2f}' for t in times[:5])} "
    f"limit={limit:.2f} max_step2_plus={max(times[1:5]):.2f}"
)
if slow:
    sys.exit(2)
PY
    then
        echo "[floor2 natural notrim cap sweep] PASS cap=${cap}"
        echo "[floor2 natural notrim cap sweep] best candidate in this grid: ${cap}"
        exit 0
    fi

    echo "[floor2 natural notrim cap sweep] FAIL cap=${cap} (run_rc=${run_rc})"
done

echo "[floor2 natural notrim cap sweep] no candidate cap passed: ${CAP_CANDIDATES}" >&2
exit 1
