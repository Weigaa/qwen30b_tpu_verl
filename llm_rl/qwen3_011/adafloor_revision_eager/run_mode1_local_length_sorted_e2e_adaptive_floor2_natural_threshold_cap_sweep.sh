#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Find the largest practical floor2 KV cap for the short natural-policy
# 16 -> 8 -> 4 -> 2 smoke test. The 147456-token cap has already been observed
# to fail after step2's 16 -> 8 warmup because HCCL could not allocate a small
# allreduce workspace. This sweep tries lower caps from high to low and stops at
# the first successful run.

CAP_CANDIDATES="${FLOOR2_CAP_CANDIDATES:-131072 114688 98304 81920 65536}"
BASE_OUTPUT_SUBDIR="${OUTPUT_SUBDIR_BASE:-mode1_length_sorted_e2e_adaptive_floor2_natural_threshold_cap_sweep}"

echo "[floor2 natural cap sweep] candidates: ${CAP_CANDIDATES}"

for cap in ${CAP_CANDIDATES}; do
    export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="$cap"
    export OUTPUT_SUBDIR="${BASE_OUTPUT_SUBDIR}_cap${cap}"
    export PLAN_DIR="$SCRIPT_DIR/$OUTPUT_SUBDIR/oracle"

    echo "[floor2 natural cap sweep] trying floor2 cap=${cap}, output=${OUTPUT_SUBDIR}"
    if "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2_natural_threshold.sh" "$@"; then
        echo "[floor2 natural cap sweep] PASS floor2 cap=${cap}"
        echo "[floor2 natural cap sweep] best candidate in this grid: ${cap}"
        exit 0
    fi
    echo "[floor2 natural cap sweep] FAIL floor2 cap=${cap}"
done

echo "[floor2 natural cap sweep] no candidate cap passed: ${CAP_CANDIDATES}" >&2
exit 1
