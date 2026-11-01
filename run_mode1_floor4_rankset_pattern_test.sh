#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  FLOOR4_PATTERN=disjoint bash run_mode1_floor4_rankset_pattern_test.sh
  FLOOR4_PATTERN=overlap  bash run_mode1_floor4_rankset_pattern_test.sh
  FLOOR4_PATTERN=repeat   bash run_mode1_floor4_rankset_pattern_test.sh
  FLOOR4_PATTERN=sliding  bash run_mode1_floor4_rankset_pattern_test.sh
  FLOOR4_PATTERN=all FLOOR4_MAX_GROUPS=8 bash run_mode1_floor4_rankset_pattern_test.sh

Purpose:
  Run the same mode=1 floor=4 live-pool test while changing only the sequence
  of explicit 4-rank survivor groups. This isolates whether the HCCL/MC2 delay
  follows repeated groups, partially-overlapping groups, or any 4-rank group.

Overrides:
  FLOOR4_RANK_SETS='12-15;10-13;8-11'
  FLOOR4_LIVE_POOL_RANK_SETS='12-15;10-13'
  FLOOR4_DISABLE_LIVE_POOL=1
  BASELINE_TOTAL_TRAINING_STEPS=5
  VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=280576
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATTERN="${FLOOR4_PATTERN:-overlap}"

case "$PATTERN" in
    disjoint)
        DEFAULT_RANK_SETS="12-15;8-11;4-7;0-3"
        ;;
    overlap)
        DEFAULT_RANK_SETS="12-15;10-13;8-11;6-9"
        ;;
    repeat)
        DEFAULT_RANK_SETS="12-15;12-15;12-15;12-15"
        ;;
    sliding)
        DEFAULT_RANK_SETS=$(python3 "$SCRIPT_DIR/generate_floor4_livepool_rank_sets.py" \
            --mode sliding --step "${FLOOR4_SLIDING_STEP:-2}" \
            --max-groups "${FLOOR4_MAX_GROUPS:-4}" | sed -n '1p')
        ;;
    all)
        # "all" is intentionally capped unless FLOOR4_MAX_GROUPS is raised.
        # A 16-rank world has 1820 floor4 combinations, which is not safe to
        # keep resident in a full training process.
        DEFAULT_RANK_SETS=$(python3 "$SCRIPT_DIR/generate_floor4_livepool_rank_sets.py" \
            --mode all --max-groups "${FLOOR4_MAX_GROUPS:-8}" | sed -n '1p')
        ;;
    custom)
        if [[ -z "${FLOOR4_RANK_SETS:-}" ]]; then
            echo "FLOOR4_PATTERN=custom requires FLOOR4_RANK_SETS." >&2
            exit 2
        fi
        DEFAULT_RANK_SETS="$FLOOR4_RANK_SETS"
        ;;
    *)
        echo "Unknown FLOOR4_PATTERN=$PATTERN" >&2
        usage >&2
        exit 2
        ;;
esac

RANK_SETS="${FLOOR4_RANK_SETS:-$DEFAULT_RANK_SETS}"
ROTATE_MODES=$(printf '%s\n' "$RANK_SETS" | awk -F';' '
{
    for (i = 1; i <= NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i)
        if ($i != "") {
            printf "%sfloor4:%s", (n++ ? "," : ""), $i
        }
    }
}')
GROUP_COUNT=$(printf '%s\n' "$RANK_SETS" | awk -F';' '
{
    n = 0
    for (i = 1; i <= NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i)
        if ($i != "") n++
    }
    print n
}')

if [[ -z "$ROTATE_MODES" || "$GROUP_COUNT" -le 0 ]]; then
    echo "No floor4 rank sets produced for pattern=$PATTERN." >&2
    exit 2
fi

export VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES="${VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES:-$ROTATE_MODES}"
if [[ "${FLOOR4_DISABLE_LIVE_POOL:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    LIVE_POOL_RANK_SETS_DEFAULT=""
else
    LIVE_POOL_RANK_SETS_DEFAULT="${FLOOR4_LIVE_POOL_RANK_SETS:-$RANK_SETS}"
fi
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS:-$LIVE_POOL_RANK_SETS_DEFAULT}"
export BASELINE_TOTAL_TRAINING_STEPS="${BASELINE_TOTAL_TRAINING_STEPS:-$GROUP_COUNT}"
# Placeholder and real active ranks may both keep a ProcessGroup wrapper.
LIVE_POOL_GROUP_COUNT=$(printf '%s\n' "$VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS" | awk -F';' '
{
    n = 0
    for (i = 1; i <= NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i)
        if ($i != "") n++
    }
    print n
}')
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_MAX_GROUPS="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_MAX_GROUPS:-$((LIVE_POOL_GROUP_COUNT * 2))}"
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_FALLBACK_ON_UNPOOLED="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_FALLBACK_ON_UNPOOLED:-0}"
if [[ "$VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_FALLBACK_ON_UNPOOLED" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_STRICT_RANK_SETS="${VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_STRICT_RANK_SETS:-0}"
fi
export RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/mode1_floor4_rankset_pattern_runs/${PATTERN}_$(date -u +%Y%m%dT%H%M%SZ)}"

printf '[mode1 floor4 rankset pattern] pattern=%s rank_sets=%s live_pool_rank_sets=%s rotate_modes=%s steps=%s run_root=%s\n' \
    "$PATTERN" \
    "$RANK_SETS" \
    "$VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS" \
    "$VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES" \
    "$BASELINE_TOTAL_TRAINING_STEPS" \
    "$RUN_ROOT"
printf '[mode1 floor4 rankset pattern] fallback_on_unpooled=%s\n' \
    "$VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_FALLBACK_ON_UNPOOLED"

exec bash "$SCRIPT_DIR/run_mode1_floor4_livepool_overlap4group_test.sh"
