#!/usr/bin/env bash
set -euo pipefail

# Six-point ACLGraph count search with count-specific HBM/KV budgets.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ACTION=${1:-plan}
case "$ACTION" in
    plan|run|summarize) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_aclgraph_count_grid_profile_derived.sh plan
  ./run_qwen3_aclgraph_count_grid_profile_derived.sh run
  ./run_qwen3_aclgraph_count_grid_profile_derived.sh summarize

Runs k={1,2,4,8,16,32} on one frozen Full16 step. Each k uses the exact
predicted minimum-padding capture set and a reserve derived from the measured
k=1..32 graph-memory profile. Fixed-work replay makes request identities,
response lengths, token IDs, reward, and KV trajectories identical.
EOF
        exit 0
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac

BASE_RUNNER="$SCRIPT_DIR/run_qwen3_aclgraph_step_bucket_profile.sh"
[[ -f "$BASE_RUNNER" ]] || { echo "missing base runner: $BASE_RUNNER" >&2; exit 2; }

ROOT=${ACLGRAPH_COUNT_GRID_ROOT:-/workspace/adafloor_graph_results/qwen3_aclgraph_count_grid_profile_derived_seed0_step4_v1}
COUNTS=1,2,4,8,16,32
LAUNCH_ORDER=1,32,2,16,4,8
PROFILE="$SCRIPT_DIR/analysis_eval/qwen3_aclgraph_step_bucket_profile_all32.json"
RESERVE_TOOL="$SCRIPT_DIR/analysis_eval/derive_aclgraph_reserve.py"

base_runner() {
    env \
        ACLGRAPH_BUCKET_PROFILE_ROOT="$ROOT" \
        ACLGRAPH_BUCKET_PROFILE_COUNTS="$COUNTS" \
        ACLGRAPH_BUCKET_PROFILE_LAUNCH_ORDER="$LAUNCH_ORDER" \
        ACLGRAPH_BUCKET_PROFILE_SOURCE_STEP=4 \
        ACLGRAPH_BUCKET_PROFILE_RESERVE_MODE=profile_derived \
        "$BASE_RUNNER" "$@"
}

# The incremental summarizer expects every candidate's deterministic protocol
# before it scans completed arms. Materialize those small contracts up front so
# an absent, not-yet-launched arm is treated as pending rather than malformed.
bootstrap_arm_protocols() {
    local count sizes measured reserve_bytes raw_bytes minimum_bytes
    local block_bytes blocks tokens full16 arm_dir candidate
    block_bytes=$((128 * 98304))
    IFS=',' read -r -a grid <<< "$COUNTS"
    for count in "${grid[@]}"; do
        sizes=$(python3 - "$ROOT/profile_plan.json" "$count" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
count = int(sys.argv[2])
for item in plan["candidates"]:
    if int(item["graph_count"]) == count:
        print(json.dumps(item["capture_sizes"], separators=(",", ":")))
        break
else:
    raise SystemExit(f"missing graph count {count}")
PY
)
        read -r measured reserve_bytes raw_bytes minimum_bytes < <(
            python3 "$RESERVE_TOOL" --profile "$PROFILE" \
                --graph-count "$count" --minimum-reserve-bytes 2684354560 \
                --quantum-bytes 536870912 --format tsv
        )
        blocks=$(((reserve_bytes + block_bytes - 1) / block_bytes))
        tokens=$((blocks * 128))
        full16=$((380800 - tokens))
        arm_dir="$ROOT/arms/$(printf 'count_%02d' "$count")"
        mkdir -p "$arm_dir"
        candidate=$(mktemp "$arm_dir/.arm_protocol.XXXXXX")
        {
            printf 'graph_count=%s\n' "$count"
            printf 'capture_sizes=%s\n' "$sizes"
            printf 'graph_reserve_mode=profile_derived\n'
            printf 'graph_reserve_bytes=%s\n' "$reserve_bytes"
            printf 'graph_reserve_blocks=%s\n' "$blocks"
            printf 'graph_reserve_tokens=%s\n' "$tokens"
            printf 'full16_kv_tokens=%s\n' "$full16"
            printf 'profile_capture_memory_gib_rank_max=%s\n' "$measured"
            printf 'profile_minimum_reserve_bytes=%s\n' "$minimum_bytes"
            printf 'profile_raw_capture_bytes=%s\n' "$raw_bytes"
        } > "$candidate"
        if [[ -f "$arm_dir/arm_protocol.env" ]]; then
            cmp -s "$candidate" "$arm_dir/arm_protocol.env" || {
                rm -f "$candidate"
                echo "immutable arm protocol changed: $arm_dir/arm_protocol.env" >&2
                exit 2
            }
            rm -f "$candidate"
        else
            mv "$candidate" "$arm_dir/arm_protocol.env"
            chmod a-w "$arm_dir/arm_protocol.env"
        fi
    done
}

if [[ "$ACTION" == plan ]]; then
    base_runner plan
    bootstrap_arm_protocols
    exit 0
fi

base_runner plan >/dev/null
bootstrap_arm_protocols
exec env \
    ACLGRAPH_BUCKET_PROFILE_ROOT="$ROOT" \
    ACLGRAPH_BUCKET_PROFILE_COUNTS="$COUNTS" \
    ACLGRAPH_BUCKET_PROFILE_LAUNCH_ORDER="$LAUNCH_ORDER" \
    ACLGRAPH_BUCKET_PROFILE_SOURCE_STEP=4 \
    ACLGRAPH_BUCKET_PROFILE_RESERVE_MODE=profile_derived \
    "$BASE_RUNNER" "$ACTION"
