#!/usr/bin/env bash
set -euo pipefail

# Compare prespecified power-of-two decode capture sets. The validated [32]
# and [16,32] arms are reused, so this runner launches only three new arms.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ACTION=${1:-plan}
case "$ACTION" in
    plan|run|summarize) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_aclgraph_power_of_two_bucket_profile.sh plan
  ./run_qwen3_aclgraph_power_of_two_bucket_profile.sh run
  ./run_qwen3_aclgraph_power_of_two_bucket_profile.sh summarize

Reuses the validated [32] and [16,32] results, then runs only:
  [8,16,32], [2,4,8,16,32], [4,8,16,32]
The launch order is k=3,5,4. All arms use the same frozen checkpoint,
step-4 dataset, fixed token trace, graph runtime, and measurement protocol.
EOF
        exit 0
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac

BASE_RUNNER="$SCRIPT_DIR/run_qwen3_aclgraph_step_bucket_profile.sh"
COMBINER="$SCRIPT_DIR/analysis_eval/combine_aclgraph_power_of_two_profile.py"
RESERVE_TOOL="$SCRIPT_DIR/analysis_eval/derive_aclgraph_reserve.py"
PROFILE="$SCRIPT_DIR/analysis_eval/qwen3_aclgraph_step_bucket_profile_all32.json"
ROOT=${ACLGRAPH_POWER2_PROFILE_ROOT:-/workspace/adafloor_graph_results/qwen3_aclgraph_power2_profile_seed0_step4_v1}
REFERENCE_ROOT=${ACLGRAPH_POWER2_REFERENCE_ROOT:-/workspace/adafloor_graph_results/qwen3_aclgraph_count_grid_profile_derived_seed0_step4_v1}
COUNTS=3,4,5
LAUNCH_ORDER=3,5,4
CAPTURE_SETS='{"3":[8,16,32],"4":[4,8,16,32],"5":[2,4,8,16,32]}'

for path in "$BASE_RUNNER" "$COMBINER" "$RESERVE_TOOL" "$PROFILE" \
        "$REFERENCE_ROOT/profile_summary.json" "$REFERENCE_ROOT/protocol.env"; do
    [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 2; }
done
python3 - "$REFERENCE_ROOT/profile_summary.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "PASS":
    raise SystemExit("reference count-grid profile is not PASS")
required = {(32,), (16, 32)}
present = {tuple(arm["capture_sizes"]) for arm in payload["arms"]}
if not required <= present:
    raise SystemExit("reference profile lacks [32] or [16,32]")
PY

base_runner() {
    env \
        ACLGRAPH_BUCKET_PROFILE_ROOT="$ROOT" \
        ACLGRAPH_BUCKET_PROFILE_COUNTS="$COUNTS" \
        ACLGRAPH_BUCKET_PROFILE_LAUNCH_ORDER="$LAUNCH_ORDER" \
        ACLGRAPH_BUCKET_PROFILE_SOURCE_STEP=4 \
        ACLGRAPH_BUCKET_PROFILE_RESERVE_MODE=profile_derived \
        ACLGRAPH_BUCKET_PROFILE_EXPLICIT_CAPTURE_SETS="$CAPTURE_SETS" \
        ACLGRAPH_BUCKET_PROFILE_EXTRA_CODE_CONTRACT="$SCRIPT_DIR/run_qwen3_aclgraph_power_of_two_bucket_profile.sh" \
        "$BASE_RUNNER" "$@"
}

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

combine_results() {
    python3 "$COMBINER" \
        --reference-root "$REFERENCE_ROOT" \
        --experiment-root "$ROOT" \
        --output-json "$ROOT/power_of_two_summary.json" \
        --output-markdown "$ROOT/power_of_two_summary.md"
}

base_runner plan >/dev/null
bootstrap_arm_protocols

if [[ "$ACTION" == plan ]]; then
    cat "$ROOT/profile_plan.json"
    exit 0
fi
if [[ "$ACTION" == summarize ]]; then
    base_runner summarize >/dev/null
    combine_results
    exit 0
fi

base_runner run
combine_results
