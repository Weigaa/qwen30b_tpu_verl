#!/usr/bin/env bash
set -euo pipefail

# Counterbalanced paper protocol for causal FULL_DECODE_ONLY versus matched
# eager comparison. Each invocation of the underlying runner executes exactly
# one arm. Graph retains the complete 21-arm matrix while matched eager runs
# only the nine prespecified representative controls.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ACTION=${1:-run}
case "$ACTION" in
    run|verify) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_paper_full_graph_vs_eager_paired.sh run
  ./run_qwen3_paper_full_graph_vs_eager_paired.sh verify

The run is resumable at completed-arm boundaries. Set PAIR_RESULT_ROOT to use
another fresh result root.
EOF
        exit 0
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac

PAIR_ROOT=${PAIR_RESULT_ROOT:-/workspace/adafloor_graph_results/qwen3_paper_full_graph_vs_eager_prompt_kv_admission_v4_minimal_eager}
GRAPH_ROOT="$PAIR_ROOT/graph"
EAGER_ROOT="$PAIR_ROOT/eager_matched"
PAIR_MANIFEST="$PAIR_ROOT/paired_launch_manifest.tsv"
PAIR_PROTOCOL="$PAIR_ROOT/paired_protocol.env"
PAIR_CODE="$PAIR_ROOT/paired_code_sha256.txt"
PAIR_ANALYSIS="$PAIR_ROOT/analysis_eval"

BASE_RUNNER="$SCRIPT_DIR/run_qwen3_paper_graph_main_from_common_epoch0.sh"
EAGER_RUNNER="$SCRIPT_DIR/run_qwen3_paper_matched_eager_tq1_from_common_epoch0.sh"
COMPARATOR="$SCRIPT_DIR/analysis_eval/compare_qwen3_paper_graph_vs_matched_eager.py"
SEEDS=(0 2 3)
GRAPH_POLICIES=(vanilla lengthsort lengthsort_guard minskew natural_f4 natural_f2 planned_f4)
MATCHED_EAGER_POLICIES=(vanilla natural_f2 planned_f4)

is_matched_eager_policy() {
    case "$1" in
        vanilla|natural_f2|planned_f4) return 0 ;;
        *) return 1 ;;
    esac
}

install_or_verify() {
    local candidate=$1 destination=$2 label=$3
    if [[ -e "$destination" ]]; then
        if [[ ! -f "$destination" || -L "$destination" ]] \
                || ! cmp -s -- "$candidate" "$destination"; then
            rm -f -- "$candidate"
            echo "immutable paired $label changed: $destination" >&2
            return 1
        fi
        rm -f -- "$candidate"
    else
        mv -- "$candidate" "$destination"
    fi
}

seal_pair_contract() {
    mkdir -p "$PAIR_ROOT"
    local code_candidate protocol_candidate created
    code_candidate=$(mktemp "$PAIR_ROOT/.paired-code.XXXXXX")
    protocol_candidate=$(mktemp "$PAIR_ROOT/.paired-protocol.XXXXXX")
    sha256sum \
        "$SCRIPT_DIR/run_qwen3_paper_full_graph_vs_eager_paired.sh" \
        "$BASE_RUNNER" "$EAGER_RUNNER" "$COMPARATOR" \
        > "$code_candidate"
    if [[ -f "$PAIR_PROTOCOL" && ! -L "$PAIR_PROTOCOL" ]]; then
        created=$(sed -n 's/^created_at_utc=//p' "$PAIR_PROTOCOL")
    else
        created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    fi
    {
        printf 'schema_version=1\n'
        printf 'created_at_utc=%s\n' "$created"
        printf 'experiment=qwen3_full_decode_only_vs_matched_eager_tq1\n'
        printf 'seeds=0,2,3\n'
        printf 'graph_policies=vanilla,lengthsort,lengthsort_guard,minskew,natural_f4,natural_f2,planned_f4\n'
        printf 'graph_epochs=1,2\n'
        printf 'matched_eager_policies=vanilla,natural_f2,planned_f4\n'
        printf 'matched_eager_epochs=1\n'
        printf 'graph_arms=21\n'
        printf 'matched_eager_arms=9\n'
        printf 'steps_per_epoch=5\n'
        printf 'counterbalanced_order=seed0_graph_eager,seed2_eager_graph,seed3_graph_eager\n'
        printf 'paired_difference=aclgraph_execution_only\n'
        printf 'paired_kv_caps=identical_graph_safe_caps\n'
        printf 'kv_admission=block_aware_prompt_plus_predicted_response_peak\n'
        printf 'kv_admission_safety_factor=1.16\n'
        printf 'kv_admission_extra_reserve_tokens=0\n'
        printf 'graph_root=%s\n' "$GRAPH_ROOT"
        printf 'eager_root=%s\n' "$EAGER_ROOT"
    } > "$protocol_candidate"
    install_or_verify "$code_candidate" "$PAIR_CODE" code
    install_or_verify "$protocol_candidate" "$PAIR_PROTOCOL" protocol
}

initialize_manifest() {
    if [[ ! -e "$PAIR_MANIFEST" ]]; then
        printf 'launch_index\ttrial_seed\tpolicy\texecution_mode\tstatus\n' \
            > "$PAIR_MANIFEST"
    fi
    [[ $(head -n 1 "$PAIR_MANIFEST") == \
        $'launch_index\ttrial_seed\tpolicy\texecution_mode\tstatus' ]] || {
        echo "invalid paired manifest header: $PAIR_MANIFEST" >&2
        exit 2
    }
}

pair_arm_complete() {
    local seed=$1 policy=$2 mode=$3
    awk -F '\t' -v s="$seed" -v p="$policy" -v m="$mode" \
        'NR>1 && $2==s && $3==p && $4==m && $5=="complete" {ok=1} END {exit !ok}' \
        "$PAIR_MANIFEST"
}

record_pair_arm() {
    local index=$1 seed=$2 policy=$3 mode=$4
    pair_arm_complete "$seed" "$policy" "$mode" && return 0
    printf '%s\t%s\t%s\t%s\tcomplete\n' \
        "$index" "$seed" "$policy" "$mode" >> "$PAIR_MANIFEST"
}

run_one() {
    local index=$1 seed=$2 policy=$3 mode=$4
    if pair_arm_complete "$seed" "$policy" "$mode"; then
        echo "[paired graph/eager] already complete index=$index seed=$seed policy=$policy mode=$mode"
        return 0
    fi
    seal_pair_contract
    if [[ "$mode" == graph ]]; then
        GRAPH_MAIN_ROOT="$GRAPH_ROOT" \
        GRAPH_MAIN_REUSE_SEED0_VANILLA=1 \
        GRAPH_MAIN_ONLY_ARM="$seed:$policy" \
            "$BASE_RUNNER" run
    else
        GRAPH_MAIN_ROOT="$EAGER_ROOT" \
        GRAPH_MAIN_ONLY_ARM="$seed:$policy" \
            "$EAGER_RUNNER" run
    fi
    seal_pair_contract
    record_pair_arm "$index" "$seed" "$policy" "$mode"
}

finalize_pair() {
    GRAPH_MAIN_ROOT="$GRAPH_ROOT" GRAPH_MAIN_REUSE_SEED0_VANILLA=1 \
        "$BASE_RUNNER" verify
    GRAPH_MAIN_ROOT="$EAGER_ROOT" "$EAGER_RUNNER" verify
    python3 "$COMPARATOR" \
        --graph-summary "$GRAPH_ROOT/analysis_eval/summary.json" \
        --eager-summary "$EAGER_ROOT/analysis_eval/summary.json" \
        --graph-protocol "$GRAPH_ROOT/protocol.env" \
        --eager-protocol "$EAGER_ROOT/protocol.env" \
        --graph-code-manifest "$GRAPH_ROOT/code_sha256.txt" \
        --eager-code-manifest "$EAGER_ROOT/code_sha256.txt" \
        --graph-pass "$GRAPH_ROOT/GRAPH_MAIN_PASS.txt" \
        --eager-pass "$EAGER_ROOT/GRAPH_MAIN_PASS.txt" \
        --output-dir "$PAIR_ANALYSIS"
    printf 'status=PASS\ngraph_arms=21\nmatched_eager_arms=9\npaired_controls=9\ncomparison_sha256=%s\n' \
        "$(sha256sum "$PAIR_ANALYSIS/full_graph_vs_matched_eager.json" | awk '{print $1}')" \
        > "$PAIR_ROOT/PAIRED_GRAPH_EAGER_PASS.txt"
    echo "[paired graph/eager] PASS result=$PAIR_ANALYSIS/full_graph_vs_matched_eager.md"
}

seal_pair_contract
initialize_manifest

if [[ "$ACTION" == verify ]]; then
    finalize_pair
    exit 0
fi

launch_index=0
for seed in "${SEEDS[@]}"; do
    for policy in "${GRAPH_POLICIES[@]}"; do
        if ! is_matched_eager_policy "$policy"; then
            order=(graph)
        elif [[ "$seed" == 2 ]]; then
            order=(eager_matched graph)
        else
            order=(graph eager_matched)
        fi
        for mode in "${order[@]}"; do
            launch_index=$((launch_index + 1))
            run_one "$launch_index" "$seed" "$policy" "$mode"
        done
    done
done
finalize_pair
