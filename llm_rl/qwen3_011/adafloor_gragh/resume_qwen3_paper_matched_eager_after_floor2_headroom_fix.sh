#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULT_ROOT=${GRAPH_MAIN_ROOT:-/workspace/adafloor_graph_results/qwen3_paper_matched_eager_tq1_prompt_kv_admission_v4_minimal_controls}
FAILED_ARM_ROOT="$RESULT_ROOT/seed_0/planned_f4/qwen3_eager_matched_planned_f4_seed0"
CORRECTION="$RESULT_ROOT/PLANNED_F4_FLOOR2_HEADROOM_CORRECTION.env"
ACTION=${1:-run}

case "$ACTION" in
    dry-run|run|verify) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./resume_qwen3_paper_matched_eager_after_floor2_headroom_fix.sh run
  ./resume_qwen3_paper_matched_eager_after_floor2_headroom_fix.sh verify

Resumes the sealed matched-eager experiment after correcting the Planned-F4
preflight for unreachable floor 2. Completed Vanilla and Natural-F2 arms are
verified and skipped. This wrapper does not modify the sealed source manifest.
EOF
        exit 0
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac

CODE_MANIFEST="$RESULT_ROOT/code_sha256.txt"
PROTOCOL="$RESULT_ROOT/protocol.env"
RUN_MANIFEST="$RESULT_ROOT/run_manifest.tsv"
EXPECTED_CODE_MANIFEST_SHA256=a166d8f04e6e05796f3ae93787a0730501f319d4341d3a8f7702edfdc673c609
EXPECTED_PROTOCOL_SHA256=5b27b12ec0ca4f1f3317c0e91589acd38d483603ed9fda6f67ea70a64610c527

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

require_hash() {
    local path=$1 expected=$2 label=$3 actual
    [[ -f "$path" && ! -L "$path" ]] || {
        echo "missing sealed $label: $path" >&2
        exit 1
    }
    actual=$(sha256_file "$path")
    [[ "$actual" == "$expected" ]] || {
        echo "sealed $label hash mismatch: expected=$expected actual=$actual" >&2
        exit 1
    }
}

require_hash "$CODE_MANIFEST" "$EXPECTED_CODE_MANIFEST_SHA256" code-manifest
require_hash "$PROTOCOL" "$EXPECTED_PROTOCOL_SHA256" protocol
sha256sum -c "$CODE_MANIFEST" >/dev/null

expected_header=$'launch_index\ttrial_seed\tpolicy\tstatus\trun_dir'
expected_vanilla=$'1\t0\tvanilla\tcomplete\t'"$RESULT_ROOT/seed_0/vanilla/qwen3_eager_matched_vanilla_seed0"
expected_natural=$'2\t0\tnatural_f2\tcomplete\t'"$RESULT_ROOT/seed_0/natural_f2/qwen3_eager_matched_natural_f2_seed0"
expected_planned=$'3\t0\tplanned_f4\tcomplete\t'"$FAILED_ARM_ROOT"
[[ "$(sed -n '1p' "$RUN_MANIFEST")" == "$expected_header" ]] || {
    echo "unexpected run manifest header" >&2
    exit 1
}
[[ "$(sed -n '2p' "$RUN_MANIFEST")" == "$expected_vanilla" ]] || {
    echo "validated seed-0 Vanilla prefix is missing" >&2
    exit 1
}
[[ "$(sed -n '3p' "$RUN_MANIFEST")" == "$expected_natural" ]] || {
    echo "validated seed-0 Natural-F2 prefix is missing" >&2
    exit 1
}
if rg -n $'\t(running|failed|incomplete)\t' "$RUN_MANIFEST" >/dev/null; then
    echo "run manifest contains a non-resumable arm state" >&2
    exit 1
fi

# The failed preflight created only this empty run directory. Never remove an
# executed or partially populated arm.
if [[ -e "$FAILED_ARM_ROOT" ]] \
        && ! grep -Fqx -- "$expected_planned" "$RUN_MANIFEST"; then
    [[ -d "$FAILED_ARM_ROOT" && ! -L "$FAILED_ARM_ROOT" ]] || {
        echo "failed arm path is not a regular directory: $FAILED_ARM_ROOT" >&2
        exit 1
    }
    if find "$FAILED_ARM_ROOT" -mindepth 1 -print -quit | grep -q .; then
        echo "refusing to remove nonempty failed arm: $FAILED_ARM_ROOT" >&2
        exit 1
    fi
    rmdir "$FAILED_ARM_ROOT"
fi

candidate=$(mktemp "$RESULT_ROOT/.floor2-correction.XXXXXX")
{
    printf 'schema_version=1\n'
    printf 'correction=planned_f4_unreachable_floor2_headroom\n'
    printf 'reason=planned_f4_reachable_floors_are_4_8_16\n'
    printf 'reachable_floors=4,8,16\n'
    printf 'unreachable_floor=2\n'
    printf 'matched_physical_floor2_cap_tokens=103680\n'
    printf 'old_unreachable_floor2_headroom_tokens=147456\n'
    printf 'corrected_unreachable_floor2_headroom_tokens=0\n'
    printf 'reachable_floor_caps_changed=false\n'
    printf 'graph_eager_pairing_changed=false\n'
    printf 'source_code_manifest_sha256=%s\n' "$EXPECTED_CODE_MANIFEST_SHA256"
    printf 'source_protocol_sha256=%s\n' "$EXPECTED_PROTOCOL_SHA256"
    printf 'resume_wrapper_sha256=%s\n' "$(sha256_file "${BASH_SOURCE[0]}")"
} > "$candidate"
if [[ -e "$CORRECTION" ]]; then
    cmp -s "$candidate" "$CORRECTION" || {
        rm -f "$candidate"
        echo "immutable floor-2 correction changed: $CORRECTION" >&2
        exit 1
    }
    rm -f "$candidate"
else
    mv "$candidate" "$CORRECTION"
fi

export GRAPH_MAIN_ROOT="$RESULT_ROOT"
export GRAPH_MAIN_EXECUTION_MODE=eager_matched
export GRAPH_MAIN_REUSE_SEED0_VANILLA=0
export VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR2=0

exec "$SCRIPT_DIR/run_qwen3_paper_matched_eager_tq1_from_common_epoch0.sh" "$ACTION"
