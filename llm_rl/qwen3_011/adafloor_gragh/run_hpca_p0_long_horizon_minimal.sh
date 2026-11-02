#!/usr/bin/env bash
set -euo pipefail

# This runner intentionally executes the paper-submitted eager stack.  It does
# not use adafloor_gragh's later ACLGraph/admission modifications.  The code
# manifest is checked before any training process is started.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAPER_EAGER_REPO="${PAPER_EAGER_REPO:-/workspace/cann-recipes-train/llm_rl/qwen3_011/qwen3_shrink_aware}"
PAPER_EAGER_MANIFEST="${PAPER_EAGER_MANIFEST:-/data/adafloor_shared_state/p0_7_natural_matching_causal_v2_20260731/code_sha256.txt}"
PAPER_EAGER_REQUIRE_CLEAN="${PAPER_EAGER_REQUIRE_CLEAN:-0}"
COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
OUTPUT_ROOT="${HPCA_P0_QUALITY_OUTPUT_ROOT:-/data/adafloor_shared_state/hpca_p0_long_horizon_minimal_$(date -u +%Y%m%dT%H%M%SZ)}"
SEED="${HPCA_P0_QUALITY_SEED:-505}"
TOTAL_EPOCHS="${HPCA_P0_QUALITY_TOTAL_EPOCHS:-5}"
VARIANTS="${HPCA_P0_QUALITY_VARIANTS:-vanilla adafloor_n_f2 adafloor_p_f4}"
DRY_RUN=0
case "${1:-run}" in
    --dry-run|dry-run) DRY_RUN=1 ;;
    run) ;;
    *) printf 'Usage: %s [run|--dry-run]\n' "$0" >&2; exit 2 ;;
esac

die() { echo "[hpca-p0-quality] $*" >&2; exit 2; }

[[ -d "$PAPER_EAGER_REPO" ]] || die "paper eager repo is missing: $PAPER_EAGER_REPO"
[[ -f "$PAPER_EAGER_MANIFEST" ]] || die "paper eager code manifest is missing: $PAPER_EAGER_MANIFEST"
[[ -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]] || die "common epoch0 marker is missing"
[[ -f "$COMMON_EPOCH0_ROOT/reuse.env" ]] || die "common epoch0 reuse.env is missing"
[[ "$SEED" =~ ^[0-9]+$ && "$TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "invalid seed or epoch count"
(( TOTAL_EPOCHS == 5 )) || die "P0 protocol requires exactly four updated epochs (TOTAL_EPOCHS=5)"
COMMON_REAL=$(realpath -e "$COMMON_EPOCH0_ROOT")
OUTPUT_REAL=$(realpath -m "$OUTPUT_ROOT")
case "$OUTPUT_REAL/" in
    "$COMMON_REAL/"*) die "output must not be inside public epoch0" ;;
esac
public_model="$COMMON_REAL/epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3"
[[ -s "$public_model/global_step_5/actor/dist_ckpt/metadata.json" ]] || die "public checkpoint metadata is missing"
[[ -f "$public_model/latest_checkpointed_iteration.txt" ]] || die "public checkpoint marker is missing"
[[ $(tr -d '[:space:]' < "$public_model/latest_checkpointed_iteration.txt") == 5 ]] || die "public checkpoint iteration is not 5"
[[ $(find "$public_model/global_step_5/actor/dist_ckpt" -maxdepth 1 -type f -name '*.distcp' -size +0c | wc -l) == 32 ]] || die "public checkpoint must have 32 nonempty shards"

echo "[hpca-p0-quality] validating paper-submitted eager code manifest"
verify_paper_manifest() {
    local digest source rel candidate actual
    while read -r digest source; do
        [[ -n "${digest:-}" && -n "${source:-}" ]] || continue
        rel="${source#/workspace/cann-recipes-train/llm_rl/qwen3_011/qwen3_shrink_aware/}"
        candidate="$PAPER_EAGER_REPO/$rel"
        [[ -f "$candidate" ]] || die "manifest source is missing in paper repo: $candidate"
        actual=$(sha256sum "$candidate" | awk '{print $1}')
        [[ "$actual" == "$digest" ]] || die "paper eager hash mismatch: $candidate"
    done < "$PAPER_EAGER_MANIFEST"
}
verify_paper_manifest
if [[ "$PAPER_EAGER_REQUIRE_CLEAN" == "1" && -d "$PAPER_EAGER_REPO/.git" ]]; then
    if [[ -n "$(git -C "$PAPER_EAGER_REPO" status --porcelain --untracked-files=all)" ]]; then
        die "paper eager repo is dirty; provide an immutable submitted-version checkout"
    fi
fi

for variant in $VARIANTS; do
    case "$variant" in
        vanilla|adafloor_n_f2|adafloor_p_f4) ;;
        *) die "unsupported variant: $variant" ;;
    esac
    candidate="$OUTPUT_REAL/hpca_p0_${variant}_seed${SEED}_epoch1_4"
    case "$candidate/" in
        "$OUTPUT_REAL/"*) ;;
        *) die "cleanup path escapes this run root" ;;
    esac
    [[ ! -e "$candidate" ]] || die "refusing to overwrite existing run: $candidate"
done

if (( DRY_RUN )); then
    printf 'dry_run=PASS\nseed=%s\nepochs=1-4\nvariants=%s\nsteps_per_epoch=5\nresponses_per_step=512\nactor_frozen=false\ngraph_mode=disabled\nTASK_QUEUE_ENABLE=2\ncommon_epoch0=%s\noutput_root=%s\ncleanup_scope=run_owned_children_only\n' \
        "$SEED" "$VARIANTS" "$COMMON_REAL" "$OUTPUT_REAL"
    exit 0
fi

mkdir -p "$OUTPUT_ROOT"
{
    printf 'schema_version=1\n'
    printf 'status=RUNNING\n'
    printf 'paper_eager_repo=%s\n' "$PAPER_EAGER_REPO"
    printf 'paper_eager_manifest=%s\n' "$PAPER_EAGER_MANIFEST"
    printf 'paper_eager_manifest_sha256=%s\n' "$(sha256sum "$PAPER_EAGER_MANIFEST" | awk '{print $1}')"
    printf 'paper_eager_require_clean=%s\n' "$PAPER_EAGER_REQUIRE_CLEAN"
    printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
    printf 'seed=%s\n' "$SEED"
    printf 'epochs=1-%s\n' "$((TOTAL_EPOCHS - 1))"
    printf 'steps_per_epoch=5\n'
    printf 'responses_per_step=512\n'
    printf 'actor_frozen=false\n'
    printf 'variants=%s\n' "$VARIANTS"
    printf 'graph_mode=disabled\n'
    printf 'paper_eager_code_contract=sha256_manifest_required\n'
} > "$OUTPUT_ROOT/protocol.env"

sha256sum "$0" "$PAPER_EAGER_MANIFEST" > "$OUTPUT_ROOT/code_sha256.txt"

echo "[hpca-p0-quality] launching paper eager runner repo=$PAPER_EAGER_REPO seed=$SEED epochs=1..$((TOTAL_EPOCHS - 1))"
for variant in $VARIANTS; do
    run_name="hpca_p0_${variant}_seed${SEED}_epoch1_$((TOTAL_EPOCHS - 1))"
    run_dir="$OUTPUT_ROOT/$run_name"
    [[ ! -e "$run_dir" ]] || die "refusing to overwrite existing run: $run_dir"
    (
        cd "$PAPER_EAGER_REPO"
        COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
        FAIR_OUTPUT_ROOT="$OUTPUT_ROOT" \
        FAIR_RUN_NAME="$run_name" \
        DYNAMIC_RUN_NAME="$run_name" \
        FAIR_START_EPOCH=1 \
        FAIR_TOTAL_EPOCHS="$TOTAL_EPOCHS" \
        FAIR_FREEZE_ACTOR=0 \
        FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
        VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
        FAIR_TRAIN_BATCH_SIZE=32 \
        FAIR_ROLLOUT_N=16 \
        FAIR_MAX_RESPONSE_LENGTH=16384 \
        DYNAMIC_TRAIN_STEPS=5 \
        DYNAMIC_PLAN_STEPS=5 \
        VLLM_ENABLE_GRAPH_MODE=0 \
        ROLLOUT_ENFORCE_EAGER=True \
        TASK_QUEUE_ENABLE=2 \
        ./run_paper_fair_epoch1_2_from_common_epoch0.sh "$variant" \
            "actor_rollout_ref.rollout.seed=$SEED"
    )
done

analysis_args=()
for variant in $VARIANTS; do
    run_name="hpca_p0_${variant}_seed${SEED}_epoch1_$((TOTAL_EPOCHS - 1))"
    analysis_args+=(--variant "$variant" "$OUTPUT_ROOT/$run_name")
done
PYTHONPATH="$PAPER_EAGER_REPO:$SCRIPT_DIR:${PYTHONPATH:-}" \
    python3 "$SCRIPT_DIR/analysis_eval/analyze_training_quality_curve.py" \
    "${analysis_args[@]}" --output-dir "$OUTPUT_ROOT/summary"

printf 'status=COMPLETE\n' >> "$OUTPUT_ROOT/protocol.env"
printf 'common_epoch0_preserved=true\n' > "$OUTPUT_ROOT/COMMON_EPOCH0_PRESERVED.txt"
echo "[hpca-p0-quality] complete output=$OUTPUT_ROOT"
