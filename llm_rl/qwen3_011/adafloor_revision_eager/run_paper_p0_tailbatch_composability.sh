#!/usr/bin/env bash
set -euo pipefail

if [[ "${ADAFLOOR_TAILBATCH_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    source_path=$(realpath "${BASH_SOURCE[0]}")
    snapshot=$(mktemp "${source_path}.run-snapshot.XXXXXX")
    cp -- "$source_path" "$snapshot"
    chmod 700 "$snapshot"
    set +e
    ADAFLOOR_TAILBATCH_SNAPSHOT_ACTIVE=1 "$snapshot" "$@"
    rc=$?
    set -e
    rm -f -- "$snapshot"
    exit "$rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
TAILBATCH_STAMP="${TAILBATCH_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
TAILBATCH_OUTPUT_ROOT="${TAILBATCH_OUTPUT_ROOT:-/data/adafloor_shared_state/p0_tailbatch_composability_$TAILBATCH_STAMP}"
TAILBATCH_SEED="${TAILBATCH_SEED:-505}"
TAILBATCH_MIN_FREE_GIB="${TAILBATCH_MIN_FREE_GIB:-180}"
TAILBATCH_ARCHIVE_INCOMPLETE="${TAILBATCH_ARCHIVE_INCOMPLETE:-0}"
TRAIN_FILE_ORIG="${TRAIN_FILE_ORIG:-/data/deepscaler/train.parquet}"
PLAN_ROOT="$TAILBATCH_OUTPUT_ROOT/plans"
FULL16_NAME="tailbatch_full16_seed${TAILBATCH_SEED}_frozen_epoch1"
NATURAL_NAME="tailbatch_adafloor_natural_f2_seed${TAILBATCH_SEED}_frozen_epoch1"
FULL16_DIR="$TAILBATCH_OUTPUT_ROOT/$FULL16_NAME"
NATURAL_DIR="$TAILBATCH_OUTPUT_ROOT/$NATURAL_NAME"

if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
      || ! -f "$COMMON_EPOCH0_ROOT/reuse.env" ]]; then
    echo "preserved common epoch0 is incomplete: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi
if ! [[ "$TAILBATCH_SEED" =~ ^[0-9]+$ \
        && "$TAILBATCH_MIN_FREE_GIB" =~ ^[0-9]+$ \
        && "$TAILBATCH_ARCHIVE_INCOMPLETE" =~ ^[01]$ ]]; then
    echo "invalid TailBatch numeric setting" >&2
    exit 2
fi

check_disk() {
    local free_kib required_kib
    free_kib=$(df --output=avail -k "$(dirname "$TAILBATCH_OUTPUT_ROOT")" \
        | tail -1 | tr -d '[:space:]')
    required_kib=$((TAILBATCH_MIN_FREE_GIB * 1024 * 1024))
    if ! [[ "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < required_kib )); then
        echo "insufficient disk for TailBatch run: free_kib=${free_kib:-unknown} required_kib=$required_kib" >&2
        return 1
    fi
    echo "[TailBatch] disk_free_gib=$((free_kib / 1024 / 1024))"
}

variant_complete() {
    local run_dir="$1" epoch_name="$2" step
    [[ -f "$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]] || return 1
    for step in 1 2 3 4 5; do
        [[ -f "$run_dir/$epoch_name/rollout_data/$step.jsonl" ]] || return 1
        [[ $(wc -l < "$run_dir/$epoch_name/rollout_data/$step.jsonl") -eq 512 ]] \
            || return 1
    done
}

prepare_variant_dir() {
    local run_dir="$1" archive_dir
    if [[ ! -e "$run_dir" ]]; then
        return 0
    fi
    if [[ "$TAILBATCH_ARCHIVE_INCOMPLETE" != "1" ]]; then
        echo "incomplete TailBatch output requires inspection: $run_dir" >&2
        return 3
    fi
    archive_dir="${run_dir}.interrupted.$(date -u +%Y%m%dT%H%M%SZ)"
    mv -- "$run_dir" "$archive_dir"
    printf '%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_dir" "$archive_dir" \
        >> "$TAILBATCH_OUTPUT_ROOT/interrupted_runs.tsv"
    echo "[TailBatch] archived incomplete output=$archive_dir"
}

run_variant() {
    local variant="$1" run_name="$2" run_dir="$3" epoch_name="$4" plan_dir="$5"
    if variant_complete "$run_dir" "$epoch_name"; then
        echo "[TailBatch] already complete variant=$variant output=$run_dir"
        return 0
    fi
    prepare_variant_dir "$run_dir"
    check_disk
    echo "[TailBatch] start variant=$variant output=$run_dir plan=$plan_dir"
    if [[ "$variant" == "lengthsort" ]]; then
        env -u FAIR_RUN_NAME -u DYNAMIC_RUN_NAME \
            COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
            FAIR_OUTPUT_ROOT="$TAILBATCH_OUTPUT_ROOT" \
            FAIR_RUN_NAME="$run_name" \
            DYNAMIC_RUN_NAME="$run_name" \
            BASELINE_PREBUILT_PLAN_DIR="$plan_dir" \
            FAIR_START_EPOCH=1 \
            FAIR_TOTAL_EPOCHS=2 \
            FAIR_FREEZE_ACTOR=1 \
            FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
            VERL_SKIP_FINAL_ACTOR_UPDATE_WHEN_FROZEN=1 \
            VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING=1 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=0 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE=0 \
            VLLM_ASCEND_DUMMY_WASTE_SELECTION_STATS=0 \
            "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
                "$variant" "actor_rollout_ref.rollout.seed=$TAILBATCH_SEED"
    else
        env -u FAIR_RUN_NAME -u DYNAMIC_RUN_NAME \
            COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
            FAIR_OUTPUT_ROOT="$TAILBATCH_OUTPUT_ROOT" \
            FAIR_RUN_NAME="$run_name" \
            DYNAMIC_RUN_NAME="$run_name" \
            PREBUILT_PLAN_DIR="$plan_dir" \
            FAIR_START_EPOCH=1 \
            FAIR_TOTAL_EPOCHS=2 \
            FAIR_FREEZE_ACTOR=1 \
            FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
            VERL_SKIP_FINAL_ACTOR_UPDATE_WHEN_FROZEN=1 \
            VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING=1 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=0 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE=0 \
            VLLM_ASCEND_DUMMY_WASTE_SELECTION_STATS=0 \
            "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
                "$variant" "actor_rollout_ref.rollout.seed=$TAILBATCH_SEED"
    fi
}

mkdir -p "$TAILBATCH_OUTPUT_ROOT"
if [[ ! -d "$PLAN_ROOT" ]]; then
    python3 -u "$SCRIPT_DIR/tools/build_tailbatch_composability_plans.py" \
        --history-dir "$COMMON_EPOCH0_ROOT/epoch_000_mode0_probe" \
        --train-file "$TRAIN_FILE_ORIG" \
        --output-root "$PLAN_ROOT"
fi
if [[ ! -f "$PLAN_ROOT/TAILBATCH_PLAN_COMPLETE.txt" ]]; then
    echo "TailBatch plan is incomplete: $PLAN_ROOT" >&2
    exit 2
fi

if [[ ! -f "$TAILBATCH_OUTPUT_ROOT/protocol.env" ]]; then
    {
        printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
        printf 'sampling_seed=%s\n' "$TAILBATCH_SEED"
        printf 'actor_frozen=true\n'
        printf 'per_request_sampling_seeds=true\n'
        printf 'epochs=1\n'
        printf 'tail_guard_enabled=false\n'
        printf 'step_membership=fixed_tailbatch\n'
        printf 'release_definition=coordinated_16_to_8_to_4_to_2\n'
    } > "$TAILBATCH_OUTPUT_ROOT/protocol.env"
    sha256sum \
        "$SCRIPT_DIR/tools/build_tailbatch_composability_plans.py" \
        "$SCRIPT_DIR/run_paper_p0_tailbatch_composability.sh" \
        "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
        "$SCRIPT_DIR/run_baseline_lengthsort_epoch1_2.sh" \
        "$SCRIPT_DIR/analysis_eval/build_rank_time_summary.py" \
        "$SCRIPT_DIR/analysis_eval/summarize_tailbatch_composability.py" \
        > "$TAILBATCH_OUTPUT_ROOT/code_sha256.txt"
fi

echo "[TailBatch] output_root=$TAILBATCH_OUTPUT_ROOT"
echo "[TailBatch] common_epoch0=$COMMON_EPOCH0_ROOT seed=$TAILBATCH_SEED"
if [[ "${TAILBATCH_DRY_RUN:-0}" == "1" ]]; then
    echo "[TailBatch] would run TailBatch-Full16 -> $FULL16_DIR"
    echo "[TailBatch] would run TailBatch+AdaFloor-N -> $NATURAL_DIR"
    echo "[TailBatch] dry run only"
    exit 0
fi

run_variant lengthsort "$FULL16_NAME" "$FULL16_DIR" \
    epoch_001_mode0_lengthsort "$PLAN_ROOT/full16"
run_variant adafloor_n_f2_noguard "$NATURAL_NAME" "$NATURAL_DIR" \
    epoch_001_mode1_natural "$PLAN_ROOT/natural_f2"

python3 - "$TAILBATCH_OUTPUT_ROOT" "$TAILBATCH_SEED" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = int(sys.argv[2])
manifest = json.loads((root / "plans/layout_manifest.json").read_text())
runs = {
    "tailbatch_full16": (
        root / f"tailbatch_full16_seed{seed}_frozen_epoch1"
        / "epoch_001_mode0_lengthsort"
    ),
    "tailbatch_adafloor_natural": (
        root / f"tailbatch_adafloor_natural_f2_seed{seed}_frozen_epoch1"
        / "epoch_001_mode1_natural"
    ),
}

def audit(epoch_dir):
    files = sorted(
        (epoch_dir / "rollout_data").glob("*.jsonl"),
        key=lambda path: int(path.stem),
    )
    if len(files) != 5:
        raise SystemExit(f"{epoch_dir}: expected five rollout files")
    by_step = []
    response_digests = {}
    scores = {}
    total_tokens = 0
    all_scores = []
    for step_index, path in enumerate(files):
        identities = set()
        plan_prompt_hashes = set()
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != 512:
            raise SystemExit(f"{path}: expected 512 responses, found {len(rows)}")
        for row in rows:
            identity = (
                row["rollout_prompt_hash"],
                int(row["rollout_sample_index"]),
                int(row["rollout_request_seed"]),
            )
            if identity in identities:
                raise SystemExit(f"{path}: duplicate identity {identity}")
            identities.add(identity)
            plan_prompt_hashes.add(
                hashlib.sha256(row["input"].encode()).hexdigest()
            )
            response_length = sum(int(value) for value in row["response_mask"])
            total_tokens += response_length
            tokens = row["responses"][:response_length]
            response_digests[identity] = (
                response_length,
                hashlib.sha256(
                    json.dumps(tokens, separators=(",", ":")).encode()
                ).hexdigest(),
            )
            score = float(row["score"])
            if not math.isfinite(score):
                raise SystemExit(f"{path}: non-finite score")
            scores[identity] = score
            all_scores.append(score)
        expected_prompts = set(manifest["step_prompt_hashes"][step_index])
        if plan_prompt_hashes != expected_prompts:
            raise SystemExit(
                f"{path}: rendered-input prompt membership differs from "
                "frozen TailBatch plan"
            )
        by_step.append(identities)
    all_identities = set().union(*by_step)
    return {
        "by_step": by_step,
        "all_identities": all_identities,
        "response_digests": response_digests,
        "scores": scores,
        "generated_response_tokens": total_tokens,
        "mean_score": sum(all_scores) / len(all_scores),
    }

audits = {name: audit(path) for name, path in runs.items()}
left = audits["tailbatch_full16"]
right = audits["tailbatch_adafloor_natural"]
if left["by_step"] != right["by_step"]:
    raise SystemExit("paired request identities differ within one or more steps")
if len(left["all_identities"]) != 2560:
    raise SystemExit("unexpected paired request cardinality")
exact_responses = sum(
    left["response_digests"][identity] == right["response_digests"][identity]
    for identity in left["all_identities"]
)
exact_scores = sum(
    left["scores"][identity] == right["scores"][identity]
    for identity in left["all_identities"]
)
digest = hashlib.sha256(
    "\n".join(repr(item) for item in sorted(left["all_identities"])).encode()
).hexdigest()
payload = {
    "schema_version": 1,
    "status": "complete",
    "actor_frozen": True,
    "tail_guard_enabled": False,
    "rollout_seed": seed,
    "prompts": 160,
    "responses_per_prompt": 16,
    "responses": 2560,
    "step_membership_digest": manifest["step_membership_digest"],
    "paired_request_digest": digest,
    "request_identity_matches": 2560,
    "exact_response_matches": exact_responses,
    "divergent_response_count": 2560 - exact_responses,
    "exact_score_matches": exact_scores,
    "token_response_equivalence_required": False,
    "variants": {name: str(path) for name, path in runs.items()},
    "variant_audits": {
        name: {
            "generated_response_tokens": audit_result["generated_response_tokens"],
            "mean_score": audit_result["mean_score"],
        }
        for name, audit_result in audits.items()
    },
    "derived_checkpoints_removed": True,
}
(root / "paired_validation.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

python3 "$SCRIPT_DIR/analysis_eval/build_rank_time_summary.py" \
    --variant "TailBatch-Full16=$FULL16_DIR" \
    --variant "TailBatch+AdaFloor-N=$NATURAL_DIR" \
    --output-dir "$TAILBATCH_OUTPUT_ROOT/rank_time"
python3 "$SCRIPT_DIR/analysis_eval/summarize_tailbatch_composability.py" \
    "$TAILBATCH_OUTPUT_ROOT"
printf 'TailBatch composability paired experiment complete\n' \
    > "$TAILBATCH_OUTPUT_ROOT/TAILBATCH_EXPERIMENT_COMPLETE.txt"
echo "[TailBatch] complete summary=$TAILBATCH_OUTPUT_ROOT/summary.md"
