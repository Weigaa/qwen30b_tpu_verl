#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

WORKLOAD_DIR="${EURUS2_CODE_WORKLOAD_DIR:-/data/eurus2_rl_code/validation_code_paired_160}"
COMMON_ROOT="${EURUS2_COMMON_ROOT:-/data/adafloor_shared_state/common_epoch0_eurus2_code_validation_frozen_gpu09_kv380800_permanent}"
RUN_STAMP="${EURUS2_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${EURUS2_OUTPUT_ROOT:-/data/adafloor_shared_state/eurus2_code_paired_single_epoch_$RUN_STAMP}"
TRAIN_FILE="$WORKLOAD_DIR/train.parquet"
TEST_FILE="$WORKLOAD_DIR/test.parquet"
ROLLOUT_SEED="${EURUS2_ROLLOUT_SEED:-401}"
RESUME_EXISTING="${EURUS2_RESUME_EXISTING:-0}"
RUN_VANILLA="${EURUS2_RUN_VANILLA:-1}"
VANILLA_SOURCE_ROOT="${EURUS2_VANILLA_SOURCE_ROOT:-}"
ADAFLOOR_PREBUILT_PLAN_DIR="${EURUS2_ADAFLOOR_PREBUILT_PLAN_DIR:-}"
REQUIRE_KV_ADMISSION_HEADROOM="${EURUS2_REQUIRE_KV_ADMISSION_HEADROOM:-0}"

if [[ ! -f "$WORKLOAD_DIR/manifest.json" \
      || ! -f "$TRAIN_FILE" || ! -f "$TEST_FILE" ]]; then
    "$SCRIPT_DIR/run_prepare_eurus2_code_workload.sh"
fi
if [[ ! -f "$COMMON_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
      || ! -f "$COMMON_ROOT/reuse.env" ]]; then
    EURUS2_COMMON_OUTPUT_ROOT=$(dirname "$COMMON_ROOT") \
    EURUS2_COMMON_RUN_NAME=$(basename "$COMMON_ROOT") \
        "$SCRIPT_DIR/run_eurus2_code_common_epoch0.sh"
fi
if [[ "$RESUME_EXISTING" != "0" && "$RESUME_EXISTING" != "1" ]]; then
    echo "[eurus2 code paired] EURUS2_RESUME_EXISTING must be 0 or 1" >&2
    exit 2
fi
if [[ "$RUN_VANILLA" != "0" && "$RUN_VANILLA" != "1" ]]; then
    echo "[eurus2 code paired] EURUS2_RUN_VANILLA must be 0 or 1" >&2
    exit 2
fi
if [[ "$REQUIRE_KV_ADMISSION_HEADROOM" != "0" \
      && "$REQUIRE_KV_ADMISSION_HEADROOM" != "1" ]]; then
    echo "[eurus2 code paired] EURUS2_REQUIRE_KV_ADMISSION_HEADROOM must be 0 or 1" >&2
    exit 2
fi
if [[ -n "$ADAFLOOR_PREBUILT_PLAN_DIR" ]]; then
    ADAFLOOR_PREBUILT_PLAN_DIR=$(realpath "$ADAFLOOR_PREBUILT_PLAN_DIR")
    for plan_file in \
        length_sorted_train.parquet \
        length_sorted_rank_plan.json \
        length_sorted_rank_plan_summary.json \
        length_sorted_length_oracle.json; do
        if [[ ! -f "$ADAFLOOR_PREBUILT_PLAN_DIR/$plan_file" ]]; then
            echo "[eurus2 code paired] incomplete AdaFloor prebuilt plan: $ADAFLOOR_PREBUILT_PLAN_DIR/$plan_file" >&2
            exit 2
        fi
    done
    python3 - \
        "$ADAFLOOR_PREBUILT_PLAN_DIR/length_sorted_rank_plan_summary.json" \
        "$REQUIRE_KV_ADMISSION_HEADROOM" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
steps = json.loads(path.read_text())
if len(steps) != 5:
    raise SystemExit(f"{path}: expected five planned steps")
for step in steps:
    if not step.get("feasible", False):
        raise SystemExit(f"{path}: step {step.get('step')} is not feasible")
    peak = float(step["max_adjusted_rank_peak_tokens"])
    cap = float(step["kv_cap"])
    admission_cap = float(step.get("kv_admission_cap", cap))
    if peak > admission_cap:
        raise SystemExit(
            f"{path}: step {step.get('step')} adjusted peak {peak} exceeds "
            f"admission cap {admission_cap}"
        )
    if admission_cap > cap:
        raise SystemExit(
            f"{path}: step {step.get('step')} admission cap {admission_cap} "
            f"exceeds runtime cap {cap}"
        )
    if int(sys.argv[2]) and "kv_admission_cap" not in step:
        raise SystemExit(
            f"{path}: step {step.get('step')} lacks strict KV admission headroom"
        )
print(
    "[eurus2 code paired] validated AdaFloor prebuilt floors="
    + ",".join(str(step["selected_floor"]) for step in steps)
    + " physical_headroom="
    + ",".join(
        f"{float(step['kv_cap']) - float(step['max_adjusted_rank_peak_tokens']):.1f}"
        for step in steps
    )
)
PY
fi
if [[ -e "$OUTPUT_ROOT" && "$RESUME_EXISTING" != "1" ]]; then
    echo "[eurus2 code paired] refusing existing output root: $OUTPUT_ROOT" >&2
    echo "[eurus2 code paired] set EURUS2_RESUME_EXISTING=1 to resume it" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"
cp -- "$WORKLOAD_DIR/manifest.json" "$OUTPUT_ROOT/eurus2_code_workload_manifest.json"

variant_complete() {
    local run_name="$1"
    local epoch_name="$2"
    local run_dir="$OUTPUT_ROOT/$run_name"
    local epoch_dir="$run_dir/$epoch_name"
    [[ -f "$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]] || return 1
    for step in 1 2 3 4 5; do
        local data_file="$epoch_dir/rollout_data/$step.jsonl"
        local length_file="$epoch_dir/rollout_length/length_$step.txt"
        [[ -f "$data_file" && -f "$length_file" ]] || return 1
        [[ "$(wc -l < "$data_file")" -eq 512 ]] || return 1
        [[ "$(wc -l < "$length_file")" -eq 512 ]] || return 1
    done
    if [[ "$run_name" == adafloor_* ]]; then
        local log_file
        log_file=$(find "$epoch_dir/logs" -maxdepth 1 -type f -name '*.txt' \
            -print -quit 2>/dev/null || true)
        [[ -n "$log_file" ]] || return 1
        ! grep -qF 'Preempting request' "$log_file" || return 1
    fi
}

run_variant() {
    local variant="$1"
    local run_name="$2"
    local epoch_name="$3"
    local run_dir="$OUTPUT_ROOT/$run_name"
    local prebuilt_plan_dir=""
    if [[ "$variant" == adafloor_* ]]; then
        prebuilt_plan_dir="$ADAFLOOR_PREBUILT_PLAN_DIR"
    fi
    if variant_complete "$run_name" "$epoch_name"; then
        echo "[eurus2 code paired] variant already complete; skipping $variant"
        return 0
    fi
    if [[ -e "$run_dir" ]]; then
        if find "$run_dir" -type f \( -path '*/rollout_data/*.jsonl' \
                -o -path '*/checkpoints/*' -o -path '*/logs/*.txt' \) \
                -print -quit | grep -q .; then
            echo "[eurus2 code paired] refusing nonempty incomplete variant: $run_dir" >&2
            exit 2
        fi
        echo "[eurus2 code paired] removing prelaunch-only incomplete variant: $run_dir"
        rm -rf -- "$run_dir"
    fi
    echo "[eurus2 code paired] starting variant=$variant output=$OUTPUT_ROOT/$run_name"
    COMMON_EPOCH0_ROOT="$COMMON_ROOT" \
    FAIR_OUTPUT_ROOT="$OUTPUT_ROOT" \
    FAIR_RUN_NAME="$run_name" \
    DYNAMIC_RUN_NAME="$run_name" \
    FAIR_START_EPOCH=1 \
    FAIR_TOTAL_EPOCHS=2 \
    FAIR_PROMPTS_PER_EPOCH=160 \
    FAIR_TRAIN_BATCH_SIZE=32 \
    FAIR_ROLLOUT_N=16 \
    FAIR_MAX_RESPONSE_LENGTH=16384 \
    FAIR_DATASET_FRACTION=1.0 \
    DYNAMIC_DATASET_FRACTION=1.0 \
    DATASET_FRACTION_FOR_ORACLE=1.0 \
    FAIR_FREEZE_ACTOR=1 \
    VERL_SKIP_FINAL_ACTOR_UPDATE_WHEN_FROZEN=1 \
    VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
    PREBUILT_PLAN_DIR="$prebuilt_plan_dir" \
    TRAIN_FILE="$TRAIN_FILE" \
    TRAIN_FILE_ORIG="$TRAIN_FILE" \
    TEST_FILE="$TEST_FILE" \
    "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" "$variant" \
        custom_reward_function.path=null \
        reward_model.reward_manager=prime \
        actor_rollout_ref.rollout.seed="$ROLLOUT_SEED"
}

if [[ -n "$ADAFLOOR_PREBUILT_PLAN_DIR" ]]; then
    {
        printf 'prebuilt_plan_dir=%s\n' "$ADAFLOOR_PREBUILT_PLAN_DIR"
        sha256sum "$ADAFLOOR_PREBUILT_PLAN_DIR"/{length_sorted_train.parquet,length_sorted_rank_plan.json,length_sorted_rank_plan_summary.json,length_sorted_length_oracle.json}
    } > "$OUTPUT_ROOT/ADAFLOOR_PREBUILT_PLAN_AUDIT.txt"
fi

if [[ "$RUN_VANILLA" == "1" ]]; then
    run_variant vanilla vanilla_frozen_seed${ROLLOUT_SEED} epoch_001_mode0_vanilla
    VANILLA_EPOCH_DIR="$OUTPUT_ROOT/vanilla_frozen_seed${ROLLOUT_SEED}/epoch_001_mode0_vanilla"
else
    if [[ -z "$VANILLA_SOURCE_ROOT" ]]; then
        echo "[eurus2 code paired] EURUS2_VANILLA_SOURCE_ROOT is required when reusing Vanilla" >&2
        exit 2
    fi
    VANILLA_EPOCH_DIR="$VANILLA_SOURCE_ROOT/vanilla_frozen_seed${ROLLOUT_SEED}/epoch_001_mode0_vanilla"
    if [[ ! -d "$VANILLA_EPOCH_DIR/rollout_data" \
          || ! -d "$VANILLA_EPOCH_DIR/logs" ]]; then
        echo "[eurus2 code paired] reusable Vanilla result is incomplete: $VANILLA_EPOCH_DIR" >&2
        exit 2
    fi
    echo "[eurus2 code paired] reusing Vanilla result from $VANILLA_EPOCH_DIR"
fi
run_variant adafloor_n_f2 adafloor_natural_f2_tailguard_frozen_seed${ROLLOUT_SEED} \
    epoch_001_mode1_natural

python3 - "$OUTPUT_ROOT" "$ROLLOUT_SEED" "$VANILLA_EPOCH_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = int(sys.argv[2])
vanilla_epoch_dir = Path(sys.argv[3])
runs = {
    "vanilla": vanilla_epoch_dir,
    "adafloor_natural_f2": (
        root
        / f"adafloor_natural_f2_tailguard_frozen_seed{seed}"
        / "epoch_001_mode1_natural"
    ),
}

def audit(epoch_dir):
    files = sorted((epoch_dir / "rollout_data").glob("*.jsonl"), key=lambda p: int(p.stem))
    if len(files) != 5:
        raise SystemExit(f"{epoch_dir}: expected five rollout files, found {len(files)}")
    identities = set()
    response_digests = {}
    prompt_hashes = set()
    total_rows = 0
    for path in files:
        rows_in_file = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows_in_file += 1
                total_rows += 1
                identity = (
                    row["rollout_prompt_hash"],
                    int(row["rollout_sample_index"]),
                    int(row["rollout_request_seed"]),
                )
                if identity in identities:
                    raise SystemExit(
                        f"{path}: duplicate paired request identity {identity}"
                    )
                identities.add(identity)
                prompt_hashes.add(row["rollout_prompt_hash"])
                response_mask = row["response_mask"]
                response_length = sum(int(value) for value in response_mask)
                response_tokens = row["responses"][:response_length]
                response_digests[identity] = (
                    response_length,
                    hashlib.sha256(
                        json.dumps(
                            response_tokens, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                )
        if rows_in_file != 512:
            raise SystemExit(f"{path}: expected 512 responses, found {rows_in_file}")
    digest = hashlib.sha256(
        "\n".join(repr(item) for item in sorted(identities)).encode()
    ).hexdigest()
    return identities, prompt_hashes, total_rows, digest, response_digests

audits = {name: audit(path) for name, path in runs.items()}
if audits["vanilla"][0] != audits["adafloor_natural_f2"][0]:
    raise SystemExit("paired request identity and seed sets differ between variants")
if len(audits["vanilla"][1]) != 160 or len(audits["vanilla"][0]) != 2560:
    raise SystemExit("unexpected paired workload cardinality")
exact_response_matches = sum(
    audits["vanilla"][4][identity]
    == audits["adafloor_natural_f2"][4][identity]
    for identity in audits["vanilla"][0]
)
divergent_response_count = len(audits["vanilla"][0]) - exact_response_matches

manifest = {
    "schema_version": 2,
    "status": "complete",
    "dataset": "PRIME-RL/Eurus-2-RL-Data",
    "split": "validation",
    "ability": "code",
    "actor_frozen": True,
    "rollout_seed": seed,
    "prompts": 160,
    "responses_per_prompt": 16,
    "responses": audits["vanilla"][2],
    "paired_request_digest": audits["vanilla"][3],
    "request_identity_matches": len(audits["vanilla"][0]),
    "exact_response_matches": exact_response_matches,
    "divergent_response_count": divergent_response_count,
    "token_response_equivalence_required": False,
    "token_response_note": (
        "Identical request seeds do not require identical sampled tokens across "
        "different EP collective topologies or TailGuard caps."
    ),
    "variants": {name: str(path) for name, path in runs.items()},
    "derived_checkpoints_removed": True,
}
(root / "paired_validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
(root / "PAIR_COMPLETE.txt").write_text(
    "Eurus-2-RL code request-identity-paired single-epoch validation complete\n"
)
print(json.dumps(manifest, indent=2))
PY

python3 "$SCRIPT_DIR/analysis_eval/summarize_eurus2_code_paired.py" "$OUTPUT_ROOT"

echo "[eurus2 code paired] complete output=$OUTPUT_ROOT"
