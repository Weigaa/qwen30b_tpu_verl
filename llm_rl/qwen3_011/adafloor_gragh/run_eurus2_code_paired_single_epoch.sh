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
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "[eurus2 code paired] refusing existing output root: $OUTPUT_ROOT" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"
cp -- "$WORKLOAD_DIR/manifest.json" "$OUTPUT_ROOT/eurus2_code_workload_manifest.json"

run_variant() {
    local variant="$1"
    local run_name="$2"
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
    VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
    TRAIN_FILE="$TRAIN_FILE" \
    TRAIN_FILE_ORIG="$TRAIN_FILE" \
    TEST_FILE="$TEST_FILE" \
    "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" "$variant" \
        custom_reward_function.path=null \
        reward_model.reward_manager=prime \
        actor_rollout_ref.rollout.seed="$ROLLOUT_SEED"
}

run_variant vanilla vanilla_frozen_seed${ROLLOUT_SEED}
run_variant adafloor_n_f2 adafloor_natural_f2_tailguard_frozen_seed${ROLLOUT_SEED}

python3 - "$OUTPUT_ROOT" "$ROLLOUT_SEED" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = int(sys.argv[2])
runs = {
    "vanilla": root / f"vanilla_frozen_seed{seed}" / "epoch_001_mode0_vanilla",
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
    responses = {}
    prompt_hashes = set()
    total_rows = 0
    for path in files:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 512:
            raise SystemExit(f"{path}: expected 512 responses, found {len(rows)}")
        total_rows += len(rows)
        for row in rows:
            identity = (
                row["rollout_prompt_hash"],
                int(row["rollout_sample_index"]),
                int(row["rollout_request_seed"]),
            )
            identities.add(identity)
            prompt_hashes.add(row["rollout_prompt_hash"])
            responses[identity] = (
                tuple(int(token) for token in row["responses"]),
                tuple(int(value) for value in row["response_mask"]),
            )
    digest = hashlib.sha256(
        "\n".join(repr(item) for item in sorted(identities)).encode()
    ).hexdigest()
    return identities, prompt_hashes, total_rows, digest, responses

audits = {name: audit(path) for name, path in runs.items()}
if audits["vanilla"][0] != audits["adafloor_natural_f2"][0]:
    raise SystemExit("paired request identity and seed sets differ between variants")
if len(audits["vanilla"][1]) != 160 or len(audits["vanilla"][0]) != 2560:
    raise SystemExit("unexpected paired workload cardinality")
if audits["vanilla"][4] != audits["adafloor_natural_f2"][4]:
    mismatched = sum(
        audits["vanilla"][4][identity]
        != audits["adafloor_natural_f2"][4][identity]
        for identity in audits["vanilla"][0]
    )
    raise SystemExit(f"paired token responses differ for {mismatched} requests")

manifest = {
    "schema_version": 1,
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
    "exact_response_matches": 2560,
    "variants": {name: str(path) for name, path in runs.items()},
    "derived_checkpoints_removed": True,
}
(root / "paired_validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
(root / "PAIR_COMPLETE.txt").write_text(
    "Eurus-2-RL code paired single-epoch validation complete\n"
)
print(json.dumps(manifest, indent=2))
PY

python3 "$SCRIPT_DIR/analysis_eval/summarize_eurus2_code_paired.py" "$OUTPUT_ROOT"

echo "[eurus2 code paired] complete output=$OUTPUT_ROOT"
