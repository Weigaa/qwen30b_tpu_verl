#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

usage() {
    cat <<'EOF'
Usage: ./run_deepseek_v2_lite_batch64_paired.sh PHASE

PHASE is one of status, semantic, common, recover-common, trigger, calibrate, authorize, gate, epoch,
verify-gate, verify-epoch, dry-run-gate, dry-run-epoch, or all.

The workflow compares frozen-checkpoint Vanilla Full16 with AdaFloor Natural
floor2 using 64 prompts, n=16, and the same measured Full16 physical KV cap.
The gate runs one step per arm. The epoch phase runs five steps per arm.
EOF
}

if (( $# != 1 )) || [[ "$1" == -h || "$1" == --help ]]; then
    usage
    [[ "${1:-}" == -h || "${1:-}" == --help ]] && exit 0
    exit 2
fi
PHASE=$1

PROFILE_PATH=${DEEPSEEK_BATCH64_WORKLOAD_PROFILE_PATH:-$SCRIPT_DIR/internal/deepseek_v2_lite_batch64_workload_profile.sh}
if [[ ! -f "$PROFILE_PATH" ]]; then
    echo "missing batch64 workload profile: $PROFILE_PATH" >&2
    exit 2
fi
PROFILE_PATH=$(realpath "$PROFILE_PATH")
# shellcheck disable=SC1090
source "$PROFILE_PATH"
PROFILE_SHA256=$(sha256sum "$PROFILE_PATH")
PROFILE_SHA256=${PROFILE_SHA256%% *}
if [[ -z "${DEEPSEEK_WORKLOAD_PROFILE_ID:-}" ]]; then
    echo "batch64 workload profile has no ID" >&2
    exit 2
fi
if [[ -n "${DEEPSEEK_WORKLOAD_PROFILE_SHA256:-}" \
      && "$DEEPSEEK_WORKLOAD_PROFILE_SHA256" != "$PROFILE_SHA256" ]]; then
    echo "batch64 workload profile SHA256 mismatch" >&2
    exit 2
fi
export DEEPSEEK_WORKLOAD_PROFILE_SHA256=$PROFILE_SHA256

CHAT_MODEL_ID=deepseek-ai/DeepSeek-V2-Lite-Chat
CHAT_MODEL_REVISION=85864749cd611b4353ce1decdb286193298f64c7
CHAT_MODEL_PATH=/data/DeepSeek-V2-Lite-Chat
CHAT_DISTCP_PATH=/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4

EXECUTION_CODE_SHA256=$(python3 \
    "$SCRIPT_DIR/tools/hash_deepseek_execution_code.py" --root "$SCRIPT_DIR")
DEFAULT_ROOT=/data/adafloor_shared_state/deepseek_v2_lite/p0_8b_batch64_natural_f2_${EXECUTION_CODE_SHA256:0:12}
ACTIVE_ROOT_RECORD=/data/adafloor_shared_state/deepseek_v2_lite/.active_batch64_experiment_root
if [[ -f "$ACTIVE_ROOT_RECORD" ]]; then
    DEFAULT_ROOT=$(tr -d '\r\n' < "$ACTIVE_ROOT_RECORD")
fi
EXPERIMENT_ROOT=${DEEPSEEK_BATCH64_EXPERIMENT_ROOT:-$DEFAULT_ROOT}
COMMON_ROOT=$EXPERIMENT_ROOT/$COMMON_EPOCH0_RUN_NAME
SEMANTIC_ROOT=$EXPERIMENT_ROOT/chat_semantic_gate
SEMANTIC_OUTPUT=$SEMANTIC_ROOT/semantic_smoke.json
ASSET_AUDIT=$SEMANTIC_ROOT/asset_audit.json
CONVERTED_GATE_RUN_NAME=converted_distcp_weight_gate
CONVERTED_GATE_ROOT=$SEMANTIC_ROOT/$CONVERTED_GATE_RUN_NAME
TRIGGER_ROOT=$EXPERIMENT_ROOT/kv_probe_trigger_batch64
CALIBRATION_ROOT=$EXPERIMENT_ROOT/kv_calibration_natural_f2
AUTHORIZATION_ROOT=$EXPERIMENT_ROOT/kv_authorization_natural_f2
CAP_ENV=$EXPERIMENT_ROOT/deepseek_v2_lite_batch64_kv_caps.env
GATE_ROOT=$EXPERIMENT_ROOT/paired_gate
EPOCH_ROOT=$EXPERIMENT_ROOT/paired_epoch
GATE_ADAFLOOR_FORCE_SELECTED_FLOOR=4
COMMON_AUDIT_MANIFEST=$COMMON_ROOT/COMMON_EPOCH0_RECOVERY_MANIFEST.json
COMMON_CHECKPOINT_SHA256=$COMMON_ROOT/FROZEN_CHECKPOINT_SHA256
SOURCE_HISTORY_ROOT=${DEEPSEEK_BATCH64_TRIGGER_SOURCE_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/threshold_actor_2step_n16_tq1_no_overlap_20260804T031350Z/epoch_000_mode0_probe}
WORKFLOW_LOCK=$EXPERIMENT_ROOT/.workflow.lock

require_file() {
    local path=$1
    local label=$2
    if [[ ! -f "$path" ]]; then
        echo "missing $label: $path" >&2
        exit 2
    fi
}

record_or_validate_contract() {
    mkdir -p "$EXPERIMENT_ROOT"
    local code_record=$EXPERIMENT_ROOT/EXECUTION_CODE_SHA256
    local continuation_record=$EXPERIMENT_ROOT/CONTINUATION_EXECUTION_CODE_SHA256
    local profile_record=$EXPERIMENT_ROOT/WORKLOAD_PROFILE_SHA256
    if [[ -f "$code_record" \
          && $(tr -d '[:space:]' < "$code_record") != "$EXECUTION_CODE_SHA256" ]]; then
        local old_code_sha256
        old_code_sha256=$(tr -d '[:space:]' < "$code_record")
        if [[ -f "$continuation_record" ]]; then
            if [[ $(tr -d '[:space:]' < "$continuation_record") \
                  != "$EXECUTION_CODE_SHA256" ]]; then
                echo "continuation execution code changed after migration" >&2
                exit 2
            fi
        else
            if [[ "${DEEPSEEK_BATCH64_MIGRATE_PRE_PAIR_CODE:-0}" == 1 ]]; then
                local migration_manifest=${DEEPSEEK_BATCH64_CODE_MIGRATION_MANIFEST:-}
                if [[ "${DEEPSEEK_BATCH64_EXPECTED_OLD_EXECUTION_CODE_SHA256:-}" \
                      != "$old_code_sha256" || ! -f "$migration_manifest" ]]; then
                    echo "pre-pair migration lacks the expected old hash or manifest" >&2
                    exit 2
                fi
                if [[ ! -f "$CALIBRATION_ROOT/COMPLETE" \
                      || ! -f "$AUTHORIZATION_ROOT/INCOMPLETE" \
                      || -f "$AUTHORIZATION_ROOT/COMPLETE" \
                      || -e "$GATE_ROOT" || -e "$EPOCH_ROOT" ]]; then
                    echo "pre-pair migration requires completed calibration, failed authorization, and no pair runs" >&2
                    exit 2
                fi
                python3 - "$migration_manifest" "$old_code_sha256" \
                    "$EXECUTION_CODE_SHA256" <<'PY_PRE_PAIR_MIGRATION_VALIDATE'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
required = {
    "status": "PASS",
    "old_execution_code_sha256": sys.argv[2],
    "continuation_execution_code_sha256": sys.argv[3],
    "scope": "pre_pair_response_mask_and_natural_audit_correction",
}
for key, expected in required.items():
    if payload.get(key) != expected:
        raise SystemExit(f"invalid pre-pair migration {key}")
changed = payload.get("changed_files")
required_files = {
    "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
    "verl/trainer/ppo/ray_trainer.py",
    "tools/audit_deepseek_n_f4_formal_run.py",
    "tools/verify_deepseek_kv_cap_run.py",
    "tools/verify_deepseek_batch64_pair.py",
}
if not isinstance(changed, list) or not required_files.issubset(set(changed)):
    raise SystemExit("pre-pair migration changed-file closure is incomplete")
if payload.get("tests", {}).get("status") != "PASS":
    raise SystemExit("pre-pair migration tests did not pass")
PY_PRE_PAIR_MIGRATION_VALIDATE
                cp "$migration_manifest" \
                    "$EXPERIMENT_ROOT/PRE_PAIR_CODE_MIGRATION.json"
                printf '%s\n' "$old_code_sha256" \
                    > "$EXPERIMENT_ROOT/COMMON_EPOCH0_ROLLOUT_EXECUTION_CODE_SHA256"
                printf '%s\n' "$EXECUTION_CODE_SHA256" > "$continuation_record"
                {
                    printf 'export DEEPSEEK_BATCH64_COMMON_ROLLOUT_EXECUTION_CODE_SHA256=%q\n' \
                        "$old_code_sha256"
                    printf 'export DEEPSEEK_BATCH64_CONTINUATION_EXECUTION_CODE_SHA256=%q\n' \
                        "$EXECUTION_CODE_SHA256"
                    printf 'export DEEPSEEK_BATCH64_CODE_MIGRATION_SCOPE=%q\n' \
                        pre_pair_response_mask_and_natural_audit_correction
                    printf 'export DEEPSEEK_BATCH64_CODE_MIGRATION_MANIFEST_SHA256=%q\n' \
                        "$(sha256sum "$EXPERIMENT_ROOT/PRE_PAIR_CODE_MIGRATION.json" | awk '{print $1}')"
                } > "$EXPERIMENT_ROOT/PRE_PAIR_CODE_MIGRATION.env"
            else
                if [[ "${DEEPSEEK_BATCH64_MIGRATE_POSTPROCESS_CODE:-0}" != 1 \
                      || "${DEEPSEEK_BATCH64_EXPECTED_OLD_EXECUTION_CODE_SHA256:-}" \
                         != "$old_code_sha256" ]]; then
                    echo "execution code changed after the batch64 workflow started" >&2
                    exit 2
                fi
                if [[ -e "$CALIBRATION_ROOT" || -e "$AUTHORIZATION_ROOT" \
                      || -e "$GATE_ROOT" || -e "$EPOCH_ROOT" ]]; then
                    echo "postprocessing code migration is forbidden after downstream work starts" >&2
                    exit 2
                fi
                printf '%s\n' "$old_code_sha256" \
                    > "$EXPERIMENT_ROOT/COMMON_EPOCH0_ROLLOUT_EXECUTION_CODE_SHA256"
                printf '%s\n' "$EXECUTION_CODE_SHA256" > "$continuation_record"
                {
                    printf 'export DEEPSEEK_BATCH64_COMMON_ROLLOUT_EXECUTION_CODE_SHA256=%q\n' \
                        "$old_code_sha256"
                    printf 'export DEEPSEEK_BATCH64_CONTINUATION_EXECUTION_CODE_SHA256=%q\n' \
                        "$EXECUTION_CODE_SHA256"
                    printf 'export DEEPSEEK_BATCH64_CODE_MIGRATION_SCOPE=%q\n' \
                        common_epoch0_postprocessing_and_duplicate_prompt_audit
                } > "$EXPERIMENT_ROOT/POSTPROCESS_CODE_MIGRATION.env"
            fi
        fi
    elif [[ -f "$continuation_record" \
            && $(tr -d '[:space:]' < "$continuation_record") \
               != "$EXECUTION_CODE_SHA256" ]]; then
        echo "continuation execution code does not match the workflow code" >&2
        exit 2
    fi
    if [[ -f "$profile_record" \
          && $(tr -d '[:space:]' < "$profile_record") != "$PROFILE_SHA256" ]]; then
        echo "workload profile changed after the batch64 workflow started" >&2
        exit 2
    fi
    if [[ ! -f "$code_record" ]]; then
        printf '%s\n' "$EXECUTION_CODE_SHA256" > "$code_record"
    fi
    if [[ ! -f "$profile_record" ]]; then
        printf '%s\n' "$PROFILE_SHA256" > "$profile_record"
    fi
}

acquire_workflow_lock() {
    exec 8>"$WORKFLOW_LOCK"
    if ! flock -n 8; then
        echo "another batch64 workflow process holds $WORKFLOW_LOCK" >&2
        exit 2
    fi
}

require_idle_npus() {
    local processes
    processes=$(npu-smi info | awk '
        /Process id/ { in_process_table=1; next }
        in_process_table && /^\|/ && !/No running processes found/ { print }
    ')
    if [[ -n "$processes" ]]; then
        echo "batch64 common epoch0 requires idle NPUs" >&2
        printf '%s\n' "$processes" >&2
        exit 2
    fi
}

common_finalized() {
    [[ -f "$COMMON_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
       && -f "$COMMON_ROOT/reuse.env" \
       && -f "$COMMON_ROOT/common_epoch0_metadata.env" \
       && -f "$COMMON_ROOT/MEASURED_GPU_KV_CACHE_TOKENS_PER_RANK" ]]
}

common_complete() {
    common_finalized && [[ -f "$COMMON_AUDIT_MANIFEST" ]] \
        && [[ -f "$COMMON_CHECKPOINT_SHA256" ]] \
        && python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); raise SystemExit(0 if data.get("status") == "PASS" else 1)' \
            "$COMMON_AUDIT_MANIFEST"
}

ensure_checkpoint_digest() {
    require_file "$COMMON_ROOT/reuse.env" "common reuse metadata"
    # shellcheck disable=SC1090
    source "$COMMON_ROOT/reuse.env"
    local observed
    observed=$(python3 "$SCRIPT_DIR/tools/hash_deepseek_checkpoint.py" \
        --checkpoint "$DYNAMIC_INITIAL_RESUME_CKPT")
    if [[ -f "$COMMON_CHECKPOINT_SHA256" ]]; then
        if [[ $(tr -d '[:space:]' < "$COMMON_CHECKPOINT_SHA256") != "$observed" ]]; then
            echo "frozen common checkpoint SHA256 mismatch" >&2
            exit 2
        fi
        return
    fi
    local temporary=$COMMON_ROOT/.FROZEN_CHECKPOINT_SHA256.tmp.$$
    printf '%s\n' "$observed" > "$temporary"
    mv "$temporary" "$COMMON_CHECKPOINT_SHA256"
}

verify_checkpoint_digest() {
    require_file "$COMMON_ROOT/reuse.env" "common reuse metadata"
    # shellcheck disable=SC1090
    source "$COMMON_ROOT/reuse.env"
    require_file "$COMMON_CHECKPOINT_SHA256" "frozen checkpoint SHA256"
    local stage_expected=${1:-}
    local expected
    local observed
    expected=$(tr -d '[:space:]' < "$COMMON_CHECKPOINT_SHA256")
    if [[ -n "$stage_expected" && "$expected" != "$stage_expected" ]]; then
        echo "frozen checkpoint SHA256 record changed during a protected stage" >&2
        echo "expected=$stage_expected observed=$expected" >&2
        exit 2
    fi
    observed=$(python3 "$SCRIPT_DIR/tools/hash_deepseek_checkpoint.py" \
        --checkpoint "$DYNAMIC_INITIAL_RESUME_CKPT")
    if [[ "$observed" != "$expected" ]]; then
        echo "frozen common checkpoint changed after commitment" >&2
        echo "expected=$expected observed=$observed" >&2
        exit 2
    fi
}

semantic_output_passes() {
    [[ -f "$SEMANTIC_OUTPUT" && -f "$ASSET_AUDIT" ]] \
        && python3 - "$SEMANTIC_OUTPUT" "$ASSET_AUDIT" \
            "$CHAT_MODEL_ID" "$CHAT_MODEL_REVISION" \
            "$CHAT_MODEL_PATH" "$CHAT_DISTCP_PATH" <<'PY_SEMANTIC_OUTPUT_VALIDATE'
import json
import sys
from collections import Counter
from pathlib import Path


def fail(message):
    print(f"invalid Chat semantic gate output: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path, name):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {name}: {exc}")
    if not isinstance(value, dict):
        fail(f"{name} must be a JSON object")
    return value


def is_exact_int(value, expected):
    return type(value) is int and value == expected


def same_path(value, expected):
    return isinstance(value, str) and Path(value).resolve() == Path(expected).resolve()


semantic_path, audit_path, model_id, revision, model_path, distcp_path = sys.argv[1:]
semantic = load_json(semantic_path, "semantic smoke")
audit = load_json(audit_path, "asset audit")

if not same_path(semantic.get("model"), model_path):
    fail("semantic smoke model path does not match the pinned Chat model")
if semantic.get("load_format") != "auto":
    fail("semantic smoke load_format must be auto")
if not is_exact_int(semantic.get("expert_parallel_size"), 1):
    fail("semantic smoke expert_parallel_size must be 1")
if not is_exact_int(semantic.get("max_tokens"), 1024):
    fail("semantic smoke max_tokens must be 1024")
if semantic.get("passed") is not True:
    fail("semantic smoke passed must be true")

records = semantic.get("records")
if not isinstance(records, list) or len(records) != 2:
    fail("semantic smoke must contain exactly two records")
if any(not isinstance(record, dict) for record in records):
    fail("every semantic smoke record must be an object")
expected_labels = Counter({"math_chat_primary": 1, "math_chat_secondary": 1})
if Counter(record.get("label") for record in records) != expected_labels:
    fail("semantic smoke must contain one record for each pinned chat prompt")
for record in records:
    label = record["label"]
    if record.get("semantic_smoke_pass") is not True:
        fail(f"{label} did not pass semantic validation")
    if record.get("answer_quality_pass") is not True:
        fail(f"{label} did not pass answer validation")
    if record.get("finish_reason") != "stop":
        fail(f"{label} did not finish with stop")
    if record.get("dialogue_continuation") is not False:
        fail(f"{label} continued the input dialogue")
    if not is_exact_int(record.get("prompt_bos_count"), 1):
        fail(f"{label} must contain exactly one prompt BOS token")

if audit.get("model_id") != model_id:
    fail("asset audit model ID does not match the pinned Chat model")
if audit.get("model_revision") != revision:
    fail("asset audit model revision does not match the pinned revision")
if not same_path(audit.get("model_path"), model_path):
    fail("asset audit model path does not match the pinned Chat model")
if not same_path(audit.get("distcp_path"), distcp_path):
    fail("asset audit distcp path does not match the pinned conversion")
if not is_exact_int(audit.get("pipeline_model_parallel_size"), 4):
    fail("asset audit pipeline parallel size must be 4")
if not is_exact_int(audit.get("expert_model_parallel_size"), 4):
    fail("asset audit expert parallel size must be 4")
PY_SEMANTIC_OUTPUT_VALIDATE
}

semantic_complete() {
    local marker=$SEMANTIC_ROOT/COMPLETE
    [[ -f "$marker" ]] || return 1
    semantic_output_passes || return 1
    if ! python3 - "$marker" "$CHAT_MODEL_ID" "$CHAT_MODEL_REVISION" \
        "$CHAT_MODEL_PATH" "$CHAT_DISTCP_PATH" <<'PY_SEMANTIC_MARKER_VALIDATE'
import sys


marker_path, model_id, revision, model_path, distcp_path = sys.argv[1:]
expected = [
    "COMPLETE DeepSeek-V2-Lite-Chat semantic gate",
    f"MODEL_ID={model_id}",
    f"MODEL_REVISION={revision}",
    f"MODEL_PATH={model_path}",
    f"DISTCP_PATH={distcp_path}",
]
try:
    with open(marker_path, encoding="utf-8") as handle:
        observed = handle.read().splitlines()
except OSError as exc:
    print(f"invalid Chat semantic COMPLETE marker: {exc}", file=sys.stderr)
    raise SystemExit(1)
if observed != expected:
    print("invalid Chat semantic COMPLETE marker identity", file=sys.stderr)
    raise SystemExit(1)
PY_SEMANTIC_MARKER_VALIDATE
    then
        return 1
    fi
    converted_weight_gate_complete || return 1
}

converted_weight_gate_complete() {
    local marker=$CONVERTED_GATE_ROOT/COMPLETE
    [[ -f "$marker" ]] && python3 - "$marker" \
        "$CHAT_MODEL_ID" "$CHAT_MODEL_REVISION" \
        "$CHAT_MODEL_PATH" "$CHAT_DISTCP_PATH" <<'PY_CONVERTED_GATE_VALIDATE'
import sys
from pathlib import Path


path = sys.argv[1]
model_id, revision, model_path, distcp_path = sys.argv[2:]
try:
    lines = open(path, encoding="utf-8").read().splitlines()
except OSError as exc:
    print(f"invalid converted-weight gate marker: {exc}", file=sys.stderr)
    raise SystemExit(1)
if not lines or lines[0] != "COMPLETE DeepSeek actor update probe":
    raise SystemExit("invalid converted-weight gate completion marker")
values = {}
for line in lines[1:]:
    key, separator, value = line.partition("=")
    if not separator or not key or key in values:
        raise SystemExit("invalid converted-weight gate contract")
    values[key] = value
expected = {
    "TRAINING_STEPS": "1",
    "TRAIN_BATCH_SIZE": "32",
    "MAX_RESPONSE_LENGTH": "32",
    "ROLLOUT_N": "1",
    "EXPECTED_ROWS": "32",
    "REQUIRE_SEMANTIC_OUTPUT": "0",
    "ROLLOUT_LOAD_FORMAT": "auto",
    "PRESERVE_INITIAL_HF_WEIGHTS": "0",
    "COMPARE_ONLINE_SYNC_TO_HF": "1",
    "MODEL_ID": model_id,
    "MODEL_REVISION": revision,
}
if any(values.get(key) != value for key, value in expected.items()):
    raise SystemExit("converted-weight gate contract does not match the pinned probe")
if Path(values.get("MODEL_PATH", "")).resolve() != Path(model_path).resolve():
    raise SystemExit("converted-weight gate model path does not match")
if Path(values.get("DISTCP_PATH", "")).resolve() != Path(distcp_path).resolve():
    raise SystemExit("converted-weight gate distcp path does not match")
PY_CONVERTED_GATE_VALIDATE
}

validate_chat_assets() {
    python3 "$SCRIPT_DIR/tools/validate_deepseek_v2_lite_assets.py" \
        --model-path "$CHAT_MODEL_PATH" \
        --distcp-path "$CHAT_DISTCP_PATH" \
        --expected-model-id "$CHAT_MODEL_ID" \
        --expected-revision "$CHAT_MODEL_REVISION" \
        --expected-pp-size 4 \
        --expected-ep-size 4
}

run_semantic() {
    if semantic_complete; then
        validate_chat_assets > "$ASSET_AUDIT"
        echo "[batch64] Chat semantic gate already complete: $SEMANTIC_ROOT"
        return
    fi
    if [[ -e "$SEMANTIC_ROOT" && ! -f "$SEMANTIC_ROOT/INCOMPLETE" ]]; then
        echo "incomplete Chat semantic gate exists: $SEMANTIC_ROOT" >&2
        exit 2
    fi
    require_idle_npus
    mkdir -p "$SEMANTIC_ROOT"
    printf '%s\n' "INCOMPLETE DeepSeek-V2-Lite-Chat semantic gate" \
        > "$SEMANTIC_ROOT/INCOMPLETE"
    validate_chat_assets > "$ASSET_AUDIT"
    if [[ ! -f "$SEMANTIC_OUTPUT" ]]; then
        env \
            MODEL_PATH="$CHAT_MODEL_PATH" \
            DEEPSEEK_SEMANTIC_SMOKE_EP_SIZE=1 \
            DEEPSEEK_SEMANTIC_SMOKE_MAX_TOKENS=1024 \
            DEEPSEEK_SEMANTIC_SMOKE_MAX_MODEL_LEN=2048 \
            DEEPSEEK_SEMANTIC_SMOKE_REQUESTS_PER_RANK=2 \
            DEEPSEEK_SEMANTIC_SMOKE_OUTPUT="$SEMANTIC_OUTPUT" \
            "$SCRIPT_DIR/run_deepseek_v2_lite_semantic_smoke.sh"
    fi
    semantic_output_passes || {
        echo "DeepSeek-V2-Lite-Chat failed the semantic gate" >&2
        exit 3
    }
    if ! converted_weight_gate_complete; then
        if [[ -e "$CONVERTED_GATE_ROOT" ]]; then
            echo "incomplete converted-weight gate exists: $CONVERTED_GATE_ROOT" >&2
            exit 2
        fi
        env \
            MODEL_PATH="$CHAT_MODEL_PATH" \
            MODEL_ID="$CHAT_MODEL_ID" \
            MODEL_REVISION="$CHAT_MODEL_REVISION" \
            DISTCP_PATH="$CHAT_DISTCP_PATH" \
            DEEPSEEK_ACTOR_PROBE_OUTPUT_ROOT="$SEMANTIC_ROOT" \
            DEEPSEEK_WEIGHT_COMPARE_SMOKE_RUN_NAME="$CONVERTED_GATE_RUN_NAME" \
            "$SCRIPT_DIR/run_deepseek_v2_lite_weight_compare_smoke.sh"
    fi
    converted_weight_gate_complete || {
        echo "DeepSeek-V2-Lite-Chat failed the converted-weight gate" >&2
        exit 3
    }
    rm -f "$SEMANTIC_ROOT/INCOMPLETE"
    printf '%s\n' \
        "COMPLETE DeepSeek-V2-Lite-Chat semantic gate" \
        "MODEL_ID=$CHAT_MODEL_ID" \
        "MODEL_REVISION=$CHAT_MODEL_REVISION" \
        "MODEL_PATH=$CHAT_MODEL_PATH" \
        "DISTCP_PATH=$CHAT_DISTCP_PATH" \
        > "$SEMANTIC_ROOT/COMPLETE"
}

audit_common() {
    common_finalized || {
        echo "batch64 common epoch0 is not finalized" >&2
        exit 2
    }
    local common_runtime_sha256
    local continuation_sha256
    local measured_kv_tokens
    common_runtime_sha256=$(tr -d '[:space:]' \
        < "$EXPERIMENT_ROOT/EXECUTION_CODE_SHA256")
    if [[ -f "$EXPERIMENT_ROOT/CONTINUATION_EXECUTION_CODE_SHA256" ]]; then
        continuation_sha256=$(tr -d '[:space:]' \
            < "$EXPERIMENT_ROOT/CONTINUATION_EXECUTION_CODE_SHA256")
    else
        continuation_sha256=$common_runtime_sha256
    fi
    measured_kv_tokens=$(tr -d '[:space:]' \
        < "$COMMON_ROOT/MEASURED_GPU_KV_CACHE_TOKENS_PER_RANK")
    python3 "$SCRIPT_DIR/tools/audit_deepseek_common_epoch0.py" \
        --common-root "$COMMON_ROOT" \
        --expected-steps "$COMMON_EPOCH0_TRAIN_STEPS" \
        --expected-batch-size "$COMMON_EPOCH0_TRAIN_BATCH_SIZE" \
        --expected-rollout-n "$COMMON_EPOCH0_ROLLOUT_N" \
        --expected-workload-profile-id "$DEEPSEEK_WORKLOAD_PROFILE_ID" \
        --expected-workload-profile-sha256 "$PROFILE_SHA256" \
        --expected-common-runtime-sha256 "$common_runtime_sha256" \
        --expected-continuation-sha256 "$continuation_sha256" \
        --expected-model-path "$CHAT_MODEL_PATH" \
        --expected-model-revision "$CHAT_MODEL_REVISION" \
        --expected-distcp-path "$CHAT_DISTCP_PATH" \
        --expected-train-file /data/deepscaler/train.parquet \
        --expected-test-file /data/deepscaler/test.parquet \
        --expected-unique-prompts 317 \
        --expected-duplicate-occurrences 3 \
        --expected-duplicate-policy latest_occurrence \
        --expected-preemption-policy "$COMMON_EPOCH0_PREEMPTION_POLICY" \
        --expected-measured-kv-tokens "$measured_kv_tokens" \
        --expected-distcp-count 32 \
        --block-size 128 \
        --max-response-length "$COMMON_EPOCH0_MAX_RESPONSE_LENGTH" \
        --max-clip-ratio 0.10 \
        --min-distinct-prompt-maxima 8 \
        --output "$COMMON_AUDIT_MANIFEST" \
        --force
}

run_common() {
    semantic_complete || {
        echo "batch64 common epoch0 requires the Chat semantic gate" >&2
        exit 2
    }
    validate_chat_assets > "$ASSET_AUDIT"
    if common_complete; then
        audit_common
        ensure_checkpoint_digest
        echo "[batch64] common epoch0 already complete: $COMMON_ROOT"
        return
    fi
    if common_finalized; then
        audit_common
        ensure_checkpoint_digest
        echo "[batch64] common epoch0 audit complete: $COMMON_ROOT"
        return
    fi
    if [[ -e "$COMMON_ROOT" ]]; then
        echo "incomplete batch64 common epoch0 exists: $COMMON_ROOT" >&2
        exit 2
    fi
    require_idle_npus
    echo "[batch64] starting common epoch0"
    env \
        DEEPSEEK_WORKLOAD_PROFILE_PATH="$PROFILE_PATH" \
        DEEPSEEK_KV_CAP_ENV="$CAP_ENV" \
        COMMON_EPOCH0_OUTPUT_ROOT="$EXPERIMENT_ROOT" \
        COMMON_EPOCH0_RUN_NAME="$COMMON_EPOCH0_RUN_NAME" \
        COMMON_EPOCH0_KV_TOKENS_PER_RANK=auto \
        COMMON_EPOCH0_ORIGINAL_EXECUTION_CODE_SHA256="$EXECUTION_CODE_SHA256" \
        MODEL_PATH="$CHAT_MODEL_PATH" \
        MODEL_REVISION="$CHAT_MODEL_REVISION" \
        DISTCP_PATH="$CHAT_DISTCP_PATH" \
        "$SCRIPT_DIR/run_deepseek_v2_lite_common_epoch0.sh"
    common_finalized || {
        echo "batch64 common epoch0 returned without finalized artifacts" >&2
        exit 3
    }
    audit_common
    ensure_checkpoint_digest
    common_complete || {
        echo "batch64 common epoch0 audit did not commit a PASS manifest" >&2
        exit 3
    }
}

recover_common() {
    semantic_complete || {
        echo "batch64 common recovery requires the Chat semantic gate" >&2
        exit 2
    }
    validate_chat_assets > "$ASSET_AUDIT"
    if common_complete; then
        audit_common
        ensure_checkpoint_digest
        echo "[batch64] common epoch0 already complete: $COMMON_ROOT"
        return
    fi
    if common_finalized; then
        audit_common
        ensure_checkpoint_digest
        echo "[batch64] common epoch0 audit complete: $COMMON_ROOT"
        return
    fi
    if [[ ! -d "$COMMON_ROOT" || ! -f "$COMMON_ROOT/INCOMPLETE" ]]; then
        echo "batch64 common epoch0 is not in a recoverable incomplete state" >&2
        exit 2
    fi
    if [[ -e "$CALIBRATION_ROOT" || -e "$AUTHORIZATION_ROOT" \
          || -e "$CAP_ENV" || -e "$GATE_ROOT" || -e "$EPOCH_ROOT" ]]; then
        echo "common epoch0 recovery is forbidden after downstream work starts" >&2
        exit 2
    fi
    local original_code_sha256
    original_code_sha256=$(tr -d '[:space:]' \
        < "$EXPERIMENT_ROOT/EXECUTION_CODE_SHA256")
    echo "[batch64] validating and recovering completed common epoch0 training"
    env \
        DEEPSEEK_WORKLOAD_PROFILE_PATH="$PROFILE_PATH" \
        DEEPSEEK_KV_CAP_ENV="$CAP_ENV" \
        COMMON_EPOCH0_OUTPUT_ROOT="$EXPERIMENT_ROOT" \
        COMMON_EPOCH0_RUN_NAME="$COMMON_EPOCH0_RUN_NAME" \
        COMMON_EPOCH0_KV_TOKENS_PER_RANK=auto \
        COMMON_EPOCH0_FINALIZE_EXISTING=1 \
        COMMON_EPOCH0_ORIGINAL_EXECUTION_CODE_SHA256="$original_code_sha256" \
        MODEL_PATH="$CHAT_MODEL_PATH" \
        MODEL_REVISION="$CHAT_MODEL_REVISION" \
        DISTCP_PATH="$CHAT_DISTCP_PATH" \
        "$SCRIPT_DIR/run_deepseek_v2_lite_common_epoch0.sh"
    common_finalized || {
        echo "batch64 common epoch0 recovery returned without finalized artifacts" >&2
        exit 3
    }
    audit_common
    ensure_checkpoint_digest
    common_complete || {
        echo "batch64 common recovery audit did not commit a PASS manifest" >&2
        exit 3
    }
}

run_trigger() {
    if [[ -f "$TRIGGER_ROOT/kv_probe_trigger_manifest.json" ]]; then
        python3 "$SCRIPT_DIR/tools/prepare_deepseek_kv_probe_trigger.py" verify \
            --output-root "$TRIGGER_ROOT" \
            --train-file /data/deepscaler/train.parquet \
            --dataset-fraction "$DEEPSEEK_KV_PROBE_DATASET_FRACTION" \
            --tokenizer-path "$CHAT_MODEL_PATH"
        echo "[batch64] trigger already complete: $TRIGGER_ROOT"
        return
    fi
    if [[ -e "$TRIGGER_ROOT" ]]; then
        echo "incomplete batch64 trigger exists: $TRIGGER_ROOT" >&2
        exit 2
    fi
    python3 "$SCRIPT_DIR/tools/prepare_deepseek_kv_probe_trigger.py" build \
        --source-root "$SOURCE_HISTORY_ROOT" \
        --output-root "$TRIGGER_ROOT" \
        --train-file /data/deepscaler/train.parquet \
        --dataset-fraction "$DEEPSEEK_KV_PROBE_DATASET_FRACTION" \
        --tokenizer-path "$CHAT_MODEL_PATH" \
        --prompt-count "$COMMON_EPOCH0_TRAIN_BATCH_SIZE" \
        --responses-per-prompt "$COMMON_EPOCH0_ROLLOUT_N" \
        --max-response 64 \
        --source-steps 1,2
    python3 "$SCRIPT_DIR/tools/prepare_deepseek_kv_probe_trigger.py" verify \
        --output-root "$TRIGGER_ROOT" \
        --train-file /data/deepscaler/train.parquet \
        --dataset-fraction "$DEEPSEEK_KV_PROBE_DATASET_FRACTION" \
        --tokenizer-path "$CHAT_MODEL_PATH"
    echo "[batch64] trigger complete: $TRIGGER_ROOT"
}

run_calibration() {
    local frozen_checkpoint_sha256
    common_complete || {
        echo "batch64 calibration requires completed common epoch0" >&2
        exit 2
    }
    audit_common
    verify_checkpoint_digest
    frozen_checkpoint_sha256=$(tr -d '[:space:]' < "$COMMON_CHECKPOINT_SHA256")
    require_file "$TRIGGER_ROOT/kv_probe_trigger_manifest.json" "batch64 trigger"
    if [[ -f "$CALIBRATION_ROOT/COMPLETE" && -f "$CAP_ENV" ]]; then
        echo "[batch64] Natural floor2 calibration already complete: $CALIBRATION_ROOT"
        return
    fi
    if [[ -e "$CALIBRATION_ROOT" || -e "$CAP_ENV" ]]; then
        echo "incomplete or conflicting batch64 calibration artifacts exist" >&2
        echo "calibration=$CALIBRATION_ROOT caps=$CAP_ENV" >&2
        exit 2
    fi
    local shared_full16
    shared_full16=$(tr -d '[:space:]' \
        < "$COMMON_ROOT/MEASURED_GPU_KV_CACHE_TOKENS_PER_RANK")
    if ! [[ "$shared_full16" =~ ^[1-9][0-9]*$ ]] \
       || (( shared_full16 % 128 != 0 )); then
        echo "invalid measured common Full16 KV capacity: $shared_full16" >&2
        exit 2
    fi
    echo "[batch64] starting Natural floor2 capacity calibration"
    env \
        DEEPSEEK_WORKLOAD_PROFILE_PATH="$PROFILE_PATH" \
        COMMON_EPOCH0_ROOT="$COMMON_ROOT" \
        DEEPSEEK_KV_PROBE_HISTORY_ROOT="$TRIGGER_ROOT" \
        DEEPSEEK_N_F2_KV_CALIBRATION_ROOT="$CALIBRATION_ROOT" \
        DEEPSEEK_N_F2_KV_CAP_ENV="$CAP_ENV" \
        DEEPSEEK_KV_CAP_ENV="$CAP_ENV" \
        DEEPSEEK_SHARED_FULL16_PHYSICAL_TOKENS="$shared_full16" \
        "$SCRIPT_DIR/run_deepseek_v2_lite_natural_f2_calibration.sh"
    verify_checkpoint_digest "$frozen_checkpoint_sha256"
    require_file "$CALIBRATION_ROOT/COMPLETE" "calibration completion marker"
    require_file "$CAP_ENV" "candidate batch64 cap file"
}

run_authorization() {
    local frozen_checkpoint_sha256
    verify_checkpoint_digest
    frozen_checkpoint_sha256=$(tr -d '[:space:]' < "$COMMON_CHECKPOINT_SHA256")
    require_file "$CALIBRATION_ROOT/COMPLETE" "batch64 calibration"
    require_file "$CAP_ENV" "candidate batch64 cap file"
    if [[ -f "$AUTHORIZATION_ROOT/COMPLETE" ]] \
       && grep -q '^export DEEPSEEK_N_F2_KV_CAPS_VERIFIED=1$' "$CAP_ENV"; then
        echo "[batch64] Natural floor2 authorization already complete: $AUTHORIZATION_ROOT"
        return
    fi
    if [[ -e "$AUTHORIZATION_ROOT" ]]; then
        if [[ ! -f "$AUTHORIZATION_ROOT/INCOMPLETE" \
              || ! -f "$EXPERIMENT_ROOT/PRE_PAIR_CODE_MIGRATION.env" ]]; then
            echo "incomplete batch64 authorization exists: $AUTHORIZATION_ROOT" >&2
            exit 2
        fi
        echo "[batch64] resuming Natural floor2 authorization audit"
        env \
            DEEPSEEK_WORKLOAD_PROFILE_PATH="$PROFILE_PATH" \
            DEEPSEEK_KV_CAP_ENV="$CAP_ENV" \
            COMMON_EPOCH0_ROOT="$COMMON_ROOT" \
            DEEPSEEK_KV_PROBE_HISTORY_ROOT="$TRIGGER_ROOT" \
            DEEPSEEK_KV_CAP_VALIDATION_OUTPUT_ROOT="$AUTHORIZATION_ROOT" \
            DEEPSEEK_KV_CAP_VALIDATION_RESUME_AUDIT=1 \
            DEEPSEEK_KV_CAP_EXPECTED_RUNTIME_EXECUTION_SHA256="$(tr -d '[:space:]' < "$EXPERIMENT_ROOT/EXECUTION_CODE_SHA256")" \
            DEEPSEEK_KV_CAP_EXPECTED_VERIFICATION_CODE_SHA256="$EXECUTION_CODE_SHA256" \
            "$SCRIPT_DIR/run_deepseek_v2_lite_kv_cap_validation.sh" natural_f2
    else
        echo "[batch64] starting strict Natural floor2 cap authorization"
        env \
            DEEPSEEK_WORKLOAD_PROFILE_PATH="$PROFILE_PATH" \
            DEEPSEEK_KV_CAP_ENV="$CAP_ENV" \
            COMMON_EPOCH0_ROOT="$COMMON_ROOT" \
            DEEPSEEK_KV_PROBE_HISTORY_ROOT="$TRIGGER_ROOT" \
            DEEPSEEK_KV_CAP_VALIDATION_OUTPUT_ROOT="$AUTHORIZATION_ROOT" \
            "$SCRIPT_DIR/run_deepseek_v2_lite_kv_cap_validation.sh" natural_f2
    fi
    verify_checkpoint_digest "$frozen_checkpoint_sha256"
    require_file "$AUTHORIZATION_ROOT/COMPLETE" "authorization completion marker"
    if ! grep -q '^export DEEPSEEK_N_F2_KV_CAPS_VERIFIED=1$' "$CAP_ENV"; then
        echo "authorization did not promote Natural floor2 caps" >&2
        exit 3
    fi
}

load_pair_contract() {
    audit_common
    require_file "$COMMON_ROOT/common_epoch0_metadata.env" "common metadata"
    require_file "$COMMON_ROOT/reuse.env" "common reuse metadata"
    require_file "$CAP_ENV" "verified batch64 caps"
    # shellcheck disable=SC1090
    source "$COMMON_ROOT/common_epoch0_metadata.env"
    # shellcheck disable=SC1090
    source "$COMMON_ROOT/reuse.env"
    # shellcheck disable=SC1090
    source "$CAP_ENV"
    verify_checkpoint_digest
    if [[ "${DEEPSEEK_N_F2_KV_CAPS_VERIFIED:-0}" != 1 ]]; then
        echo "Natural floor2 caps are not authorized" >&2
        exit 2
    fi
    if [[ "${DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID:-}" \
          != "$DEEPSEEK_WORKLOAD_PROFILE_ID" \
          || "${DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256:-}" \
             != "$PROFILE_SHA256" ]]; then
        echo "verified caps do not match the batch64 workload profile" >&2
        exit 2
    fi
}

assert_execution_code_unchanged() {
    local observed
    observed=$(python3 "$SCRIPT_DIR/tools/hash_deepseek_execution_code.py" \
        --root "$SCRIPT_DIR")
    if [[ "$observed" != "$EXECUTION_CODE_SHA256" ]]; then
        echo "execution code changed while the batch64 workflow was running" >&2
        echo "started=$EXECUTION_CODE_SHA256 observed=$observed" >&2
        exit 2
    fi
}

arm_run_name() {
    case "$1" in
        vanilla) printf '%s' deepseek_v2_lite_vanilla_common_epoch0_epoch1_2 ;;
        adafloor) printf '%s' deepseek_v2_lite_adafloor_n_f2_common_epoch0_epoch1_2 ;;
        *) echo "invalid pair arm: $1" >&2; exit 2 ;;
    esac
}

pair_dataset_fraction() {
    if [[ "$1" == gate ]]; then
        printf '%s' "$DEEPSEEK_KV_PROBE_DATASET_FRACTION"
    else
        printf '%s' "$COMMON_EPOCH0_DATASET_FRACTION"
    fi
}

write_pair_manifest() {
    local arm=$1
    local phase=$2
    local run_root=$3
    local manifest=$run_root/batch64_pair_manifest.env
    local temporary
    local cap_sha256
    cap_sha256=$(sha256sum "$CAP_ENV")
    cap_sha256=${cap_sha256%% *}
    temporary=$(mktemp "$run_root/.batch64_pair_manifest.env.tmp.XXXXXX")
    {
        printf 'export DEEPSEEK_BATCH64_ARM=%q\n' "$arm"
        printf 'export DEEPSEEK_BATCH64_PHASE=%q\n' "$phase"
        printf 'export DEEPSEEK_WORKLOAD_PROFILE_ID=%q\n' "$DEEPSEEK_WORKLOAD_PROFILE_ID"
        printf 'export DEEPSEEK_WORKLOAD_PROFILE_SHA256=%q\n' "$PROFILE_SHA256"
        printf 'export DEEPSEEK_BATCH64_COMMON_ROOT=%q\n' "$COMMON_ROOT"
        printf 'export DEEPSEEK_BATCH64_FROZEN_CHECKPOINT=%q\n' "$DYNAMIC_INITIAL_RESUME_CKPT"
        printf 'export DEEPSEEK_BATCH64_MODEL_PATH=%q\n' "$COMMON_EPOCH0_MODEL_PATH"
        printf 'export DEEPSEEK_BATCH64_MODEL_REVISION=%q\n' "$COMMON_EPOCH0_MODEL_REVISION"
        printf 'export DEEPSEEK_BATCH64_EXECUTION_PROFILE=%q\n' "$COMMON_EPOCH0_EXECUTION_PROFILE_USED"
        printf 'export DEEPSEEK_BATCH64_CAP_ENV_SHA256=%q\n' "$cap_sha256"
        printf 'export DEEPSEEK_BATCH64_EXECUTION_CODE_SHA256=%q\n' \
            "$EXECUTION_CODE_SHA256"
        printf 'export DEEPSEEK_BATCH64_FROZEN_CHECKPOINT_SHA256=%q\n' \
            "$(tr -d '[:space:]' < "$COMMON_CHECKPOINT_SHA256")"
        if [[ -f "$EXPERIMENT_ROOT/COMMON_EPOCH0_ROLLOUT_EXECUTION_CODE_SHA256" ]]; then
            printf 'export DEEPSEEK_BATCH64_COMMON_ROLLOUT_EXECUTION_CODE_SHA256=%q\n' \
                "$(tr -d '[:space:]' < "$EXPERIMENT_ROOT/COMMON_EPOCH0_ROLLOUT_EXECUTION_CODE_SHA256")"
        fi
        printf 'export DEEPSEEK_BATCH64_PAIRED_REQUEST_SAMPLING_SEEDS=1\n'
        printf 'export DEEPSEEK_BATCH64_TRAIN_BATCH_SIZE=%s\n' "$COMMON_EPOCH0_TRAIN_BATCH_SIZE"
        printf 'export DEEPSEEK_BATCH64_ROLLOUT_N=%s\n' "$COMMON_EPOCH0_ROLLOUT_N"
        printf 'export DEEPSEEK_BATCH64_MAX_NUM_SEQS=%s\n' "$COMMON_EPOCH0_MAX_NUM_SEQS"
        printf 'export DEEPSEEK_BATCH64_MAX_PROMPT_LENGTH=%s\n' "$COMMON_EPOCH0_MAX_PROMPT_LENGTH"
        printf 'export DEEPSEEK_BATCH64_MAX_RESPONSE_LENGTH=%s\n' "$COMMON_EPOCH0_MAX_RESPONSE_LENGTH"
        printf 'export DEEPSEEK_BATCH64_MAX_NUM_BATCHED_TOKENS=%s\n' "$COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS"
        printf 'export DEEPSEEK_BATCH64_FULL16_PHYSICAL_TOKENS=%s\n' "$DEEPSEEK_KV_CAP_SHARED_FULL16_PHYSICAL_TOKENS"
        printf 'export DEEPSEEK_BATCH64_TEMPERATURE=0.9\n'
        printf 'export DEEPSEEK_BATCH64_TOP_P=0.9\n'
        printf 'export DEEPSEEK_BATCH64_TOP_K=50\n'
        printf 'export DEEPSEEK_BATCH64_DATASET_FRACTION=%s\n' \
            "$(pair_dataset_fraction "$phase")"
        if [[ "$phase" == gate && "$arm" == adafloor ]]; then
            printf 'export DEEPSEEK_BATCH64_FORCED_SELECTED_FLOOR=%s\n' \
                "$GATE_ADAFLOOR_FORCE_SELECTED_FLOOR"
        else
            printf 'export DEEPSEEK_BATCH64_FORCED_SELECTED_FLOOR=none\n'
        fi
    } > "$temporary"
    if [[ -f "$manifest" ]]; then
        if ! cmp -s "$temporary" "$manifest"; then
            rm -f "$temporary"
            echo "refusing to rewrite a stale pair manifest: $manifest" >&2
            exit 2
        fi
        rm -f "$temporary"
    else
        mv "$temporary" "$manifest"
    fi
}

run_pair_arm() {
    local arm=$1
    local phase=$2
    local phase_root=$3
    local prompts=$4
    local steps=$5
    local variant
    local run_name
    local run_root
    local dataset_fraction
    local force_floor_env=()
    if [[ "$arm" == vanilla ]]; then
        variant=vanilla
    else
        variant=adafloor_n_f2
    fi
    run_name=$(arm_run_name "$arm")
    run_root=$phase_root/$run_name
    dataset_fraction=$(pair_dataset_fraction "$phase")
    if [[ "$phase" == gate && "$arm" == adafloor ]]; then
        force_floor_env+=(
            "DEEPSEEK_FAIR_FORCE_SELECTED_FLOOR=$GATE_ADAFLOOR_FORCE_SELECTED_FLOOR"
        )
    fi
    assert_execution_code_unchanged
    if [[ -f "$run_root/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; then
        echo "[batch64] $phase $arm arm already complete: $run_root"
        write_pair_manifest "$arm" "$phase" "$run_root"
        return
    fi
    if [[ -e "$run_root" ]]; then
        echo "incomplete batch64 $phase $arm arm exists: $run_root" >&2
        exit 2
    fi
    echo "[batch64] starting $phase $arm arm"
    env \
        "${force_floor_env[@]}" \
        DEEPSEEK_FAIR_DATASET_FRACTION="$dataset_fraction" \
        DEEPSEEK_WORKLOAD_PROFILE_PATH="$PROFILE_PATH" \
        DEEPSEEK_KV_CAP_ENV="$CAP_ENV" \
        COMMON_EPOCH0_ROOT="$COMMON_ROOT" \
        FAIR_OUTPUT_ROOT="$phase_root" \
        FAIR_START_EPOCH=1 \
        FAIR_TOTAL_EPOCHS=2 \
        FAIR_FREEZE_ACTOR=1 \
        FAIR_PROMPTS_PER_EPOCH="$prompts" \
        FAIR_EXPECTED_STEPS="$steps" \
        FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
        "$SCRIPT_DIR/run_deepseek_v2_lite_fair_compare.sh" "$variant"
    assert_execution_code_unchanged
    verify_checkpoint_digest
    require_file "$run_root/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" \
        "$phase $arm validation record"
    write_pair_manifest "$arm" "$phase" "$run_root"
}

verify_pair() {
    local phase=$1
    local phase_root
    local summary
    local vanilla_root
    local adafloor_root
    if [[ "$phase" == gate ]]; then
        phase_root=$GATE_ROOT
        summary=$phase_root/paired_gate_summary.json
    else
        phase_root=$EPOCH_ROOT
        summary=$phase_root/paired_epoch_summary.json
    fi
    vanilla_root=$phase_root/$(arm_run_name vanilla)
    adafloor_root=$phase_root/$(arm_run_name adafloor)
    assert_execution_code_unchanged
    python3 "$SCRIPT_DIR/tools/verify_deepseek_batch64_pair.py" \
        --phase "$phase" \
        --vanilla-run-dir "$vanilla_root" \
        --adafloor-run-dir "$adafloor_root" \
        --common-root "$COMMON_ROOT" \
        --cap-env "$CAP_ENV" \
        --workload-profile-env "$PROFILE_PATH" \
        --expected-execution-code-sha256 "$EXECUTION_CODE_SHA256" \
        --output "$summary"
    assert_execution_code_unchanged
    echo "[batch64] paired $phase summary: $summary"
}

require_verified_gate() {
    if [[ ! -f "$GATE_ROOT/paired_gate_summary.json" ]]; then
        echo "batch64 epoch requires a completed one-step paired gate" >&2
        exit 2
    fi
    verify_pair gate
}

run_pair() {
    local phase=$1
    local phase_root
    local prompts
    local steps
    local dataset_fraction
    load_pair_contract
    if [[ "$phase" == epoch ]]; then
        require_verified_gate
    fi
    if [[ "$phase" == gate ]]; then
        phase_root=$GATE_ROOT
        prompts=$COMMON_EPOCH0_TRAIN_BATCH_SIZE
        steps=1
    else
        phase_root=$EPOCH_ROOT
        prompts=$COMMON_EPOCH0_PROMPTS_TOTAL
        steps=$COMMON_EPOCH0_TRAIN_STEPS
    fi
    mkdir -p "$phase_root"
    run_pair_arm vanilla "$phase" "$phase_root" "$prompts" "$steps"
    run_pair_arm adafloor "$phase" "$phase_root" "$prompts" "$steps"
    verify_pair "$phase"
}

dry_run_pair() {
    local phase=$1
    local prompts
    local steps
    load_pair_contract
    if [[ "$phase" == gate ]]; then
        prompts=$COMMON_EPOCH0_TRAIN_BATCH_SIZE
        steps=1
    else
        prompts=$COMMON_EPOCH0_PROMPTS_TOTAL
        steps=$COMMON_EPOCH0_TRAIN_STEPS
    fi
    dataset_fraction=$(pair_dataset_fraction "$phase")
    for variant in vanilla adafloor_n_f2; do
        local force_floor_env=()
        if [[ "$phase" == gate && "$variant" == adafloor_n_f2 ]]; then
            force_floor_env+=(
                "DEEPSEEK_FAIR_FORCE_SELECTED_FLOOR=$GATE_ADAFLOOR_FORCE_SELECTED_FLOOR"
            )
        fi
        env \
            "${force_floor_env[@]}" \
            DEEPSEEK_FAIR_DATASET_FRACTION="$dataset_fraction" \
            DEEPSEEK_WORKLOAD_PROFILE_PATH="$PROFILE_PATH" \
            DEEPSEEK_KV_CAP_ENV="$CAP_ENV" \
            COMMON_EPOCH0_ROOT="$COMMON_ROOT" \
            FAIR_OUTPUT_ROOT="$EXPERIMENT_ROOT/dry_run_$phase" \
            FAIR_START_EPOCH=1 \
            FAIR_TOTAL_EPOCHS=2 \
            FAIR_FREEZE_ACTOR=1 \
            FAIR_PROMPTS_PER_EPOCH="$prompts" \
            FAIR_EXPECTED_STEPS="$steps" \
            DEEPSEEK_FAIR_DRY_RUN=1 \
            DEEPSEEK_VALIDATE_ASSETS_ON_LAUNCH=0 \
            "$SCRIPT_DIR/run_deepseek_v2_lite_fair_compare.sh" "$variant"
    done
}

pair_summary_complete() {
    local summary=$1
    local expected_phase=$2
    [[ -f "$summary" ]] || return 1
    python3 - "$summary" "$expected_phase" "$EXECUTION_CODE_SHA256" <<'PY'
import json
import sys

try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
if payload.get("status") != "PASS" or payload.get("phase") != sys.argv[2]:
    raise SystemExit(1)
if payload.get("provenance", {}).get("execution_code_sha256") != sys.argv[3]:
    raise SystemExit(1)
PY
}

show_status() {
    local gate_status=pending
    local epoch_status=pending
    if pair_summary_complete "$GATE_ROOT/paired_gate_summary.json" gate; then
        gate_status=complete
    elif [[ -f "$GATE_ROOT/paired_gate_summary.json" ]]; then
        gate_status=failed_or_stale
    fi
    if pair_summary_complete "$EPOCH_ROOT/paired_epoch_summary.json" epoch; then
        epoch_status=complete
    elif [[ -f "$EPOCH_ROOT/paired_epoch_summary.json" ]]; then
        epoch_status=failed_or_stale
    fi
    printf '%s\n' \
        "experiment_root=$EXPERIMENT_ROOT" \
        "execution_code_sha256=$EXECUTION_CODE_SHA256" \
        "common_rollout_execution_code_sha256=$([[ -f "$EXPERIMENT_ROOT/EXECUTION_CODE_SHA256" ]] && tr -d '[:space:]' < "$EXPERIMENT_ROOT/EXECUTION_CODE_SHA256" || echo unrecorded)" \
        "continuation_execution_code_sha256=$([[ -f "$EXPERIMENT_ROOT/CONTINUATION_EXECUTION_CODE_SHA256" ]] && tr -d '[:space:]' < "$EXPERIMENT_ROOT/CONTINUATION_EXECUTION_CODE_SHA256" || echo none)" \
        "workload_profile_sha256=$PROFILE_SHA256" \
        "semantic=$(semantic_complete && echo complete || echo pending)" \
        "common=$(common_complete && echo complete || echo pending)" \
        "trigger=$([[ -f "$TRIGGER_ROOT/kv_probe_trigger_manifest.json" ]] && echo complete || echo pending)" \
        "calibration=$([[ -f "$CALIBRATION_ROOT/COMPLETE" ]] && echo complete || echo pending)" \
        "authorization=$([[ -f "$AUTHORIZATION_ROOT/COMPLETE" ]] && echo complete || echo pending)" \
        "gate=$gate_status" \
        "epoch=$epoch_status"
}

case "$PHASE" in
    status)
        show_status
        exit 0
        ;;
    semantic|common|recover-common|trigger|calibrate|authorize|gate|epoch|verify-gate|verify-epoch|dry-run-gate|dry-run-epoch|all)
        ;;
    *)
        echo "unknown batch64 phase: $PHASE" >&2
        usage >&2
        exit 2
        ;;
esac

record_or_validate_contract
acquire_workflow_lock

case "$PHASE" in
    semantic) run_semantic ;;
    common) run_common ;;
    recover-common) recover_common ;;
    trigger) run_trigger ;;
    calibrate) run_calibration ;;
    authorize) run_authorization ;;
    gate) run_pair gate ;;
    epoch) run_pair epoch ;;
    verify-gate)
        load_pair_contract
        verify_pair gate
        ;;
    verify-epoch)
        load_pair_contract
        require_verified_gate
        verify_pair epoch
        ;;
    dry-run-gate) dry_run_pair gate ;;
    dry-run-epoch) dry_run_pair epoch ;;
    all)
        run_semantic
        run_common
        run_trigger
        run_calibration
        run_authorization
        dry_run_pair gate
        run_pair gate
        dry_run_pair epoch
        run_pair epoch
        ;;
esac

show_status
