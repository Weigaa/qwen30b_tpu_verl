#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

usage() {
    cat <<'EOF'
Usage: ./run_deepseek_v2_lite_batch64_fixed_work.sh PHASE

PHASE is one of status, prepare, dry-run-gate, gate, natural-epoch,
fixed-epoch, epoch, verify-gate, verify-epoch, or all.

The fixed-work arms replay the same per-request response lengths on Full16 and
AdaFloor. Natural-generation arms remain separate quality and stability runs.
EOF
}

if (( $# != 1 )) || [[ "$1" == -h || "$1" == --help ]]; then
    usage
    [[ "${1:-}" == -h || "${1:-}" == --help ]] && exit 0
    exit 2
fi
PHASE=$1

ACTIVE_SOURCE_RECORD=/data/adafloor_shared_state/deepseek_v2_lite/.active_batch64_experiment_root
if [[ ! -f "$ACTIVE_SOURCE_RECORD" ]]; then
    echo "missing source batch64 experiment record: $ACTIVE_SOURCE_RECORD" >&2
    exit 2
fi
SOURCE_ROOT=${DEEPSEEK_FIXED_WORK_SOURCE_ROOT:-$(tr -d '\r\n' < "$ACTIVE_SOURCE_RECORD")}
COMMON_RUN_NAME=common_epoch0_deepseek_v2_lite_chat_batch64_n16_tq1_no_overlap_gpu09
PROFILE_PATH=${DEEPSEEK_BATCH64_WORKLOAD_PROFILE_PATH:-$SCRIPT_DIR/internal/deepseek_v2_lite_batch64_workload_profile.sh}
PROFILE_PATH=$(realpath "$PROFILE_PATH")
EXECUTION_CODE_SHA256=$(python3 "$SCRIPT_DIR/tools/hash_deepseek_execution_code.py" --root "$SCRIPT_DIR")
OUTPUT_ROOT=${DEEPSEEK_FIXED_WORK_OUTPUT_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/p0_8b_batch64_fixed_work_${EXECUTION_CODE_SHA256:0:12}}
CONTRACT_ROOT=$OUTPUT_ROOT/contracts
SOURCE_NATURAL_EXECUTION_CODE_SHA256=${DEEPSEEK_FIXED_WORK_NATURAL_SOURCE_EXECUTION_SHA256:-b0f33dbe107e700fc606c6416065f2ae14ef80fbf4a3f68e7d16f6d38e870ca5}
SOURCE_NATURAL_RECOVERY_SHA256=${DEEPSEEK_FIXED_WORK_NATURAL_SOURCE_RECOVERY_SHA256:-de30e3fa6f9c72b1a4b4391190f42e194b1ec85c95eed747bb19b0610300a19b}
NATURAL_EPOCH_ROOT=${DEEPSEEK_FIXED_WORK_NATURAL_SOURCE_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/p0_8b_batch64_fixed_work_b0f33dbe107e/natural_epoch}
SOURCE_VALIDATION=$CONTRACT_ROOT/immutable_natural_source.json
SOURCE_CAP_ENV=
COMMON_ROOT=
CAP_ENV=$CONTRACT_ROOT/deepseek_v2_lite_batch64_kv_caps.env
MIGRATION_MANIFEST=$CONTRACT_ROOT/cap_provenance_migration.json
GATE_TRACE=$CONTRACT_ROOT/gate_fixed_work_trace.json
EPOCH_TRACE=$CONTRACT_ROOT/epoch_fixed_work_trace.json
GATE_PREFLIGHT_PLAN=$SOURCE_ROOT/audit_migration_20260807/gate_floor4_planner_preflight/length_sorted_rank_plan.json
EPOCH_PREFLIGHT_PLAN=$SOURCE_ROOT/audit_migration_20260807/epoch_five_step_planner_preflight/length_sorted_rank_plan.json
SOURCE_GATE_ADAFLOOR_RUN=$SOURCE_ROOT/paired_gate/deepseek_v2_lite_adafloor_n_f2_common_epoch0_epoch1_2
FIXED_GATE_ROOT=$OUTPUT_ROOT/fixed_gate
FIXED_EPOCH_ROOT=$OUTPUT_ROOT/fixed_epoch
WORKFLOW_LOCK=$OUTPUT_ROOT/.workflow.lock

require_file() {
    local path=$1
    local label=$2
    if [[ ! -f "$path" ]]; then
        echo "missing $label: $path" >&2
        exit 2
    fi
}

require_adafloor_audit() {
    local run_root=$1
    local audit=$run_root/DEEPSEEK_PLAN_RUNTIME_AUDIT.json
    require_file "$audit" "AdaFloor plan/runtime audit"
    python3 - "$audit" <<'PY_AUDIT_STATUS'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("status") != "PASS" or payload.get("lifecycle") != "natural_f2":
    raise SystemExit(f"invalid AdaFloor plan/runtime audit: {path}")
PY_AUDIT_STATUS
}

require_idle_npus() {
    local processes
    processes=$(npu-smi info | awk '
        /Process id/ { in_process_table=1; next }
        in_process_table && /^\|/ && !/No running processes found/ { print }
    ')
    if [[ -n "$processes" ]]; then
        echo "fixed-work experiment requires idle NPUs" >&2
        printf '%s\n' "$processes" >&2
        exit 2
    fi
}

acquire_lock() {
    mkdir -p "$OUTPUT_ROOT"
    exec 8>"$WORKFLOW_LOCK"
    if ! flock -n 8; then
        echo "another fixed-work process holds $WORKFLOW_LOCK" >&2
        exit 2
    fi
}

assert_code_unchanged() {
    local observed
    observed=$(python3 "$SCRIPT_DIR/tools/hash_deepseek_execution_code.py" --root "$SCRIPT_DIR")
    if [[ "$observed" != "$EXECUTION_CODE_SHA256" ]]; then
        echo "execution code changed during the fixed-work workflow" >&2
        echo "started=$EXECUTION_CODE_SHA256 observed=$observed" >&2
        exit 2
    fi
}

load_common_contract() {
    require_file "$COMMON_ROOT/common_epoch0_metadata.env" "common epoch0 metadata"
    require_file "$COMMON_ROOT/reuse.env" "common epoch0 reuse metadata"
    require_file "$COMMON_ROOT/FROZEN_CHECKPOINT_SHA256" "frozen checkpoint digest"
    require_file "$COMMON_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" "protected checkpoint marker"
    require_file "$PROFILE_PATH" "batch64 workload profile"
    # shellcheck disable=SC1090
    source "$PROFILE_PATH"
    # shellcheck disable=SC1090
    source "$COMMON_ROOT/common_epoch0_metadata.env"
    # shellcheck disable=SC1090
    source "$COMMON_ROOT/reuse.env"
    local expected_checkpoint_sha256
    local observed_checkpoint_sha256
    expected_checkpoint_sha256=$(tr -d '[:space:]' < "$COMMON_ROOT/FROZEN_CHECKPOINT_SHA256")
    observed_checkpoint_sha256=$(python3 "$SCRIPT_DIR/tools/hash_deepseek_checkpoint.py" \
        --checkpoint "$DYNAMIC_INITIAL_RESUME_CKPT")
    if [[ "$observed_checkpoint_sha256" != "$expected_checkpoint_sha256" ]]; then
        echo "protected DeepSeek checkpoint digest changed" >&2
        exit 2
    fi
    if [[ "$COMMON_EPOCH0_TRAIN_BATCH_SIZE" != 64 \
          || "$COMMON_EPOCH0_ROLLOUT_N" != 16 \
          || "$COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP" != 1024 \
          || "$COMMON_EPOCH0_MAX_NUM_SEQS" != 64 \
          || "$COMMON_EPOCH0_MAX_RESPONSE_LENGTH" != 16384 ]]; then
        echo "source common epoch0 is not the batch64 contract" >&2
        exit 2
    fi
}

validate_immutable_natural_source() {
    mkdir -p "$CONTRACT_ROOT"
    python3 "$SCRIPT_DIR/tools/validate_deepseek_fixed_work_source.py" \
        --source-natural-root "$NATURAL_EPOCH_ROOT" \
        --expected-execution-sha256 "$SOURCE_NATURAL_EXECUTION_CODE_SHA256" \
        --expected-recovery-sha256 "$SOURCE_NATURAL_RECOVERY_SHA256" \
        --output "$SOURCE_VALIDATION" >/dev/null
    mapfile -t source_contract_values < <(
        python3 - "$SOURCE_VALIDATION" <<'PY_SOURCE_CONTRACT'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "PASS":
    raise SystemExit("immutable Natural source validation did not pass")
print(payload["cap_env"])
print(payload["common_root"])
print(payload["adafloor_run"])
print(payload["adafloor_actual_plan"])
PY_SOURCE_CONTRACT
    )
    if (( ${#source_contract_values[@]} != 4 )); then
        echo "immutable Natural source validation returned incomplete paths" >&2
        exit 2
    fi
    SOURCE_CAP_ENV=${source_contract_values[0]}
    COMMON_ROOT=${source_contract_values[1]}
    SOURCE_NATURAL_ADAFLOOR_RUN=${source_contract_values[2]}
    SOURCE_NATURAL_ACTUAL_PLAN=${source_contract_values[3]}
}

prepare_cap_migration() {
    require_file "$SOURCE_CAP_ENV" "source KV cap authorization"
    mkdir -p "$CONTRACT_ROOT"
    python3 "$SCRIPT_DIR/tools/migrate_deepseek_fixed_work_cap.py" \
        --source "$SOURCE_CAP_ENV" \
        --output "$CAP_ENV" \
        --manifest "$MIGRATION_MANIFEST" \
        --current-execution-sha256 "$EXECUTION_CODE_SHA256" \
        --common-root "$COMMON_ROOT"
    # shellcheck disable=SC1090
    source "$CAP_ENV"
    if [[ "${DEEPSEEK_EXECUTION_CODE_SHA256:-}" != "$EXECUTION_CODE_SHA256" \
          || "${DEEPSEEK_KV_CAP_CONTINUATION_EXECUTION_CODE_SHA256:-}" \
             != "$EXECUTION_CODE_SHA256" \
          || -z "${DEEPSEEK_KV_CAP_AUTHORIZED_RUNTIME_EXECUTION_CODE_SHA256:-}" \
          || "${DEEPSEEK_N_F2_KV_CAPS_VERIFIED:-0}" != 1 \
          || "${DEEPSEEK_KV_CAP_TRAIN_BATCH_SIZE:-}" != 64 \
          || "${DEEPSEEK_KV_CAP_ROLLOUT_N:-}" != 16 \
          || "${DEEPSEEK_KV_CAP_MAX_NUM_SEQS:-}" != 64 ]]; then
        echo "migrated cap contract is invalid" >&2
        exit 2
    fi
}

prepare_contract() {
    assert_code_unchanged
    validate_immutable_natural_source
    load_common_contract
    prepare_cap_migration
    require_file "$GATE_PREFLIGHT_PLAN" "gate AdaFloor preflight plan"
    require_file "$EPOCH_PREFLIGHT_PLAN" "epoch AdaFloor preflight plan"
    require_file "$SOURCE_GATE_ADAFLOOR_RUN/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" \
        "source natural AdaFloor gate completion marker"
    require_adafloor_audit "$SOURCE_GATE_ADAFLOOR_RUN"
}

arm_run_name() {
    case "$1" in
        vanilla) printf '%s' deepseek_v2_lite_vanilla_common_epoch0_epoch1_2 ;;
        adafloor) printf '%s' deepseek_v2_lite_adafloor_n_f2_common_epoch0_epoch1_2 ;;
        *) echo "invalid fixed-work arm: $1" >&2; exit 2 ;;
    esac
}

run_name_for() {
    local arm=$1
    local phase=$2
    local mode=$3
    if [[ "$arm" == vanilla && "$mode" == fixed ]]; then
        printf '%s' deepseek_v2_lite_lengthsort_common_epoch0_epoch1_2
    else
        arm_run_name "$arm"
    fi
}

phase_dataset_fraction() {
    if [[ "$1" == gate ]]; then
        printf '%s' "$DEEPSEEK_KV_PROBE_DATASET_FRACTION"
    else
        printf '%s' "$COMMON_EPOCH0_DATASET_FRACTION"
    fi
}

write_pair_manifest() {
    local arm=$1
    local phase=$2
    local mode=$3
    local run_root=$4
    local trace=${5:-}
    local trace_sha256=${6:-}
    local manifest=$run_root/batch64_pair_manifest.env
    local temporary
    local cap_sha256
    cap_sha256=$(sha256sum "$CAP_ENV" | awk '{print $1}')
    temporary=$(mktemp "$run_root/.batch64_pair_manifest.env.tmp.XXXXXX")
    {
        printf 'export DEEPSEEK_BATCH64_ARM=%q\n' "$arm"
        printf 'export DEEPSEEK_BATCH64_PHASE=%q\n' "$phase"
        printf 'export DEEPSEEK_WORKLOAD_PROFILE_ID=%q\n' "$DEEPSEEK_WORKLOAD_PROFILE_ID"
        printf 'export DEEPSEEK_WORKLOAD_PROFILE_SHA256=%q\n' \
            "$(sha256sum "$PROFILE_PATH" | awk '{print $1}')"
        printf 'export DEEPSEEK_BATCH64_COMMON_ROOT=%q\n' "$COMMON_ROOT"
        printf 'export DEEPSEEK_BATCH64_FROZEN_CHECKPOINT=%q\n' "$DYNAMIC_INITIAL_RESUME_CKPT"
        printf 'export DEEPSEEK_BATCH64_MODEL_PATH=%q\n' "$COMMON_EPOCH0_MODEL_PATH"
        printf 'export DEEPSEEK_BATCH64_MODEL_REVISION=%q\n' "$COMMON_EPOCH0_MODEL_REVISION"
        printf 'export DEEPSEEK_BATCH64_EXECUTION_PROFILE=%q\n' "$COMMON_EPOCH0_EXECUTION_PROFILE_USED"
        printf 'export DEEPSEEK_BATCH64_CAP_ENV_SHA256=%q\n' "$cap_sha256"
        printf 'export DEEPSEEK_BATCH64_EXECUTION_CODE_SHA256=%q\n' "$EXECUTION_CODE_SHA256"
        printf 'export DEEPSEEK_BATCH64_FROZEN_CHECKPOINT_SHA256=%q\n' \
            "$(tr -d '[:space:]' < "$COMMON_ROOT/FROZEN_CHECKPOINT_SHA256")"
        printf 'export DEEPSEEK_BATCH64_PAIRED_REQUEST_SAMPLING_SEEDS=1\n'
        printf 'export DEEPSEEK_BATCH64_TRAIN_BATCH_SIZE=%s\n' "$COMMON_EPOCH0_TRAIN_BATCH_SIZE"
        printf 'export DEEPSEEK_BATCH64_ROLLOUT_N=%s\n' "$COMMON_EPOCH0_ROLLOUT_N"
        printf 'export DEEPSEEK_BATCH64_MAX_NUM_SEQS=%s\n' "$COMMON_EPOCH0_MAX_NUM_SEQS"
        printf 'export DEEPSEEK_BATCH64_MAX_PROMPT_LENGTH=%s\n' "$COMMON_EPOCH0_MAX_PROMPT_LENGTH"
        printf 'export DEEPSEEK_BATCH64_MAX_RESPONSE_LENGTH=%s\n' "$COMMON_EPOCH0_MAX_RESPONSE_LENGTH"
        printf 'export DEEPSEEK_BATCH64_MAX_NUM_BATCHED_TOKENS=%s\n' "$COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS"
        printf 'export DEEPSEEK_BATCH64_FULL16_PHYSICAL_TOKENS=%s\n' \
            "$DEEPSEEK_KV_CAP_SHARED_FULL16_PHYSICAL_TOKENS"
        printf 'export DEEPSEEK_BATCH64_TEMPERATURE=0.9\n'
        printf 'export DEEPSEEK_BATCH64_TOP_P=0.9\n'
        printf 'export DEEPSEEK_BATCH64_TOP_K=50\n'
        printf 'export DEEPSEEK_BATCH64_DATASET_FRACTION=%s\n' \
            "$(phase_dataset_fraction "$phase")"
        if [[ "$phase" == gate && "$arm" == adafloor ]]; then
            printf 'export DEEPSEEK_BATCH64_FORCED_SELECTED_FLOOR=4\n'
        else
            printf 'export DEEPSEEK_BATCH64_FORCED_SELECTED_FLOOR=none\n'
        fi
        printf 'export DEEPSEEK_BATCH64_RUN_MODE=%q\n' "$mode"
        if [[ "$mode" == fixed ]]; then
            printf 'export DEEPSEEK_BATCH64_FIXED_WORK_PROTOCOL=%q\n' \
                deepseek_batch64_fixed_work_replay_v3
            printf 'export DEEPSEEK_BATCH64_FIXED_WORK_TRACE=%q\n' "$(realpath "$trace")"
            printf 'export DEEPSEEK_BATCH64_FIXED_WORK_TRACE_SHA256=%q\n' "$trace_sha256"
        fi
    } > "$temporary"
    if [[ -f "$manifest" ]]; then
        if ! cmp -s "$temporary" "$manifest"; then
            rm -f "$temporary"
            echo "refusing to replace stale pair manifest: $manifest" >&2
            exit 2
        fi
        rm -f "$temporary"
    else
        mv "$temporary" "$manifest"
    fi
}

build_trace() {
    local phase=$1
    local source_run=$2
    local plan=$3
    local trace=$4
    local expected_steps
    local source_epoch
    if [[ "$phase" == gate ]]; then expected_steps=1; else expected_steps=5; fi
    source_epoch=$(resolve_rollout_epoch "$source_run")
    local rebuilt_trace
    rebuilt_trace=$(mktemp "$CONTRACT_ROOT/.fixed_work_trace.rebuilt.XXXXXX")
    if ! python3 "$SCRIPT_DIR/tools/build_deepseek_fixed_work_replay.py" \
        --source-run-dir "$source_epoch" \
        --adafloor-plan "$plan" \
        --output "$rebuilt_trace" \
        --force >&2; then
        rm -f -- "$rebuilt_trace"
        exit 2
    fi
    if [[ -f "$trace" ]]; then
        if ! cmp -s "$rebuilt_trace" "$trace"; then
            rm -f -- "$rebuilt_trace"
            echo "existing fixed-work trace differs from source rebuild: $trace" >&2
            exit 2
        fi
        rm -f -- "$rebuilt_trace"
    else
        mv -- "$rebuilt_trace" "$trace"
    fi
    local trace_sha256
    trace_sha256=$(sha256sum "$trace" | awk '{print $1}')
    python3 - "$trace" "$trace_sha256" "$expected_steps" <<'PY_TRACE' >&2
import sys
from verl.utils.fixed_work_replay import load_fixed_work_replay

trace = load_fixed_work_replay(sys.argv[1], expected_sha256=sys.argv[2])
expected_steps = int(sys.argv[3])
if trace.steps != tuple(range(1, expected_steps + 1)):
    raise SystemExit(f"fixed-work trace has steps={trace.steps}")
print(
    f"fixed-work trace validated steps={trace.steps} "
    f"source_tokens={trace.source_generated_tokens} "
    f"target_tokens={trace.target_generated_tokens}"
)
PY_TRACE
    python3 - "$trace" "$source_epoch" "$plan" <<'PY_TRACE_SOURCE' >&2
import hashlib
import json
import os
import sys

trace_path, expected_source, expected_plan = sys.argv[1:]
payload = json.load(open(trace_path, encoding="utf-8"))
expected_source = os.path.realpath(expected_source)
expected_plan = os.path.realpath(expected_plan)
plan_source = payload.get("adafloor_plan_source")
if payload.get("source_run_dir") != expected_source:
    raise SystemExit("fixed-work trace points to another source rollout")
if not isinstance(plan_source, dict) or plan_source.get("path") != expected_plan:
    raise SystemExit("fixed-work trace points to another source plan")
actual_sha256 = hashlib.sha256(open(expected_plan, "rb").read()).hexdigest()
if plan_source.get("sha256") != actual_sha256:
    raise SystemExit("fixed-work trace source plan SHA256 is stale")
PY_TRACE_SOURCE
    printf '%s' "$trace_sha256"
}

resolve_rollout_epoch() {
    local run_root=$1
    local candidates=()
    if [[ -d "$run_root/rollout_data" ]]; then
        realpath "$run_root"
        return
    fi
    mapfile -d '' -t candidates < <(
        find "$run_root" -mindepth 1 -maxdepth 1 -type d \
            -name 'epoch_001*' -print0
    )
    if (( ${#candidates[@]} != 1 )); then
        echo "expected one epoch_001 directory under $run_root, found ${#candidates[@]}" >&2
        exit 2
    fi
    require_file "${candidates[0]}/rollout_data/1.jsonl" \
        "fixed-work source rollout step 1"
    realpath "${candidates[0]}"
}

resolve_actual_plan() {
    local run_root=$1
    local epoch_dir
    epoch_dir=$(resolve_rollout_epoch "$run_root")
    require_file "$epoch_dir/oracle/length_sorted_rank_plan.json" \
        "actual Natural AdaFloor plan"
    realpath "$epoch_dir/oracle/length_sorted_rank_plan.json"
}

run_arm() (
    local arm=$1
    local phase=$2
    local mode=$3
    local phase_root=$4
    local prompts=$5
    local steps=$6
    local trace=${7:-}
    local trace_sha256=${8:-}
    local variant
    local run_name
    local run_root
    local dataset_fraction
    local retained_marker
    local removed_marker
    local cleanup_pending=$phase_root/.pair_verified_checkpoint_cleanup_pending.json
    local cleanup_committed=$phase_root/PAIR_VERIFIED_CHECKPOINT_CLEANUP.json
    if [[ "$arm" == vanilla ]]; then
        if [[ "$mode" == fixed ]]; then
            variant=lengthsort
        else
            variant=vanilla
        fi
    else
        variant=adafloor_n_f2
    fi
    run_name=$(run_name_for "$arm" "$phase" "$mode")
    run_root=$phase_root/$run_name
    retained_marker=$run_root/CHECKPOINTS_RETAINED_AFTER_VALIDATION.txt
    removed_marker=$run_root/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt
    dataset_fraction=$(phase_dataset_fraction "$phase")
    if [[ -f "$retained_marker" ]]; then
        if [[ "$arm" == adafloor ]]; then
            require_adafloor_audit "$run_root"
        fi
        echo "[fixed-work] $mode $phase $arm validated with checkpoints retained: $run_root"
        write_pair_manifest "$arm" "$phase" "$mode" "$run_root" "$trace" "$trace_sha256"
        exit 0
    fi
    if [[ -f "$removed_marker" ]]; then
        python3 "$SCRIPT_DIR/tools/manage_deepseek_fixed_work_cleanup.py" \
            allows-removed \
            --pending "$cleanup_pending" \
            --committed "$cleanup_committed" \
            --arm "$arm" \
            --arm-run "$run_root" \
            --execution-code-sha256 "$EXECUTION_CODE_SHA256"
        if [[ "$arm" == adafloor ]]; then
            require_adafloor_audit "$run_root"
        fi
        echo "[fixed-work] $mode $phase $arm already verified and cleaned: $run_root"
        write_pair_manifest "$arm" "$phase" "$mode" "$run_root" "$trace" "$trace_sha256"
        exit 0
    fi
    if [[ -e "$run_root" ]]; then
        echo "incomplete $mode $phase $arm run exists: $run_root" >&2
        exit 2
    fi
    unset VERL_FIXED_WORK_REPLAY_TRACE VERL_FIXED_WORK_REPLAY_SHA256
    unset VERL_FIXED_WORK_REPLAY_REQUIRE_PLAN_CAP
    if [[ "$mode" == fixed ]]; then
        export VERL_FIXED_WORK_REPLAY_TRACE=$trace
        export VERL_FIXED_WORK_REPLAY_SHA256=$trace_sha256
        if [[ "$arm" == adafloor ]]; then
            export VERL_FIXED_WORK_REPLAY_REQUIRE_PLAN_CAP=1
        else
            export VERL_FIXED_WORK_REPLAY_REQUIRE_PLAN_CAP=0
        fi
    fi
    local force_floor=()
    if [[ "$phase" == gate && "$arm" == adafloor ]]; then
        force_floor+=(DEEPSEEK_FAIR_FORCE_SELECTED_FLOOR=4)
    fi
    env \
        "${force_floor[@]}" \
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
        FAIR_FINALIZE_EXISTING=0 \
        FAIR_KEEP_COMPLETED_CHECKPOINTS=1 \
        DEEPSEEK_FIXED_WORK_LAUNCH_PROTOCOL=deepseek_batch64_fixed_work_replay_v3 \
        DEEPSEEK_FIXED_WORK_PHASE="$phase" \
        DEEPSEEK_FIXED_WORK_EXPECTED_STEPS="$steps" \
        DEEPSEEK_FIXED_WORK_EXECUTION_CODE_SHA256="$EXECUTION_CODE_SHA256" \
        "$SCRIPT_DIR/run_deepseek_v2_lite_fair_compare.sh" "$variant"
    require_file "$retained_marker" "$mode $phase $arm retained-checkpoint marker"
    if [[ -e "$removed_marker" ]]; then
        echo "$mode $phase $arm was cleaned before pair verification" >&2
        exit 2
    fi
    if [[ "$arm" == adafloor ]]; then
        require_adafloor_audit "$run_root"
    fi
    write_pair_manifest "$arm" "$phase" "$mode" "$run_root" "$trace" "$trace_sha256"
)

verify_fixed_pair() {
    local phase=$1
    local phase_root=$2
    local trace=$3
    local summary=$4
    local trace_sha256
    trace_sha256=$(sha256sum "$trace" | awk '{print $1}')
    python3 "$SCRIPT_DIR/tools/verify_deepseek_fixed_work_pair.py" \
        --phase "$phase" \
        --vanilla-run-dir "$phase_root/$(run_name_for vanilla "$phase" fixed)" \
        --adafloor-run-dir "$phase_root/$(run_name_for adafloor "$phase" fixed)" \
        --common-root "$COMMON_ROOT" \
        --cap-env "$CAP_ENV" \
        --workload-profile-env "$PROFILE_PATH" \
        --trace "$trace" \
        --trace-sha256 "$trace_sha256" \
        --expected-execution-code-sha256 "$EXECUTION_CODE_SHA256" \
        --output "$summary"
}

finalize_arm_checkpoints() (
    local arm=$1
    local phase=$2
    local mode=$3
    local phase_root=$4
    local prompts=$5
    local steps=$6
    local variant
    local run_name
    local run_root
    local dataset_fraction
    local force_floor=()
    if [[ "$arm" == vanilla ]]; then
        if [[ "$mode" == fixed ]]; then
            variant=lengthsort
        else
            variant=vanilla
        fi
    else
        variant=adafloor_n_f2
    fi
    if [[ "$phase" == gate && "$arm" == adafloor ]]; then
        force_floor+=(DEEPSEEK_FAIR_FORCE_SELECTED_FLOOR=4)
    fi
    run_name=$(run_name_for "$arm" "$phase" "$mode")
    run_root=$phase_root/$run_name
    dataset_fraction=$(phase_dataset_fraction "$phase")
    env \
        -u VERL_FIXED_WORK_REPLAY_TRACE \
        -u VERL_FIXED_WORK_REPLAY_SHA256 \
        -u VERL_FIXED_WORK_REPLAY_REQUIRE_PLAN_CAP \
        "${force_floor[@]}" \
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
        FAIR_FINALIZE_EXISTING=1 \
        FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
        DEEPSEEK_FAIR_DRY_RUN=0 \
        DEEPSEEK_VALIDATE_ASSETS_ON_LAUNCH=0 \
        "$SCRIPT_DIR/run_deepseek_v2_lite_fair_compare.sh" "$variant"
    require_file "$run_root/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" \
        "$mode $phase $arm checkpoint cleanup marker"
    if [[ -e "$run_root/CHECKPOINTS_RETAINED_AFTER_VALIDATION.txt" ]] \
       || find "$run_root" -mindepth 2 -maxdepth 2 -type d -name checkpoints \
            -print -quit | grep -q .; then
        echo "$mode $phase $arm checkpoint cleanup is incomplete" >&2
        exit 2
    fi
    if [[ "$arm" == adafloor ]]; then
        require_adafloor_audit "$run_root"
    fi
)

cleanup_verified_pair() {
    local phase=$1
    local mode=$2
    local phase_root=$3
    local summary=$4
    local prompts=$5
    local steps=$6
    shift 6
    local vanilla_run=$phase_root/$(run_name_for vanilla "$phase" "$mode")
    local adafloor_run=$phase_root/$(run_name_for adafloor "$phase" "$mode")
    local adafloor_audit=$adafloor_run/DEEPSEEK_PLAN_RUNTIME_AUDIT.json
    local pending=$phase_root/.pair_verified_checkpoint_cleanup_pending.json
    local committed=$phase_root/PAIR_VERIFIED_CHECKPOINT_CLEANUP.json
    local status_args=()
    local status
    for status in "$@"; do
        status_args+=(--allowed-summary-status "$status")
    done
    python3 "$SCRIPT_DIR/tools/manage_deepseek_fixed_work_cleanup.py" prepare \
        --pending "$pending" \
        --committed "$committed" \
        --phase "$phase" \
        --mode "$mode" \
        --summary "$summary" \
        "${status_args[@]}" \
        --cap-env "$CAP_ENV" \
        --execution-code-sha256 "$EXECUTION_CODE_SHA256" \
        --vanilla-run "$vanilla_run" \
        --adafloor-run "$adafloor_run" \
        --adafloor-audit "$adafloor_audit"
    finalize_arm_checkpoints vanilla "$phase" "$mode" "$phase_root" "$prompts" "$steps"
    finalize_arm_checkpoints adafloor "$phase" "$mode" "$phase_root" "$prompts" "$steps"
    python3 "$SCRIPT_DIR/tools/manage_deepseek_fixed_work_cleanup.py" commit \
        --pending "$pending" \
        --committed "$committed" >/dev/null
}

require_cleanup_commit() {
    local phase_root=$1
    python3 "$SCRIPT_DIR/tools/manage_deepseek_fixed_work_cleanup.py" commit \
        --pending "$phase_root/.pair_verified_checkpoint_cleanup_pending.json" \
        --committed "$phase_root/PAIR_VERIFIED_CHECKPOINT_CLEANUP.json" \
        >/dev/null
}

summarize_natural_epoch() {
    python3 "$SCRIPT_DIR/tools/verify_deepseek_batch64_pair.py" \
        --phase epoch \
        --vanilla-run-dir "$NATURAL_EPOCH_ROOT/$(arm_run_name vanilla)" \
        --adafloor-run-dir "$NATURAL_EPOCH_ROOT/$(arm_run_name adafloor)" \
        --common-root "$COMMON_ROOT" \
        --cap-env "$CAP_ENV" \
        --workload-profile-env "$PROFILE_PATH" \
        --expected-execution-code-sha256 "$EXECUTION_CODE_SHA256" \
        --allow-layout-induced-work-divergence \
        --output "$NATURAL_EPOCH_ROOT/natural_epoch_summary.json"
}

fixed_summary_passes() {
    local summary=$1
    local phase=$2
    [[ -f "$summary" ]] || return 1
    python3 - "$summary" "$phase" "$EXECUTION_CODE_SHA256" <<'PY_STATUS'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "PASS" or value.get("phase") != sys.argv[2]:
    raise SystemExit(1)
if value.get("provenance", {}).get("execution_code_sha256") != sys.argv[3]:
    raise SystemExit(1)
if value.get("fixed_work", {}).get("arms_exactly_equal") is not True:
    raise SystemExit(1)
PY_STATUS
}

run_fixed_gate() {
    prepare_contract
    local actual_plan
    local trace_sha256
    actual_plan=$(resolve_actual_plan "$SOURCE_GATE_ADAFLOOR_RUN")
    trace_sha256=$(build_trace gate "$SOURCE_GATE_ADAFLOOR_RUN" "$actual_plan" "$GATE_TRACE")
    require_idle_npus
    mkdir -p "$FIXED_GATE_ROOT"
    run_arm vanilla gate fixed "$FIXED_GATE_ROOT" 64 1 "$GATE_TRACE" "$trace_sha256"
    assert_code_unchanged
    load_common_contract
    run_arm adafloor gate fixed "$FIXED_GATE_ROOT" 64 1 "$GATE_TRACE" "$trace_sha256"
    assert_code_unchanged
    load_common_contract
    verify_fixed_pair gate "$FIXED_GATE_ROOT" "$GATE_TRACE" \
        "$FIXED_GATE_ROOT/fixed_gate_summary.json"
    cleanup_verified_pair gate fixed "$FIXED_GATE_ROOT" \
        "$FIXED_GATE_ROOT/fixed_gate_summary.json" 64 1 PASS
}

require_fixed_gate() {
    if ! fixed_summary_passes "$FIXED_GATE_ROOT/fixed_gate_summary.json" gate; then
        echo "five-step work requires a passing fixed-work gate" >&2
        exit 2
    fi
    verify_fixed_pair gate "$FIXED_GATE_ROOT" "$GATE_TRACE" \
        "$FIXED_GATE_ROOT/fixed_gate_summary.json"
    require_cleanup_commit "$FIXED_GATE_ROOT"
}

run_natural_epoch() {
    prepare_contract
    validate_immutable_natural_source
    echo "[fixed-work] immutable Natural epoch source is verified: $NATURAL_EPOCH_ROOT"
}

run_fixed_epoch() {
    prepare_contract
    require_fixed_gate
    validate_immutable_natural_source
    local natural_adafloor=$SOURCE_NATURAL_ADAFLOOR_RUN
    local actual_plan
    local trace_sha256
    actual_plan=$(resolve_actual_plan "$natural_adafloor")
    if [[ "$actual_plan" != "$SOURCE_NATURAL_ACTUAL_PLAN" ]]; then
        echo "resolved Natural plan differs from immutable source validation" >&2
        exit 2
    fi
    trace_sha256=$(build_trace epoch "$natural_adafloor" "$actual_plan" "$EPOCH_TRACE")
    require_idle_npus
    mkdir -p "$FIXED_EPOCH_ROOT"
    run_arm vanilla epoch fixed "$FIXED_EPOCH_ROOT" 320 5 "$EPOCH_TRACE" "$trace_sha256"
    assert_code_unchanged
    load_common_contract
    run_arm adafloor epoch fixed "$FIXED_EPOCH_ROOT" 320 5 "$EPOCH_TRACE" "$trace_sha256"
    assert_code_unchanged
    load_common_contract
    verify_fixed_pair epoch "$FIXED_EPOCH_ROOT" "$EPOCH_TRACE" \
        "$FIXED_EPOCH_ROOT/fixed_epoch_summary.json"
    cleanup_verified_pair epoch fixed "$FIXED_EPOCH_ROOT" \
        "$FIXED_EPOCH_ROOT/fixed_epoch_summary.json" 320 5 PASS
}

dry_run_gate() {
    prepare_contract
    local actual_plan
    local trace_sha256
    actual_plan=$(resolve_actual_plan "$SOURCE_GATE_ADAFLOOR_RUN")
    trace_sha256=$(build_trace gate "$SOURCE_GATE_ADAFLOOR_RUN" "$actual_plan" "$GATE_TRACE")
    for arm in vanilla adafloor; do
        local variant=lengthsort
        local force_floor=()
        if [[ "$arm" == adafloor ]]; then
            variant=adafloor_n_f2
            force_floor+=(DEEPSEEK_FAIR_FORCE_SELECTED_FLOOR=4)
        fi
        env \
            "${force_floor[@]}" \
            VERL_FIXED_WORK_REPLAY_TRACE="$GATE_TRACE" \
            VERL_FIXED_WORK_REPLAY_SHA256="$trace_sha256" \
            VERL_FIXED_WORK_REPLAY_REQUIRE_PLAN_CAP=$([[ "$arm" == adafloor ]] && echo 1 || echo 0) \
            DEEPSEEK_FAIR_DATASET_FRACTION="$DEEPSEEK_KV_PROBE_DATASET_FRACTION" \
            DEEPSEEK_WORKLOAD_PROFILE_PATH="$PROFILE_PATH" \
            DEEPSEEK_KV_CAP_ENV="$CAP_ENV" \
            COMMON_EPOCH0_ROOT="$COMMON_ROOT" \
            FAIR_OUTPUT_ROOT="$OUTPUT_ROOT/dry_run_gate" \
            FAIR_START_EPOCH=1 FAIR_TOTAL_EPOCHS=2 FAIR_FREEZE_ACTOR=1 \
            FAIR_PROMPTS_PER_EPOCH=64 FAIR_EXPECTED_STEPS=1 \
            DEEPSEEK_FIXED_WORK_LAUNCH_PROTOCOL=deepseek_batch64_fixed_work_replay_v3 \
            DEEPSEEK_FIXED_WORK_PHASE=gate \
            DEEPSEEK_FIXED_WORK_EXPECTED_STEPS=1 \
            DEEPSEEK_FIXED_WORK_EXECUTION_CODE_SHA256="$EXECUTION_CODE_SHA256" \
            DEEPSEEK_FAIR_DRY_RUN=1 DEEPSEEK_VALIDATE_ASSETS_ON_LAUNCH=0 \
            "$SCRIPT_DIR/run_deepseek_v2_lite_fair_compare.sh" "$variant"
    done
}

show_status() {
    local gate=pending
    local natural=none
    local epoch=pending
    if fixed_summary_passes "$FIXED_GATE_ROOT/fixed_gate_summary.json" gate \
       && [[ -f "$FIXED_GATE_ROOT/PAIR_VERIFIED_CHECKPOINT_CLEANUP.json" ]]; then
        gate=complete
    fi
    if [[ -f "$NATURAL_EPOCH_ROOT/$(arm_run_name vanilla)/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; then
        natural=vanilla_complete
    fi
    if [[ -f "$NATURAL_EPOCH_ROOT/$(arm_run_name adafloor)/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; then
        natural=pair_complete
    fi
    if fixed_summary_passes "$FIXED_EPOCH_ROOT/fixed_epoch_summary.json" epoch \
       && [[ -f "$FIXED_EPOCH_ROOT/PAIR_VERIFIED_CHECKPOINT_CLEANUP.json" ]]; then
        epoch=complete
    fi
    printf '%s\n' \
        "source_root=$SOURCE_ROOT" \
        "output_root=$OUTPUT_ROOT" \
        "execution_code_sha256=$EXECUTION_CODE_SHA256" \
        "prepared=$([[ -f "$MIGRATION_MANIFEST" ]] && echo yes || echo no)" \
        "fixed_gate=$gate" \
        "natural_epoch=$natural" \
        "fixed_epoch=$epoch"
}

case "$PHASE" in
    status)
        show_status
        exit 0
        ;;
    prepare|dry-run-gate|gate|natural-epoch|fixed-epoch|epoch|verify-gate|verify-epoch|all) ;;
    *)
        echo "unknown phase: $PHASE" >&2
        usage >&2
        exit 2
        ;;
esac

acquire_lock
case "$PHASE" in
    prepare) prepare_contract ;;
    dry-run-gate) dry_run_gate ;;
    gate) run_fixed_gate ;;
    natural-epoch) run_natural_epoch ;;
    fixed-epoch) run_fixed_epoch ;;
    epoch)
        run_natural_epoch
        run_fixed_epoch
        ;;
    verify-gate)
        prepare_contract
        verify_fixed_pair gate "$FIXED_GATE_ROOT" "$GATE_TRACE" \
            "$FIXED_GATE_ROOT/fixed_gate_summary.json"
        require_cleanup_commit "$FIXED_GATE_ROOT"
        ;;
    verify-epoch)
        prepare_contract
        require_fixed_gate
        verify_fixed_pair epoch "$FIXED_EPOCH_ROOT" "$EPOCH_TRACE" \
            "$FIXED_EPOCH_ROOT/fixed_epoch_summary.json"
        require_cleanup_commit "$FIXED_EPOCH_ROOT"
        ;;
    all)
        dry_run_gate
        run_fixed_gate
        run_natural_epoch
        run_fixed_epoch
        ;;
esac
show_status
