#!/usr/bin/env bash
set -euo pipefail

# Freeze the launcher for the full queue so live edits cannot alter later arms.
if [[ "${ADAFLOOR_P0_7_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    source_path=$(realpath "${BASH_SOURCE[0]}")
    snapshot=$(mktemp "${source_path}.run-snapshot.XXXXXX")
    cp -- "$source_path" "$snapshot"
    chmod 700 "$snapshot"
    set +e
    ADAFLOOR_P0_7_SNAPSHOT_ACTIVE=1 "$snapshot" "$@"
    rc=$?
    set -e
    rm -f -- "$snapshot"
    exit "$rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
P0_7_PLANNING_HISTORY_DIR="${P0_7_PLANNING_HISTORY_DIR:-/data/adafloor_shared_state/adafloor_p_f4_low_peak_optimizer_epoch2_20260725T_low_peak_optimizer_v1/adafloor_planned_floor4_tailguard_epoch2_low_peak_optimizer/epoch_002_mode1_planned}"
P0_7_OUTPUT_ROOT="${P0_7_OUTPUT_ROOT:-/data/adafloor_shared_state/p0_7_natural_matching_$(date -u +%Y%m%dT%H%M%SZ)}"
P0_7_TRIAL_SEEDS="${P0_7_TRIAL_SEEDS:-601 602 603}"
P0_7_FORCE_SELECTED_FLOORS="${P0_7_FORCE_SELECTED_FLOORS:-2,2,4,4,16}"
P0_7_MIN_FREE_GIB="${P0_7_MIN_FREE_GIB:-180}"
P0_7_DRY_RUN="${P0_7_DRY_RUN:-0}"
P0_7_TRAIN_FILE=/data/deepscaler/train.parquet
P0_7_TEST_FILE=/data/deepscaler/test.parquet
P0_7_DATASET_FRACTION=0.005
P0_7_SUMMARY_DIR="$P0_7_OUTPUT_ROOT/summary"
P0_7_PLAN_ROOT="$P0_7_OUTPUT_ROOT/prebuilt_plans"
P0_7_RELEASE_PLAN="$P0_7_PLAN_ROOT/release_area"
P0_7_MINSKEW_PLAN="$P0_7_PLAN_ROOT/min_skew"
P0_7_PLAN_AUDIT="$P0_7_OUTPUT_ROOT/prebuilt_plan_audit.json"
MANIFEST="$P0_7_OUTPUT_ROOT/trial_manifest.tsv"

if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
      || ! -f "$COMMON_EPOCH0_ROOT/reuse.env" ]]; then
    echo "preserved common epoch0 is incomplete: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi
if [[ ! -d "$P0_7_PLANNING_HISTORY_DIR/rollout_data" \
      || ! -d "$P0_7_PLANNING_HISTORY_DIR/rollout_length" \
      || ! -f "$P0_7_PLANNING_HISTORY_DIR/offline_planning_history.json" ]]; then
    echo "candidate planning history is incomplete: $P0_7_PLANNING_HISTORY_DIR" >&2
    exit 2
fi
if [[ ! -f "$P0_7_TRAIN_FILE" || ! -f "$P0_7_TEST_FILE" ]]; then
    echo "fixed DeepScaleR workload is incomplete: $P0_7_TRAIN_FILE $P0_7_TEST_FILE" >&2
    exit 2
fi
if ! [[ "$P0_7_MIN_FREE_GIB" =~ ^[0-9]+$ ]]; then
    echo "P0_7_MIN_FREE_GIB must be a nonnegative integer: $P0_7_MIN_FREE_GIB" >&2
    exit 2
fi
case "$P0_7_DRY_RUN" in
    0|1) ;;
    *) echo "P0_7_DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac

IFS=',' read -r -a forced_floors <<< "$P0_7_FORCE_SELECTED_FLOORS"
if (( ${#forced_floors[@]} != 5 )); then
    echo "P0_7_FORCE_SELECTED_FLOORS must contain five entries" >&2
    exit 2
fi
for floor in "${forced_floors[@]}"; do
    case "$floor" in
        2|4|8|16) ;;
        *) echo "unsupported forced floor: $floor" >&2; exit 2 ;;
    esac
done
normalized_floors=$(IFS=,; echo "${forced_floors[*]}")
if [[ "$normalized_floors" != "$P0_7_FORCE_SELECTED_FLOORS" ]]; then
    echo "invalid forced floor format: $P0_7_FORCE_SELECTED_FLOORS" >&2
    exit 2
fi

declare -A seen_seeds=()
for seed in $P0_7_TRIAL_SEEDS; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "invalid trial seed: $seed" >&2
        exit 2
    fi
    if [[ -n "${seen_seeds[$seed]:-}" ]]; then
        echo "duplicate trial seed: $seed" >&2
        exit 2
    fi
    seen_seeds[$seed]=1
done
if (( ${#seen_seeds[@]} < 1 )); then
    echo "P0_7_TRIAL_SEEDS is empty" >&2
    exit 2
fi

mkdir -p "$P0_7_OUTPUT_ROOT" "$P0_7_SUMMARY_DIR" "$P0_7_PLAN_ROOT"
if [[ ! -f "$MANIFEST" ]]; then
    printf 'trial_seed\tpolicy\toutput_dir\tstatus\n' > "$MANIFEST"
fi

sha256sum \
    "$SCRIPT_DIR/run_paper_p0_7_causal_matching_from_history.sh" \
    "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
    "$SCRIPT_DIR/run_baseline_minskew_epoch1_2.sh" \
    "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh" \
    "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2.sh" \
    "$SCRIPT_DIR/tools/build_mode1_length_sorted_e2e_plan.py" \
    "$SCRIPT_DIR/analysis_eval/analyze_p0_matching_ablation.py" \
    "$SCRIPT_DIR/verl/utils/rollout_seeding.py" \
    > "$P0_7_OUTPUT_ROOT/code_sha256.txt"
{
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
    printf 'planning_history_dir=%s\n' "$P0_7_PLANNING_HISTORY_DIR"
    printf 'trial_seeds=%s\n' "$P0_7_TRIAL_SEEDS"
    printf 'forced_floors=%s\n' "$P0_7_FORCE_SELECTED_FLOORS"
    printf 'train_file=%s\n' "$P0_7_TRAIN_FILE"
    printf 'train_file_sha256=%s\n' "$(sha256sum "$P0_7_TRAIN_FILE" | cut -d' ' -f1)"
    printf 'test_file=%s\n' "$P0_7_TEST_FILE"
    printf 'test_file_sha256=%s\n' "$(sha256sum "$P0_7_TEST_FILE" | cut -d' ' -f1)"
    printf 'dataset_fraction=%s\n' "$P0_7_DATASET_FRACTION"
    printf 'actor_frozen=true\n'
    printf 'per_request_sampling_seeds=true\n'
    printf 'epochs_per_trial=1\n'
    printf 'steps_per_epoch=5\n'
    printf 'responses_per_step=512\n'
    printf 'same_natural_floor2_lifecycle=true\n'
} > "$P0_7_OUTPUT_ROOT/protocol.env"

build_plan() {
    local policy="$1"
    local plan_dir="$2"
    local complete_files=0
    local plan_file
    local inputs_file="$plan_dir/plan_inputs.env"
    local planner_sha
    local train_sha
    planner_sha=$(sha256sum "$SCRIPT_DIR/tools/build_mode1_length_sorted_e2e_plan.py" | cut -d' ' -f1)
    train_sha=$(sha256sum "$P0_7_TRAIN_FILE" | cut -d' ' -f1)

    for plan_file in \
        length_sorted_train.parquet \
        length_sorted_rank_plan.json \
        length_sorted_rank_plan_summary.json \
        length_sorted_length_oracle.json; do
        if [[ -f "$plan_dir/$plan_file" ]]; then
            complete_files=$((complete_files + 1))
        fi
    done
    if (( complete_files == 4 )) \
       && [[ -f "$inputs_file" ]] \
       && grep -qFx "policy=$policy" "$inputs_file" \
       && grep -qFx "planning_history=$P0_7_PLANNING_HISTORY_DIR" "$inputs_file" \
       && grep -qFx "forced_floors=$P0_7_FORCE_SELECTED_FLOORS" "$inputs_file" \
       && grep -qFx "train_sha256=$train_sha" "$inputs_file" \
       && grep -qFx "planner_sha256=$planner_sha" "$inputs_file"; then
        echo "[P0.7] reuse prebuilt policy=$policy plan=$plan_dir"
        return 0
    fi
    if [[ -e "$plan_dir" ]]; then
        if [[ "$plan_dir" != "$P0_7_PLAN_ROOT/"* ]]; then
            echo "refusing unsafe partial-plan cleanup: $plan_dir" >&2
            return 2
        fi
        rm -r -- "$plan_dir"
    fi
    mkdir -p "$plan_dir"
    echo "[P0.7] build prebuilt policy=$policy plan=$plan_dir"
    python3 -u "$SCRIPT_DIR/tools/build_mode1_length_sorted_e2e_plan.py" \
        --baseline-dir "$P0_7_PLANNING_HISTORY_DIR" \
        --length-ema-decay 0.3 \
        --train-file "$P0_7_TRAIN_FILE" \
        --output-train "$plan_dir/length_sorted_train.parquet" \
        --output-plan "$plan_dir/length_sorted_rank_plan.json" \
        --output-summary "$plan_dir/length_sorted_rank_plan_summary.json" \
        --output-oracle "$plan_dir/length_sorted_length_oracle.json" \
        --steps 5 \
        --batch-size 32 \
        --responses-per-prompt 16 \
        --dataset-fraction "$P0_7_DATASET_FRACTION" \
        --max-rank-peak-tokens 380800 \
        --adaptive-floor \
        --min-adaptive-floor 2 \
        --force-selected-floors "$P0_7_FORCE_SELECTED_FLOORS" \
        --floor-kv-caps 2:131072,4:280576,8:315648,16:380800 \
        --rank-matching-policy "$policy" \
        --active-peak-safety-factor 1.16 \
        --max-response-len 16384 \
        --max-cross-step-repair-swaps 8 \
        --repair-candidate-limit 8 \
        --tail-guard-ratio-quantile 0.95 \
        --tail-guard-ratio-window 3 \
        --tail-guard-default-ratio 1.20 \
        --tail-guard-min-cap 4096 \
        --tail-guard-round-to 512 \
        --require-compact-history \
        > "$plan_dir/planner.stdout.log"
    {
        printf 'policy=%s\n' "$policy"
        printf 'planning_history=%s\n' "$P0_7_PLANNING_HISTORY_DIR"
        printf 'forced_floors=%s\n' "$P0_7_FORCE_SELECTED_FLOORS"
        printf 'train_sha256=%s\n' "$train_sha"
        printf 'planner_sha256=%s\n' "$planner_sha"
    } > "$inputs_file"
}

build_plan release_area "$P0_7_RELEASE_PLAN"
build_plan min_skew "$P0_7_MINSKEW_PLAN"

python3 - "$P0_7_RELEASE_PLAN" "$P0_7_MINSKEW_PLAN" \
    "$P0_7_FORCE_SELECTED_FLOORS" "$P0_7_PLAN_AUDIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

release_dir = Path(sys.argv[1])
minskew_dir = Path(sys.argv[2])
expected_floors = [int(value) for value in sys.argv[3].split(",")]
output = Path(sys.argv[4])


def load(directory: Path) -> tuple[list[dict], list[dict]]:
    summary = json.loads(
        (directory / "length_sorted_rank_plan_summary.json").read_text()
    )
    plan = json.loads((directory / "length_sorted_rank_plan.json").read_text())
    if len(summary) != 5 or len(plan) != 5:
        raise SystemExit(f"{directory}: expected five plan steps")
    return summary, plan


def source_to_rank(step: dict) -> dict[int, int]:
    result = {}
    for rank, prompts in step["rank_to_source_idx"].items():
        for prompt in prompts:
            prompt = int(prompt)
            if prompt in result:
                raise SystemExit(f"duplicate prompt {prompt} in rank plan")
            result[prompt] = int(rank)
    return result


release_summary, release_plan = load(release_dir)
minskew_summary, minskew_plan = load(minskew_dir)
steps = []
positive_candidates = []
for index, (rel_sum, min_sum, rel_plan, min_plan) in enumerate(
    zip(release_summary, minskew_summary, release_plan, minskew_plan, strict=True),
    start=1,
):
    rel_floor = int(rel_sum["selected_floor"])
    min_floor = int(min_sum["selected_floor"])
    if rel_floor != expected_floors[index - 1] or min_floor != rel_floor:
        raise SystemExit(
            f"step {index}: floors release/minskew={rel_floor}/{min_floor}, "
            f"expected {expected_floors[index - 1]}"
        )
    for label, summary, policy in (
        ("release", rel_sum, "release_area"),
        ("minskew", min_sum, "min_skew"),
    ):
        if summary["rank_matching_policy"] != policy:
            raise SystemExit(
                f"step {index}: {label} policy={summary['rank_matching_policy']}"
            )
        if not summary["feasible"]:
            raise SystemExit(f"step {index}: {label} plan is infeasible")
        if float(summary["max_adjusted_rank_peak_tokens"]) > float(summary["kv_cap"]):
            raise SystemExit(f"step {index}: {label} adjusted peak exceeds KV cap")
    for field in ("kv_cap", "tail_guard_response_cap", "tail_guard_enabled"):
        if rel_sum[field] != min_sum[field]:
            raise SystemExit(f"step {index}: paired {field} differs")
    rel_mapping = source_to_rank(rel_plan)
    min_mapping = source_to_rank(min_plan)
    if set(rel_mapping) != set(min_mapping):
        raise SystemExit(f"step {index}: paired prompt sets differ")
    moved = sum(rel_mapping[prompt] != min_mapping[prompt] for prompt in rel_mapping)
    gain = float(rel_sum["release_area"]) - float(min_sum["release_area"])
    row = {
        "step": index,
        "floor": rel_floor,
        "moved_prompts": moved,
        "release_predicted_area": float(rel_sum["release_area"]),
        "minskew_predicted_area": float(min_sum["release_area"]),
        "predicted_area_gain": gain,
        "kv_cap": float(rel_sum["kv_cap"]),
        "release_adjusted_peak": float(rel_sum["max_adjusted_rank_peak_tokens"]),
        "minskew_adjusted_peak": float(min_sum["max_adjusted_rank_peak_tokens"]),
        "tail_guard_response_cap": int(rel_sum["tail_guard_response_cap"]),
    }
    steps.append(row)
    if rel_floor < 16 and moved > 0 and gain > 0.0:
        positive_candidates.append(index)

if not positive_candidates:
    raise SystemExit("prebuilt plans have no positive-release mapping difference")
train_hashes = {}
for label, directory in (("release", release_dir), ("minskew", minskew_dir)):
    payload = (directory / "length_sorted_train.parquet").read_bytes()
    train_hashes[label] = hashlib.sha256(payload).hexdigest()

payload = {
    "schema_version": 1,
    "expected_floors": expected_floors,
    "positive_candidate_steps": positive_candidates,
    "ordered_train_sha256": train_hashes,
    "steps": steps,
}
output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

policy_run_name() {
    local policy="$1"
    local seed="$2"
    printf 'p0_natural_%s_seed%s_frozen_epoch1' "$policy" "$seed"
}

policy_variant() {
    case "$1" in
        release) printf 'adafloor_n_f2' ;;
        minskew) printf 'minskew' ;;
        *) return 1 ;;
    esac
}

policy_plan_dir() {
    case "$1" in
        release) printf '%s' "$P0_7_RELEASE_PLAN" ;;
        minskew) printf '%s' "$P0_7_MINSKEW_PLAN" ;;
        *) return 1 ;;
    esac
}

policy_order_for_seed() {
    if (( $1 % 2 == 0 )); then
        printf '%s\n' minskew release
    else
        printf '%s\n' release minskew
    fi
}

record_status() {
    local seed="$1" policy="$2" output_dir="$3" status="$4"
    local tmp="$MANIFEST.tmp"
    awk -F '\t' -v seed="$seed" -v policy="$policy" \
        'NR == 1 || !($1 == seed && $2 == policy)' "$MANIFEST" > "$tmp"
    printf '%s\t%s\t%s\t%s\n' "$seed" "$policy" "$output_dir" "$status" >> "$tmp"
    mv -f -- "$tmp" "$MANIFEST"
}

check_disk() {
    local free_kib required_kib
    free_kib=$(df --output=avail -k "$P0_7_OUTPUT_ROOT" | tail -1 | tr -d '[:space:]')
    required_kib=$((P0_7_MIN_FREE_GIB * 1024 * 1024))
    if ! [[ "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < required_kib )); then
        echo "insufficient disk before next P0.7 trial: free_kib=${free_kib:-unknown} required_kib=$required_kib" >&2
        return 1
    fi
    echo "[P0.7] disk_free_gib=$((free_kib / 1024 / 1024))"
}

analyze_pairs() {
    local incomplete_flag=()
    if [[ "$1" == "allow-incomplete" ]]; then
        incomplete_flag+=(--allow-incomplete)
    fi
    # shellcheck disable=SC2086
    python3 "$SCRIPT_DIR/analysis_eval/analyze_p0_matching_ablation.py" \
        --root "$P0_7_OUTPUT_ROOT" \
        --output-dir "$P0_7_SUMMARY_DIR" \
        --release-run-template 'p0_natural_release_seed{seed}_frozen_epoch1' \
        --minskew-run-template 'p0_natural_minskew_seed{seed}_frozen_epoch1' \
        --title 'Natural F2 causal matching ablation' \
        --expected-seeds $P0_7_TRIAL_SEEDS \
        --require-different-release-step \
        "${incomplete_flag[@]}"
}

run_policy() {
    local seed="$1" policy="$2"
    local variant run_name trial_root run_dir cleanup_record plan_dir rc
    variant=$(policy_variant "$policy")
    plan_dir=$(policy_plan_dir "$policy")
    run_name=$(policy_run_name "$policy" "$seed")
    trial_root="$P0_7_OUTPUT_ROOT/trial_seed_${seed}"
    run_dir="$trial_root/$run_name"
    cleanup_record="$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
    mkdir -p "$trial_root"

    if [[ -f "$cleanup_record" ]] \
       && grep -qFx "validated_epochs=001" "$cleanup_record" \
       && ! find "$run_dir" -mindepth 2 -maxdepth 2 -type d -name checkpoints \
            -print -quit | grep -q .; then
        echo "[P0.7] already complete seed=$seed policy=$policy output=$run_dir"
        record_status "$seed" "$policy" "$run_dir" complete
        return 0
    fi
    if [[ -e "$run_dir" ]]; then
        echo "incomplete existing P0.7 output requires inspection: $run_dir" >&2
        record_status "$seed" "$policy" "$run_dir" incomplete
        return 3
    fi
    if [[ "$P0_7_DRY_RUN" == "1" ]]; then
        echo "[P0.7] dry run seed=$seed policy=$policy variant=$variant plan=$plan_dir output=$run_dir"
        return 0
    fi

    check_disk
    echo "[P0.7] start seed=$seed policy=$policy variant=$variant plan=$plan_dir"
    record_status "$seed" "$policy" "$run_dir" running
    set +e
    env -u FAIR_RUN_NAME \
        COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
        FAIR_PLANNING_HISTORY_DIR="$P0_7_PLANNING_HISTORY_DIR" \
        FAIR_OUTPUT_ROOT="$trial_root" \
        DYNAMIC_RUN_NAME="$run_name" \
        FAIR_START_EPOCH=1 \
        FAIR_TOTAL_EPOCHS=2 \
        FAIR_FREEZE_ACTOR=1 \
        FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
        VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
        TRAIN_FILE_ORIG="$P0_7_TRAIN_FILE" \
        TEST_FILE="$P0_7_TEST_FILE" \
        DATASET_FRACTION_FOR_ORACLE="$P0_7_DATASET_FRACTION" \
        PREBUILT_PLAN_DIR="$plan_dir" \
        DYNAMIC_FORCE_SELECTED_FLOORS="$P0_7_FORCE_SELECTED_FLOORS" \
        FORCE_SELECTED_FLOORS="$P0_7_FORCE_SELECTED_FLOORS" \
        MIN_SKEW_FORCE_SELECTED_FLOORS="$P0_7_FORCE_SELECTED_FLOORS" \
        RANK_MATCHING_POLICY="$([[ "$policy" == release ]] && echo release_area || echo min_skew)" \
        "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
            "$variant" "actor_rollout_ref.rollout.seed=$seed"
    rc=$?
    set -e
    if (( rc != 0 )); then
        record_status "$seed" "$policy" "$run_dir" failed
        echo "[P0.7] failed seed=$seed policy=$policy rc=$rc" >&2
        return "$rc"
    fi
    record_status "$seed" "$policy" "$run_dir" complete
}

echo "[P0.7] output_root=$P0_7_OUTPUT_ROOT"
echo "[P0.7] common_epoch0=$COMMON_EPOCH0_ROOT"
echo "[P0.7] planning_history=$P0_7_PLANNING_HISTORY_DIR"
echo "[P0.7] seeds=$P0_7_TRIAL_SEEDS floors=$P0_7_FORCE_SELECTED_FLOORS"

for seed in $P0_7_TRIAL_SEEDS; do
    while IFS= read -r policy; do
        run_policy "$seed" "$policy"
    done < <(policy_order_for_seed "$seed")
    if [[ "$P0_7_DRY_RUN" != "1" ]]; then
        analyze_pairs allow-incomplete
    fi
done

if [[ "$P0_7_DRY_RUN" == "1" ]]; then
    echo "[P0.7] dry run complete plan_audit=$P0_7_PLAN_AUDIT"
    exit 0
fi
analyze_pairs complete
echo "[P0.7] all trials complete summary=$P0_7_SUMMARY_DIR/summary.md"
