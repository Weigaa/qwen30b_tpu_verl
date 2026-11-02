#!/usr/bin/env bash
set -euo pipefail

if [[ "${ADAFLOOR_P0_MATCHED_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    source_path=$(realpath "${BASH_SOURCE[0]}")
    snapshot=$(mktemp "${source_path}.run-snapshot.XXXXXX")
    cp -- "$source_path" "$snapshot"
    chmod 700 "$snapshot"
    set +e
    ADAFLOOR_P0_MATCHED_SNAPSHOT_ACTIVE=1 "$snapshot" "$@"
    rc=$?
    set -e
    rm -f -- "$snapshot"
    exit "$rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
P0_OUTPUT_ROOT="${P0_OUTPUT_ROOT:-/data/adafloor_shared_state/p0_matched_trials_common_epoch0_$(date -u +%Y%m%dT%H%M%SZ)}"
P0_TRIAL_SEEDS="${P0_TRIAL_SEEDS:-101 202 303}"
P0_MIN_FREE_GIB="${P0_MIN_FREE_GIB:-180}"
P0_SUMMARY_DIR="$P0_OUTPUT_ROOT/summary"
MANIFEST="$P0_OUTPUT_ROOT/trial_manifest.tsv"

if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
      || ! -f "$COMMON_EPOCH0_ROOT/reuse.env" ]]; then
    echo "preserved common epoch0 is incomplete: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi
if ! [[ "$P0_MIN_FREE_GIB" =~ ^[0-9]+$ ]]; then
    echo "P0_MIN_FREE_GIB must be a nonnegative integer: $P0_MIN_FREE_GIB" >&2
    exit 2
fi

mkdir -p "$P0_OUTPUT_ROOT" "$P0_SUMMARY_DIR"
if [[ ! -f "$MANIFEST" ]]; then
    printf 'trial_seed\tpolicy\toutput_dir\tstatus\n' > "$MANIFEST"
fi

sha256sum \
    "$SCRIPT_DIR/run_paper_p0_matched_trials_from_common_epoch0.sh" \
    "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
    "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh" \
    "$SCRIPT_DIR/tools/build_mode1_length_sorted_e2e_plan.py" \
    "$SCRIPT_DIR/verl/utils/rollout_seeding.py" \
    "$SCRIPT_DIR/verl/trainer/ppo/ray_trainer.py" \
    "$SCRIPT_DIR/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py" \
    > "$P0_OUTPUT_ROOT/code_sha256.txt"
{
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
    printf 'trial_seeds=%s\n' "$P0_TRIAL_SEEDS"
    printf 'actor_frozen=true\n'
    printf 'per_request_sampling_seeds=true\n'
    printf 'epochs_per_trial=1\n'
    printf 'steps_per_epoch=5\n'
} > "$P0_OUTPUT_ROOT/protocol.env"

policy_run_name() {
    local policy="$1"
    local seed="$2"
    printf 'p0_%s_seed%s_frozen_epoch1' "$policy" "$seed"
}

policy_fair_variant() {
    case "$1" in
        vanilla) printf 'vanilla' ;;
        lengthsort_guard) printf 'lengthsort_guard' ;;
        planned_release) printf 'adafloor_p_f4' ;;
        planned_minskew) printf 'adafloor_p_f4_minskew' ;;
        *) return 1 ;;
    esac
}

policy_order_for_seed() {
    case "$1" in
        101) printf '%s\n' lengthsort_guard planned_release planned_minskew vanilla ;;
        202) printf '%s\n' planned_release planned_minskew vanilla lengthsort_guard ;;
        303) printf '%s\n' planned_minskew vanilla lengthsort_guard planned_release ;;
        *) printf '%s\n' vanilla lengthsort_guard planned_release planned_minskew ;;
    esac
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
    free_kib=$(df --output=avail -k "$P0_OUTPUT_ROOT" | tail -1 | tr -d '[:space:]')
    required_kib=$((P0_MIN_FREE_GIB * 1024 * 1024))
    if ! [[ "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < required_kib )); then
        echo "insufficient disk before next P0 trial: free_kib=${free_kib:-unknown} required_kib=$required_kib" >&2
        return 1
    fi
    echo "[P0 matched] disk_free_gib=$((free_kib / 1024 / 1024))"
}

run_policy() {
    local seed="$1" policy="$2"
    local fair_variant run_name trial_root run_dir cleanup_record
    fair_variant=$(policy_fair_variant "$policy")
    run_name=$(policy_run_name "$policy" "$seed")
    trial_root="$P0_OUTPUT_ROOT/trial_seed_${seed}"
    run_dir="$trial_root/$run_name"
    cleanup_record="$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
    mkdir -p "$trial_root"

    if [[ -f "$cleanup_record" ]] \
       && grep -qFx "validated_epochs=001" "$cleanup_record" \
       && ! find "$run_dir" -mindepth 2 -maxdepth 2 -type d -name checkpoints \
            -print -quit | grep -q .; then
        echo "[P0 matched] already complete seed=$seed policy=$policy output=$run_dir"
        record_status "$seed" "$policy" "$run_dir" complete
        return 0
    fi
    if [[ -e "$run_dir" ]]; then
        echo "incomplete existing P0 output requires inspection: $run_dir" >&2
        record_status "$seed" "$policy" "$run_dir" incomplete
        return 3
    fi

    check_disk
    echo "[P0 matched] start seed=$seed policy=$policy variant=$fair_variant"
    record_status "$seed" "$policy" "$run_dir" running
    if [[ "$policy" == "vanilla" ]]; then
        env -u DYNAMIC_RUN_NAME \
            COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
            FAIR_OUTPUT_ROOT="$trial_root" \
            FAIR_RUN_NAME="$run_name" \
            FAIR_START_EPOCH=1 \
            FAIR_TOTAL_EPOCHS=2 \
            FAIR_FREEZE_ACTOR=1 \
            FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
            VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
            "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
                "$fair_variant" "actor_rollout_ref.rollout.seed=$seed"
    else
        env -u FAIR_RUN_NAME \
            COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
            FAIR_OUTPUT_ROOT="$trial_root" \
            DYNAMIC_RUN_NAME="$run_name" \
            FAIR_START_EPOCH=1 \
            FAIR_TOTAL_EPOCHS=2 \
            FAIR_FREEZE_ACTOR=1 \
            FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
            VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
            "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
                "$fair_variant" "actor_rollout_ref.rollout.seed=$seed"
    fi
    record_status "$seed" "$policy" "$run_dir" complete
    python3 "$SCRIPT_DIR/analysis_eval/summarize_p0_matched_trials.py" \
        --root "$P0_OUTPUT_ROOT" --allow-incomplete \
        --output-dir "$P0_SUMMARY_DIR"
}

echo "[P0 matched] output_root=$P0_OUTPUT_ROOT"
echo "[P0 matched] common_epoch0=$COMMON_EPOCH0_ROOT"
echo "[P0 matched] seeds=$P0_TRIAL_SEEDS"

for seed in $P0_TRIAL_SEEDS; do
    while IFS= read -r policy; do
        run_policy "$seed" "$policy"
    done < <(policy_order_for_seed "$seed")
done

python3 "$SCRIPT_DIR/analysis_eval/summarize_p0_matched_trials.py" \
    --root "$P0_OUTPUT_ROOT" --output-dir "$P0_SUMMARY_DIR"
echo "[P0 matched] all trials complete summary=$P0_SUMMARY_DIR/summary.md"
