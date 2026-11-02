#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

P0_ROOT="${P0_ROOT:-/data/adafloor_shared_state/p0_matched_trials_common_epoch0_20260728T221830Z}"
FOLLOWUP_ROOT="${FOLLOWUP_ROOT:-$P0_ROOT/revision_followups}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-259200}"
EXPECTED_TRIALS="${EXPECTED_TRIALS:-4}"
ACCOUNTING_SEED="${ACCOUNTING_SEED:-404}"
ORACLE_TRIAL_SEEDS="${ORACLE_TRIAL_SEEDS:-101}"
COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"

if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ \
        && "$WAIT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ \
        && "$EXPECTED_TRIALS" =~ ^[1-9][0-9]*$ \
        && "$ACCOUNTING_SEED" =~ ^[0-9]+$ ]]; then
    echo "invalid numeric follow-up setting" >&2
    exit 2
fi
if [[ ! -f "$P0_ROOT/trial_manifest.tsv" ]]; then
    echo "missing P0 manifest: $P0_ROOT/trial_manifest.tsv" >&2
    exit 2
fi
if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]]; then
    echo "missing preserved common epoch0 marker: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi

mkdir -p "$FOLLOWUP_ROOT"
{
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'p0_root=%s\n' "$P0_ROOT"
    printf 'expected_trials=%s\n' "$EXPECTED_TRIALS"
    printf 'accounting_seed=%s\n' "$ACCOUNTING_SEED"
    printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
} > "$FOLLOWUP_ROOT/protocol.env"

completed_count() {
    awk -F '\t' 'NR > 1 && $4 == "complete" {count++} END {print count + 0}' \
        "$P0_ROOT/trial_manifest.tsv"
}

validate_completed_trials() {
    local count=0 output_dir marker
    while IFS=$'\t' read -r seed policy output_dir status; do
        [[ "$seed" == "trial_seed" ]] && continue
        [[ "$status" == "complete" ]] || continue
        marker="$output_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
        if [[ ! -f "$marker" || -n "$(find "$output_dir" -mindepth 2 -maxdepth 2 \
                -type d -name checkpoints -print -quit)" ]]; then
            echo "invalid completed P0 trial: seed=$seed policy=$policy output=$output_dir" >&2
            return 1
        fi
        count=$((count + 1))
    done < "$P0_ROOT/trial_manifest.tsv"
    if (( count != EXPECTED_TRIALS )); then
        echo "P0 completion count mismatch: got=$count expected=$EXPECTED_TRIALS" >&2
        return 1
    fi
}

waited=0
while (( $(completed_count) < EXPECTED_TRIALS )); do
    if (( waited >= WAIT_TIMEOUT_SECONDS )); then
        echo "timed out waiting for P0 after ${waited}s" >&2
        exit 3
    fi
    if grep -q $'\tincomplete$' "$P0_ROOT/trial_manifest.tsv"; then
        echo "P0 manifest contains an incomplete trial" >&2
        exit 3
    fi
    if (( waited >= 120 )) \
       && ! pgrep -f '[r]un_paper_p0_matched_trials_from_common_epoch0.sh' >/dev/null; then
        echo "P0 runner exited before all trials completed" >&2
        exit 3
    fi
    printf '[revision follow-up] waiting completed=%s/%s elapsed_s=%s\n' \
        "$(completed_count)" "$EXPECTED_TRIALS" "$waited"
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
done

validate_completed_trials
printf '[revision follow-up] all P0 trials validated\n'

python3 "$SCRIPT_DIR/analysis_eval/summarize_p0_matched_trials.py" \
    --root "$P0_ROOT" --allow-incomplete --output-dir "$P0_ROOT/summary"
python3 "$SCRIPT_DIR/analysis_eval/analyze_p0_matching_ablation.py" \
    --root "$P0_ROOT" --output-dir "$P0_ROOT/matching_ablation"

oracle_summary="$FOLLOWUP_ROOT/oracle_replay/summary/oracle_summary.md"
if [[ -s "$oracle_summary" ]]; then
    echo "[revision follow-up] reuse completed oracle replay: $oracle_summary"
else
    P0_ROOT="$P0_ROOT" \
    ORACLE_OUTPUT_ROOT="$FOLLOWUP_ROOT/oracle_replay" \
    TRIAL_SEEDS="$ORACLE_TRIAL_SEEDS" \
        "$SCRIPT_DIR/run_paper_oracle_replay_from_p0.sh"
fi

COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
ACCOUNTING_OUTPUT_ROOT="$FOLLOWUP_ROOT/rank_time_accounting" \
ACCOUNTING_SEED="$ACCOUNTING_SEED" \
    "$SCRIPT_DIR/run_paper_rank_time_accounting_from_common_epoch0.sh"

{
    printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'p0_summary=%s\n' "$P0_ROOT/summary/summary.md"
    printf 'matching_summary=%s\n' "$P0_ROOT/matching_ablation/summary.md"
    printf 'oracle_summary=%s\n' "$oracle_summary"
    printf 'rank_time_summary=%s\n' "$FOLLOWUP_ROOT/rank_time_accounting/summary/rank_time_summary.md"
} > "$FOLLOWUP_ROOT/FOLLOWUPS_COMPLETE.txt"
printf '[revision follow-up] complete marker=%s\n' "$FOLLOWUP_ROOT/FOLLOWUPS_COMPLETE.txt"
