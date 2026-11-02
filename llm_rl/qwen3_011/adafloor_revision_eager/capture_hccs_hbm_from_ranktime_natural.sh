#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

P0_ROOT="${P0_ROOT:-/data/adafloor_shared_state/p0_matched_trials_common_epoch0_20260728T221830Z}"
RANKTIME_ROOT="${RANKTIME_ROOT:-$P0_ROOT/revision_followups/rank_time_accounting}"
ARCH_OUTPUT_ROOT="${ARCH_OUTPUT_ROOT:-$P0_ROOT/revision_remaining/architecture_hccs_hbm}"
NATURAL_RUN_NAME="${NATURAL_RUN_NAME:-ranktime_adafloor_natural_floor2_seed404_frozen_epoch1}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-43200}"
POLL_SECONDS="${POLL_SECONDS:-2}"
RUN_DIR="$RANKTIME_ROOT/$NATURAL_RUN_NAME"
SAMPLE_ROOT="$ARCH_OUTPUT_ROOT/hccs_samples"
HCCS_SUMMARY="$ARCH_OUTPUT_ROOT/hccs_summary.md"
EXISTING_HBM_SUMMARY="${EXISTING_HBM_SUMMARY:-$P0_ROOT/hbm_breakdown_seed101/hbm_summary.md}"

finalize_architecture_evidence() {
    local log_file="$1"
    local hbm_summary="$ARCH_OUTPUT_ROOT/hbm/hbm_summary.md"
    local hbm_source="current_ranktime_log"

    if grep -q 'Mode1 comm cache compact:.*free_bytes=' "$log_file"; then
        python3 "$SCRIPT_DIR/analysis_eval/analyze_mode1_hbm_breakdown.py" \
            --run natural_floor2_ranktime "$log_file" \
            --output-dir "$ARCH_OUTPUT_ROOT/hbm"
    else
        if [[ ! -s "$EXISTING_HBM_SUMMARY" ]]; then
            echo "rank-time log has no contextualized HBM snapshots and reusable HBM evidence is missing" >&2
            return 5
        fi
        hbm_summary="$EXISTING_HBM_SUMMARY"
        hbm_source="reused_seed101_completed_hbm_breakdown"
        {
            printf 'reason=current_ranktime_log_has_no_contextualized_hbm_snapshots\n'
            printf 'source=%s\n' "$EXISTING_HBM_SUMMARY"
            printf 'new_npu_runs=0\n'
        } > "$ARCH_OUTPUT_ROOT/HBM_REUSED_EXISTING.txt"
        echo "[architecture reuse] reused existing HBM evidence=$EXISTING_HBM_SUMMARY"
    fi
    {
        printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'reused_run_dir=%s\n' "$RUN_DIR"
        printf 'log=%s\n' "$log_file"
        printf 'hccs_summary=%s\n' "$HCCS_SUMMARY"
        printf 'hbm_summary=%s\n' "$hbm_summary"
        printf 'hbm_source=%s\n' "$hbm_source"
        printf 'additional_npu_runs=0\n'
    } > "$ARCH_OUTPUT_ROOT/ARCHITECTURE_PROBE_COMPLETE.txt"
    echo "[architecture reuse] complete output=$ARCH_OUTPUT_ROOT"
}

if [[ -f "$ARCH_OUTPUT_ROOT/ARCHITECTURE_PROBE_COMPLETE.txt" ]]; then
    echo "[architecture reuse] already complete output=$ARCH_OUTPUT_ROOT"
    exit 0
fi
if ! [[ "$WAIT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ \
        && "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid architecture reuse timeout" >&2
    exit 2
fi
mkdir -p "$ARCH_OUTPUT_ROOT"
if [[ -s "$HCCS_SUMMARY" \
      && -f "$RUN_DIR/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; then
    completed_log=$(find "$RUN_DIR" -path '*/logs/*.txt' -type f -print 2>/dev/null \
        | sort | tail -1 || true)
    if [[ -z "$completed_log" || ! -f "$completed_log" ]]; then
        echo "validated rank-time run is missing its log: $RUN_DIR" >&2
        exit 3
    fi
    echo "[architecture reuse] preserving completed HCCS samples=$SAMPLE_ROOT"
    finalize_architecture_evidence "$completed_log"
    exit 0
fi
if [[ -d "$SAMPLE_ROOT" ]] && find "$SAMPLE_ROOT" -mindepth 1 -print -quit | grep -q .; then
    interrupted_samples="${SAMPLE_ROOT}.interrupted.$(date -u +%Y%m%dT%H%M%SZ)"
    mv -- "$SAMPLE_ROOT" "$interrupted_samples"
    echo "[architecture reuse] archived incomplete samples=$interrupted_samples"
fi
mkdir -p "$SAMPLE_ROOT"

log_file=""
waited=0
while [[ -z "$log_file" ]]; do
    log_file=$(find "$RUN_DIR" -path '*/logs/*.txt' -type f -print 2>/dev/null \
        | sort | tail -1 || true)
    if [[ -n "$log_file" ]]; then
        break
    fi
    if (( waited >= WAIT_TIMEOUT_SECONDS )); then
        echo "timed out waiting for Natural F2 rank-time log" >&2
        exit 3
    fi
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
done
echo "[architecture reuse] sampling rank-time log=$log_file"

python3 "$SCRIPT_DIR/analysis_eval/sample_hccs_bw.py" \
    --output-dir "$SAMPLE_ROOT/decode" --label decode \
    --samples 5 --duration 1 --watch-log "$log_file" \
    --watch-regex 'Shrink-aware tail-guard response cap:' \
    --watch-timeout "$WAIT_TIMEOUT_SECONDS" > "$SAMPLE_ROOT/decode.stdout" 2>&1 &
decode_pid=$!
python3 "$SCRIPT_DIR/analysis_eval/sample_hccs_bw.py" \
    --output-dir "$SAMPLE_ROOT/shrink" --label shrink \
    --samples 5 --duration 1 --watch-log "$log_file" \
    --watch-regex 'Elastic EP shrink rpc enter:' \
    --watch-timeout "$WAIT_TIMEOUT_SECONDS" > "$SAMPLE_ROOT/shrink.stdout" 2>&1 &
shrink_pid=$!
python3 "$SCRIPT_DIR/analysis_eval/sample_hccs_bw.py" \
    --output-dir "$SAMPLE_ROOT/restore" --label restore \
    --samples 5 --duration 1 --watch-log "$log_file" \
    --watch-regex 'Mode1 step timeline: driver_restore_rpc_start' \
    --watch-timeout "$WAIT_TIMEOUT_SECONDS" > "$SAMPLE_ROOT/restore.stdout" 2>&1 &
restore_pid=$!

set +e
wait "$decode_pid"; decode_rc=$?
wait "$shrink_pid"; shrink_rc=$?
wait "$restore_pid"; restore_rc=$?
set -e
if (( decode_rc != 0 || shrink_rc != 0 || restore_rc != 0 )); then
    echo "HCCS sampling failed decode=$decode_rc shrink=$shrink_rc restore=$restore_rc" >&2
    exit 4
fi

waited=0
while [[ ! -f "$RUN_DIR/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; do
    if (( waited >= WAIT_TIMEOUT_SECONDS )); then
        echo "timed out waiting for validated Natural F2 completion" >&2
        exit 3
    fi
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
done

python3 "$SCRIPT_DIR/analysis_eval/summarize_hccs_probe.py" \
    --root "$SAMPLE_ROOT" --output "$HCCS_SUMMARY"
finalize_architecture_evidence "$log_file"
