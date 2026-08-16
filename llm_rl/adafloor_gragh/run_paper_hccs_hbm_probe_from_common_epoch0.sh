#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
ARCH_OUTPUT_ROOT="${ARCH_OUTPUT_ROOT:-/data/adafloor_shared_state/architecture_hccs_hbm_probe_$(date -u +%Y%m%dT%H%M%SZ)}"
ARCH_RUN_NAME="${ARCH_RUN_NAME:-natural_floor2_hccs_hbm_seed606}"
ARCH_SEED="${ARCH_SEED:-606}"
RUN_DIR="$ARCH_OUTPUT_ROOT/$ARCH_RUN_NAME"
SAMPLE_ROOT="$ARCH_OUTPUT_ROOT/hccs_samples"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite architecture probe: $RUN_DIR" >&2
    exit 2
fi
if pgrep -f '[p]ython3 -m verl\.trainer\.main_ppo|[r]ay::TaskRunner\.run' >/dev/null; then
    echo "another training process is active" >&2
    exit 3
fi
mkdir -p "$ARCH_OUTPUT_ROOT" "$SAMPLE_ROOT"

set +e
COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
FAIR_OUTPUT_ROOT="$ARCH_OUTPUT_ROOT" \
DYNAMIC_RUN_NAME="$ARCH_RUN_NAME" \
FAIR_START_EPOCH=1 \
FAIR_TOTAL_EPOCHS=2 \
FAIR_FREEZE_ACTOR=1 \
FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
    "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
        adafloor_n_f2 "actor_rollout_ref.rollout.seed=$ARCH_SEED" &
run_pid=$!
set -e

log_file=""
for _ in $(seq 1 1800); do
    log_file=$(find "$RUN_DIR" -path '*/logs/*.txt' -type f -print 2>/dev/null \
        | sort | tail -1)
    [[ -n "$log_file" ]] && break
    if ! kill -0 "$run_pid" 2>/dev/null; then
        wait "$run_pid"
        exit $?
    fi
    sleep 1
done
if [[ -z "$log_file" ]]; then
    echo "timed out waiting for architecture probe log" >&2
    kill "$run_pid" 2>/dev/null || true
    wait "$run_pid" || true
    exit 3
fi
echo "[architecture probe] log=$log_file"

python3 "$SCRIPT_DIR/analysis_eval/sample_hccs_bw.py" \
    --output-dir "$SAMPLE_ROOT/decode" --label decode \
    --samples 5 --duration 1 --watch-log "$log_file" \
    --watch-regex 'Shrink-aware tail-guard response cap:' \
    --watch-timeout 10800 > "$SAMPLE_ROOT/decode.stdout" 2>&1 &
decode_pid=$!
python3 "$SCRIPT_DIR/analysis_eval/sample_hccs_bw.py" \
    --output-dir "$SAMPLE_ROOT/shrink" --label shrink \
    --samples 5 --duration 1 --watch-log "$log_file" \
    --watch-regex 'Elastic EP shrink rpc enter:' \
    --watch-timeout 10800 > "$SAMPLE_ROOT/shrink.stdout" 2>&1 &
shrink_pid=$!
python3 "$SCRIPT_DIR/analysis_eval/sample_hccs_bw.py" \
    --output-dir "$SAMPLE_ROOT/restore" --label restore \
    --samples 5 --duration 1 --watch-log "$log_file" \
    --watch-regex 'Mode1 step timeline: driver_restore_rpc_start' \
    --watch-timeout 10800 > "$SAMPLE_ROOT/restore.stdout" 2>&1 &
restore_pid=$!

set +e
wait "$run_pid"
run_rc=$?
wait "$decode_pid"; decode_rc=$?
wait "$shrink_pid"; shrink_rc=$?
wait "$restore_pid"; restore_rc=$?
set -e
if (( run_rc != 0 )); then
    echo "architecture probe training failed rc=$run_rc" >&2
    exit "$run_rc"
fi
if (( decode_rc != 0 || shrink_rc != 0 || restore_rc != 0 )); then
    echo "HCCS sampling failed decode=$decode_rc shrink=$shrink_rc restore=$restore_rc" >&2
    exit 4
fi

python3 "$SCRIPT_DIR/analysis_eval/summarize_hccs_probe.py" \
    --root "$SAMPLE_ROOT" --output "$ARCH_OUTPUT_ROOT/hccs_summary.md"
python3 "$SCRIPT_DIR/analysis_eval/analyze_mode1_hbm_breakdown.py" \
    --run natural_floor2 "$log_file" --output-dir "$ARCH_OUTPUT_ROOT/hbm"
{
    printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'run_dir=%s\n' "$RUN_DIR"
    printf 'log=%s\n' "$log_file"
    printf 'hccs_summary=%s\n' "$ARCH_OUTPUT_ROOT/hccs_summary.md"
    printf 'hbm_summary=%s\n' "$ARCH_OUTPUT_ROOT/hbm/hbm_summary.md"
} > "$ARCH_OUTPUT_ROOT/ARCHITECTURE_PROBE_COMPLETE.txt"
echo "[architecture probe] complete output=$ARCH_OUTPUT_ROOT"
