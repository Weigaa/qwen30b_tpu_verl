#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Usage:
  FLOOR=2 bash run_mode1_floor_kv_cap_sweep.sh
  FLOOR=4 bash run_mode1_floor_kv_cap_sweep.sh

Find the largest safe KV cache cap for mode1 floor=2 or floor=4 on the current
machine/runtime. Each candidate runs the corresponding threshold multistep smoke
test in an isolated OUTPUT_SUBDIR, then appends a CSV row under:

  mode1_kv_cap_sweeps/<timestamp>_floor<FLOOR>/results.csv

Important env overrides:
  FLOOR=2|4
  KV_CANDIDATES="147456 139264 131072 122880 114688 100000"
  STOP_ON_FIRST_PASS=1       # default: stop after the first passing cap
  PRESERVE_CANDIDATE_ORDER=0  # default: sort candidates numerically high->low
  FAIL_ON_TRACEBACK=0         # default: treat traceback as warning if exit_code=0
  MODE1_FLOOR2_SUCCESS_LOGIC=1 # floor=2 default: match known-good 147456 run
  DATASET_FRACTION=0.005
  TRAINER_TOTAL_EPOCHS=1
  MAX_RESPONSE_LENGTH=640
  VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=...

Default candidates:
  floor=2: 147456 139264 135168 131072 122880 114688 100000
  floor=4: 321408 300000 280576 260000 240000 220000 200000

Pass criteria:
  - process exit_code is 0
  - the completion validator sees EXPECTED_STEPS complete start/done/metric groups
  - no RuntimeError/strict-OOM/Memory_Allocation/aclnnMoeDistributeDispatchV4/Preempting
  - Traceback is recorded separately; set FAIL_ON_TRACEBACK=1 to make it fatal
  - no severe rollout regression:
      max(step2..end) <= max(ROLLOUT_REGRESSION_ABS_S, step1 + ROLLOUT_REGRESSION_DELTA_S)

EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

FLOOR="${FLOOR:-}"
if [[ "$FLOOR" != "2" && "$FLOOR" != "4" ]]; then
    echo "FLOOR must be 2 or 4; got '${FLOOR:-<unset>}'" >&2
    usage >&2
    exit 2
fi

STOP_ON_FIRST_PASS="${STOP_ON_FIRST_PASS:-1}"
PRESERVE_CANDIDATE_ORDER="${PRESERVE_CANDIDATE_ORDER:-0}"
FAIL_ON_TRACEBACK="${FAIL_ON_TRACEBACK:-0}"
EXPECTED_STEPS="${EXPECTED_STEPS:-5}"
ROLLOUT_REGRESSION_ABS_S="${ROLLOUT_REGRESSION_ABS_S:-180}"
ROLLOUT_REGRESSION_DELTA_S="${ROLLOUT_REGRESSION_DELTA_S:-45}"

if [[ -z "${KV_CANDIDATES:-}" ]]; then
    if [[ "$FLOOR" == "2" ]]; then
        KV_CANDIDATES="147456 139264 135168 131072 122880 114688 100000"
    else
        KV_CANDIDATES="321408 300000 280576 260000 240000 220000 200000"
    fi
fi

if [[ "$PRESERVE_CANDIDATE_ORDER" != "1" ]]; then
    KV_CANDIDATES=$(printf '%s\n' $KV_CANDIDATES | sort -nr | tr '\n' ' ' | sed 's/[[:space:]]*$//')
fi

STAMP="${SWEEP_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/mode1_kv_cap_sweeps/${STAMP}_floor${FLOOR}}"
mkdir -p "$SWEEP_DIR"
RESULTS="$SWEEP_DIR/results.csv"

if [[ ! -f "$RESULTS" ]]; then
    printf 'floor,kv,actual_kv,kv_actual_ok,output_subdir,exit_code,rollout_count,step1_s,max_later_s,regression,hard_errors,tracebacks,completion_ok,pass,failure_reason,log_path,analysis_path\n' >"$RESULTS"
fi

if [[ "$FLOOR" == "2" ]]; then
    RUNNER="$SCRIPT_DIR/run_mode1_local_baseline_random_batch_floor2_threshold_multistep.sh"
else
    RUNNER="$SCRIPT_DIR/run_mode1_local_baseline_random_batch_threshold_multistep.sh"
fi

printf '[kv sweep] floor=%s candidates="%s" sweep_dir=%s runner=%s\n' \
    "$FLOOR" "$KV_CANDIDATES" "$SWEEP_DIR" "$RUNNER"

for kv in $KV_CANDIDATES; do
    output_subdir="mode1_kv_cap_sweep_floor${FLOOR}_kv${kv}_${STAMP}"
    run_log="$SWEEP_DIR/floor${FLOOR}_kv${kv}.launcher.log"
    printf '\n[kv sweep] start floor=%s kv=%s output=%s\n' "$FLOOR" "$kv" "$output_subdir" | tee -a "$run_log"

    set +e
    if [[ "$FLOOR" == "2" ]]; then
        OUTPUT_SUBDIR="$output_subdir" \
        KV_TOKENS="$kv" \
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="$kv" \
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="$kv" \
        MODE1_FLOOR2_SUCCESS_LOGIC="${MODE1_FLOOR2_SUCCESS_LOGIC:-1}" \
        MODE1_EXPECTED_ROLLOUT_STEPS="$EXPECTED_STEPS" \
        bash "$RUNNER" "$@" 2>&1 | tee -a "$run_log"
    else
        OUTPUT_SUBDIR="$output_subdir" \
        KV_TOKENS="$kv" \
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="$kv" \
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="$kv" \
        VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR="${VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR:-0}" \
        VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM="${VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM:-0}" \
        MODE1_EXPECTED_ROLLOUT_STEPS="$EXPECTED_STEPS" \
        bash "$RUNNER" "$@" 2>&1 | tee -a "$run_log"
    fi
    rc=${PIPESTATUS[0]}
    set -e

    log_path=$(find "$SCRIPT_DIR/$output_subdir/logs" -maxdepth 1 -type f \
        -name 'wjqwen30b-a3b-record_graph_save4eagle3_[0-9]*.txt' \
        ! -name '*_analysis.txt' \
        ! -name '*_analysis_summary.txt' \
        ! -name '*_analysis_summary_analysis.txt' \
        2>/dev/null | sort | tail -1 || true)
    analysis_path=""
    rollout_count=0
    actual_kv=""
    kv_actual_ok="unknown"
    step1_s=""
    max_later_s=""
    regression="unknown"
    hard_errors="missing_log"
    tracebacks="missing_log"
    completion_ok="0"
    failure_reason="missing_log"
    pass="0"

    if [[ -n "$log_path" && -f "$log_path" ]]; then
        analysis_path="${log_path%.txt}_analysis.txt"
        actual_kv=$(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$log_path" | sed -E 's/.*: ([0-9,]+) tokens/\1/; s/,//g' || true)
        if [[ -n "$actual_kv" && "$actual_kv" =~ ^[0-9]+$ ]]; then
            if [[ "$actual_kv" -ge "$kv" ]]; then
                kv_actual_ok="1"
            else
                kv_actual_ok="0"
            fi
        fi
        python "$SCRIPT_DIR/tools/analyze_mode1_step_regression.py" "$log_path" >"$analysis_path" || true
        mapfile -t rollout_times < <(grep -oE 'rollout_output_time_s: [0-9.]+' "$log_path" | awk '{print $2}')
        rollout_count="${#rollout_times[@]}"
        if [[ "$rollout_count" -gt 0 ]]; then
            step1_s="${rollout_times[0]}"
        fi
        if [[ "$rollout_count" -gt 1 ]]; then
            max_later_s=$(printf '%s\n' "${rollout_times[@]:1}" | awk 'BEGIN{m=0} {if($1>m)m=$1} END{printf "%.6f", m}')
            regression=$(awk -v s1="$step1_s" -v mx="$max_later_s" \
                -v abs="$ROLLOUT_REGRESSION_ABS_S" -v delta="$ROLLOUT_REGRESSION_DELTA_S" \
                'BEGIN{limit=(abs > s1+delta ? abs : s1+delta); print (mx > limit ? "yes" : "no")}')
        else
            max_later_s=""
            regression="unknown"
        fi

        # Keep OOM matching strict: case-insensitive "OOM" also matches benign
        # words such as "headroom", which made successful runs look failed.
        hard_error_pattern='RuntimeError|Memory_Allocation|aclnnMoeDistributeDispatchV4|Preempting|(^|[^[:alpha:]])OOM([^[:alpha:]]|$)|OutOfMemory|out of memory|NPU out of memory'
        hard_error_count=$(grep -Eic "$hard_error_pattern" "$log_path" || true)
        traceback_count=$(grep -Eic 'Traceback' "$log_path" || true)
        hard_errors="$hard_error_count"
        tracebacks="$traceback_count"
        set +e
        completion_output=$(python "$SCRIPT_DIR/tools/validate_mode1_run_completion.py" \
            "$log_path" --expected-steps "$EXPECTED_STEPS" --require-run-done 2>&1)
        completion_rc=$?
        set -e
        if [[ "$completion_rc" -eq 0 ]]; then
            completion_ok="1"
        fi
        fatal_traceback=0
        if [[ "$FAIL_ON_TRACEBACK" == "1" && "$traceback_count" -gt 0 ]]; then
            fatal_traceback=1
        fi
        failure_reason=""
        if [[ "$rc" -ne 0 ]]; then
            failure_reason="${failure_reason:+$failure_reason; }exit_code_${rc}"
        fi
        if [[ "$completion_ok" != "1" ]]; then
            failure_reason="${failure_reason:+$failure_reason; }completion_validation_failed:${completion_output//$'\n'/ }"
        fi
        if [[ "$kv_actual_ok" == "0" ]]; then
            failure_reason="${failure_reason:+$failure_reason; }actual_kv_${actual_kv}_below_requested_${kv}"
        fi
        if [[ "$hard_error_count" -gt 0 ]]; then
            failure_reason="${failure_reason:+$failure_reason; }hard_errors_${hard_error_count}"
        fi
        if [[ "$fatal_traceback" -ne 0 ]]; then
            failure_reason="${failure_reason:+$failure_reason; }tracebacks_${traceback_count}"
        fi
        if [[ "$regression" == "yes" ]]; then
            failure_reason="${failure_reason:+$failure_reason; }rollout_regression"
        fi
        if [[ -z "$failure_reason" ]]; then
            failure_reason="none"
        fi
        if [[ "$rc" -eq 0 && "$completion_ok" == "1" && "$kv_actual_ok" != "0" && "$hard_error_count" -eq 0 && "$fatal_traceback" -eq 0 && "$regression" != "yes" ]]; then
            pass="1"
        fi
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%q,%s,%s\n' \
        "$FLOOR" "$kv" "$actual_kv" "$kv_actual_ok" "$output_subdir" "$rc" "$rollout_count" \
        "$step1_s" "$max_later_s" "$regression" "$hard_errors" "$tracebacks" "$completion_ok" "$pass" "$failure_reason" \
        "$log_path" "$analysis_path" | tee -a "$RESULTS"

    if [[ "$pass" == "1" ]]; then
        printf '[kv sweep] PASS floor=%s kv=%s log=%s\n' "$FLOOR" "$kv" "$log_path" | tee -a "$run_log"
        if [[ "$STOP_ON_FIRST_PASS" == "1" ]]; then
            printf '[kv sweep] stopping on first pass. results=%s\n' "$RESULTS"
            exit 0
        fi
    else
        printf '[kv sweep] FAIL floor=%s kv=%s actual_kv=%s kv_actual_ok=%s exit=%s rollouts=%s completion_ok=%s hard_errors=%s tracebacks=%s regression=%s reason=%s\n' \
            "$FLOOR" "$kv" "${actual_kv:-}" "$kv_actual_ok" "$rc" "$rollout_count" "$completion_ok" "$hard_errors" "$tracebacks" "$regression" "$failure_reason" | tee -a "$run_log"
    fi
done

printf '[kv sweep] finished without a passing cap. results=%s\n' "$RESULTS"
exit 1
