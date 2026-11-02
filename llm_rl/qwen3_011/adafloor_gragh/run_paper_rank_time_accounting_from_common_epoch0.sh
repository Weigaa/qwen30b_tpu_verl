#!/usr/bin/env bash
set -euo pipefail

if [[ "${ADAFLOOR_RANK_TIME_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    source_path=$(realpath "${BASH_SOURCE[0]}")
    snapshot=$(mktemp "${source_path}.run-snapshot.XXXXXX")
    cp -- "$source_path" "$snapshot"
    chmod 700 "$snapshot"
    set +e
    ADAFLOOR_RANK_TIME_SNAPSHOT_ACTIVE=1 "$snapshot" "$@"
    rc=$?
    set -e
    rm -f -- "$snapshot"
    exit "$rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
ACCOUNTING_OUTPUT_ROOT="${ACCOUNTING_OUTPUT_ROOT:-/data/adafloor_shared_state/rank_time_accounting_common_epoch0_$(date -u +%Y%m%dT%H%M%SZ)}"
ACCOUNTING_SEED="${ACCOUNTING_SEED:-404}"
ACCOUNTING_MIN_FREE_GIB="${ACCOUNTING_MIN_FREE_GIB:-180}"
ACCOUNTING_ARCHIVE_INCOMPLETE="${ACCOUNTING_ARCHIVE_INCOMPLETE:-0}"
VANILLA_NAME="ranktime_vanilla_seed${ACCOUNTING_SEED}_frozen_epoch1"
ADAFLOOR_NAME="ranktime_adafloor_natural_floor2_seed${ACCOUNTING_SEED}_frozen_epoch1"
VANILLA_DIR="$ACCOUNTING_OUTPUT_ROOT/$VANILLA_NAME"
ADAFLOOR_DIR="$ACCOUNTING_OUTPUT_ROOT/$ADAFLOOR_NAME"
SUMMARY_DIR="$ACCOUNTING_OUTPUT_ROOT/summary"

if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
      || ! -f "$COMMON_EPOCH0_ROOT/reuse.env" ]]; then
    echo "preserved common epoch0 is incomplete: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi
if ! [[ "$ACCOUNTING_SEED" =~ ^[0-9]+$ \
        && "$ACCOUNTING_MIN_FREE_GIB" =~ ^[0-9]+$ \
        && "$ACCOUNTING_ARCHIVE_INCOMPLETE" =~ ^[01]$ ]]; then
    echo "invalid rank-time numeric setting" >&2
    exit 2
fi

mkdir -p "$ACCOUNTING_OUTPUT_ROOT" "$SUMMARY_DIR"
{
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
    printf 'sampling_seed=%s\n' "$ACCOUNTING_SEED"
    printf 'actor_frozen=true\n'
    printf 'per_request_sampling_seeds=true\n'
    printf 'epochs=1\n'
    printf 'dummy_timing=host_enqueue_unsynchronized\n'
    printf 'release_definition=coordinated_16_to_8_to_4_to_2\n'
    printf 'sidecar_enabled=false\n'
} > "$ACCOUNTING_OUTPUT_ROOT/protocol.env"
sha256sum \
    "$SCRIPT_DIR/run_paper_rank_time_accounting_from_common_epoch0.sh" \
    "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
    "$SCRIPT_DIR/analysis_eval/build_rank_time_summary.py" \
    > "$ACCOUNTING_OUTPUT_ROOT/code_sha256.txt"

check_disk() {
    local free_kib required_kib
    free_kib=$(df --output=avail -k "$ACCOUNTING_OUTPUT_ROOT" | tail -1 | tr -d '[:space:]')
    required_kib=$((ACCOUNTING_MIN_FREE_GIB * 1024 * 1024))
    if ! [[ "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < required_kib )); then
        echo "insufficient disk for rank-time run: free_kib=${free_kib:-unknown} required_kib=$required_kib" >&2
        return 1
    fi
    echo "[rank-time] disk_free_gib=$((free_kib / 1024 / 1024))"
}

run_variant() {
    local variant="$1" run_name="$2" run_dir="$3"
    local archive_dir
    if [[ -f "$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; then
        echo "[rank-time] already complete variant=$variant output=$run_dir"
        return 0
    fi
    if [[ -e "$run_dir" ]]; then
        if [[ "$ACCOUNTING_ARCHIVE_INCOMPLETE" != "1" ]]; then
            echo "incomplete existing rank-time output requires inspection: $run_dir" >&2
            return 3
        fi
        archive_dir="${run_dir}.interrupted.$(date -u +%Y%m%dT%H%M%SZ)"
        mv -- "$run_dir" "$archive_dir"
        printf '%s\t%s\t%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_dir" "$archive_dir" \
            >> "$ACCOUNTING_OUTPUT_ROOT/interrupted_runs.tsv"
        echo "[rank-time] archived incomplete output=$archive_dir"
    fi
    check_disk
    echo "[rank-time] start variant=$variant output=$run_dir"
    if [[ "$variant" == "vanilla" ]]; then
        env -u DYNAMIC_RUN_NAME \
            COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
            FAIR_OUTPUT_ROOT="$ACCOUNTING_OUTPUT_ROOT" \
            FAIR_RUN_NAME="$run_name" \
            FAIR_START_EPOCH=1 \
            FAIR_TOTAL_EPOCHS=2 \
            FAIR_FREEZE_ACTOR=1 \
            FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
            VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING=1 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=0 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE=0 \
            VLLM_ASCEND_DUMMY_WASTE_SELECTION_STATS=0 \
            "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
                "$variant" "actor_rollout_ref.rollout.seed=$ACCOUNTING_SEED"
    else
        env -u FAIR_RUN_NAME \
            COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
            FAIR_OUTPUT_ROOT="$ACCOUNTING_OUTPUT_ROOT" \
            DYNAMIC_RUN_NAME="$run_name" \
            FAIR_START_EPOCH=1 \
            FAIR_TOTAL_EPOCHS=2 \
            FAIR_FREEZE_ACTOR=1 \
            FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
            VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING=1 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=0 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE=0 \
            VLLM_ASCEND_DUMMY_WASTE_SELECTION_STATS=0 \
            "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
                "$variant" "actor_rollout_ref.rollout.seed=$ACCOUNTING_SEED"
    fi
}

validate_dummy_timing() {
    local run_dir="$1"
    local log_file
    log_file=$(find "$run_dir" -path '*/logs/*.txt' -type f -print \
        | sort | tail -1)
    if [[ ! -f "$log_file" ]] || ! grep -qF 'Dummy waste timing:' "$log_file"; then
        echo "dummy timing instrumentation produced no records: $run_dir" >&2
        return 4
    fi
}

echo "[rank-time] output_root=$ACCOUNTING_OUTPUT_ROOT"
echo "[rank-time] common_epoch0=$COMMON_EPOCH0_ROOT"
echo "[rank-time] seed=$ACCOUNTING_SEED"
if [[ "${ACCOUNTING_DRY_RUN:-0}" == "1" ]]; then
    echo "[rank-time] would run vanilla -> $VANILLA_DIR"
    echo "[rank-time] would run adafloor_n_f2 -> $ADAFLOOR_DIR"
    echo "[rank-time] dry run only"
    exit 0
fi

run_variant vanilla "$VANILLA_NAME" "$VANILLA_DIR"
validate_dummy_timing "$VANILLA_DIR"
run_variant adafloor_n_f2 "$ADAFLOOR_NAME" "$ADAFLOOR_DIR"
validate_dummy_timing "$ADAFLOOR_DIR"

python3 "$SCRIPT_DIR/analysis_eval/build_rank_time_summary.py" \
    --variant "Vanilla=$VANILLA_DIR" \
    --variant "AdaFloor-N-F2=$ADAFLOOR_DIR" \
    --output-dir "$SUMMARY_DIR"
echo "[rank-time] complete summary=$SUMMARY_DIR/rank_time_summary.md"
