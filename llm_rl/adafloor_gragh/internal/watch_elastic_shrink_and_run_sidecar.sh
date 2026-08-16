#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_LOG=${1:-${VERL_SIDECAR_TRAIN_LOG:-}}
if [[ -z "${TRAIN_LOG}" ]]; then
    echo "usage: $0 /path/to/train.log" >&2
    echo "or set VERL_SIDECAR_TRAIN_LOG" >&2
    exit 2
fi
TRAIN_LOG=$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "${TRAIN_LOG}")

SIDECAR_SCRIPT=${VERL_SIDECAR_SCRIPT:-"${ROOT_DIR}/internal/run_elastic_sidecar_infer.sh"}
LEASE_LOG=${VERL_SIDECAR_LEASE_LOG:-"${ROOT_DIR}/sidecar_lease_$(date +%Y%m%d%H%M%S).log"}
EXPECTED_ACTIVE_RANKS=${VERL_SIDECAR_EXPECTED_ACTIVE_RANKS:-8}
WORLD_SIZE=${VERL_SIDECAR_WORLD_SIZE:-16}
POLL_INTERVAL=${VERL_SIDECAR_WATCH_POLL_INTERVAL:-1}
START_ONCE=${VERL_SIDECAR_START_ONCE:-1}
GRACEFUL_KILL_SECONDS=${VERL_SIDECAR_GRACEFUL_KILL_SECONDS:-3}
MAX_SHRINK_TOTAL_MS=${VERL_SIDECAR_MAX_SHRINK_TOTAL_MS:-0}
START_TRIGGER=${VERL_SIDECAR_START_TRIGGER:-shrink_done}
REQUIRE_ACTIVE_LEASE=${VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE:-0}
REQUIRE_SHRINK_QUORUM=${VERL_SIDECAR_REQUIRE_SHRINK_QUORUM:-0}
SHRINK_QUORUM_SIZE=${VERL_SIDECAR_SHRINK_QUORUM_SIZE:-${WORLD_SIZE}}
MANUAL_NPU_DEVICES=${VERL_SIDECAR_NPU_DEVICES:-}
if [[ "${REQUIRE_SHRINK_QUORUM}" == "1" ]]; then
    if [[ "${START_TRIGGER}" != "shrink_done" ]]; then
        echo "VERL_SIDECAR_REQUIRE_SHRINK_QUORUM=1 requires START_TRIGGER=shrink_done" >&2
        exit 2
    fi
    if [[ ! "${SHRINK_QUORUM_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
       (( SHRINK_QUORUM_SIZE > WORLD_SIZE )); then
        echo "invalid sidecar shrink quorum size: ${SHRINK_QUORUM_SIZE} world_size=${WORLD_SIZE}" >&2
        exit 2
    fi
fi
SIDECAR_STOP_FILE=${VERL_SIDECAR_STOP_FILE:-}
if [[ -z "${SIDECAR_STOP_FILE}" ]]; then
    if [[ -n "${VERL_SIDECAR_OUTPUT_FILE:-}" ]]; then
        SIDECAR_STOP_FILE="${VERL_SIDECAR_OUTPUT_FILE}.stop_requested"
    else
        SIDECAR_STOP_FILE="${LEASE_LOG}.stop_requested"
    fi
fi
export VERL_SIDECAR_STOP_FILE="${SIDECAR_STOP_FILE}"

RESTORE_HANDSHAKE_DIR=${VERL_SIDECAR_RESTORE_HANDSHAKE_DIR:-}
if [[ -z "${RESTORE_HANDSHAKE_DIR}" ]]; then
    if [[ -n "${VERL_SIDECAR_LOG_DIR:-}" ]]; then
        RESTORE_HANDSHAKE_DIR="${VERL_SIDECAR_LOG_DIR}/restore_handshake"
    else
        RESTORE_HANDSHAKE_DIR="$(dirname "${LEASE_LOG}")/restore_handshake"
    fi
fi
ACTIVE_LEASE_FILE=${VERL_SIDECAR_ACTIVE_LEASE_FILE:-"${RESTORE_HANDSHAKE_DIR}/active_lease"}
STOP_REQUEST_FILE=${VERL_SIDECAR_STOP_REQUEST_FILE:-"${RESTORE_HANDSHAKE_DIR}/stop_request"}
STOP_ACK_FILE=${VERL_SIDECAR_STOP_ACK_FILE:-"${RESTORE_HANDSHAKE_DIR}/stop_ack"}

mkdir -p "$(dirname "${LEASE_LOG}")"
mkdir -p "$(dirname "${VERL_SIDECAR_STOP_FILE}")"
mkdir -p "${RESTORE_HANDSHAKE_DIR}"
mkdir -p "$(dirname "${ACTIVE_LEASE_FILE}")"
mkdir -p "$(dirname "${STOP_REQUEST_FILE}")"
mkdir -p "$(dirname "${STOP_ACK_FILE}")"
rm -f "${ACTIVE_LEASE_FILE}" "${STOP_REQUEST_FILE}" "${STOP_ACK_FILE}"

echo "watch_start_time=$(date +%s.%N)" | tee -a "${LEASE_LOG}"
echo "watch_train_log=${TRAIN_LOG}" | tee -a "${LEASE_LOG}"
echo "watch_sidecar_script=${SIDECAR_SCRIPT}" | tee -a "${LEASE_LOG}"
echo "watch_expected_active_ranks=${EXPECTED_ACTIVE_RANKS}" | tee -a "${LEASE_LOG}"
echo "watch_world_size=${WORLD_SIZE}" | tee -a "${LEASE_LOG}"
echo "watch_start_trigger=${START_TRIGGER}" | tee -a "${LEASE_LOG}"
echo "watch_require_active_lease=${REQUIRE_ACTIVE_LEASE}" | tee -a "${LEASE_LOG}"
echo "watch_require_shrink_quorum=${REQUIRE_SHRINK_QUORUM}" | tee -a "${LEASE_LOG}"
echo "watch_shrink_quorum_size=${SHRINK_QUORUM_SIZE}" | tee -a "${LEASE_LOG}"
echo "watch_graceful_kill_seconds=${GRACEFUL_KILL_SECONDS}" | tee -a "${LEASE_LOG}"
echo "watch_max_shrink_total_ms=${MAX_SHRINK_TOTAL_MS}" | tee -a "${LEASE_LOG}"
echo "watch_sidecar_stop_file=${VERL_SIDECAR_STOP_FILE}" | tee -a "${LEASE_LOG}"
echo "watch_active_lease_file=${ACTIVE_LEASE_FILE}" | tee -a "${LEASE_LOG}"
echo "watch_stop_request_file=${STOP_REQUEST_FILE}" | tee -a "${LEASE_LOG}"
echo "watch_stop_ack_file=${STOP_ACK_FILE}" | tee -a "${LEASE_LOG}"

sidecar_pid=""
sidecar_started=0
sidecar_done=0
sidecar_lease_count=0
stop_acknowledged=0
last_stop_request_id=""
line_offset=0
train_log_missing_logged=0
declare -A shrink_quorum_seen=()
declare -A shrink_quorum_counts=()

reset_shrink_quorum() {
    shrink_quorum_seen=()
    shrink_quorum_counts=()
}

write_active_lease() {
    local state=$1
    local pid=${2:-}
    local devices=${3:-}
    local temporary
    temporary=$(mktemp "${ACTIVE_LEASE_FILE}.tmp.XXXXXX")
    {
        printf 'lease_id=%s\n' "${sidecar_lease_count}"
        printf 'state=%s\n' "${state}"
        printf 'pid=%s\n' "${pid}"
        printf 'devices=%s\n' "${devices}"
        printf 'update_time=%s\n' "$(date +%s.%N)"
    } > "${temporary}"
    mv -f "${temporary}" "${ACTIVE_LEASE_FILE}"
}

write_stop_ack() {
    local request_id=$1
    local lease_id=$2
    local status=$3
    local reason=$4
    local request_time=${5:-}
    local ack_time=${6:-$(date +%s.%N)}
    local temporary
    temporary=$(mktemp "${STOP_ACK_FILE}.tmp.XXXXXX")
    {
        printf 'request_id=%s\n' "${request_id}"
        printf 'lease_id=%s\n' "${lease_id}"
        printf 'status=%s\n' "${status}"
        printf 'reason=%s\n' "${reason}"
        printf 'request_time=%s\n' "${request_time}"
        printf 'ack_time=%s\n' "${ack_time}"
    } > "${temporary}"
    mv -f "${temporary}" "${STOP_ACK_FILE}"
}

read_handshake_field() {
    local path=$1
    local key=$2
    sed -n "s/^${key}=//p" "${path}" 2>/dev/null | head -1
}

sidecar_process_alive() {
    [[ -n "${sidecar_pid}" ]] || return 1
    ps -eo pgid=,stat= 2>/dev/null | awk -v pgid="${sidecar_pid}" '
        $1 == pgid && $2 !~ /^Z/ { found = 1 }
        END { exit(found ? 0 : 1) }
    ' && return 0
    local leader_state
    leader_state=$(ps -o stat= -p "${sidecar_pid}" 2>/dev/null | awk 'NR == 1 { print $1 }' || true)
    [[ -n "${leader_state}" && "${leader_state}" != Z* ]]
}

write_active_lease "armed"

merge_sidecar_outputs() {
    local output_file=${VERL_SIDECAR_OUTPUT_FILE:-}
    [[ -n "${output_file}" ]] || return 0
    local output_dir
    output_dir=$(dirname "${output_file}")
    shopt -s nullglob
    local shard_outputs=("${output_file}".shard*)
    shopt -u nullglob
    if (( ${#shard_outputs[@]} == 0 )); then
        return 0
    fi
    mkdir -p "${output_dir}"
    : > "${output_file}"
    for shard_output in "${shard_outputs[@]}"; do
        if [[ -f "${shard_output}" ]]; then
            cat "${shard_output}" >> "${output_file}"
        fi
    done
    echo "sidecar_output_merge_time=$(date +%s.%N) output=${output_file} shards=${#shard_outputs[@]}" | tee -a "${LEASE_LOG}"
}

sync_sidecar_artifacts() {
    local output_file=${VERL_SIDECAR_OUTPUT_FILE:-}
    local state_dir=${VERL_SIDECAR_STATE_DIR:-}
    python3 - "${output_file}" "${state_dir}" <<'PY'
import os
import sys
from pathlib import Path

paths = []
output_arg, state_arg = sys.argv[1:]
if output_arg:
    output = Path(output_arg)
    if output.is_file():
        paths.append(output)
if state_arg:
    state = Path(state_arg)
    if state.is_dir():
        paths.extend(path for path in state.rglob("*") if path.is_file())

for path in paths:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

directories = {path.parent for path in paths}
if state_arg and Path(state_arg).is_dir():
    directories.add(Path(state_arg))
for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY
    echo "sidecar_artifacts_durable_time=$(date +%s.%N) output=${output_file:-none} state_dir=${state_dir:-none}" | tee -a "${LEASE_LOG}"
}

stop_sidecar() {
    local reason=${1:-unknown}
    if sidecar_process_alive; then
        echo "sidecar_stop_request_time=$(date +%s.%N) reason=${reason} pid=${sidecar_pid} stop_file=${VERL_SIDECAR_STOP_FILE}" | tee -a "${LEASE_LOG}"
        printf "time=%s\nreason=%s\npid=%s\n" "$(date +%s.%N)" "${reason}" "${sidecar_pid}" > "${VERL_SIDECAR_STOP_FILE}"
        local deadline=$((SECONDS + GRACEFUL_KILL_SECONDS))
        while sidecar_process_alive && (( SECONDS < deadline )); do
            sleep 0.2
        done
        if sidecar_process_alive; then
            echo "sidecar_kill_time=$(date +%s.%N) reason=${reason} pid=${sidecar_pid} after_soft_stop=1" | tee -a "${LEASE_LOG}"
            kill -- -"${sidecar_pid}" 2>/dev/null || kill "${sidecar_pid}" 2>/dev/null || true
            local term_deadline=$((SECONDS + 2))
            while sidecar_process_alive && (( SECONDS < term_deadline )); do
                sleep 0.2
            done
        fi
        if sidecar_process_alive; then
            echo "sidecar_force_kill_time=$(date +%s.%N) reason=${reason} pid=${sidecar_pid}" | tee -a "${LEASE_LOG}"
            kill -9 -- -"${sidecar_pid}" 2>/dev/null || kill -9 "${sidecar_pid}" 2>/dev/null || true
            local force_deadline=$((SECONDS + 2))
            while sidecar_process_alive && (( SECONDS < force_deadline )); do
                sleep 0.1
            done
        fi
    fi
    if [[ -n "${sidecar_pid}" ]] && ! sidecar_process_alive; then
        wait "${sidecar_pid}" 2>/dev/null || true
    fi
    if sidecar_process_alive; then
        echo "sidecar_release_failed_time=$(date +%s.%N) reason=${reason} pid=${sidecar_pid}" | tee -a "${LEASE_LOG}"
        return 1
    fi
    merge_sidecar_outputs
    if ! sync_sidecar_artifacts; then
        echo "sidecar_artifacts_durable_failed_time=$(date +%s.%N) reason=${reason}" | tee -a "${LEASE_LOG}"
        return 1
    fi
    echo "sidecar_release_confirmed_time=$(date +%s.%N) reason=${reason} pid=${sidecar_pid:-none} process_group_alive=0" | tee -a "${LEASE_LOG}"
}

cleanup() {
    stop_sidecar "watcher_exit" || true
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

count_active_ranks() {
    local line=$1
    local ranks
    ranks=$(sed -n 's/.*active_ranks=\[\([^]]*\)\].*/\1/p' <<< "${line}" | head -1)
    if [[ -z "${ranks// /}" ]]; then
        echo 0
        return
    fi
    awk -F',' '{print NF}' <<< "${ranks}"
}

extract_active_ranks_csv() {
    local line=$1
    sed -n 's/.*active_ranks=\[\([^]]*\)\].*/\1/p' <<< "${line}" \
        | head -1 \
        | tr -d ' '
}

extract_shrink_total_ms() {
    local line=$1
    sed -n 's/.*total_ms=\([0-9.][0-9.]*\).*/\1/p' <<< "${line}" | head -1
}

shrink_total_allowed() {
    local total_ms=$1
    python3 - "$MAX_SHRINK_TOTAL_MS" "$total_ms" <<'PY'
import sys

limit = float(sys.argv[1] or 0)
value_arg = sys.argv[2].strip()
if limit <= 0 or not value_arg:
    raise SystemExit(0)
value = float(value_arg)
raise SystemExit(0 if value <= limit else 1)
PY
}

derive_inactive_devices_csv() {
    local active_csv=$1
    python3 - "$WORLD_SIZE" "$active_csv" <<'PY'
import sys

world_size = int(sys.argv[1])
active_csv = sys.argv[2].strip()
active = set()
if active_csv:
    active = {int(x) for x in active_csv.split(",") if x != ""}
inactive = [str(rank) for rank in range(world_size) if rank not in active]
print(",".join(inactive))
PY
}

rearm_sidecar() {
    sidecar_pid=""
    sidecar_started=0
    sidecar_done=0
    if [[ -z "${MANUAL_NPU_DEVICES}" ]]; then
        unset VERL_SIDECAR_NPU_DEVICES
    fi
    stop_acknowledged=0
    reset_shrink_quorum
    write_active_lease "armed"
    echo "sidecar_rearm_time=$(date +%s.%N) completed_leases=${sidecar_lease_count}" | tee -a "${LEASE_LOG}"
}

start_sidecar() {
    local active_csv=$1
    local devices_csv
    if [[ -n "${MANUAL_NPU_DEVICES}" ]]; then
        devices_csv="${MANUAL_NPU_DEVICES}"
        echo "sidecar_devices_source=manual" | tee -a "${LEASE_LOG}"
    else
        devices_csv=$(derive_inactive_devices_csv "${active_csv}")
        if [[ -z "${devices_csv}" ]]; then
            echo "sidecar_skip_reason=no_inactive_devices active_ranks=${active_csv}" | tee -a "${LEASE_LOG}"
            return
        fi
        export VERL_SIDECAR_NPU_DEVICES="${devices_csv}"
        echo "sidecar_devices_source=auto_from_inactive_ranks" | tee -a "${LEASE_LOG}"
    fi
    rm -f "${VERL_SIDECAR_STOP_FILE}"
    sidecar_lease_count=$((sidecar_lease_count + 1))
    sidecar_started=1
    sidecar_done=0
    echo "sidecar_start_time=$(date +%s.%N)" | tee -a "${LEASE_LOG}"
    echo "sidecar_lease_index=${sidecar_lease_count}" | tee -a "${LEASE_LOG}"
    echo "sidecar_active_ranks=${active_csv}" | tee -a "${LEASE_LOG}"
    echo "sidecar_devices=${devices_csv}" | tee -a "${LEASE_LOG}"
    setsid "${SIDECAR_SCRIPT}" &
    sidecar_pid=$!
    write_active_lease "running" "${sidecar_pid}" "${devices_csv}"
    echo "sidecar_pid=${sidecar_pid}" | tee -a "${LEASE_LOG}"
}

process_stop_request() {
    [[ -f "${STOP_REQUEST_FILE}" ]] || return 1
    local request_id
    local requested_lease
    local trainer_request_time
    request_id=$(read_handshake_field "${STOP_REQUEST_FILE}" request_id)
    requested_lease=$(read_handshake_field "${STOP_REQUEST_FILE}" lease_id)
    trainer_request_time=$(read_handshake_field "${STOP_REQUEST_FILE}" request_time)
    [[ -n "${request_id}" && -n "${requested_lease}" && \
       -n "${trainer_request_time}" ]] || return 1
    [[ "${request_id}" != "${last_stop_request_id}" ]] || return 1
    last_stop_request_id=${request_id}

    echo "sidecar_handshake_request_time=$(date +%s.%N) request_id=${request_id} requested_lease=${requested_lease} current_lease=${sidecar_lease_count}" | tee -a "${LEASE_LOG}"
    echo "trainer_stop_request_time=${trainer_request_time} watcher_observed_time=$(date +%s.%N) request_id=${request_id} lease_id=${requested_lease}" | tee -a "${LEASE_LOG}"
    if [[ "${REQUIRE_ACTIVE_LEASE}" == "1" && \
          ( "${requested_lease}" == "0" || "${sidecar_started}" != "1" || \
            "${sidecar_done}" == "1" ) ]]; then
        write_stop_ack "${request_id}" "${requested_lease}" "error" \
            "active_running_lease_required" "${trainer_request_time}"
        echo "sidecar_handshake_error_time=$(date +%s.%N) request_id=${request_id} reason=active_running_lease_required" | tee -a "${LEASE_LOG}"
        return 0
    fi
    if [[ "${requested_lease}" != "${sidecar_lease_count}" ]]; then
        write_stop_ack "${request_id}" "${requested_lease}" "error" \
            "lease_mismatch_current_${sidecar_lease_count}" "${trainer_request_time}"
        echo "sidecar_handshake_error_time=$(date +%s.%N) request_id=${request_id} reason=lease_mismatch" | tee -a "${LEASE_LOG}"
        return 0
    fi

    local release_ok=1
    if ! stop_sidecar "pre_restore_handshake"; then
        release_ok=0
    fi
    sidecar_done=1
    if [[ "${release_ok}" == "1" ]] && ! sidecar_process_alive; then
        local exit_confirmed_time
        local restore_ack_time
        exit_confirmed_time=$(date +%s.%N)
        echo "sidecar_exit_confirmed_time=${exit_confirmed_time} reason=pre_restore_handshake pid=${sidecar_pid:-none} process_group_alive=0 request_id=${request_id} lease_id=${requested_lease}" | tee -a "${LEASE_LOG}"
        write_active_lease "released" "${sidecar_pid:-}" "${VERL_SIDECAR_NPU_DEVICES:-}"
        restore_ack_time=$(date +%s.%N)
        echo "sidecar_handshake_ack_time=${restore_ack_time} request_id=${request_id} lease_id=${requested_lease} status=released" | tee -a "${LEASE_LOG}"
        echo "watcher_restore_ack_time=${restore_ack_time} request_id=${request_id} lease_id=${requested_lease} status=released" | tee -a "${LEASE_LOG}"
        write_stop_ack "${request_id}" "${requested_lease}" "released" \
            "sidecar_process_group_exited" "${trainer_request_time}" \
            "${restore_ack_time}"
        stop_acknowledged=1
        if [[ "${START_ONCE}" == "0" ]]; then
            if [[ -f "${TRAIN_LOG}" ]]; then
                line_offset=$(wc -l < "${TRAIN_LOG}" || echo "${line_offset}")
            fi
            rearm_sidecar
        fi
    else
        write_stop_ack "${request_id}" "${requested_lease}" "error" \
            "sidecar_process_group_still_alive" "${trainer_request_time}"
        echo "sidecar_handshake_error_time=$(date +%s.%N) request_id=${request_id} reason=process_group_still_alive" | tee -a "${LEASE_LOG}"
    fi
    return 0
}

while [[ ! -f "${TRAIN_LOG}" ]]; do
    sleep "${POLL_INTERVAL}"
done

is_sidecar_start_line() {
    local line=$1
    case "${START_TRIGGER}" in
        shrink_done)
            [[ "${line}" == *"Elastic parallel shrink done"* ]]
            ;;
        detach_or_shrink)
            [[ "${line}" == *"Elastic parallel detach done"* ]] || \
                [[ "${line}" == *"Elastic parallel shrink done"* ]]
            ;;
        *)
            echo "unsupported_start_trigger=${START_TRIGGER}" | tee -a "${LEASE_LOG}"
            return 1
            ;;
    esac
}

is_shrink_quorum_line() {
    local line=$1
    [[ "${line}" == *"Elastic parallel shrink done:"* ]] || \
        [[ "${line}" == *"Elastic parallel shrink rpc done:"* ]]
}

extract_shrink_reporter_rank() {
    local line=$1
    local reporter
    reporter=$(sed -n 's/.*Elastic parallel shrink rpc done: global_rank=\([0-9][0-9]*\).*/\1/p' <<< "${line}" | head -1)
    if [[ -z "${reporter}" ]]; then
        reporter=$(sed -n 's/.*Elastic parallel shrink done: rank=\([0-9][0-9]*\).*/\1/p' <<< "${line}" | head -1)
    fi
    printf '%s\n' "${reporter}"
}

quorum_ranks_csv() {
    local stage_key=$1
    local rank
    local ranks=()
    for ((rank = 0; rank < WORLD_SIZE; rank++)); do
        if [[ -n "${shrink_quorum_seen[${stage_key}|${rank}]:-}" ]]; then
            ranks+=("${rank}")
        fi
    done
    local joined
    joined=$(IFS=,; echo "${ranks[*]}")
    printf '%s\n' "${joined}"
}

observe_shrink_quorum() {
    local line=$1
    local active_csv
    local active_count
    local reporter
    local total_ms
    local seen_key
    local count
    active_csv=$(extract_active_ranks_csv "${line}")
    active_count=$(count_active_ranks "${line}")
    [[ "${active_count}" == "${EXPECTED_ACTIVE_RANKS}" ]] || return 0
    reporter=$(extract_shrink_reporter_rank "${line}")
    if [[ -z "${reporter}" || "${reporter}" -lt 0 || \
          "${reporter}" -ge "${WORLD_SIZE}" ]]; then
        echo "shrink_quorum_reject_time=$(date +%s.%N) reason=invalid_reporter active_ranks=${active_csv} reporter=${reporter:-missing}" | tee -a "${LEASE_LOG}"
        return 0
    fi
    total_ms=$(extract_shrink_total_ms "${line}")
    if ! shrink_total_allowed "${total_ms}"; then
        echo "sidecar_skip_reason=shrink_total_too_slow shrink_total_ms=${total_ms} max_shrink_total_ms=${MAX_SHRINK_TOTAL_MS}" | tee -a "${LEASE_LOG}"
        return 0
    fi

    seen_key="${active_csv}|${reporter}"
    if [[ -n "${shrink_quorum_seen[${seen_key}]:-}" ]]; then
        echo "shrink_quorum_duplicate_time=$(date +%s.%N) active_ranks=${active_csv} reporter=${reporter} quorum_count=${shrink_quorum_counts[${active_csv}]:-0} quorum_required=${SHRINK_QUORUM_SIZE}" | tee -a "${LEASE_LOG}"
        return 0
    fi
    shrink_quorum_seen["${seen_key}"]=1
    count=$(( ${shrink_quorum_counts[${active_csv}]:-0} + 1 ))
    shrink_quorum_counts["${active_csv}"]=${count}
    echo "shrink_quorum_progress_time=$(date +%s.%N) active_ranks=${active_csv} reporter=${reporter} quorum_count=${count} quorum_required=${SHRINK_QUORUM_SIZE}" | tee -a "${LEASE_LOG}"
    if (( count < SHRINK_QUORUM_SIZE )); then
        return 0
    fi

    local ranks_csv
    local coordinated_time
    ranks_csv=$(quorum_ranks_csv "${active_csv}")
    coordinated_time=$(date +%s.%N)
    echo "quorum_ranks=${ranks_csv} active_ranks=${active_csv}" | tee -a "${LEASE_LOG}"
    echo "quorum_count=${count} quorum_required=${SHRINK_QUORUM_SIZE} active_ranks=${active_csv}" | tee -a "${LEASE_LOG}"
    echo "coordinated_start_time=${coordinated_time} active_ranks=${active_csv} quorum_ranks=${ranks_csv} quorum_count=${count}" | tee -a "${LEASE_LOG}"
    echo "shrink_window_detected_time=${coordinated_time} active_count=${active_count} coordinated=1" | tee -a "${LEASE_LOG}"
    echo "shrink_window_line=${line}" | tee -a "${LEASE_LOG}"
    start_sidecar "${active_csv}"
}

while true; do
    process_stop_request || true
    if [[ "${stop_acknowledged}" == "1" && "${START_ONCE}" == "1" ]]; then
        echo "watch_end_time=$(date +%s.%N)" | tee -a "${LEASE_LOG}"
        exit 0
    fi
    if [[ ! -f "${TRAIN_LOG}" ]]; then
        if [[ "${train_log_missing_logged}" == "0" ]]; then
            echo "train_log_missing_time=$(date +%s.%N) path=${TRAIN_LOG}" | tee -a "${LEASE_LOG}"
            train_log_missing_logged=1
        fi
        sleep "${POLL_INTERVAL}"
        continue
    fi
    train_log_missing_logged=0
    total_lines=$(wc -l < "${TRAIN_LOG}" || echo 0)
    if (( total_lines > line_offset )); then
        while IFS= read -r line; do
            if [[ "${sidecar_started}" == "0" ]]; then
                if [[ "${REQUIRE_SHRINK_QUORUM}" == "1" ]]; then
                    if is_shrink_quorum_line "${line}"; then
                        observe_shrink_quorum "${line}"
                    elif is_sidecar_start_line "${line}"; then
                        echo "shrink_quorum_reject_time=$(date +%s.%N) reason=unrecognized_done_format line=${line}" | tee -a "${LEASE_LOG}"
                    fi
                elif is_sidecar_start_line "${line}"; then
                    active_count=$(count_active_ranks "${line}")
                    if [[ "${active_count}" == "${EXPECTED_ACTIVE_RANKS}" ]]; then
                        shrink_total_ms=$(extract_shrink_total_ms "${line}")
                        if ! shrink_total_allowed "${shrink_total_ms}"; then
                            echo "sidecar_skip_reason=shrink_total_too_slow shrink_total_ms=${shrink_total_ms} max_shrink_total_ms=${MAX_SHRINK_TOTAL_MS}" | tee -a "${LEASE_LOG}"
                            sidecar_done=1
                            continue
                        fi
                        active_csv=$(extract_active_ranks_csv "${line}")
                        echo "shrink_window_detected_time=$(date +%s.%N) active_count=${active_count} coordinated=0" | tee -a "${LEASE_LOG}"
                        echo "shrink_window_line=${line}" | tee -a "${LEASE_LOG}"
                        start_sidecar "${active_csv}"
                    fi
                fi
            fi

            if [[ "${sidecar_started}" == "1" ]] && \
               { [[ "${line}" == *"Elastic parallel restore requested before rollout restore rpc"* ]] || \
                 [[ "${line}" == *"Elastic parallel restore requested after rollout"* ]] || \
                 [[ "${line}" == *"Elastic parallel restore"* ]] || \
                 [[ "${line}" == *"rollout_output_time_s"* ]]; }; then
                if [[ "${REQUIRE_ACTIVE_LEASE}" == "1" ]]; then
                    echo "sidecar_deadline_signal_deferred_time=$(date +%s.%N) reason=trainer_stop_request_is_authoritative" | tee -a "${LEASE_LOG}"
                    echo "sidecar_deadline_signal_deferred_line=${line}" | tee -a "${LEASE_LOG}"
                else
                    echo "sidecar_deadline_signal_time=$(date +%s.%N)" | tee -a "${LEASE_LOG}"
                    echo "sidecar_deadline_signal_line=${line}" | tee -a "${LEASE_LOG}"
                    sidecar_done=1
                    if stop_sidecar "training_rollout_done_or_restore"; then
                        write_active_lease "released" "${sidecar_pid:-}" "${VERL_SIDECAR_NPU_DEVICES:-}"
                    else
                        write_active_lease "stopping" "${sidecar_pid:-}" "${VERL_SIDECAR_NPU_DEVICES:-}"
                    fi
                fi
            fi
        done < <(tail -n +$((line_offset + 1)) "${TRAIN_LOG}")
        line_offset=${total_lines}
    fi

    if [[ "${sidecar_started}" == "1" && "${sidecar_done}" == "0" && -n "${sidecar_pid}" ]]; then
        if ! sidecar_process_alive; then
            set +e
            wait "${sidecar_pid}" 2>/dev/null
            rc=$?
            set -e
            echo "sidecar_exit_time=$(date +%s.%N) sidecar_exit_code=${rc}" | tee -a "${LEASE_LOG}"
            sidecar_done=1
            merge_sidecar_outputs
            write_active_lease "released" "${sidecar_pid}" "${VERL_SIDECAR_NPU_DEVICES:-}"
        fi
    fi

    if [[ "${sidecar_done}" == "1" && "${START_ONCE}" == "1" && \
          "${stop_acknowledged}" == "1" ]]; then
        echo "watch_end_time=$(date +%s.%N)" | tee -a "${LEASE_LOG}"
        exit 0
    fi

    sleep "${POLL_INTERVAL}"
done
