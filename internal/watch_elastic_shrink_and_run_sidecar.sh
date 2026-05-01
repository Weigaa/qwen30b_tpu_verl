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

mkdir -p "$(dirname "${LEASE_LOG}")"

echo "watch_start_time=$(date +%s.%N)" | tee -a "${LEASE_LOG}"
echo "watch_train_log=${TRAIN_LOG}" | tee -a "${LEASE_LOG}"
echo "watch_sidecar_script=${SIDECAR_SCRIPT}" | tee -a "${LEASE_LOG}"
echo "watch_expected_active_ranks=${EXPECTED_ACTIVE_RANKS}" | tee -a "${LEASE_LOG}"
echo "watch_world_size=${WORLD_SIZE}" | tee -a "${LEASE_LOG}"
echo "watch_graceful_kill_seconds=${GRACEFUL_KILL_SECONDS}" | tee -a "${LEASE_LOG}"

sidecar_pid=""
sidecar_started=0
sidecar_done=0
line_offset=0
train_log_missing_logged=0

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

kill_sidecar() {
    local reason=${1:-unknown}
    if [[ -n "${sidecar_pid}" ]] && kill -0 "${sidecar_pid}" 2>/dev/null; then
        echo "sidecar_kill_time=$(date +%s.%N) reason=${reason} pid=${sidecar_pid}" | tee -a "${LEASE_LOG}"
        kill -- -"${sidecar_pid}" 2>/dev/null || kill "${sidecar_pid}" 2>/dev/null || true
        local deadline=$((SECONDS + GRACEFUL_KILL_SECONDS))
        while kill -0 "${sidecar_pid}" 2>/dev/null && (( SECONDS < deadline )); do
            sleep 0.2
        done
        if kill -0 "${sidecar_pid}" 2>/dev/null; then
            echo "sidecar_force_kill_time=$(date +%s.%N) reason=${reason} pid=${sidecar_pid}" | tee -a "${LEASE_LOG}"
            kill -9 -- -"${sidecar_pid}" 2>/dev/null || kill -9 "${sidecar_pid}" 2>/dev/null || true
        fi
        wait "${sidecar_pid}" 2>/dev/null || true
    fi
    merge_sidecar_outputs
}

cleanup() {
    kill_sidecar "watcher_exit"
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

start_sidecar() {
    local active_csv=$1
    local devices_csv
    if [[ -n "${VERL_SIDECAR_NPU_DEVICES:-}" ]]; then
        devices_csv="${VERL_SIDECAR_NPU_DEVICES}"
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
    sidecar_started=1
    echo "sidecar_start_time=$(date +%s.%N)" | tee -a "${LEASE_LOG}"
    echo "sidecar_active_ranks=${active_csv}" | tee -a "${LEASE_LOG}"
    echo "sidecar_devices=${devices_csv}" | tee -a "${LEASE_LOG}"
    setsid "${SIDECAR_SCRIPT}" &
    sidecar_pid=$!
    echo "sidecar_pid=${sidecar_pid}" | tee -a "${LEASE_LOG}"
}

while [[ ! -f "${TRAIN_LOG}" ]]; do
    sleep "${POLL_INTERVAL}"
done

while true; do
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
            if [[ "${sidecar_started}" == "0" ]] && \
               { [[ "${line}" == *"Elastic parallel detach done"* ]] || [[ "${line}" == *"Elastic parallel shrink done"* ]]; }; then
                active_count=$(count_active_ranks "${line}")
                if [[ "${active_count}" == "${EXPECTED_ACTIVE_RANKS}" ]]; then
                    active_csv=$(extract_active_ranks_csv "${line}")
                    echo "shrink_window_detected_time=$(date +%s.%N) active_count=${active_count}" | tee -a "${LEASE_LOG}"
                    echo "shrink_window_line=${line}" | tee -a "${LEASE_LOG}"
                    start_sidecar "${active_csv}"
                fi
            fi

            if [[ "${sidecar_started}" == "1" && "${sidecar_done}" == "0" ]] && \
               { [[ "${line}" == *"Elastic parallel restore requested after rollout"* ]] || \
                 [[ "${line}" == *"Elastic parallel restore"* ]] || \
                 [[ "${line}" == *"rollout_output_time_s"* ]]; }; then
                echo "sidecar_deadline_signal_time=$(date +%s.%N)" | tee -a "${LEASE_LOG}"
                echo "sidecar_deadline_signal_line=${line}" | tee -a "${LEASE_LOG}"
                kill_sidecar "training_rollout_done_or_restore"
                sidecar_done=1
            fi
        done < <(tail -n +$((line_offset + 1)) "${TRAIN_LOG}")
        line_offset=${total_lines}
    fi

    if [[ "${sidecar_started}" == "1" && "${sidecar_done}" == "0" && -n "${sidecar_pid}" ]]; then
        if ! kill -0 "${sidecar_pid}" 2>/dev/null; then
            set +e
            wait "${sidecar_pid}" 2>/dev/null
            rc=$?
            set -e
            echo "sidecar_exit_time=$(date +%s.%N) sidecar_exit_code=${rc}" | tee -a "${LEASE_LOG}"
            sidecar_done=1
        fi
    fi

    if [[ "${sidecar_done}" == "1" && "${START_ONCE}" == "1" ]]; then
        echo "watch_end_time=$(date +%s.%N)" | tee -a "${LEASE_LOG}"
        exit 0
    fi

    sleep "${POLL_INTERVAL}"
done
