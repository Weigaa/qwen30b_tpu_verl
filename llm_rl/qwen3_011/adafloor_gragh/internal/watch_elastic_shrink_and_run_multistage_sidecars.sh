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
LOG_DIR=${VERL_SIDECAR_LOG_DIR:-"${ROOT_DIR}/multistage_sidecar_$(date +%Y%m%d%H%M%S)"}
LEASE_LOG=${VERL_SIDECAR_LEASE_LOG:-"${LOG_DIR}/lease.log"}
STATE_DIR=${VERL_SIDECAR_STATE_DIR:-"${LOG_DIR}/state"}
WORLD_SIZE=${VERL_SIDECAR_WORLD_SIZE:-16}
TARGET_FLOORS=${VERL_SIDECAR_TARGET_FLOORS:-8,4,2}
POLL_INTERVAL=${VERL_SIDECAR_WATCH_POLL_INTERVAL:-1}
GRACEFUL_KILL_SECONDS=${VERL_SIDECAR_GRACEFUL_KILL_SECONDS:-5}
MAX_SHRINK_TOTAL_MS=${VERL_SIDECAR_MAX_SHRINK_TOTAL_MS:-0}
MASTER_PORT_BASE=${VERL_SIDECAR_MULTI_STAGE_MASTER_PORT_BASE:-24300}
MASTER_PORT_WINDOW_STRIDE=${VERL_SIDECAR_MULTI_STAGE_MASTER_PORT_WINDOW_STRIDE:-1024}
MASTER_PORT_STAGE_STRIDE=${VERL_SIDECAR_MULTI_STAGE_MASTER_PORT_STAGE_STRIDE:-256}

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

log_event() {
    echo "$*" | tee -a "${LEASE_LOG}"
}

full_active_csv() {
    python3 - "${WORLD_SIZE}" <<'PY'
import sys
print(",".join(map(str, range(int(sys.argv[1])))))
PY
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

csv_count() {
    local csv=$1
    if [[ -z "${csv}" ]]; then
        echo 0
    else
        awk -F',' '{print NF}' <<< "${csv}"
    fi
}

is_target_floor() {
    local active_count=$1
    [[ ",${TARGET_FLOORS}," == *",${active_count},"* ]]
}

shrink_total_allowed() {
    local total_ms=$1
    python3 - "${MAX_SHRINK_TOTAL_MS}" "${total_ms}" <<'PY'
import sys

limit = float(sys.argv[1] or 0)
value_arg = sys.argv[2].strip()
if limit <= 0 or not value_arg:
    raise SystemExit(0)
raise SystemExit(0 if float(value_arg) <= limit else 1)
PY
}

released_ranks_csv() {
    local previous_csv=$1
    local current_csv=$2
    python3 - "${previous_csv}" "${current_csv}" <<'PY'
import sys

previous = [int(item) for item in sys.argv[1].split(",") if item]
current = {int(item) for item in sys.argv[2].split(",") if item}
if not current.issubset(set(previous)):
    raise SystemExit(2)
print(",".join(str(rank) for rank in previous if rank not in current))
PY
}

is_shrink_line() {
    local line=$1
    [[ "${line}" == *"Elastic parallel shrink done"* ]] || \
        [[ "${line}" == *"Elastic parallel shrink rpc done"* ]]
}

is_restore_line() {
    local line=$1
    [[ "${line}" == *"Elastic parallel restore requested before rollout restore rpc"* ]] || \
        [[ "${line}" == *"Elastic parallel restore requested after rollout"* ]] || \
        [[ "${line}" == *"Elastic parallel restore"* ]] || \
        [[ "${line}" == *"rollout_output_time_s"* ]]
}

declare -a SIDECAR_PIDS=()
declare -a SIDECAR_STOP_FILES=()
declare -a SIDECAR_LABELS=()
declare -a SIDECAR_DEVICES=()
declare -a SIDECAR_EXIT_LOGGED=()

stop_all_sidecars() {
    local reason=${1:-unknown}
    local index pid stop_file
    if (( ${#SIDECAR_PIDS[@]} == 0 )); then
        return 0
    fi

    log_event "multistage_stop_all_request_time=$(date +%s.%N) reason=${reason} count=${#SIDECAR_PIDS[@]}"
    for index in "${!SIDECAR_PIDS[@]}"; do
        pid=${SIDECAR_PIDS[$index]}
        stop_file=${SIDECAR_STOP_FILES[$index]}
        printf "time=%s\nreason=%s\npid=%s\nstage=%s\n" \
            "$(date +%s.%N)" "${reason}" "${pid}" "${SIDECAR_LABELS[$index]}" \
            > "${stop_file}"
    done

    local deadline=$((SECONDS + GRACEFUL_KILL_SECONDS))
    while (( SECONDS < deadline )); do
        local alive=0
        for pid in "${SIDECAR_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                alive=1
                break
            fi
        done
        (( alive == 0 )) && break
        sleep 0.2
    done

    for index in "${!SIDECAR_PIDS[@]}"; do
        pid=${SIDECAR_PIDS[$index]}
        if kill -0 "${pid}" 2>/dev/null; then
            log_event "multistage_sidecar_kill_time=$(date +%s.%N) reason=${reason} stage=${SIDECAR_LABELS[$index]} pid=${pid}"
            kill -- -"${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
        fi
    done
    sleep 0.2
    for index in "${!SIDECAR_PIDS[@]}"; do
        pid=${SIDECAR_PIDS[$index]}
        if kill -0 "${pid}" 2>/dev/null; then
            log_event "multistage_sidecar_force_kill_time=$(date +%s.%N) reason=${reason} stage=${SIDECAR_LABELS[$index]} pid=${pid}"
            kill -9 -- -"${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
        fi
        wait "${pid}" 2>/dev/null || true
    done

    SIDECAR_PIDS=()
    SIDECAR_STOP_FILES=()
    SIDECAR_LABELS=()
    SIDECAR_DEVICES=()
    SIDECAR_EXIT_LOGGED=()
    log_event "multistage_stop_all_done_time=$(date +%s.%N) reason=${reason}"
}

CURRENT_FULL_ACTIVE=$(full_active_csv)
previous_active_csv=${CURRENT_FULL_ACTIVE}
window_index=0
stage_index=0
line_offset=0
train_log_missing_logged=0

start_stage_sidecar() {
    local previous_count=$1
    local active_count=$2
    local devices_csv=$3
    local replica_count
    replica_count=$(csv_count "${devices_csv}")
    (( replica_count > 0 )) || return 0

    if (( stage_index == 0 )); then
        window_index=$((window_index + 1))
    fi
    stage_index=$((stage_index + 1))
    local stage_label
    stage_label=$(printf "window_%03d_stage_%02d_%s_to_%s" \
        "${window_index}" "${stage_index}" "${previous_count}" "${active_count}")
    local stage_dir=${LOG_DIR}/${stage_label}
    local output_file=${stage_dir}/outputs.jsonl
    local infer_log=${stage_dir}/infer.log
    local launcher_log=${stage_dir}/launcher.log
    local stop_file=${stage_dir}/stop_requested
    local master_port=$((MASTER_PORT_BASE + (window_index - 1) * MASTER_PORT_WINDOW_STRIDE + (stage_index - 1) * MASTER_PORT_STAGE_STRIDE))
    mkdir -p "${stage_dir}"
    rm -f "${stop_file}"

    (
        export VERL_SIDECAR_NPU_DEVICES="${devices_csv}"
        export VERL_SIDECAR_PARALLEL_MODE=dp
        export VERL_SIDECAR_TENSOR_PARALLEL_SIZE=1
        export VERL_SIDECAR_DATA_PARALLEL_SIZE=1
        export VERL_SIDECAR_REPLICA_COUNT="${replica_count}"
        export VERL_SIDECAR_ENABLE_EXPERT_PARALLEL=0
        export VERL_SIDECAR_GLOBAL_NUM_SHARDS="${WORLD_SIZE}"
        export VERL_SIDECAR_GLOBAL_SHARD_INDICES="${devices_csv}"
        export VERL_SIDECAR_MASTER_PORT="${master_port}"
        export VERL_SIDECAR_LOG_FILE="${infer_log}"
        export VERL_SIDECAR_OUTPUT_FILE="${output_file}"
        export VERL_SIDECAR_STOP_FILE="${stop_file}"
        export VERL_SIDECAR_STATE_DIR="${STATE_DIR}"
        unset VERL_SIDECAR_DEVICE_GROUPS VERL_SIDECAR_UNUSED_DEVICES
        unset VERL_SIDECAR_MASTER_PORTS VERL_SIDECAR_DATA_PARALLEL_RPC_PORTS
        unset VERL_SIDECAR_HCCL_IF_BASE_PORTS VERL_SIDECAR_MAX_PROMPTS
        exec setsid "${SIDECAR_SCRIPT}"
    ) >> "${launcher_log}" 2>&1 &
    local pid=$!

    SIDECAR_PIDS+=("${pid}")
    SIDECAR_STOP_FILES+=("${stop_file}")
    SIDECAR_LABELS+=("${stage_label}")
    SIDECAR_DEVICES+=("${devices_csv}")
    SIDECAR_EXIT_LOGGED+=(0)
    log_event "multistage_sidecar_start_time=$(date +%s.%N) stage=${stage_label} pid=${pid} released_devices=${devices_csv} replicas=${replica_count} global_shards=${devices_csv}/${WORLD_SIZE} master_port=${master_port}"
}

cleanup() {
    stop_all_sidecars "watcher_exit"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

log_event "multistage_watch_start_time=$(date +%s.%N)"
log_event "multistage_watch_train_log=${TRAIN_LOG}"
log_event "multistage_watch_sidecar_script=${SIDECAR_SCRIPT}"
log_event "multistage_watch_world_size=${WORLD_SIZE}"
log_event "multistage_watch_target_floors=${TARGET_FLOORS}"
log_event "multistage_watch_state_dir=${STATE_DIR}"

while [[ ! -f "${TRAIN_LOG}" ]]; do
    sleep "${POLL_INTERVAL}"
done

while true; do
    if [[ ! -f "${TRAIN_LOG}" ]]; then
        if [[ "${train_log_missing_logged}" == "0" ]]; then
            log_event "multistage_train_log_missing_time=$(date +%s.%N) path=${TRAIN_LOG}"
            train_log_missing_logged=1
        fi
        sleep "${POLL_INTERVAL}"
        continue
    fi
    train_log_missing_logged=0
    total_lines=$(wc -l < "${TRAIN_LOG}" || echo 0)
    if (( total_lines > line_offset )); then
        while IFS= read -r line; do
            if is_restore_line "${line}"; then
                if (( ${#SIDECAR_PIDS[@]} > 0 )); then
                    log_event "multistage_restore_signal_time=$(date +%s.%N) line=${line}"
                    stop_all_sidecars "training_rollout_done_or_restore"
                fi
                previous_active_csv=${CURRENT_FULL_ACTIVE}
                stage_index=0
                continue
            fi

            is_shrink_line "${line}" || continue
            active_csv=$(extract_active_ranks_csv "${line}")
            [[ -n "${active_csv}" ]] || continue
            active_count=$(csv_count "${active_csv}")
            is_target_floor "${active_count}" || continue

            shrink_total_ms=$(extract_shrink_total_ms "${line}")
            if ! shrink_total_allowed "${shrink_total_ms}"; then
                log_event "multistage_skip_reason=shrink_total_too_slow active_count=${active_count} shrink_total_ms=${shrink_total_ms} max_shrink_total_ms=${MAX_SHRINK_TOTAL_MS}"
                previous_active_csv=${active_csv}
                continue
            fi

            previous_count=$(csv_count "${previous_active_csv}")
            set +e
            released_csv=$(released_ranks_csv "${previous_active_csv}" "${active_csv}")
            released_rc=$?
            set -e
            if [[ "${released_rc}" != "0" ]]; then
                log_event "multistage_skip_reason=non_monotonic_active_set previous=${previous_active_csv} current=${active_csv}"
                previous_active_csv=${active_csv}
                continue
            fi
            if [[ -z "${released_csv}" ]]; then
                continue
            fi

            log_event "multistage_shrink_detected_time=$(date +%s.%N) transition=${previous_count}_to_${active_count} previous_active=${previous_active_csv} active=${active_csv} newly_released=${released_csv}"
            start_stage_sidecar "${previous_count}" "${active_count}" "${released_csv}"
            previous_active_csv=${active_csv}
        done < <(tail -n +$((line_offset + 1)) "${TRAIN_LOG}")
        line_offset=${total_lines}
    fi

    for index in "${!SIDECAR_PIDS[@]}"; do
        pid=${SIDECAR_PIDS[$index]}
        if [[ "${SIDECAR_EXIT_LOGGED[$index]}" == "0" ]] && \
                ! kill -0 "${pid}" 2>/dev/null; then
            log_event "multistage_sidecar_not_running_time=$(date +%s.%N) stage=${SIDECAR_LABELS[$index]} pid=${pid} devices=${SIDECAR_DEVICES[$index]}"
            SIDECAR_EXIT_LOGGED[$index]=1
        fi
    done
    sleep "${POLL_INTERVAL}"
done
