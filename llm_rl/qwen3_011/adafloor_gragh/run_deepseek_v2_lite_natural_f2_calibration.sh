#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
WORKLOAD_PROFILE_PATH=${DEEPSEEK_WORKLOAD_PROFILE_PATH:-}
if [[ -n "$WORKLOAD_PROFILE_PATH" ]]; then
    if [[ ! -f "$WORKLOAD_PROFILE_PATH" ]]; then
        echo "missing DeepSeek workload profile: $WORKLOAD_PROFILE_PATH" >&2
        exit 2
    fi
    WORKLOAD_PROFILE_PATH=$(realpath "$WORKLOAD_PROFILE_PATH")
    # shellcheck disable=SC1090
    source "$WORKLOAD_PROFILE_PATH"
    workload_profile_sha256=$(sha256sum "$WORKLOAD_PROFILE_PATH")
    workload_profile_sha256=${workload_profile_sha256%% *}
    if [[ -z "${DEEPSEEK_WORKLOAD_PROFILE_ID:-}" ]]; then
        echo "DeepSeek workload profile has no DEEPSEEK_WORKLOAD_PROFILE_ID" >&2
        exit 2
    fi
    if [[ -n "${DEEPSEEK_WORKLOAD_PROFILE_SHA256:-}" \
          && "$DEEPSEEK_WORKLOAD_PROFILE_SHA256" != "$workload_profile_sha256" ]]; then
        echo "DeepSeek workload profile SHA256 mismatch" >&2
        exit 2
    fi
    export DEEPSEEK_WORKLOAD_PROFILE_SHA256=$workload_profile_sha256
fi
COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/common_epoch0_deepseek_v2_lite_tq1_no_overlap_gpu09}
CALIBRATION_ROOT=${DEEPSEEK_N_F2_KV_CALIBRATION_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/kv_calibration_natural_f2_$timestamp}
CAP_ENV=${DEEPSEEK_N_F2_KV_CAP_ENV:-${DEEPSEEK_KV_CAP_ENV:-$SCRIPT_DIR/deepseek_v2_lite_kv_caps.env}}
MODEL_PATH=/data/DeepSeek-V2-Lite-Chat
PROBE_HISTORY_ROOT=${DEEPSEEK_KV_PROBE_HISTORY_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/kv_probe_positive_release_trigger_v2}
RUNTIME_PROFILE_PATH=$SCRIPT_DIR/internal/deepseek_v2_lite_natural_f2_runtime_profile.sh
NPU_EXCLUSIVE_LOCK=/data/adafloor_shared_state/deepseek_v2_lite/.adafloor_npu_exclusive.lock
PROBE_TAIL_GUARD_MIN_CAP=64
PROBE_TAIL_GUARD_ROUND_TO=64
PROBE_EXPECTED_PLAN_RESPONSE_CAP=128
# shellcheck disable=SC1091
source "$RUNTIME_PROFILE_PATH"
profile_hash_args=()
IFS=, read -r -a runtime_profile_files <<< "$DEEPSEEK_N_F2_RUNTIME_PROFILE_FILES"
for runtime_profile_file in "${runtime_profile_files[@]}"; do
    profile_hash_args+=(--profile "$runtime_profile_file")
done
RUNTIME_PROFILE_SHA256=$(python3 \
    "$SCRIPT_DIR/tools/hash_deepseek_runtime_profile.py" \
    --root "$SCRIPT_DIR" "${profile_hash_args[@]}")
EXECUTION_CODE_SHA256=$(python3 \
    "$SCRIPT_DIR/tools/hash_deepseek_execution_code.py" \
    --root "$SCRIPT_DIR")

if (( $# != 0 )); then
    echo "DeepSeek Natural floor2 KV calibration does not accept workload overrides" >&2
    exit 2
fi

if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]]; then
    echo "missing completed DeepSeek common epoch0: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi
if [[ -e "$CALIBRATION_ROOT" ]]; then
    echo "refusing to overwrite DeepSeek KV calibration: $CALIBRATION_ROOT" >&2
    exit 2
fi
mkdir -p "$(dirname "$NPU_EXCLUSIVE_LOCK")"
exec 9>"$NPU_EXCLUSIVE_LOCK"
if ! flock -n 9; then
    echo "another AdaFloor NPU calibration holds $NPU_EXCLUSIVE_LOCK" >&2
    exit 2
fi

npu_processes=$(npu-smi info | awk '
    /Process id/ { in_process_table=1; next }
    in_process_table && /^\|/ && !/No running processes found/ { print }
')
if [[ -n "$npu_processes" ]]; then
    echo "DeepSeek KV calibration requires exclusive idle NPUs" >&2
    printf '%s\n' "$npu_processes" >&2
    exit 2
fi

echo "[DeepSeek Natural floor2 KV calibration] validating common epoch0 protocol"
COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
DEEPSEEK_KV_PROBE_HISTORY_ROOT="$PROBE_HISTORY_ROOT" \
DEEPSEEK_PROBE_TAIL_GUARD_MIN_CAP="$PROBE_TAIL_GUARD_MIN_CAP" \
DEEPSEEK_PROBE_TAIL_GUARD_ROUND_TO="$PROBE_TAIL_GUARD_ROUND_TO" \
DEEPSEEK_PROBE_DRY_RUN=1 \
"$SCRIPT_DIR/run_deepseek_v2_lite_kv_probe.sh" natural_f2 16

mkdir -p "$CALIBRATION_ROOT"
printf '%s\n' "INCOMPLETE DeepSeek Natural floor2 KV calibration" > "$CALIBRATION_ROOT/INCOMPLETE"

for floor in 16 8 4 2; do
    echo "[DeepSeek Natural floor2 KV calibration] plan-only preflight floor$floor"
    COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    DEEPSEEK_KV_PROBE_HISTORY_ROOT="$PROBE_HISTORY_ROOT" \
    DEEPSEEK_KV_PROBE_OUTPUT_ROOT="$CALIBRATION_ROOT" \
    DYNAMIC_RUN_NAME="natural_f2_floor${floor}" \
    DEEPSEEK_PROBE_TAIL_GUARD_MIN_CAP="$PROBE_TAIL_GUARD_MIN_CAP" \
    DEEPSEEK_PROBE_TAIL_GUARD_ROUND_TO="$PROBE_TAIL_GUARD_ROUND_TO" \
    DEEPSEEK_PROBE_PLAN_ONLY=1 \
    "$SCRIPT_DIR/run_deepseek_v2_lite_kv_probe.sh" natural_f2 "$floor"
done

for floor in 16 8 4 2; do
    echo "[DeepSeek Natural floor2 KV calibration] probing floor$floor"
    COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    DEEPSEEK_KV_PROBE_HISTORY_ROOT="$PROBE_HISTORY_ROOT" \
    DEEPSEEK_KV_PROBE_OUTPUT_ROOT="$CALIBRATION_ROOT" \
    DYNAMIC_RUN_NAME="natural_f2_floor${floor}" \
    DEEPSEEK_PROBE_TAIL_GUARD_MIN_CAP="$PROBE_TAIL_GUARD_MIN_CAP" \
    DEEPSEEK_PROBE_TAIL_GUARD_ROUND_TO="$PROBE_TAIL_GUARD_ROUND_TO" \
    "$SCRIPT_DIR/run_deepseek_v2_lite_kv_probe.sh" natural_f2 "$floor"
done

python3 "$SCRIPT_DIR/tools/generate_deepseek_kv_caps.py" \
    --lifecycle natural_f2 \
    --common-epoch0-root "$COMMON_EPOCH0_ROOT" \
    --model-path "$MODEL_PATH" \
    --floor2-summary "$CALIBRATION_ROOT/natural_f2_floor2/kv_probe_summary.json" \
    --floor4-summary "$CALIBRATION_ROOT/natural_f2_floor4/kv_probe_summary.json" \
    --floor8-summary "$CALIBRATION_ROOT/natural_f2_floor8/kv_probe_summary.json" \
    --floor16-summary "$CALIBRATION_ROOT/natural_f2_floor16/kv_probe_summary.json" \
    --output "$CAP_ENV" \
    --runtime-profile "$DEEPSEEK_N_F2_RUNTIME_PROFILE_ID" \
    --runtime-profile-sha256 "$RUNTIME_PROFILE_SHA256" \
    --execution-code-sha256 "$EXECUTION_CODE_SHA256" \
    --probe-history-root "$PROBE_HISTORY_ROOT" \
    --expected-plan-response-cap "$PROBE_EXPECTED_PLAN_RESPONSE_CAP" \
    --target-ratio 1.0 \
    --merge-existing

rm -f "$CALIBRATION_ROOT/INCOMPLETE"
printf '%s\n' \
    "COMPLETE DeepSeek Natural floor2 KV calibration" \
    "COMMON_EPOCH0_ROOT=$COMMON_EPOCH0_ROOT" \
    "WORKLOAD_PROFILE_PATH=${WORKLOAD_PROFILE_PATH:-unspecified}" \
    "WORKLOAD_PROFILE_ID=${DEEPSEEK_WORKLOAD_PROFILE_ID:-unspecified}" \
    "WORKLOAD_PROFILE_SHA256=${DEEPSEEK_WORKLOAD_PROFILE_SHA256:-unspecified}" \
    "PROBE_HISTORY_ROOT=$PROBE_HISTORY_ROOT" \
    "PROBE_TAIL_GUARD_MIN_CAP=$PROBE_TAIL_GUARD_MIN_CAP" \
    "PROBE_TAIL_GUARD_ROUND_TO=$PROBE_TAIL_GUARD_ROUND_TO" \
    "PROBE_EXPECTED_PLAN_RESPONSE_CAP=$PROBE_EXPECTED_PLAN_RESPONSE_CAP" \
    "CAP_ENV=$CAP_ENV" \
    > "$CALIBRATION_ROOT/COMPLETE"

echo "[DeepSeek Natural floor2 KV calibration] candidate caps=$CAP_ENV"
echo "[DeepSeek Natural floor2 KV calibration] root=$CALIBRATION_ROOT"
