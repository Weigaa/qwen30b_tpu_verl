#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/common_epoch0_deepseek_v2_lite_tq1_no_overlap_gpu09}
CALIBRATION_ROOT=${DEEPSEEK_P_F2_KV_CALIBRATION_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/kv_calibration_planned_f2_$timestamp}
CAP_ENV=${DEEPSEEK_P_F2_KV_CAP_ENV:-${DEEPSEEK_KV_CAP_ENV:-$SCRIPT_DIR/deepseek_v2_lite_kv_caps.env}}
MODEL_PATH=/data/DeepSeek-V2-Lite-Chat
PROBE_HISTORY_ROOT=${DEEPSEEK_KV_PROBE_HISTORY_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/kv_probe_positive_release_trigger_v2}
RUNTIME_PROFILE_PATH=$SCRIPT_DIR/internal/deepseek_v2_lite_planned_f2_runtime_profile.sh
NPU_EXCLUSIVE_LOCK=/data/adafloor_shared_state/deepseek_v2_lite/.adafloor_npu_exclusive.lock
PROBE_TAIL_GUARD_MIN_CAP=64
PROBE_TAIL_GUARD_ROUND_TO=64
PROBE_EXPECTED_PLAN_RESPONSE_CAP=128

if (( $# != 0 )); then
    echo "DeepSeek Planned floor2 KV calibration does not accept workload overrides" >&2
    exit 2
fi
if ! [[ "${DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB must be an explicitly measured positive integer" >&2
    exit 2
fi

# shellcheck disable=SC1091
source "$RUNTIME_PROFILE_PATH"
IFS=, read -r -a runtime_profile_files <<< "$DEEPSEEK_P_F2_RUNTIME_PROFILE_FILES"
profile_hash_args=()
for runtime_profile_file in "${runtime_profile_files[@]}"; do
    profile_hash_args+=(--profile "$runtime_profile_file")
done
RUNTIME_PROFILE_SHA256=$(python3 \
    "$SCRIPT_DIR/tools/hash_deepseek_runtime_profile.py" \
    --root "$SCRIPT_DIR" "${profile_hash_args[@]}")
EXECUTION_CODE_SHA256=$(python3 \
    "$SCRIPT_DIR/tools/hash_deepseek_execution_code.py" \
    --root "$SCRIPT_DIR")

if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]]; then
    echo "missing completed DeepSeek common epoch0: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi
if [[ -e "$CALIBRATION_ROOT" ]]; then
    echo "refusing to overwrite DeepSeek Planned floor2 calibration: $CALIBRATION_ROOT" >&2
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

probe_env=(
    COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT"
    DEEPSEEK_KV_PROBE_HISTORY_ROOT="$PROBE_HISTORY_ROOT"
    DEEPSEEK_PROBE_TAIL_GUARD_MIN_CAP="$PROBE_TAIL_GUARD_MIN_CAP"
    DEEPSEEK_PROBE_TAIL_GUARD_ROUND_TO="$PROBE_TAIL_GUARD_ROUND_TO"
    DEEPSEEK_P_F2_HEADROOM_FLOOR2="$DEEPSEEK_P_F2_HEADROOM_FLOOR2"
    DEEPSEEK_P_F2_HEADROOM_FLOOR4="$DEEPSEEK_P_F2_HEADROOM_FLOOR4"
    DEEPSEEK_P_F2_HEADROOM_FLOOR8="$DEEPSEEK_P_F2_HEADROOM_FLOOR8"
    DEEPSEEK_P_F2_HEADROOM_FLOOR16="$DEEPSEEK_P_F2_HEADROOM_FLOOR16"
    DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB="$DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB"
)

echo "[DeepSeek Planned floor2 calibration] validating common epoch0 protocol"
env "${probe_env[@]}" DEEPSEEK_PROBE_DRY_RUN=1 \
    "$SCRIPT_DIR/run_deepseek_v2_lite_kv_probe.sh" planned_f2 16

mkdir -p "$CALIBRATION_ROOT"
printf '%s\n' "INCOMPLETE DeepSeek Planned floor2 KV calibration" > "$CALIBRATION_ROOT/INCOMPLETE"

for floor in 16 8 4 2; do
    echo "[DeepSeek Planned floor2 calibration] plan-only preflight floor$floor"
    env "${probe_env[@]}" \
        DEEPSEEK_KV_PROBE_OUTPUT_ROOT="$CALIBRATION_ROOT" \
        DYNAMIC_RUN_NAME="planned_f2_floor${floor}" \
        DEEPSEEK_PROBE_PLAN_ONLY=1 \
        "$SCRIPT_DIR/run_deepseek_v2_lite_kv_probe.sh" planned_f2 "$floor"
done

for floor in 16 8 4 2; do
    echo "[DeepSeek Planned floor2 calibration] probing floor$floor"
    env "${probe_env[@]}" \
        DEEPSEEK_KV_PROBE_OUTPUT_ROOT="$CALIBRATION_ROOT" \
        DYNAMIC_RUN_NAME="planned_f2_floor${floor}" \
        "$SCRIPT_DIR/run_deepseek_v2_lite_kv_probe.sh" planned_f2 "$floor"
done

python3 "$SCRIPT_DIR/tools/generate_deepseek_kv_caps.py" \
    --lifecycle planned_f2 \
    --common-epoch0-root "$COMMON_EPOCH0_ROOT" \
    --model-path "$MODEL_PATH" \
    --floor2-summary "$CALIBRATION_ROOT/planned_f2_floor2/kv_probe_summary.json" \
    --floor4-summary "$CALIBRATION_ROOT/planned_f2_floor4/kv_probe_summary.json" \
    --floor8-summary "$CALIBRATION_ROOT/planned_f2_floor8/kv_probe_summary.json" \
    --floor16-summary "$CALIBRATION_ROOT/planned_f2_floor16/kv_probe_summary.json" \
    --output "$CAP_ENV" \
    --runtime-profile "$DEEPSEEK_P_F2_RUNTIME_PROFILE_ID" \
    --runtime-profile-sha256 "$RUNTIME_PROFILE_SHA256" \
    --execution-code-sha256 "$EXECUTION_CODE_SHA256" \
    --probe-history-root "$PROBE_HISTORY_ROOT" \
    --expected-plan-response-cap "$PROBE_EXPECTED_PLAN_RESPONSE_CAP" \
    --planned-headroom-floor2 "$DEEPSEEK_P_F2_HEADROOM_FLOOR2" \
    --planned-headroom-floor4 "$DEEPSEEK_P_F2_HEADROOM_FLOOR4" \
    --planned-headroom-floor8 "$DEEPSEEK_P_F2_HEADROOM_FLOOR8" \
    --planned-headroom-floor16 "$DEEPSEEK_P_F2_HEADROOM_FLOOR16" \
    --training-min-free-mib "$DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB" \
    --target-ratio 1.0 \
    --merge-existing

rm -f "$CALIBRATION_ROOT/INCOMPLETE"
printf '%s\n' \
    "COMPLETE DeepSeek Planned floor2 KV calibration" \
    "COMMON_EPOCH0_ROOT=$COMMON_EPOCH0_ROOT" \
    "PROBE_HISTORY_ROOT=$PROBE_HISTORY_ROOT" \
    "RUNTIME_PROFILE_ID=$DEEPSEEK_P_F2_RUNTIME_PROFILE_ID" \
    "RUNTIME_PROFILE_FILES=$DEEPSEEK_P_F2_RUNTIME_PROFILE_FILES" \
    "RUNTIME_PROFILE_SHA256=$RUNTIME_PROFILE_SHA256" \
    "EXECUTION_CODE_SHA256=$EXECUTION_CODE_SHA256" \
    "PROBE_TAIL_GUARD_MIN_CAP=$PROBE_TAIL_GUARD_MIN_CAP" \
    "PROBE_TAIL_GUARD_ROUND_TO=$PROBE_TAIL_GUARD_ROUND_TO" \
    "PROBE_EXPECTED_PLAN_RESPONSE_CAP=$PROBE_EXPECTED_PLAN_RESPONSE_CAP" \
    "PLANNED_HEADROOM_FLOOR2=$DEEPSEEK_P_F2_HEADROOM_FLOOR2" \
    "PLANNED_HEADROOM_FLOOR4=$DEEPSEEK_P_F2_HEADROOM_FLOOR4" \
    "PLANNED_HEADROOM_FLOOR8=$DEEPSEEK_P_F2_HEADROOM_FLOOR8" \
    "PLANNED_HEADROOM_FLOOR16=$DEEPSEEK_P_F2_HEADROOM_FLOOR16" \
    "TRAINING_MIN_FREE_MIB=$DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB" \
    "CAP_ENV=$CAP_ENV" \
    > "$CALIBRATION_ROOT/COMPLETE"

echo "[DeepSeek Planned floor2 calibration] candidate caps=$CAP_ENV"
echo "[DeepSeek Planned floor2 calibration] root=$CALIBRATION_ROOT"
