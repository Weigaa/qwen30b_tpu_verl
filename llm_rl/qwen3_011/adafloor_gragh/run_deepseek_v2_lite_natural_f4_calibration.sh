#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/common_epoch0_deepseek_v2_lite_tq1_no_overlap_gpu09}
CALIBRATION_ROOT=${DEEPSEEK_KV_CALIBRATION_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/kv_calibration_natural_f4_$timestamp}
CAP_ENV=${DEEPSEEK_KV_CAP_ENV:-$SCRIPT_DIR/deepseek_v2_lite_kv_caps.env}
MODEL_PATH=/data/DeepSeek-V2-Lite-Chat
PROBE_HISTORY_ROOT=${DEEPSEEK_KV_PROBE_HISTORY_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite/kv_probe_positive_release_trigger_v2}
RUNTIME_PROFILE_PATH=$SCRIPT_DIR/internal/deepseek_v2_lite_natural_f4_runtime_profile.sh
# shellcheck disable=SC1091
source "$RUNTIME_PROFILE_PATH"
profile_hash_args=()
IFS=, read -r -a runtime_profile_files <<< "$DEEPSEEK_N_F4_RUNTIME_PROFILE_FILES"
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
    echo "DeepSeek KV calibration does not accept workload overrides" >&2
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
NPU_EXCLUSIVE_LOCK=${DEEPSEEK_EXCLUSIVE_NPU_LOCK:-/data/adafloor_shared_state/deepseek_v2_lite/.adafloor_npu_exclusive.lock}
mkdir -p "$(dirname "$NPU_EXCLUSIVE_LOCK")"
exec 9>"$NPU_EXCLUSIVE_LOCK"
if ! flock -n 9; then
    echo "another DeepSeek calibration or authorization holds $NPU_EXCLUSIVE_LOCK" >&2
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

echo "[DeepSeek KV calibration] validating common epoch0 protocol"
COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
DEEPSEEK_KV_PROBE_HISTORY_ROOT="$PROBE_HISTORY_ROOT" \
DEEPSEEK_PROBE_DRY_RUN=1 \
"$SCRIPT_DIR/run_deepseek_v2_lite_kv_probe.sh" natural_f4 16

mkdir -p "$CALIBRATION_ROOT"
printf '%s\n' "INCOMPLETE DeepSeek Natural floor4 KV calibration" > "$CALIBRATION_ROOT/INCOMPLETE"

for floor in 16 8 4; do
    echo "[DeepSeek KV calibration] plan-only preflight Natural floor$floor"
    COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    DEEPSEEK_KV_PROBE_HISTORY_ROOT="$PROBE_HISTORY_ROOT" \
    DEEPSEEK_KV_PROBE_OUTPUT_ROOT="$CALIBRATION_ROOT" \
    DYNAMIC_RUN_NAME="natural_f4_floor${floor}" \
    DEEPSEEK_PROBE_PLAN_ONLY=1 \
    "$SCRIPT_DIR/run_deepseek_v2_lite_kv_probe.sh" natural_f4 "$floor"
done

for floor in 16 8 4; do
    echo "[DeepSeek KV calibration] probing Natural floor$floor"
    COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    DEEPSEEK_KV_PROBE_HISTORY_ROOT="$PROBE_HISTORY_ROOT" \
    DEEPSEEK_KV_PROBE_OUTPUT_ROOT="$CALIBRATION_ROOT" \
    DYNAMIC_RUN_NAME="natural_f4_floor${floor}" \
    "$SCRIPT_DIR/run_deepseek_v2_lite_kv_probe.sh" natural_f4 "$floor"
done

python3 "$SCRIPT_DIR/tools/generate_deepseek_kv_caps.py" \
    --lifecycle natural_f4 \
    --common-epoch0-root "$COMMON_EPOCH0_ROOT" \
    --model-path "$MODEL_PATH" \
    --floor4-summary "$CALIBRATION_ROOT/natural_f4_floor4/kv_probe_summary.json" \
    --floor8-summary "$CALIBRATION_ROOT/natural_f4_floor8/kv_probe_summary.json" \
    --floor16-summary "$CALIBRATION_ROOT/natural_f4_floor16/kv_probe_summary.json" \
    --output "$CAP_ENV" \
    --runtime-profile "$DEEPSEEK_N_F4_RUNTIME_PROFILE_ID" \
    --runtime-profile-sha256 "$RUNTIME_PROFILE_SHA256" \
    --execution-code-sha256 "$EXECUTION_CODE_SHA256" \
    --probe-history-root "$PROBE_HISTORY_ROOT" \
    --target-ratio 1.0 \
    --merge-existing

rm -f "$CALIBRATION_ROOT/INCOMPLETE"
printf '%s\n' \
    "COMPLETE DeepSeek Natural floor4 KV calibration" \
    "COMMON_EPOCH0_ROOT=$COMMON_EPOCH0_ROOT" \
    "PROBE_HISTORY_ROOT=$PROBE_HISTORY_ROOT" \
    "CAP_ENV=$CAP_ENV" \
    > "$CALIBRATION_ROOT/COMPLETE"

echo "[DeepSeek KV calibration] candidate caps=$CAP_ENV"
echo "[DeepSeek KV calibration] root=$CALIBRATION_ROOT"
