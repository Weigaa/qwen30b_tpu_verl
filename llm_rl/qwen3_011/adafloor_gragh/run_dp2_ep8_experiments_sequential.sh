#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR}"
RANDOM_RUN_NAME="${RANDOM_RUN_NAME:-dp2_ep8_random_adafloor_epoch1}"
TLTLIKE_RUN_NAME="${TLTLIKE_RUN_NAME:-dp2_ep8_tltlike_adafloor_epoch1}"

echo "[dp2-ep8 suite] phase=1 random grouping"
OUTPUT_ROOT="$OUTPUT_ROOT" \
DP2_EP8_RUN_NAME="$RANDOM_RUN_NAME" \
"$SCRIPT_DIR/run_dp2_ep8_random_adafloor_epoch1.sh" "$@"

if [[ "${DP2_EP8_PLAN_ONLY:-0}" == "1" ]]; then
    required_marker=PLAN_ONLY_COMPLETE
else
    required_marker=STRICT_RUN_COMPLETE
fi
if [[ ! -f "$OUTPUT_ROOT/$RANDOM_RUN_NAME/$required_marker" ]]; then
    echo "random DP2 EP8 run did not pass strict validation" >&2
    exit 4
fi

echo "[dp2-ep8 suite] phase=2 TLT-like worker reuse plus AdaFloor"
OUTPUT_ROOT="$OUTPUT_ROOT" \
DP2_EP8_RUN_NAME="$TLTLIKE_RUN_NAME" \
"$SCRIPT_DIR/run_dp2_ep8_tltlike_adafloor_epoch1.sh" "$@"

echo "[dp2-ep8 suite] complete random=$OUTPUT_ROOT/$RANDOM_RUN_NAME tltlike=$OUTPUT_ROOT/$TLTLIKE_RUN_NAME"
