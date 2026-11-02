#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export FAST_STEP_SUBSET="${FAST_STEP_SUBSET:-1}"
export FAST_STEP_SUBSET_STEPS="${FAST_STEP_SUBSET_STEPS:-1,5}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode1_length_sorted_e2e_adaptive_floor4_fast15}"

exec "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh" "$@"
