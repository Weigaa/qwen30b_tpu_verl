#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash run_mode1_floor4_overlap4group_destroy_test.sh

Purpose:
  Test whether floor=4 HCCL/MC2 destroy latency appears when consecutive
  4-rank groups partially overlap instead of being identical or disjoint.

Default pattern:
  - step 1 floor4 survivor tends to [12,13,14,15]
  - step 2 floor4 survivor tends to [10,11,12,13]
  - overlap is [12,13]

Useful overrides:
  BASELINE_TOTAL_TRAINING_STEPS=3
  VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES=tail,shift2
  VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE=0
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BASELINE_TOTAL_TRAINING_STEPS="${BASELINE_TOTAL_TRAINING_STEPS:-3}"
export VERL_ELASTIC_TAIL_VALIDATE_ROTATE_BUCKETS="${VERL_ELASTIC_TAIL_VALIDATE_ROTATE_BUCKETS:-1}"
export VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES="${VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES:-tail,shift2}"
export RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/mode1_floor4_overlap4group_destroy_runs/$(date -u +%Y%m%dT%H%M%SZ)}"

printf '[mode1 floor4 overlap4 destroy test] modes=%s steps=%s run_root=%s\n' \
    "$VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES" \
    "$BASELINE_TOTAL_TRAINING_STEPS" \
    "$RUN_ROOT"

exec bash "$SCRIPT_DIR/run_mode1_floor4_rotate4group_destroy_test.sh"
