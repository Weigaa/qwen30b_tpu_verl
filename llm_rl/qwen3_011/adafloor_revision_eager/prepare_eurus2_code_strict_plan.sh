#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

COMMON_ROOT="${EURUS2_COMMON_ROOT:-/data/adafloor_shared_state/common_epoch0_eurus2_code_validation_frozen_gpu09_kv380800_permanent}"
WORKLOAD_DIR="${EURUS2_CODE_WORKLOAD_DIR:-/data/eurus2_rl_code/validation_code_paired_160}"
OUTPUT_DIR="${EURUS2_STRICT_PLAN_DIR:-$SCRIPT_DIR/analysis_eval/generated/eurus2_code_strict_plan_headroom025}"
BASELINE_DIR="$COMMON_ROOT/epoch_000_mode0_probe"
TRAIN_FILE="$WORKLOAD_DIR/train.parquet"

# The failed run reached a 373607-token realized response proxy before one
# full-world preemption. Admit floor16 plans below that observation at a
# 128-token boundary. Other floors retain their calibrated capacities. Runtime
# still allocates the complete physical floor16 capacity.
ADMISSION_CAPS="2:147456,4:280576,8:377344,16:371200"
PHYSICAL_CAPS="2:147456,4:280576,8:377344,16:380800"

if [[ ! -f "$BASELINE_DIR/offline_planning_history.json" ]]; then
    echo "missing compact epoch0 history: $BASELINE_DIR" >&2
    exit 2
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "missing Eurus-2-RL workload: $TRAIN_FILE" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
python3 -u tools/build_mode1_length_sorted_e2e_plan.py \
    --baseline-dir "$BASELINE_DIR" \
    --train-file "$TRAIN_FILE" \
    --output-train "$OUTPUT_DIR/length_sorted_train.parquet" \
    --output-plan "$OUTPUT_DIR/length_sorted_rank_plan.json" \
    --output-summary "$OUTPUT_DIR/length_sorted_rank_plan_summary.json" \
    --output-oracle "$OUTPUT_DIR/length_sorted_length_oracle.json" \
    --steps 5 \
    --batch-size 32 \
    --responses-per-prompt 16 \
    --dataset-fraction 1.0 \
    --max-rank-peak-tokens 371200 \
    --adaptive-floor \
    --min-adaptive-floor 2 \
    --floor-kv-caps "$ADMISSION_CAPS" \
    --rank-matching-policy release_area \
    --active-peak-safety-factor 1.16 \
    --max-response-len 16384 \
    --ignore-tail-ties-at-response-cap \
    --max-cross-step-repair-swaps 8 \
    --repair-candidate-limit 8 \
    --tail-guard-ratio-quantile 0.95 \
    --tail-guard-ratio-window 3 \
    --tail-guard-default-ratio 1.20 \
    --tail-guard-min-cap 4096 \
    --tail-guard-round-to 512 \
    --require-compact-history \
    --timing-output "$OUTPUT_DIR/timing.json"

python3 tools/promote_kv_admission_caps.py \
    --plan "$OUTPUT_DIR/length_sorted_rank_plan.json" \
    --summary "$OUTPUT_DIR/length_sorted_rank_plan_summary.json" \
    --physical-caps "$PHYSICAL_CAPS"

sha256sum \
    "$TRAIN_FILE" \
    "$BASELINE_DIR/offline_planning_history.json" \
    "$OUTPUT_DIR/length_sorted_train.parquet" \
    "$OUTPUT_DIR/length_sorted_rank_plan.json" \
    "$OUTPUT_DIR/length_sorted_rank_plan_summary.json" \
    "$OUTPUT_DIR/length_sorted_length_oracle.json" \
    > "$OUTPUT_DIR/input_output_sha256.txt"

printf 'admission_caps=%s\nphysical_caps=%s\n' \
    "$ADMISSION_CAPS" "$PHYSICAL_CAPS" \
    > "$OUTPUT_DIR/kv_admission_protocol.txt"

echo "[eurus2 strict plan] output=$OUTPUT_DIR"
