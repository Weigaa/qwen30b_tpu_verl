#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

P0_ROOT="${P0_ROOT:-/data/adafloor_shared_state/p0_matched_trials_common_epoch0_20260728T221830Z}"
ORACLE_OUTPUT_ROOT="${ORACLE_OUTPUT_ROOT:-$P0_ROOT/oracle_replay}"
TRAIN_FILE="${TRAIN_FILE:-/data/deepscaler/train.parquet}"
TRIAL_SEEDS="${TRIAL_SEEDS:-101 202 303}"
DATASET_FRACTION="${DATASET_FRACTION:-0.005}"

mkdir -p "$ORACLE_OUTPUT_ROOT"

run_plan() {
    local seed="$1" name="$2" baseline_dir="$3"
    local min_floor="$4" floors="$5" matching="$6" forced_floor="$7"
    local repair_swaps="$8" allow_infeasible="$9"
    local output="$ORACLE_OUTPUT_ROOT/seed_${seed}/oracle_${name}"
    local extra=()
    mkdir -p "$output"
    if [[ "$forced_floor" != "0" ]]; then
        extra+=(--force-selected-floor "$forced_floor")
    fi
    if [[ "$allow_infeasible" == "1" ]]; then
        extra+=(--allow-infeasible)
    fi
    python3 -u "$SCRIPT_DIR/tools/build_mode1_length_sorted_e2e_plan.py" \
        --baseline-dir "$baseline_dir" \
        --length-ema-decay 0.3 \
        --train-file "$TRAIN_FILE" \
        --output-train "$output/length_sorted_train.parquet" \
        --output-plan "$output/length_sorted_rank_plan.json" \
        --output-summary "$output/length_sorted_rank_plan_summary.json" \
        --output-oracle "$output/length_sorted_length_oracle.json" \
        --steps 5 \
        --batch-size 32 \
        --responses-per-prompt 16 \
        --dataset-fraction "$DATASET_FRACTION" \
        --max-rank-peak-tokens 380800 \
        --adaptive-floor \
        --min-adaptive-floor "$min_floor" \
        --floor-kv-caps "$floors" \
        --rank-matching-policy "$matching" \
        --active-peak-safety-factor 1.0 \
        --max-response-len 16384 \
        --tail-guard-ratio-quantile 0.95 \
        --tail-guard-ratio-window 3 \
        --tail-guard-default-ratio 1.20 \
        --tail-guard-min-cap 4096 \
        --tail-guard-round-to 512 \
        --max-cross-step-repair-swaps "$repair_swaps" \
        --repair-candidate-limit 8 \
        --require-compact-history \
        "${extra[@]}"
}

for seed in $TRIAL_SEEDS; do
    vanilla_run="$P0_ROOT/trial_seed_${seed}/p0_vanilla_seed${seed}_frozen_epoch1"
    marker="$vanilla_run/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
    if [[ ! -f "$marker" ]]; then
        echo "P0 Vanilla trial is incomplete for seed $seed: $vanilla_run" >&2
        exit 3
    fi
    vanilla_epoch=""
    for candidate in "$vanilla_run"/epoch_001_*; do
        if [[ -d "$candidate" ]]; then
            vanilla_epoch="$candidate"
            break
        fi
    done
    if [[ -z "$vanilla_epoch" ]]; then
        echo "missing Vanilla epoch for seed $seed" >&2
        exit 3
    fi
    baseline_dir="$ORACLE_OUTPUT_ROOT/seed_${seed}/realized_vanilla_history"
    mkdir -p "$baseline_dir"
    if [[ ! -L "$baseline_dir/rollout_data" ]]; then
        ln -s "$vanilla_epoch/rollout_data" "$baseline_dir/rollout_data"
    fi
    if [[ ! -L "$baseline_dir/rollout_length" ]]; then
        ln -s "$vanilla_epoch/rollout_length" "$baseline_dir/rollout_length"
    fi
    python3 "$SCRIPT_DIR/tools/build_offline_planning_history.py" \
        --baseline-dir "$baseline_dir" --steps 5 --responses-per-prompt 16

    run_plan "$seed" full16_contiguous "$baseline_dir" 16 \
        '16:380800' contiguous 16 0 1
    run_plan "$seed" natural_f2_release "$baseline_dir" 2 \
        '2:131072,4:280576,8:315648,16:380800' release_area 0 8 0
    run_plan "$seed" planned_f4_release "$baseline_dir" 4 \
        '4:133120,8:262656,16:380800' release_area 0 8 0
done

python3 "$SCRIPT_DIR/analysis_eval/summarize_oracle_replay.py" \
    --root "$ORACLE_OUTPUT_ROOT" \
    --output-dir "$ORACLE_OUTPUT_ROOT/summary"
echo "[oracle replay] complete summary=$ORACLE_OUTPUT_ROOT/summary/oracle_summary.md"
