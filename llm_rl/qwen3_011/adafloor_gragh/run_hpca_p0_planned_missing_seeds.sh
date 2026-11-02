#!/usr/bin/env bash
set -euo pipefail
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAPER=/workspace/cann-recipes-train/llm_rl/qwen3_011/qwen3_shrink_aware
PAPER_MANIFEST=/data/adafloor_shared_state/p0_7_natural_matching_causal_v2_20260731/code_sha256.txt
COMMON=/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent
HISTORICAL=/data/adafloor_shared_state/p0_matched_trials_common_epoch0_20260728T221830Z
OUT=${HPCA_P0_PLANNED_OUTPUT_ROOT:-/workspace/hpca_p0_planned_frozen_missing_seeds_20261007}
ACTION=${1:-run}
[[ "$ACTION" == run || "$ACTION" == --dry-run ]] || exit 2
sha256sum -c --quiet "$PAPER_MANIFEST"
[[ -f "$COMMON/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" && -s "$COMMON/reuse.env" ]]
PYTHONPATH="$SELF_DIR/analysis_eval" python3 -c \
    'from pathlib import Path; from finalize_hpca_p0_result import verify_public_checkpoint; verify_public_checkpoint(Path("/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent"))'
out_real=$(realpath -m "$OUT")
case "$out_real/" in "$COMMON/"*) exit 2 ;; esac
for policy in planned_release planned_minskew; do
    [[ -s "$HISTORICAL/trial_seed_101/p0_${policy}_seed101_frozen_epoch1/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]
done
if [[ "$ACTION" == --dry-run ]]; then
    printf 'dry_run=PASS\nseeds=202,303\npolicies=planned_release,planned_minskew\nchild_runs=4\nepochs_per_run=1\nsteps=5\nresponses_per_step=512\nactor_frozen=true\nTQ=2\ngraph=disabled\noutput=%s\n' "$OUT"
    exit 0
fi
mkdir -p "$OUT/summary" "$OUT/trial_seed_101"
MANIFEST="$OUT/trial_manifest.tsv"
[[ -f "$MANIFEST" ]] || printf 'trial_seed\tpolicy\toutput_dir\tstatus\n' > "$MANIFEST"
record() {
    local seed=$1 policy=$2 path=$3 state=$4
    awk -F '\t' -v s="$seed" -v p="$policy" 'NR==1 || !($1==s && $2==p)' "$MANIFEST" > "$MANIFEST.tmp"
    printf '%s\t%s\t%s\t%s\n' "$seed" "$policy" "$path" "$state" >> "$MANIFEST.tmp"
    mv -f "$MANIFEST.tmp" "$MANIFEST"
}
for policy in planned_release planned_minskew; do
    target="$HISTORICAL/trial_seed_101/p0_${policy}_seed101_frozen_epoch1"
    link="$OUT/trial_seed_101/p0_${policy}_seed101_frozen_epoch1"
    [[ -e "$link" ]] || ln -s "$target" "$link"
    [[ $(realpath -e "$link") == "$target" ]]
    record 101 "$policy" "$target" historical_complete
done
printf 'status=RUNNING\nactor_frozen=true\nper_request_sampling_seeds=true\ncommon_epoch0_root=%s\ntrial_seeds=101 202 303\nnew_seeds=202 303\nepochs_per_trial=1\nsteps_per_epoch=5\nresponses_per_step=512\nTASK_QUEUE_ENABLE=2\ngraph_mode=disabled\nhistorical_seed101_same_full_code=false\n' "$COMMON" > "$OUT/protocol.env"
sha256sum "$0" "$PAPER_MANIFEST" "$PAPER/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
    "$PAPER/tools/build_mode1_length_sorted_e2e_plan.py" "$PAPER/verl/trainer/ppo/ray_trainer.py" \
    "$PAPER/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py" > "$OUT/code_sha256.txt"
cp "$HISTORICAL/code_sha256.txt" "$OUT/historical_seed101_code_sha256.txt"
for pair in '202 planned_release' '202 planned_minskew' '303 planned_minskew' '303 planned_release'; do
    read -r seed policy <<< "$pair"
    run_name="p0_${policy}_seed${seed}_frozen_epoch1"
    trial_root="$OUT/trial_seed_$seed"
    run_dir="$trial_root/$run_name"
    if [[ -s "$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; then
        record "$seed" "$policy" "$run_dir" complete
        continue
    fi
    [[ ! -e "$run_dir" ]] || { record "$seed" "$policy" "$run_dir" incomplete; exit 3; }
    mkdir -p "$trial_root"
    free=$(df -PB1 "$OUT" | awk 'NR==2 {print $4}')
    (( free > 180 * 1024 * 1024 * 1024 )) || { printf 'Insufficient space: %s bytes\n' "$free" >&2; exit 2; }
    sha256sum -c --quiet "$PAPER_MANIFEST"
    case "$policy" in planned_release) variant=adafloor_p_f4 ;; planned_minskew) variant=adafloor_p_f4_minskew ;; esac
    record "$seed" "$policy" "$run_dir" running
    printf '[P0 planned] start seed=%s policy=%s\n' "$seed" "$policy"
    set +e
    (
        cd "$PAPER"
        env -u FAIR_RUN_NAME COMMON_EPOCH0_ROOT="$COMMON" FAIR_OUTPUT_ROOT="$trial_root" \
            DYNAMIC_RUN_NAME="$run_name" FAIR_START_EPOCH=1 FAIR_TOTAL_EPOCHS=2 \
            FAIR_FREEZE_ACTOR=1 FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
            FAIR_TRAIN_BATCH_SIZE=32 FAIR_ROLLOUT_N=16 FAIR_MAX_RESPONSE_LENGTH=16384 \
            VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 TASK_QUEUE_ENABLE=2 \
            VLLM_ENABLE_GRAPH_MODE=0 ROLLOUT_ENFORCE_EAGER=True \
            ./run_paper_fair_epoch1_2_from_common_epoch0.sh "$variant" "actor_rollout_ref.rollout.seed=$seed"
    ) > "$OUT/${seed}_${policy}.launch.log" 2>&1
    rc=$?
    set -e
    if (( rc != 0 )); then
        record "$seed" "$policy" "$run_dir" "failed_rc_$rc"
        printf 'status=FAILED\nfailed_seed=%s\nfailed_policy=%s\n' "$seed" "$policy" >> "$OUT/protocol.env"
        exit "$rc"
    fi
    [[ -s "$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]
    record "$seed" "$policy" "$run_dir" complete
    printf '[P0 planned] complete seed=%s policy=%s; checkpoints cleaned\n' "$seed" "$policy"
done
python3 "$PAPER/analysis_eval/summarize_p0_matched_trials.py" --root "$OUT" --allow-incomplete --output-dir "$OUT/summary"
python3 "$SELF_DIR/analysis_eval/summarize_hpca_planner_ablation.py" --root "$OUT" --output-dir "$OUT/hpca_ablation"
python3 "$SELF_DIR/analysis_eval/finalize_hpca_planned_frozen.py" --root "$OUT" \
    --result-md "$SELF_DIR/analysis_eval/hpca_p0_result.md" --refine-md "$SELF_DIR/analysis_eval/future_refine.md"
printf 'status=COMPLETE\n' >> "$OUT/protocol.env"
printf '[P0 planned] done root=%s\n' "$OUT"
