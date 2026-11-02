#!/usr/bin/env bash
set -euo pipefail

# Waits for the already-running Planned-F4 job, then performs one final,
# machine-checked post-processing pass. It never launches a training job.
RUN_ROOT=${1:?usage: $0 PLANNED_RUN_ROOT}
BASE_ROOT=${2:?usage: $0 PLANNED_RUN_ROOT BASE_ROOT}
LAUNCHER_PID=${3:?a live outer launcher PID is required}
[[ "$LAUNCHER_PID" =~ ^[0-9]+$ ]] || exit 2
LAUNCHER_START=$(awk '{print $22}' "/proc/$LAUNCHER_PID/stat")
REPO=/workspace/cann-recipes-train/llm_rl/qwen3_011/adafloor_gragh
LOG="$BASE_ROOT/finalizer_watch.log"
mkdir -p "$BASE_ROOT"
touch "$LOG"

live_run_process() {
  [[ -r "/proc/$LAUNCHER_PID/stat" ]] || return 1
  [[ "$(awk '{print $22}' "/proc/$LAUNCHER_PID/stat")" == "$LAUNCHER_START" ]]
}

while live_run_process; do
  sleep 300
done

if [[ ! -d "$RUN_ROOT/epoch_004_mode1_planned" ]]; then
  echo "$(date -u +%FT%TZ) planned process ended without epoch004; no finalization" >>"$LOG"
  exit 3
fi

natural=/data/adafloor_shared_state/hpca_p0_long_horizon_retry_20261006/summary_natural_final/quality_curve_summary.json
vanilla=/data/adafloor_shared_state/hpca_p0_long_horizon_minimal_20261005/summary_vanilla_final/quality_curve_summary.json
out="$BASE_ROOT/final_summary"

echo "$(date -u +%FT%TZ) starting finalizer" >>"$LOG"
if PYTHONPATH="$REPO" python3 "$REPO/analysis_eval/finalize_hpca_p0_result.py" \
    --planned-run "$RUN_ROOT" \
    --natural-summary "$natural" \
    --vanilla-summary "$vanilla" \
    --natural-run /data/adafloor_shared_state/hpca_p0_long_horizon_retry_20261006/hpca_p0_adafloor_n_f2_seed505_epoch1_4 \
    --vanilla-run /data/adafloor_shared_state/hpca_p0_long_horizon_minimal_20261005/hpca_p0_vanilla_seed505_epoch1_4 \
    --output-dir "$out" \
    --result-md "$REPO/analysis_eval/hpca_p0_result.md" \
    --common-epoch0 /data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent \
    --paper-manifest /data/adafloor_shared_state/p0_7_natural_matching_causal_v2_20260731/code_sha256.txt \
    --cleanup-relocated-checkpoints \
    --analyzer "$REPO/analysis_eval/analyze_training_quality_curve.py" >>"$LOG" 2>&1; then
  printf 'quality_observations_finalized=true\n' >>"$BASE_ROOT/protocol.env"
  printf 'common_epoch0_preserved=true\n' >"$BASE_ROOT/COMMON_EPOCH0_PRESERVED.txt"
  echo "$(date -u +%FT%TZ) finalization complete" >>"$LOG"
else
  echo "$(date -u +%FT%TZ) finalization failed; inspect log" >>"$LOG"
  exit 4
fi
