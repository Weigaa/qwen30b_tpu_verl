#!/usr/bin/env bash
set -euo pipefail
cd /workspace/cann-recipes-train/llm_rl/qwen3_shrink_aware
natural_root="mode1_dynamic_length_aware_adaptive_floor4_natural_tailguard_full3"
planner_root="mode1_dynamic_length_aware_adaptive_floor4_planned_tailguard_full3"
watch_log="$natural_root/watch_then_planner_$(date -u +%Y%m%dT%H%M%SZ).log"
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$watch_log"; }
complete_epoch() {
  local dir="$1"
  local jsonl_count=0
  [[ -d "$dir/rollout_data" ]] && jsonl_count=$(find "$dir/rollout_data" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
  [[ "$jsonl_count" -ge 5 && -d "$dir/checkpoints/qwen3moe_for_eagle3/global_step_5" ]]
}
log "watching natural epoch2 completion"
while ! complete_epoch "$natural_root/epoch_002_mode1_natural"; do
  if ! pgrep -af "${natural_root}.*(main_ppo|TaskRunner|wj_train|run_mode1)" >/dev/null; then
    log "natural process not found and epoch2 is incomplete; not starting planner"
    exit 2
  fi
  jsonl_count=0
  [[ -d "$natural_root/epoch_002_mode1_natural/rollout_data" ]] && jsonl_count=$(find "$natural_root/epoch_002_mode1_natural/rollout_data" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
  log "natural epoch2 not complete yet: rollout_jsonl=$jsonl_count"
  sleep 300
done
log "natural epoch2 complete; checking tailguard evidence"
if rg -q "Shrink-aware tail-guard response cap" "$natural_root/epoch_001_mode1_natural/logs" "$natural_root/epoch_002_mode1_natural/logs"; then
  log "natural tailguard evidence found"
else
  log "WARNING: natural tailguard evidence not found in logs"
fi
log "starting planner tailguard full3: $planner_root"
DYNAMIC_SHORT_STEP_CAP_ENABLE=1 \
./run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh \
  > "$planner_root.driver_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
status=$?
log "planner finished status=$status"
if [[ "$status" -eq 0 ]] && rg -q "Shrink-aware tail-guard response cap" "$planner_root"; then
  log "planner tailguard evidence found"
else
  log "WARNING: planner tailguard evidence missing or planner failed"
fi
exit "$status"
