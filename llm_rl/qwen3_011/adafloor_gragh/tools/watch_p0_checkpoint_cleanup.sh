#!/usr/bin/env bash
set -euo pipefail

# Storage guard: relocate completed predecessors, NEVER delete their payloads.
# Keep the original checkpoint path usable for the paper runner's final audit.
RUN_ROOT=${1:?usage: $0 RUN_ROOT [LOG]}
LOG=${2:-$RUN_ROOT/checkpoint_watch.log}
LAUNCHER_PID=${3:?a live outer launcher PID is required}
[[ "$LAUNCHER_PID" =~ ^[0-9]+$ ]] || exit 2
LAUNCHER_START=$(awk '{print $22}' "/proc/$LAUNCHER_PID/stat")
RUN_ROOT=$(realpath -e "$RUN_ROOT")
case "$RUN_ROOT" in /data/adafloor_shared_state/hpca_p0_long_horizon_*/hpca_p0_adafloor_p_f4_*) ;; *) exit 2;; esac
ARCHIVE_ROOT="/workspace/hpca_p0_checkpoint_archive/$(basename "$(dirname "$RUN_ROOT")")"
mkdir -p "$RUN_ROOT"
touch "$LOG"

has_live_run_process() {
  [[ -r "/proc/$LAUNCHER_PID/stat" ]] || return 1
  [[ "$(awk '{print $22}' "/proc/$LAUNCHER_PID/stat")" == "$LAUNCHER_START" ]]
}

successor_uses() {
  local target="$1" proc pid cmd
  for proc in /proc/[0-9]*; do
    [[ -r "$proc/cmdline" ]] || continue
    pid=${proc##*/}
    [[ "$pid" == "$$" ]] && continue
    cmd=$(tr '\0' ' ' <"$proc/cmdline" 2>/dev/null || true)
    [[ "$cmd" == *"resume_from_path=$target"* ]] && return 0
  done
  return 1
}

successor_ready() {
  local n="$1" target="$2" next_tag next_dir
  next_tag=$(printf '%03d' "$((n + 1))")
  next_dir=$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d \
    -name "epoch_${next_tag}_*" -print -quit)
  [[ -n "$next_dir" ]] || return 1
  # Require an ACTUALLY COMPLETED successor training step, not startup intent.
  [[ -s "$next_dir/rollout_data/1.jsonl" ]] || return 1
  find "$next_dir/logs" -maxdepth 1 -type f -name '*.txt' -print0 2>/dev/null \
    | xargs -0 -r rg -q 'step:1 - .*actor/lr:'
}

while has_live_run_process; do
  for n in 1 2 3; do
    tag=$(printf '%03d' "$n")
    epoch_dir=$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d \
      -name "epoch_${tag}_*" -print -quit)
    [[ -n "$epoch_dir" ]] || continue
    checkpoint_dir="$epoch_dir/checkpoints"
    marker="$checkpoint_dir/qwen3moe_for_eagle3/latest_checkpointed_iteration.txt"
    target="$checkpoint_dir/qwen3moe_for_eagle3/global_step_5"
    audit="$epoch_dir/CHECKPOINT_RELOCATION.txt"
    model_dir="$checkpoint_dir/qwen3moe_for_eagle3"
    [[ -f "$marker" && -d "$target" && ! -L "$model_dir" && ! -e "$audit" ]] || continue
    successor_uses "$target" || continue
    successor_ready "$n" "$target" || continue

    # Re-check the path is strictly under this run root before deletion.
    case "$checkpoint_dir" in
      "$RUN_ROOT"/epoch_*/checkpoints) ;;
      *) echo "$(date -u +%FT%TZ) refusing unexpected path $checkpoint_dir" >>"$LOG"; continue ;;
    esac
    [[ "$(tr -d '[:space:]' < "$marker")" == 5 ]] || continue
    [[ -s "$target/actor/dist_ckpt/metadata.json" ]] || continue
    [[ "$(find "$target/actor/dist_ckpt" -maxdepth 1 -type f -name '*.distcp' -size +0c | wc -l)" == 32 ]] || continue
    archive="$ARCHIVE_ROOT/epoch_$tag/qwen3moe_for_eagle3"
    [[ ! -e "$archive" ]] || { echo "archive already exists: $archive" >>"$LOG"; exit 2; }
    mkdir -p "$(dirname "$archive")"
    need=$(du -sb "$model_dir" | awk '{print $1}')
    free=$(df -PB1 "$ARCHIVE_ROOT" | awk 'NR==2 {print $4}')
    (( free > need + 10737418240 )) || { echo "insufficient archive space; retaining source" >>"$LOG"; exit 2; }
    mv -T -- "$model_dir" "$archive"
    ln -s -- "$archive" "$model_dir"
    printf 'relocated_at=%s\noriginal=%s\narchive=%s\npayload_deleted=false\npublic_epoch0_untouched=true\n' \
      "$(date -u +%FT%TZ)" "$model_dir" "$archive" >"$audit"
    echo "$(date -u +%FT%TZ) relocated $model_dir to $archive after successor completed step1" >>"$LOG"
  done
  sleep 60
done
echo "$(date -u +%FT%TZ) no live run process; watcher exits" >>"$LOG"
