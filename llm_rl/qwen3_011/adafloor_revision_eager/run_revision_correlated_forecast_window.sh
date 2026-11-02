#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMON=/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe
OUT=${REVISION_WINDOW_ROOT:-/workspace/adafloor_revision_results/correlated_forecast_window_v1}
[[ ! -e "$OUT" ]] || exit 2
[[ -z $(pgrep -f '^python3 -m verl.trainer.main_ppo' || true) ]] || exit 3
mkdir -p "$OUT"
python3 "$REPO/tools/create_revision_forecast_fault.py" --history "$COMMON/offline_planning_history.json" \
    --output-dir "$OUT/derived_history" --scale 0.6
printf 'purpose=correlated_forecast_safety_window\nforecast_scale=0.6\nsampling_response_limit=4096\nactor_frozen=true\nsteps=1\nresponses=512\ngraph_mode=disabled\nTQ=2\nsidecar=%s\n' "${VERL_SIDECAR_ENABLE:-0}" > "$OUT/protocol.env"
sha256sum "$REPO/verl/trainer/ppo/ray_trainer.py" "$REPO/verl/utils/revision_kv_guard.py" \
    "$REPO/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py" \
    "$REPO/tools/create_revision_forecast_fault.py" "$0" > "$OUT/code_sha256.txt"
if [[ "${VERL_SIDECAR_ENABLE:-0}" == "1" ]]; then
    sha256sum "$REPO/verl/utils/sidecar_restore_handshake.py" \
        "$REPO/internal/watch_elastic_shrink_and_run_sidecar.sh" \
        "$REPO/internal/run_elastic_sidecar_infer.sh" \
        "$REPO/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh" \
        "$REPO/run_revision_active_sidecar_safety_window.sh" >> "$OUT/code_sha256.txt"
fi
cd "$REPO"
export BASELINE_DIR="$OUT/derived_history" BASELINE_DIRS="$OUT/derived_history"
export PLAN_DIR="$OUT/oracle" OUTPUT_ROOT="$OUT" OUTPUT_SUBDIR=child
export VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1 VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1
export ROLLOUT_ENFORCE_EAGER=True VLLM_ENABLE_GRAPH_MODE=0 TASK_QUEUE_ENABLE=2
export ADAFLOOR_REVISION_HARD_KV_GUARD=1
export ADAFLOOR_REVISION_EFFECTIVE_CAPS='2:0,4:133120,8:262656,16:380800'
export FLOOR_KV_CAPS='4:133120,8:262656,16:380800'
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=planned
export VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING=1
export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1 VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=1
export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB=28672
export VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1
export VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0
export SAVE_CKPT_ENABLE=0 TRAINER_SAVE_FREQ=-1
export MAX_RESPONSE_LENGTH=4096 MAX_RESPONSE_LEN=4096 PLAN_STEPS=5 ROLLOUT_N=16
export VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE=0
set +e
bash ./run_mode1_local_length_sorted_e2e_adaptive_floor4.sh \
    trainer.resume_mode=resume_path trainer.resume_from_path="$COMMON/checkpoints/qwen3moe_for_eagle3/global_step_5" \
    trainer.total_training_steps=1 actor_rollout_ref.actor.optim.lr=0.0 actor_rollout_ref.rollout.seed=707
rc=$?
set -e
printf 'exit_code=%s\n' "$rc" >> "$OUT/protocol.env"
exit "$rc"
