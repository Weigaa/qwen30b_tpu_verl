#!/usr/bin/env bash
# One frozen-policy, same-Full16 cap-on/off pair; not a convergence benchmark.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMON=/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe
CKPT="$COMMON/checkpoints/qwen3moe_for_eagle3/global_step_5"
ROOT=${REVISION_CAP_PAIR_ROOT:-/workspace/adafloor_revision_results/tailguard_cap_pair_v1}
SEED=${REVISION_CAP_PAIR_SEED:-707}
ACTION=${1:-dry-run}
[[ "$ACTION" == dry-run || "$ACTION" == run ]] || exit 2
[[ -s "$CKPT/actor/dist_ckpt/metadata.json" && -s "$COMMON/offline_planning_history.json" ]]
[[ $(realpath -m "$ROOT") == /workspace/adafloor_revision_results/* ]]
[[ ! -e "$ROOT" ]] || { echo "refusing existing pair root: $ROOT" >&2; exit 2; }
printf 'purpose=one_cap_pair_not_training_equivalence\nseed=%s\nactor_frozen=true\nvariants=cap_on,cap_off\nsteps_per_arm=1\nprompts=32\nresponses_per_prompt=8\nresponses_per_arm=256\ncommon_checkpoint=%s\nfloor=16\nTQ=2\ngraph=disabled\nglobal_response_limit=16384\nguard_cap=paper_formula_from_history\nworkload=first_length_sorted_step_from_common_epoch0\n' "$SEED" "$CKPT"
[[ "$ACTION" == run ]] || exit 0
[[ -z $(pgrep -f '^python3 -m verl.trainer.main_ppo' || true) ]] || { echo 'primary job still live' >&2; exit 3; }
mkdir -p "$ROOT"
printf 'seed=%s\nactor_frozen=true\nfloor=16\nsteps=1\nresponses_per_prompt=8\nprompts=32\nTQ=2\ngraph=disabled\norder=cap_on,cap_off\nmetadata_capture=explicit_vllm_completion\ncommon_checkpoint=%s\nworkload=first_length_sorted_step_from_common_epoch0\nmax_response_length=16384\nguard_cap=paper_formula_from_history\n' "$SEED" "$CKPT" > "$ROOT/protocol.env"
sha256sum "$0" "$REPO/verl/trainer/ppo/ray_trainer.py" \
    "$REPO/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py" \
    "$REPO/verl/utils/revision_censored_history.py" "$REPO/tools/build_mode1_length_sorted_e2e_plan.py" \
    "$REPO/tools/build_revision_censored_history.py" "$REPO/verl/utils/revision_completion_audit.py" \
    "$REPO/tools/build_mode1_optimized_rank_plan.py" "$REPO/verl/utils/rollout_seeding.py" \
    "$REPO/analysis_eval/analyze_revision_cap_pair.py" > "$ROOT/code_sha256.txt"
sha256sum "$COMMON/offline_planning_history.json" "$CKPT/actor/dist_ckpt/metadata.json" \
    /data/deepscaler/train.parquet > "$ROOT/input_sha256.txt"
cd "$REPO"
export BASELINE_DIR="$COMMON" BASELINE_DIRS="$COMMON" VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1
export ROLLOUT_ENFORCE_EAGER=True VLLM_ENABLE_GRAPH_MODE=0 TASK_QUEUE_ENABLE=2
export VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 ADAFLOOR_REVISION_COMPLETION_AUDIT=1
export VERL_SKIP_FINAL_ACTOR_UPDATE_WHEN_FROZEN=1
export ADAFLOOR_REVISION_EXPERT_AUDIT=0 VERL_SIDECAR_ENABLE=0
export ADAFLOOR_REVISION_HARD_KV_GUARD=1
export ADAFLOOR_REVISION_EFFECTIVE_CAPS='2:0,4:133120,8:262656,16:380800'
export FLOOR_KV_CAPS='4:133120,8:262656,16:380800' FORCE_SELECTED_FLOOR=16
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=planned
export VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING=1
export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1 VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=1
export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB=28672
export VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1
export VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0
export SAVE_CKPT_ENABLE=0 TRAINER_SAVE_FREQ=-1 MAX_RESPONSE_LENGTH=16384 MAX_RESPONSE_LEN=16384
export PLAN_STEPS=5 ROLLOUT_N=8
for arm in cap_on cap_off; do
    export OUTPUT_ROOT="$ROOT/$arm" OUTPUT_SUBDIR=child PLAN_DIR="$ROOT/$arm/oracle"
    export VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE=$([[ "$arm" == cap_on ]] && echo 1 || echo 0)
    mkdir -p "$OUTPUT_ROOT"
    set +e
    bash ./run_mode1_local_length_sorted_e2e_adaptive_floor4.sh \
        trainer.resume_mode=resume_path trainer.resume_from_path="$CKPT" \
        trainer.total_training_steps=1 actor_rollout_ref.actor.optim.lr=0.0 \
        actor_rollout_ref.rollout.seed="$SEED" > "$OUTPUT_ROOT/launch.log" 2>&1
    rc=$?
    set -e
    printf '%s_exit_code=%s\n' "$arm" "$rc" >> "$ROOT/protocol.env"
    [[ "$rc" == 0 ]] || { echo "cap pair failed: arm=$arm rc=$rc; evidence retained" >&2; exit "$rc"; }
done
# Full verification includes matching request IDs/seeds, explicit finish metadata,
# zero faults and source immutability; do not mark PASS on exit code alone.
python3 "$REPO/analysis_eval/analyze_revision_cap_pair.py" --root "$ROOT"
sha256sum -c "$ROOT/input_sha256.txt"
