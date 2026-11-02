#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMON=/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe
CKPT="$COMMON/checkpoints/qwen3moe_for_eagle3/global_step_5"
OUT=${REVISION_SMOKE_ROOT:-/workspace/adafloor_revision_results/hard_guard_smoke_v1}
[[ ! -e "$OUT" ]] || { printf 'Refusing existing output: %s\n' "$OUT" >&2; exit 2; }
[[ -s "$CKPT/actor/dist_ckpt/metadata.json" ]]
[[ -z $(pgrep -f '^python3 -m verl.trainer.main_ppo' || true) ]] || exit 3
mkdir -p "$OUT"
printf 'purpose=functional_smoke_not_accuracy_or_robustness_evidence\ngraph_mode=disabled\nTQ=2\nactor_frozen=false\nsteps=2\nresponse_limit=1024\ncommon_checkpoint=%s\nrevision_guard=hard_cap_bound\n' "$CKPT" > "$OUT/protocol.env"
sha256sum "$REPO/verl/trainer/ppo/ray_trainer.py" \
    "$REPO/verl/utils/revision_kv_guard.py" \
    "$REPO/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py" \
    "$REPO/PAPER_SOURCE_SNAPSHOT.json" "$0" > "$OUT/code_sha256.txt"
cd "$REPO"
export BASELINE_DIR="$COMMON" BASELINE_DIRS="$COMMON"
export OUTPUT_ROOT="$OUT" OUTPUT_SUBDIR=child
export ROLLOUT_ENFORCE_EAGER=True VLLM_ENABLE_GRAPH_MODE=0 TASK_QUEUE_ENABLE=2
export VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1
export ADAFLOOR_REVISION_HARD_KV_GUARD=1
export ADAFLOOR_REVISION_EFFECTIVE_CAPS='2:0,4:133120,8:262656,16:380800'
export FLOOR_KV_CAPS='4:133120,8:262656,16:380800'
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=planned
export VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING=1
export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1
export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=1
export VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB=28672
export VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1
export VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0
export SAVE_CKPT_ENABLE=0 TRAINER_SAVE_FREQ=-1
export MAX_RESPONSE_LENGTH=1024 MAX_RESPONSE_LEN=1024 PLAN_STEPS=5 ROLLOUT_N=16
export VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE=1
set +e
bash ./run_mode1_local_length_sorted_e2e_adaptive_floor4.sh \
    trainer.resume_mode=resume_path trainer.resume_from_path="$CKPT" \
    trainer.total_training_steps=2 actor_rollout_ref.rollout.seed=707
rc=$?
set -e
printf 'exit_code=%s\n' "$rc" >> "$OUT/protocol.env"
exit "$rc"
