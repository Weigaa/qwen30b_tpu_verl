#!/usr/bin/env bash
# Minimal one-floor mechanism control; remote expert sources are NOT sidecars.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMON=/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe
ROOT=${REVISION_MODE4_ROOT:-/workspace/adafloor_revision_results/mode4_control_pair_v1}
ACTION=${1:-dry-run}
[[ "$ACTION" == dry-run || "$ACTION" == run ]] || exit 2
[[ -s "$COMMON/checkpoints/qwen3moe_for_eagle3/global_step_5/actor/dist_ckpt/metadata.json" ]]
[[ $(realpath -m "$ROOT") == /workspace/adafloor_revision_results/* && ! -e "$ROOT" ]]
printf 'experiment=one_floor4_expert_mechanism_pair\nmodes=resident1,remote4\nactor_frozen=true\ncommon_checkpoint=%s\nsteps_per_arm=1\nresponse_caps=256,512,640,768,896\nignore_eos=true\nexpected_tokens=225280\nsidecar=false\nTQ=2\ngraph=false\n' "$COMMON"
[[ "$ACTION" == run ]] || exit 0
[[ -z $(pgrep -f '^python3 -m (verl.trainer.main_ppo|torch.distributed.run)' || true) ]] || exit 3
mkdir -p "$ROOT"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" CONFIG_DIR="$REPO/verl/trainer/config"
export MODEL_PATH=/data/Qwen3-30B-A3B DISTCP_PATH=/data/Qwen3-30B-A3B_megatron
export TRAIN_FILE=/data/deepscaler/train.parquet TEST_FILE=/data/deepscaler/test.parquet
export SAVE_CKPT_ENABLE=0 TRAINER_SAVE_FREQ=-1 VERL_SKIP_FINAL_ACTOR_UPDATE_WHEN_FROZEN=1
export VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1 VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1
export VERL_SIDECAR_ENABLE=0 ADAFLOOR_REVISION_EXPERT_AUDIT=0 ADAFLOOR_REVISION_WEIGHT_TIMING=0
export VLLM_ASCEND_MODE1_UPDATE_WEIGHTS_DIAG=0 ADAFLOOR_REVISION_COMPLETION_AUDIT=1
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896
export ROLLOUT_ENFORCE_EAGER=True TASK_QUEUE_ENABLE=2 VLLM_ENABLE_GRAPH_MODE=0
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4 VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE=4
export VLLM_ASCEND_SHRINK_AWARE_ENABLE=0
sha256sum "$0" "$REPO/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh" \
    "$REPO/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py" > "$ROOT/code_sha256.txt"
printf 'actor_frozen=true\nmodes=1,4\nfloor=4\nsteps=1\nignore_eos=true\nexpected_generated_tokens=225280\nTQ=2\ngraph=false\n' > "$ROOT/protocol.env"
for mode in 1 4; do
    export VLLM_ASCEND_ELASTIC_EXECUTION_MODE="$mode" OUTPUT_ROOT="$ROOT/mode$mode" OUTPUT_SUBDIR=child
    mkdir -p "$OUTPUT_ROOT"
    set +e
    bash "$REPO/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh" \
        trainer.resume_mode=resume_path trainer.resume_from_path="$COMMON/checkpoints/qwen3moe_for_eagle3/global_step_5" \
        trainer.total_training_steps=1 trainer.total_epochs=1 actor_rollout_ref.actor.optim.lr=0.0 \
        actor_rollout_ref.rollout.seed=707 actor_rollout_ref.rollout.ignore_eos=True \
        data.max_response_length=1024 actor_rollout_ref.rollout.response_length=1024 \
        actor_rollout_ref.rollout.max_model_len=2048 actor_rollout_ref.rollout.max_num_batched_tokens=5120 \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=1040 \
        trainer.rollout_data_dir="$OUTPUT_ROOT/rollout_data" \
        > "$OUTPUT_ROOT/launch.log" 2>&1
    rc=$?
    set -e
    printf 'mode%s_exit_code=%s\n' "$mode" "$rc" >> "$ROOT/protocol.env"
    [[ "$rc" == 0 ]] || { echo "mechanism arm failed mode=$mode; evidence retained" >&2; exit "$rc"; }
done
echo 'MECHANISM_ARMS_FINISHED_REQUIRES_TOKEN_AND_TRANSITION_AUDIT'
