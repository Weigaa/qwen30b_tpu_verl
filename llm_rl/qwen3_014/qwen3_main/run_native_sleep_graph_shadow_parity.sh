#!/usr/bin/env bash
# Diagnostic, not a throughput experiment: first FULL decode forwards are
# compared with uncaptured forwards at identical inputs, weights and KV state.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
PARITY_MODE=${1:-graph}
case "$PARITY_MODE" in
    graph) PARITY_CUDAGRAPH_MODE=FULL_DECODE_ONLY; PARITY_EAGER=False ;;
    eager-reference) PARITY_CUDAGRAPH_MODE=NONE; PARITY_EAGER=True ;;
    *) echo "usage: $0 [graph|eager-reference]" >&2; exit 2 ;;
esac
export OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/adafloor_graph_results}
export OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-qwen3_sleep_${PARITY_MODE}_shadow_$(date -u +%Y%m%dT%H%M%SZ)}
export TRAINER_TOTAL_EPOCHS=1 TRAINER_TOTAL_TRAINING_STEPS=1
export TRAIN_BATCH_SIZE=32 ROLLOUT_N=16 MAX_RESPONSE_LENGTH=128
export ROLLOUT_MAX_NUM_SEQS=32 ROLLOUT_MAX_NUM_BATCHED_TOKENS=17408
export DATA_SHUFFLE=False DATASET_FRACTION=0.005
export ROLLOUT_ENFORCE_EAGER=$PARITY_EAGER
export VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1 TASK_QUEUE_ENABLE=1
export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
export VLLM_ROLLOUT_SLEEP_LEVEL=${VLLM_ROLLOUT_SLEEP_LEVEL:-2}
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
export VLLM_ROLLOUT_VERIFY_WEIGHT_COVERAGE=1
export VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
export VLLM_ASCEND_FULL_GRAPH_PARITY_STEPS=4
export VLLM_ASCEND_RUNTIME_PROGRESS_INTERVAL=32
export VLLM_ROLLOUT_ASYNC_SCHEDULING=false
export VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC=1
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=20480 ACTOR_RECOMPUTE_NUM_LAYERS=13
export VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1
export SAVE_CKPT_ENABLE=0 GIT_TRACK_RUN_LOGS=0
export VERL_LOGGING_LEVEL=INFO
export RAY_TMPDIR=/tmp/qwen3_graph_shadow_$$
COMMON_CKPT=/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5
exec bash run_mode0_no_shrink_baseline.sh \
    trainer.resume_mode=resume_path "trainer.resume_from_path=$COMMON_CKPT" \
    trainer.total_training_steps=1 trainer.critic_warmup=999999 \
    "actor_rollout_ref.rollout.cudagraph_mode=$PARITY_CUDAGRAPH_MODE" \
    'actor_rollout_ref.rollout.cudagraph_capture_sizes=[32]' \
    actor_rollout_ref.rollout.async_scheduling=false \
    actor_rollout_ref.rollout.seed=0 actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.rollout.ignore_eos=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=2761 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
