#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# DeepSeek-V2-Lite is a 27-layer MLA MoE with 64 routed experts and two
# shared experts. AdaFloor redistributes only the routed FusedMoE weights.
export MODEL_PATH=${MODEL_PATH:-/data/DeepSeek-V2-Lite-Chat}
export MODEL_REVISION=${MODEL_REVISION:-85864749cd611b4353ce1decdb286193298f64c7}
export DISTCP_PATH=${DISTCP_PATH:-/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4}
export TRAIN_FILE=${TRAIN_FILE:-/data/deepscaler/train.parquet}
export TEST_FILE=${TEST_FILE:-/data/deepscaler/test.parquet}

export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=${VLLM_ASCEND_REGISTER_CUSTOM_MODELS:-1}

require_topology_value() {
    local name=$1
    local expected=$2
    local actual=${!name:-$expected}
    if [[ "$actual" != "$expected" ]]; then
        echo "[DeepSeek-V2-Lite] $name must be $expected for the EP16 experiments, got $actual" >&2
        exit 2
    fi
    printf -v "$name" '%s' "$expected"
    export "$name"
}

# Keep this launcher fail-closed against variables left by the retired DP2EP8
# experiments. The rollout world is one 16-rank expert-parallel instance.
require_topology_value HIERARCHICAL_DP2_EP8 0
require_topology_value VLLM_DP_SIZE 16
require_topology_value ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE 1

# Keep HCCL listeners below the host's ephemeral TCP range. Four isolated
# 4096-port phase windows begin at this base.
export HCCL_IF_BASE_PORT=${DEEPSEEK_HCCL_IF_BASE_PORT:-12000}
export VERL_HCCL_IF_BASE_PORT_START=$HCCL_IF_BASE_PORT
export MASTER_PORT=${DEEPSEEK_MASTER_PORT:-30000}
export VERL_MASTER_PORT_START=$MASTER_PORT
export VERL_MEGATRON_EAGER_WEIGHT_SYNC_GROUP_INIT=${VERL_MEGATRON_EAGER_WEIGHT_SYNC_GROUP_INIT:-1}

# The training world is PP4 x EP4. TP remains one because MLA and the routed
# experts are already partitioned by PP and EP on the 16-rank node.
require_topology_value MCORE_SEQUENCE_PARALLEL False
require_topology_value MCORE_TENSOR_MODEL_PARALLEL_SIZE 1
require_topology_value MCORE_PIPELINE_MODEL_PARALLEL_SIZE 4
require_topology_value MCORE_EXPERT_MODEL_PARALLEL_SIZE 4
require_topology_value MCORE_EXPERT_TENSOR_PARALLEL_SIZE 1
export MCORE_PIPELINE_NUM_TRANSFORMER_LAYERS=${MCORE_PIPELINE_NUM_TRANSFORMER_LAYERS:-'[[6],[7],[7],[7]]'}
export MCORE_FIRST_PIPELINE_NUM_LAYERS=${MCORE_FIRST_PIPELINE_NUM_LAYERS:-6}
export MCORE_LAST_PIPELINE_NUM_LAYERS=${MCORE_LAST_PIPELINE_NUM_LAYERS:-7}
export TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-deepseek_v2_lite_adafloor}
export CHECKPOINT_MODEL_DIR_NAME=${CHECKPOINT_MODEL_DIR_NAME:-deepseek_v2_lite}
export TRAIN_LOG_PREFIX=${TRAIN_LOG_PREFIX:-deepseek-v2-lite-adafloor}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-17408}
export ROLLOUT_N=${ROLLOUT_N:-16}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.9}

# A single DeepSeek request contains at most 1024 prompt and 16384 response
# tokens. Keeping dynamic training batches at that bound avoids combining a
# maximum-length request with extra sequences on the output pipeline stage.
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-17408}
export ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-17408}
TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
if [[ "$TASK_QUEUE_ENABLE" != 1 && "$TASK_QUEUE_ENABLE" != 2 ]]; then
    echo "[DeepSeek-V2-Lite] TASK_QUEUE_ENABLE must be 1 or 2" >&2
    exit 2
fi
export TASK_QUEUE_ENABLE

normalize_bool() {
    case "${1,,}" in
        1|true|yes|on) printf '%s' True ;;
        0|false|no|off) printf '%s' False ;;
        *) return 1 ;;
    esac
}

if ! MCORE_MOE_ALLTOALL_OVERLAP_COMM=$(normalize_bool "${DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM:-True}"); then
    echo "[DeepSeek-V2-Lite] invalid DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM" >&2
    exit 2
fi
if ! MCORE_MOE_SHARED_EXPERT_OVERLAP=$(normalize_bool "${DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP:-True}"); then
    echo "[DeepSeek-V2-Lite] invalid DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP" >&2
    exit 2
fi
if ! MCORE_DEALLOCATE_PIPELINE_OUTPUTS=$(normalize_bool "${DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS:-False}"); then
    echo "[DeepSeek-V2-Lite] invalid DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS" >&2
    exit 2
fi
if [[ "$MCORE_MOE_ALLTOALL_OVERLAP_COMM" == True \
      && "$MCORE_MOE_SHARED_EXPERT_OVERLAP" != True ]]; then
    echo "[DeepSeek-V2-Lite] shared expert overlap must remain enabled with all-to-all overlap" >&2
    exit 2
fi
export MCORE_MOE_ALLTOALL_OVERLAP_COMM
export MCORE_MOE_SHARED_EXPERT_OVERLAP
export MCORE_DEALLOCATE_PIPELINE_OUTPUTS

DEEPSEEK_ACTOR_RECOMPUTE_METHOD=${DEEPSEEK_ACTOR_RECOMPUTE_METHOD:-uniform}
DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS=${DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS:-1}
if [[ "$DEEPSEEK_ACTOR_RECOMPUTE_METHOD" != uniform \
      || "$DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS" != 1 ]]; then
    echo "[DeepSeek-V2-Lite] actor recompute must be full/uniform/1 for comparable EP16 runs" >&2
    exit 2
fi

for required_path in "$MODEL_PATH" "$DISTCP_PATH" "$TRAIN_FILE" "$TEST_FILE"; do
    if [[ ! -e "$required_path" ]]; then
        echo "[DeepSeek-V2-Lite] missing required input: $required_path" >&2
        echo "Run prepare_deepseek_v2_lite_assets.sh before launching an experiment." >&2
        exit 2
    fi
done

python3 "$REPO_ROOT/tools/validate_deepseek_v2_lite_assets.py" \
    --model-path "$MODEL_PATH" \
    --distcp-path "$DISTCP_PATH" \
    --expected-revision "$MODEL_REVISION" \
    --expected-pp-size 4 \
    --expected-ep-size 4

echo "[DeepSeek-V2-Lite] training_weight_source=megatron_distcp path=$DISTCP_PATH pp=4 ep=4 revision=$MODEL_REVISION"
echo "[DeepSeek-V2-Lite] task_queue=$TASK_QUEUE_ENABLE recompute=$DEEPSEEK_ACTOR_RECOMPUTE_METHOD/$DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS moe_overlap=$MCORE_MOE_ALLTOALL_OVERLAP_COMM/$MCORE_MOE_SHARED_EXPERT_OVERLAP deallocate_pipeline_outputs=$MCORE_DEALLOCATE_PIPELINE_OUTPUTS"

exec "$SCRIPT_DIR/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.multi_head_latent_attention=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.rope_scaling_type=yarn \
    +actor_rollout_ref.actor.megatron.override_transformer_config.yarn_scaling_factor=40 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.rope_scaling_mscale=0.707 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.rope_scaling_mscale_all_dim=0.707 \
    "$@" \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method="$DEEPSEEK_ACTOR_RECOMPUTE_METHOD" \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers="$DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS" \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="$DISTCP_PATH" \
    actor_rollout_ref.ref.load_weight=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="$DISTCP_PATH"
