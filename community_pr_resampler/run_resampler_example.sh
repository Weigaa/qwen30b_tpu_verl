#!/usr/bin/env bash
set -ex

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

# Resampler controls.
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=${VLLM_EPOCH_LENGTH_REGROUP_ENABLE:-1}
export VLLM_EPOCH_LENGTH_REGROUP_BUCKET_SIZE=${VLLM_EPOCH_LENGTH_REGROUP_BUCKET_SIZE:-1024}
export VLLM_EPOCH_LENGTH_REGROUP_EMA_DECAY=${VLLM_EPOCH_LENGTH_REGROUP_EMA_DECAY:-0.7}
export VLLM_EPOCH_LENGTH_REGROUP_SHUFFLE_BLOCKS=${VLLM_EPOCH_LENGTH_REGROUP_SHUFFLE_BLOCKS:-1}

# Optional long-tail guard.
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=${VLLM_ROLLOUT_EARLY_STOP_ENABLE:-1}
export VLLM_ROLLOUT_EARLY_STOP_FACTOR=${VLLM_ROLLOUT_EARLY_STOP_FACTOR:-2.0}
export VLLM_ROLLOUT_EARLY_STOP_MIN_TOKENS=${VLLM_ROLLOUT_EARLY_STOP_MIN_TOKENS:-10000}

MODEL_PATH=${MODEL_PATH:-"/path/to/model"}
TRAIN_FILE=${TRAIN_FILE:-"/path/to/train.parquet"}
TEST_FILE=${TEST_FILE:-"/path/to/test.parquet"}
CONFIG_DIR=${CONFIG_DIR:-"${ROOT_DIR}/verl/trainer/config"}

SAMPLER_CLASS_PATH=null
SAMPLER_CLASS_NAME=null
SAMPLER_EXTRA_ARGS=""

if [ "${VLLM_EPOCH_LENGTH_REGROUP_ENABLE}" = "1" ]; then
    SAMPLER_CLASS_PATH=pkg://verl.experimental.dataset.length_bucket_sampler
    SAMPLER_CLASS_NAME=LengthAwareEpochSampler
    SAMPLER_EXTRA_ARGS="+data.sampler.bucket_size=${VLLM_EPOCH_LENGTH_REGROUP_BUCKET_SIZE} +data.sampler.ema_decay=${VLLM_EPOCH_LENGTH_REGROUP_EMA_DECAY} +data.sampler.shuffle_batch_blocks=${VLLM_EPOCH_LENGTH_REGROUP_SHUFFLE_BLOCKS}"
fi

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_DIR}" \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    data.dataloader_num_workers=0 \
    data.sampler.class_path=${SAMPLER_CLASS_PATH} \
    data.sampler.class_name=${SAMPLER_CLASS_NAME} \
    ${SAMPLER_EXTRA_ARGS} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.ignore_eos=False \
    trainer.logger=['console','tensorboard'] \
    trainer.total_epochs=3 \
    "$@"
