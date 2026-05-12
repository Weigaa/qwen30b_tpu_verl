#!/usr/bin/env bash
# Eager-control run for the ziyi-aligned graph+resampler diagnosis.
#
# This keeps the same high-level rollout shape as the qwen3 true-graph probe:
# TP=4, runtime EP, batch=64, n=8, max_response=8192, top_k=-1, prefix cache on.
# The only intentional semantic switch is disabling true graph capture and
# forcing eager rollout so length distributions can be compared prompt-by-prompt.
set -ex

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="${SCRIPT_DIR}/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_async_oldcfg.sh"

export VLLM_ROLLOUT_PARALLEL_MODE=${VLLM_ROLLOUT_PARALLEL_MODE:-tp_ep}
export VLLM_ROLLOUT_ZIYI_ALIGN=${VLLM_ROLLOUT_ZIYI_ALIGN:-1}
export VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER=0
export VLLM_ROLLOUT_UNSAFE_TRUE_GRAPH_WITH_RESAMPLER=0

# Match the ziyi-aligned graph diagnostic batch shape and sampling knobs.
export VLLM_ROLLOUT_FAST_DEBUG=${VLLM_ROLLOUT_FAST_DEBUG:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-64}
export DATASET_FRACTION=${DATASET_FRACTION:-0.004}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
export ROLLOUT_N=${ROLLOUT_N:-8}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
export ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=${VLLM_ROLLOUT_ENABLE_PREFIX_CACHING:-true}
export VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=${VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL:-true}

# Keep scheduling and memory knobs close to the graph run for a clean
# correctness comparison.  The eager fast path can be re-enabled separately.
export VLLM_ROLLOUT_ASYNC_SCHEDULING=${VLLM_ROLLOUT_ASYNC_SCHEDULING:-false}
export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=${VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.87}
export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=${VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE:-0}
export VLLM_ROLLOUT_SLEEP_LEVEL=${VLLM_ROLLOUT_SLEEP_LEVEL:-1}
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=${VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD:-0}
export VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=${VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS:-0}

# Stop after rollout diagnostics so this can be compared against the graph run
# without paying actor-update memory/time.
export VLLM_ROLLOUT_DEBUG_GENERATION=${VLLM_ROLLOUT_DEBUG_GENERATION:-1}
export VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE=${VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE:-1}
export VLLM_ROLLOUT_DEBUG_GENERATION_SAMPLES=${VLLM_ROLLOUT_DEBUG_GENERATION_SAMPLES:-8}
export VLLM_ROLLOUT_DEBUG_GROUPS=${VLLM_ROLLOUT_DEBUG_GROUPS:-8}
export TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-1}
export TRAINER_TOTAL_TRAINING_STEPS=${TRAINER_TOTAL_TRAINING_STEPS:-1}

# Make the log header explicit even though the base script will also set these
# through the non-graph eager branch.
export VLLM_ENABLE_GRAPH_MODE=0
export ROLLOUT_ENFORCE_EAGER=True
export VLLM_ASCEND_EAGER_COMPILE=${VLLM_ASCEND_EAGER_COMPILE:-1}

exec bash "${BASE_SCRIPT}" "$@"
