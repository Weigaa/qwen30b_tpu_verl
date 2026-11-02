#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Configuration-matched Vanilla baseline for the final AdaFloor evaluation.
# It runs from the base checkpoint for three consecutive epochs (15 steps).
# For training-age-aligned comparisons, use:
#   Vanilla log epoch 0 (steps 1-5)  <-> AdaFloor epoch_001
#   Vanilla log epoch 1 (steps 6-10) <-> AdaFloor epoch_002
# Vanilla log epoch 2 is retained as an additional baseline epoch.

OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-mode0_no_shrink_baseline_full3_gpu_memory_09}"
RECORD_DIR="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}"
VANILLA_KV_TOKENS_PER_RANK="${VANILLA_KV_TOKENS_PER_RANK:-380800}"
VLLM_KV_BLOCK_SIZE="${VLLM_KV_BLOCK_SIZE:-128}"
SAVE_CKPT_ENABLE="${SAVE_CKPT_ENABLE:-0}"

if [[ "$SAVE_CKPT_ENABLE" == "1" ]]; then
    TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-15}"
else
    TRAINER_SAVE_FREQ=-1
fi

if (( VANILLA_KV_TOKENS_PER_RANK <= 0 || VLLM_KV_BLOCK_SIZE <= 0 )); then
    echo "KV token capacity and block size must both be positive" >&2
    exit 2
fi
if (( VANILLA_KV_TOKENS_PER_RANK % VLLM_KV_BLOCK_SIZE != 0 )); then
    echo "VANILLA_KV_TOKENS_PER_RANK must be divisible by VLLM_KV_BLOCK_SIZE" >&2
    exit 2
fi
VANILLA_KV_BLOCKS=$((VANILLA_KV_TOKENS_PER_RANK / VLLM_KV_BLOCK_SIZE))

if [[ -e "$RECORD_DIR" && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
    echo "output already exists: $RECORD_DIR" >&2
    echo "Set OUTPUT_SUBDIR to a new directory, or ALLOW_EXISTING_OUTPUT=1 explicitly." >&2
    exit 2
fi

echo "[mode0 full3 gpu-memory-09] output=$RECORD_DIR"
echo "[mode0 full3 gpu-memory-09] epochs=3 steps_per_epoch=5 total_steps=15"
echo "[mode0 full3 gpu-memory-09] rollout_gpu_memory_utilization=0.9"
echo "[mode0 full3 gpu-memory-09] kv_tokens_per_rank=$VANILLA_KV_TOKENS_PER_RANK block_size=$VLLM_KV_BLOCK_SIZE num_gpu_blocks_override=$VANILLA_KV_BLOCKS"
echo "[mode0 full3 gpu-memory-09] save_checkpoint=$SAVE_CKPT_ENABLE trainer_save_freq=$TRAINER_SAVE_FREQ"
echo "[mode0 full3 gpu-memory-09] comparison: vanilla_epoch0->AdaFloor_epoch001 vanilla_epoch1->AdaFloor_epoch002"

exec env \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    OUTPUT_SUBDIR="$OUTPUT_SUBDIR" \
    RECORD_DIR="$RECORD_DIR" \
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 \
    SAVE_CKPT_ENABLE="$SAVE_CKPT_ENABLE" \
    TRAINER_SAVE_FREQ="$TRAINER_SAVE_FREQ" \
    "$SCRIPT_DIR/run_mode0_no_shrink_baseline_full3.sh" \
    trainer.resume_mode=disable \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override="$VANILLA_KV_BLOCKS" \
    "$@"
