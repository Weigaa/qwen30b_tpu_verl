#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_rollout_decode_batchsize_benchmark.sh [extra hydra overrides...]

VERL-flow rollout-only benchmark. This script reuses the mode0/no-shrink
GRPO launcher path and stops inside RayPPOTrainer.fit() after a limited
number of rollout generate_sequences() calls.

Defaults:
  BATCH_SIZES="16 32 64 128 256 512 1024 2048 4096 8192"
  ROLLOUT_N=16
  DECODE_TOKENS=32
  MAX_PROMPT_LENGTH=1024
  ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
  VERL_ROLLOUT_BENCH_WARMUP_STEPS=1
  VERL_ROLLOUT_BENCH_MEASURE_STEPS=3
  DATASET_FRACTION=1.0
  OUTPUT_ROOT=<repo>/rollout_decode_batchsize_benchmarks
  OUTPUT_SUBDIR=<timestamped run>

Examples:
  BATCH_SIZES="16 32 64" VERL_ROLLOUT_BENCH_MEASURE_STEPS=5 ./run_rollout_decode_batchsize_benchmark.sh
  ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 MAX_PROMPT_LENGTH=512 ./run_rollout_decode_batchsize_benchmark.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"

BATCH_SIZES="${BATCH_SIZES:-16 32 64 128 256 512 1024 2048 4096 8192}"
ROLLOUT_N="${ROLLOUT_N:-16}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
VERL_ROLLOUT_BENCH_WARMUP_STEPS="${VERL_ROLLOUT_BENCH_WARMUP_STEPS:-1}"
VERL_ROLLOUT_BENCH_MEASURE_STEPS="${VERL_ROLLOUT_BENCH_MEASURE_STEPS:-3}"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
export OUTPUT_ROOT="${OUTPUT_ROOT:-${script_dir}/rollout_decode_batchsize_benchmarks}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-${stamp}_verl_rollout_decode_bs_sweep}"
export VERL_ROLLOUT_BENCH_OUTPUT_DIR="${VERL_ROLLOUT_BENCH_OUTPUT_DIR:-${OUTPUT_ROOT}/${OUTPUT_SUBDIR}}"
export VERL_ROLLOUT_BENCH_CSV="${VERL_ROLLOUT_BENCH_CSV:-${VERL_ROLLOUT_BENCH_OUTPUT_DIR}/summary.csv}"
LOG_DIR="${VERL_ROLLOUT_BENCH_OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

export VERL_ROLLOUT_BENCHMARK_ONLY=1
export VERL_ROLLOUT_BENCH_WARMUP_STEPS
export VERL_ROLLOUT_BENCH_MEASURE_STEPS
export MAX_RESPONSE_LENGTH="${DECODE_TOKENS}"
export DATASET_FRACTION="${DATASET_FRACTION:-1.0}"
export DATA_SHUFFLE="${DATA_SHUFFLE:-False}"
export TRAINER_TOTAL_EPOCHS=1
export TRAINER_LOGGER="${TRAINER_LOGGER:-['console']}"
export MODE0_SAVE_ROLLOUT_ARTIFACTS=0
export SAVE_CKPT_ENABLE=0
export SAVE_DRAFT_HIDDEN_ENABLE=0
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=0
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=0
export VLLM_EAGER_BASELINE_NO_RESAMPLE=1
export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=0
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=0
export VLLM_ASCEND_SHRINK_AWARE_ENABLE=0
export ROLLOUT_N
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"

echo "[rollout decode bench] verl-flow output_dir=${VERL_ROLLOUT_BENCH_OUTPUT_DIR}"
echo "[rollout decode bench] summary_csv=${VERL_ROLLOUT_BENCH_CSV}"
echo "[rollout decode bench] batch_sizes=${BATCH_SIZES} n=${ROLLOUT_N} prompt_len=${MAX_PROMPT_LENGTH} decode_tokens=${DECODE_TOKENS}"
echo "[rollout decode bench] warmup_steps=${VERL_ROLLOUT_BENCH_WARMUP_STEPS} measure_steps=${VERL_ROLLOUT_BENCH_MEASURE_STEPS}"

for batch_size in ${BATCH_SIZES}; do
    export TRAIN_BATCH_SIZE="${batch_size}"
    export ROLLOUT_MAX_NUM_SEQS="${batch_size}"
    export VERL_ROLLOUT_BENCH_BATCH_SIZE="${batch_size}"

    log_file="${LOG_DIR}/batch_${batch_size}.log"
    echo "[rollout decode bench] start batch_size=${batch_size} max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} log=${log_file}"
    set +e
    ./run_mode0_no_shrink_baseline.sh \
        actor_rollout_ref.actor.use_kl_loss=False \
        algorithm.use_kl_in_reward=False \
        "$@" 2>&1 | tee "${log_file}"
    rc=${PIPESTATUS[0]}
    set -e
    if [[ "${rc}" -ne 0 ]]; then
        echo "[rollout decode bench] batch_size=${batch_size} failed rc=${rc}; continuing sweep"
    fi
done

echo "[rollout decode bench] done"
echo "[rollout decode bench] summary_csv=${VERL_ROLLOUT_BENCH_CSV}"
