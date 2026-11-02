#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_mode1_mode2_fair_perf_compare.sh
  MODES=2 ./run_mode1_mode2_fair_perf_compare.sh
  MODES=1,2,3,4,5 ./run_mode1_mode2_fair_perf_compare.sh

This is a one-step, configuration-matched expert-availability comparison. All arms
use the same floor, prompts, sampling configuration, response caps, and fixed
KV block count. The script does not reuse mode-specific auto-sized KV caches.

Environment overrides:
  MODES=1,2                         Arms to run. Valid modes are 1 through 5.
  COMPARE_FLOOR=4                  Elastic floor. Recommended: 4.
  COMMON_KV_TOKENS_PER_RANK=277120 Fixed per-rank KV token capacity.
  VLLM_KV_BLOCK_SIZE=128           KV tokens per block.
  MODE2_RESIDENT_EXPERT_SLOTS=8    Fixed mode2 NPU resident slots per rank.
  OUTPUT_ROOT=/path/to/output      Parent directory for the comparison.
  COMMON_EPOCH0_ROOT=/path         Preserved checkpoint and rollout history.
  COMPARE_SEED=808                 Per-request sampling seed base.
  CLEAN_RAY_BETWEEN_RUNS=1         Stop Ray before each arm.

The default workload is inherited from the existing perf-clean experiments:
32 prompts, rollout.n=16, max response length 16384, and one training step.
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER="$SCRIPT_DIR/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -gt 0 ]]; then
    echo "unknown argument: $1" >&2
    usage >&2
    exit 2
fi

MODES="${MODES:-1,2}"
COMPARE_FLOOR="${COMPARE_FLOOR:-4}"
COMMON_KV_TOKENS_PER_RANK="${COMMON_KV_TOKENS_PER_RANK:-277120}"
VLLM_KV_BLOCK_SIZE="${VLLM_KV_BLOCK_SIZE:-128}"
MODE2_RESIDENT_EXPERT_SLOTS="${MODE2_RESIDENT_EXPERT_SLOTS:-8}"
CLEAN_RAY_BETWEEN_RUNS="${CLEAN_RAY_BETWEEN_RUNS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/adafloor_shared_state/mode1_mode2_fair_compare_$(date -u +%Y%m%dT%H%M%SZ)}"
COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
COMPARE_SEED="${COMPARE_SEED:-808}"
REUSE_ENV="$COMMON_EPOCH0_ROOT/reuse.env"

case "$COMPARE_FLOOR" in
    2|4|8) ;;
    *)
        echo "COMPARE_FLOOR must be one of 2, 4, or 8" >&2
        exit 2
        ;;
esac
if (( COMMON_KV_TOKENS_PER_RANK <= 0 || VLLM_KV_BLOCK_SIZE <= 0 )); then
    echo "KV token capacity and block size must be positive" >&2
    exit 2
fi
if (( COMMON_KV_TOKENS_PER_RANK % VLLM_KV_BLOCK_SIZE != 0 )); then
    echo "COMMON_KV_TOKENS_PER_RANK must be divisible by VLLM_KV_BLOCK_SIZE" >&2
    exit 2
fi
if (( MODE2_RESIDENT_EXPERT_SLOTS <= 0 )); then
    echo "MODE2_RESIDENT_EXPERT_SLOTS must be positive" >&2
    exit 2
fi
if ! [[ "$COMPARE_SEED" =~ ^[0-9]+$ ]]; then
    echo "COMPARE_SEED must be a nonnegative integer" >&2
    exit 2
fi
if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
      || ! -f "$REUSE_ENV" ]]; then
    echo "preserved common epoch0 is incomplete: $COMMON_EPOCH0_ROOT" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$REUSE_ENV"
if [[ ! -d "$DYNAMIC_INITIAL_RESUME_CKPT/actor" \
      || ! -f "$DYNAMIC_INITIAL_RESUME_CKPT/.PRESERVE_COMMON_EPOCH0" ]]; then
    echo "reuse.env does not reference the protected epoch0 checkpoint" >&2
    exit 2
fi
if [[ ! -x "$LAUNCHER" ]]; then
    echo "launcher is missing or not executable: $LAUNCHER" >&2
    exit 2
fi

IFS=',' read -r -a mode_list <<< "$MODES"
for mode in "${mode_list[@]}"; do
    if [[ ! "$mode" =~ ^[1-5]$ ]]; then
        echo "MODES accepts integer modes 1 through 5, got: $mode" >&2
        exit 2
    fi
done

COMMON_KV_BLOCKS=$((COMMON_KV_TOKENS_PER_RANK / VLLM_KV_BLOCK_SIZE))
mkdir -p "$OUTPUT_ROOT"
SUMMARY_CSV="$OUTPUT_ROOT/summary.csv"
SUMMARY_MD="$OUTPUT_ROOT/summary.md"
printf 'mode,floor,kv_tokens_per_rank,kv_blocks,resident_slots,rollout_output_time_s,response_length_mean,reward_mean,reported_kv_tokens,preemption_count,oom_count,exit_code,log\n' > "$SUMMARY_CSV"

cat > "$OUTPUT_ROOT/run_config.env" <<EOF
MODES=$MODES
COMPARE_FLOOR=$COMPARE_FLOOR
COMMON_KV_TOKENS_PER_RANK=$COMMON_KV_TOKENS_PER_RANK
VLLM_KV_BLOCK_SIZE=$VLLM_KV_BLOCK_SIZE
COMMON_KV_BLOCKS=$COMMON_KV_BLOCKS
MODE2_RESIDENT_EXPERT_SLOTS=$MODE2_RESIDENT_EXPERT_SLOTS
COMMON_EPOCH0_ROOT=$COMMON_EPOCH0_ROOT
COMMON_EPOCH0_CHECKPOINT=$DYNAMIC_INITIAL_RESUME_CKPT
COMPARE_SEED=$COMPARE_SEED
ACTOR_LEARNING_RATE=0.0
PER_REQUEST_SAMPLING_SEEDS=1
TRAIN_BATCH_SIZE=32
ROLLOUT_N=16
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=16384
ROLLOUT_MAX_NUM_SEQS=32
ROLLOUT_GPU_MEMORY_UTILIZATION=0.9
VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896
EOF

echo "[mode1/mode2 fair compare] output=$OUTPUT_ROOT"
echo "[mode1/mode2 fair compare] modes=$MODES floor=$COMPARE_FLOOR"
echo "[mode1/mode2 fair compare] kv_tokens_per_rank=$COMMON_KV_TOKENS_PER_RANK blocks=$COMMON_KV_BLOCKS block_size=$VLLM_KV_BLOCK_SIZE"
echo "[mode1/mode2 fair compare] mode2_resident_slots=$MODE2_RESIDENT_EXPERT_SLOTS"

run_arm() {
    local mode="$1"
    shift
    local arm_dir="$OUTPUT_ROOT/mode${mode}_floor${COMPARE_FLOOR}"
    local wrapper_log="$arm_dir/launcher.log"
    local hccl_port=$((47641 + mode * 100))
    local master_port=$((26640 + mode * 100))

    mkdir -p "$arm_dir/record"
    if [[ "$CLEAN_RAY_BETWEEN_RUNS" == "1" ]] && command -v ray >/dev/null 2>&1; then
        ray stop --force >/dev/null 2>&1 || true
    fi

    echo "[mode1/mode2 fair compare] starting mode=$mode arm=$arm_dir"
    set +e
    (
        cd "$arm_dir"
        env \
            HOME="$arm_dir" \
            CONFIG_DIR="$SCRIPT_DIR/verl/trainer/config" \
            PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            RECORD_DIR="$arm_dir/record" \
            VLLM_ASCEND_ELASTIC_EXECUTION_MODE="$mode" \
            VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="$COMPARE_FLOOR" \
            VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS="$MODE2_RESIDENT_EXPERT_SLOTS" \
            VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=0 \
            VLLM_ASCEND_CUSTOM_MODE1_DEBUG=0 \
            VLLM_ASCEND_CUSTOM_MODE1_TIMING_EVENTS=0 \
            VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG=0 \
            VLLM_ASCEND_MODE3_TRANSFER_LOG=0 \
            VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG=0 \
            VLLM_ASCEND_MODE3_TIMING_LOG=0 \
            VLLM_ASCEND_MODE3_TIMING_SYNC=0 \
            VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS=0 \
            VLLM_ASCEND_BUCKET_OP_PROFILE=0 \
            VLLM_ASCEND_DUMMY_WASTE_TIMING=0 \
            VERL_SIDECAR_ENABLE=0 \
            VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
            TRAINER_TOTAL_EPOCHS=1 \
            TRAIN_BATCH_SIZE=32 \
            MAX_PROMPT_LENGTH=1024 \
            MAX_RESPONSE_LENGTH=16384 \
            ROLLOUT_N=16 \
            ROLLOUT_MAX_NUM_SEQS=32 \
            ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 \
            VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896 \
            HCCL_IF_BASE_PORT="$hccl_port" \
            MASTER_PORT="$master_port" \
            VERL_HCCL_IF_BASE_PORT_START="$hccl_port" \
            "$LAUNCHER" \
            trainer.resume_mode=resume_path \
            "trainer.resume_from_path=$DYNAMIC_INITIAL_RESUME_CKPT" \
            actor_rollout_ref.actor.optim.lr=0.0 \
            "actor_rollout_ref.rollout.seed=$COMPARE_SEED" \
            data.shuffle=False \
            +actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override="$COMMON_KV_BLOCKS" \
            "$@"
    ) 2>&1 | tee "$wrapper_log"
    local run_rc=${PIPESTATUS[0]}
    set -e

    local run_log
    run_log=$(find "$arm_dir" -maxdepth 1 -type f -name 'wjeagerqwen30b-a3b-with_draft_*.txt' -printf '%T@ %p\n' \
        | sort -nr | head -n 1 | cut -d' ' -f2-)
    local rollout_time=""
    local response_length_mean=""
    local reward_mean=""
    local reported_kv=""
    local preemption_count=0
    local oom_count=0
    local logged_rc="$run_rc"
    if [[ -n "$run_log" && -f "$run_log" ]]; then
        rollout_time=$(sed -n 's/.*rollout_output_time_s: \([0-9.][0-9.]*\).*/\1/p' "$run_log" | tail -n 1)
        response_length_mean=$(sed -n 's/.*response_length\/mean:\([^ ]*\).*/\1/p' "$run_log" | tail -n 1)
        reward_mean=$(sed -n 's/.*critic\/rewards\/mean:\([^ ]*\).*/\1/p' "$run_log" | tail -n 1)
        reported_kv=$(sed -n 's/.*GPU KV cache size: \([0-9,][0-9,]*\) tokens.*/\1/p' "$run_log" | tail -n 1 | tr -d ',')
        preemption_count=$(grep -Eic 'preempting request|request preempted' "$run_log" || true)
        oom_count=$(grep -Eic 'NPU out of memory|Memory_Allocation_Failure|Failed to allocate.*NPU memory' "$run_log" || true)
        logged_rc=$(sed -n 's/.*\[run\].*exit_code=\([0-9][0-9]*\).*/\1/p' "$run_log" | tail -n 1)
        logged_rc="${logged_rc:-$run_rc}"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$mode" "$COMPARE_FLOOR" "$COMMON_KV_TOKENS_PER_RANK" "$COMMON_KV_BLOCKS" \
        "$MODE2_RESIDENT_EXPERT_SLOTS" "$rollout_time" "$response_length_mean" "$reward_mean" \
        "$reported_kv" "$preemption_count" "$oom_count" "$logged_rc" "$run_log" \
        >> "$SUMMARY_CSV"
    echo "[mode1/mode2 fair compare] finished mode=$mode rc=$logged_rc rollout_s=${rollout_time:-NA} response_mean=${response_length_mean:-NA} preemptions=$preemption_count oom=$oom_count"
    if [[ "$logged_rc" != "0" || -z "$rollout_time" || -z "$response_length_mean" \
          || "$preemption_count" != "0" || "$oom_count" != "0" ]]; then
        echo "mode=$mode did not pass the fair-comparison validation" >&2
        return 4
    fi
    if grep -qE 'response/aborted_ratio:(0\.[0-9]*[1-9]|[1-9])' "$run_log"; then
        echo "mode=$mode produced aborted responses" >&2
        return 4
    fi
}

for mode in "${mode_list[@]}"; do
    run_arm "$mode"
done

{
    echo '# Mode1 vs Mode2 Fair Performance Comparison'
    echo
    echo "Common floor: \`$COMPARE_FLOOR\`"
    echo
    echo "Common KV capacity: \`$COMMON_KV_TOKENS_PER_RANK\` tokens per rank, \`$COMMON_KV_BLOCKS\` blocks"
    echo
    echo '| Mode | Floor | Rollout time (s) | Response mean | Reward mean | KV tokens | Preemptions | OOM | Exit |'
    echo '|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
    tail -n +2 "$SUMMARY_CSV" | while IFS=',' read -r mode floor _tokens _blocks _slots rollout_time response_mean reward_mean reported_kv preemptions oom exit_code _log; do
        echo "| $mode | $floor | ${rollout_time:-NA} | ${response_mean:-NA} | ${reward_mean:-NA} | ${reported_kv:-NA} | $preemptions | $oom | $exit_code |"
    done
    echo
    echo 'Use this as paper evidence only if both arms complete without OOM, preemption, or output-quality errors.'
} > "$SUMMARY_MD"

echo "[mode1/mode2 fair compare] summary=$SUMMARY_MD"
