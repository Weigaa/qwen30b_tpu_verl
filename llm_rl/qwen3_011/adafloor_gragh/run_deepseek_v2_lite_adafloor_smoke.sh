#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT=${DEEPSEEK_ADAFLOOR_SMOKE_OUTPUT_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite}
TARGET_FLOOR=${DEEPSEEK_ADAFLOOR_SMOKE_FLOOR:-8}
SHRINK_POLICY=${DEEPSEEK_ADAFLOOR_SMOKE_POLICY:-natural}
RUN_NAME=${DEEPSEEK_ADAFLOOR_SMOKE_RUN_NAME:-adafloor_ep16_floor${TARGET_FLOOR}_smoke_$timestamp}
RUN_ROOT="$OUTPUT_ROOT/$RUN_NAME"
BASELINE_DIR=${DEEPSEEK_ADAFLOOR_SMOKE_BASELINE_DIR:-$OUTPUT_ROOT/weight_sync_continuous_2step_20260803T114900Z/epoch_000_mode0_probe}
KV_TOKENS=${DEEPSEEK_ADAFLOOR_SMOKE_KV_TOKENS:-16384}
KV_BLOCK_SIZE=${VLLM_KV_BLOCK_SIZE:-128}
TRAINING_STEPS=${DEEPSEEK_ADAFLOOR_SMOKE_TRAINING_STEPS:-1}
TRAIN_BATCH_SIZE=${DEEPSEEK_ADAFLOOR_SMOKE_TRAIN_BATCH_SIZE:-32}
ROLLOUT_N=${DEEPSEEK_ADAFLOOR_SMOKE_ROLLOUT_N:-1}
MAX_PROMPT_LENGTH=${DEEPSEEK_ADAFLOOR_SMOKE_MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${DEEPSEEK_ADAFLOOR_SMOKE_MAX_RESPONSE_LENGTH:-64}
TAIL_VALIDATE_LEVEL_TOKENS=${DEEPSEEK_ADAFLOOR_SMOKE_TAIL_VALIDATE_LEVEL_TOKENS:-4,16,32,64,64}
TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP=${DEEPSEEK_ADAFLOOR_SMOKE_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP:-}
TASK_QUEUE_LEVEL=${DEEPSEEK_ADAFLOOR_SMOKE_TASK_QUEUE_ENABLE:-1}
RECOMPUTE_METHOD=${DEEPSEEK_ADAFLOOR_SMOKE_RECOMPUTE_METHOD:-uniform}
RECOMPUTE_NUM_LAYERS=${DEEPSEEK_ADAFLOOR_SMOKE_RECOMPUTE_NUM_LAYERS:-1}
MOE_ALLTOALL_OVERLAP=${DEEPSEEK_ADAFLOOR_SMOKE_MOE_ALLTOALL_OVERLAP_COMM:-False}
MOE_SHARED_EXPERT_OVERLAP=${DEEPSEEK_ADAFLOOR_SMOKE_MOE_SHARED_EXPERT_OVERLAP:-False}
DEALLOCATE_PIPELINE_OUTPUTS=${DEEPSEEK_ADAFLOOR_SMOKE_DEALLOCATE_PIPELINE_OUTPUTS:-False}
PLANNED_MIN_FREE_MIB=0

case "$SHRINK_POLICY" in
    natural) ;;
    planned)
        for name in \
            VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE \
            VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS \
            VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS \
            VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE \
            VLLM_ASCEND_MODE1_PARITY_PREFILL_PLANNED_EXPERT_SLOTS \
            VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD \
            VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT; do
            if [[ "${!name:-0}" != 1 ]]; then
                echo "planned DeepSeek AdaFloor smoke requires $name=1 from its runtime profile" >&2
                exit 2
            fi
        done
        if [[ "${VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY:-}" != planned ]]; then
            echo "planned DeepSeek AdaFloor smoke requires the Planned runtime profile" >&2
            exit 2
        fi
        PLANNED_MIN_FREE_MIB=${VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB:-0}
        if ! [[ "$PLANNED_MIN_FREE_MIB" =~ ^[1-9][0-9]*$ ]]; then
            echo "planned DeepSeek AdaFloor smoke requires an explicit positive training HBM guard" >&2
            exit 2
        fi
        ;;
    *)
        echo "DeepSeek AdaFloor smoke policy must be natural or planned" >&2
        exit 2
        ;;
esac
if [[ "${ALLOW_INFEASIBLE_PLAN:-0}" != 0 ]]; then
    echo "ALLOW_INFEASIBLE_PLAN is forbidden for the DeepSeek AdaFloor smoke" >&2
    exit 2
fi
unset ALLOW_INFEASIBLE_PLAN
export BASELINE_ALLOW_INFEASIBLE_PLAN=0

if [[ -e "$RUN_ROOT" ]]; then
    echo "refusing to overwrite DeepSeek AdaFloor smoke: $RUN_ROOT" >&2
    exit 2
fi
case "$TARGET_FLOOR" in
    8)
        SHRINK_STAGES=8
        ACTIVE_GROUPS='8,9,10,11,12,13,14,15'
        FINAL_GROUP='8,9,10,11,12,13,14,15'
        ;;
    4)
        SHRINK_STAGES=8,4
        ACTIVE_GROUPS='8,9,10,11,12,13,14,15;12,13,14,15'
        FINAL_GROUP='12,13,14,15'
        ;;
    2)
        SHRINK_STAGES=8,4,2
        ACTIVE_GROUPS='8,9,10,11,12,13,14,15;12,13,14,15;14,15'
        FINAL_GROUP='14,15'
        ;;
    *)
        echo "DeepSeek AdaFloor smoke floor must be 8, 4, or 2" >&2
        exit 2
        ;;
esac
if [[ ! -d "$BASELINE_DIR/rollout_data" \
      || ! -d "$BASELINE_DIR/rollout_length" ]]; then
    echo "missing DeepSeek baseline history: $BASELINE_DIR" >&2
    exit 2
fi
if ! [[ "$KV_TOKENS" =~ ^[1-9][0-9]*$ ]] \
   || (( KV_TOKENS % KV_BLOCK_SIZE != 0 )); then
    echo "smoke KV capacity must be a positive block multiple" >&2
    exit 2
fi
for value in \
    "$TRAINING_STEPS" \
    "$TRAIN_BATCH_SIZE" \
    "$ROLLOUT_N" \
    "$MAX_PROMPT_LENGTH" \
    "$MAX_RESPONSE_LENGTH"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "DeepSeek AdaFloor smoke sizes must be positive integers" >&2
        exit 2
    fi
done
if (( (TRAIN_BATCH_SIZE * ROLLOUT_N) % 16 != 0 )); then
    echo "DeepSeek AdaFloor smoke outputs must divide evenly across EP16" >&2
    exit 2
fi
if [[ "$TASK_QUEUE_LEVEL" != 1 && "$TASK_QUEUE_LEVEL" != 2 ]]; then
    echo "DeepSeek AdaFloor smoke TASK_QUEUE_ENABLE must be 1 or 2" >&2
    exit 2
fi
if [[ "$RECOMPUTE_METHOD" != block && "$RECOMPUTE_METHOD" != uniform ]]; then
    echo "DeepSeek AdaFloor smoke recompute method must be block or uniform" >&2
    exit 2
fi
if ! [[ "$RECOMPUTE_NUM_LAYERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "DeepSeek AdaFloor smoke recompute layers must be a positive integer" >&2
    exit 2
fi

validate_cap_group() {
    local group="$1"
    local -a caps=()
    IFS=',' read -r -a caps <<< "$group"
    if (( ${#caps[@]} != 5 )); then
        echo "DeepSeek AdaFloor smoke requires five repeated-halving caps: $group" >&2
        exit 2
    fi
    local cap
    for cap in "${caps[@]}"; do
        if ! [[ "$cap" =~ ^[1-9][0-9]*$ ]] || (( cap > MAX_RESPONSE_LENGTH )); then
            echo "invalid DeepSeek AdaFloor smoke cap $cap for max response $MAX_RESPONSE_LENGTH" >&2
            exit 2
        fi
    done
}

validate_cap_group "$TAIL_VALIDATE_LEVEL_TOKENS"
if [[ -z "$TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP" ]]; then
    for (( step = 1; step <= TRAINING_STEPS; step++ )); do
        if [[ -n "$TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP" ]]; then
            TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP+=';'
        fi
        TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP+="$TAIL_VALIDATE_LEVEL_TOKENS"
    done
else
    IFS=';' read -r -a step_cap_groups <<< "$TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP"
    if (( ${#step_cap_groups[@]} != TRAINING_STEPS )); then
        echo "DeepSeek AdaFloor smoke needs one cap group per training step" >&2
        exit 2
    fi
    for cap_group in "${step_cap_groups[@]}"; do
        validate_cap_group "$cap_group"
    done
fi
EXPECTED_ROWS=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
EXPECTED_OUTPUTS_PER_RANK=$((EXPECTED_ROWS / 16))

mkdir -p "$RUN_ROOT"
printf '%s\n' "INCOMPLETE DeepSeek EP16 $SHRINK_POLICY AdaFloor smoke" > "$RUN_ROOT/INCOMPLETE"

export MODEL_PATH=${MODEL_PATH:-/data/DeepSeek-V2-Lite-Chat}
export DISTCP_PATH=${DISTCP_PATH:-/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4}
export LOCAL_TEST_LAUNCHER=${LOCAL_TEST_LAUNCHER:-$SCRIPT_DIR/internal/wj_train_grpo_deepseek_v2_lite_16die_adafloor.sh}
export BASELINE_LAUNCHER=${BASELINE_LAUNCHER:-$LOCAL_TEST_LAUNCHER}
export HIERARCHICAL_DP2_EP8=0
export VLLM_DP_SIZE=16
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
export TASK_QUEUE_ENABLE=$TASK_QUEUE_LEVEL
export DEEPSEEK_ACTOR_RECOMPUTE_METHOD=$RECOMPUTE_METHOD
export DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS=$RECOMPUTE_NUM_LAYERS
export DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM=$MOE_ALLTOALL_OVERLAP
export DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP=$MOE_SHARED_EXPERT_OVERLAP
export DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS=$DEALLOCATE_PIPELINE_OUTPUTS
export CHECKPOINT_MODEL_DIR_NAME=deepseek_v2_lite
export TRAIN_LOG_PREFIX=deepseek-v2-lite-adafloor-smoke
export PLANNER_TOKENIZER_PATH=$MODEL_PATH

export DYNAMIC_OUTPUT_ROOT=$OUTPUT_ROOT
export DYNAMIC_RUN_NAME=$RUN_NAME
export DYNAMIC_SKIP_MODE0_PROBE=1
export DYNAMIC_INITIAL_BASELINE_DIR=$BASELINE_DIR
export DYNAMIC_START_EPOCH=1
export DYNAMIC_TOTAL_EPOCHS=2
export DYNAMIC_PLAN_STEPS=$TRAINING_STEPS
export DYNAMIC_TRAIN_STEPS=$TRAINING_STEPS
export DYNAMIC_ENABLE_CKPT_CHAIN=0
export DYNAMIC_BUILD_OFFLINE_PLANNING_HISTORY=1
export DYNAMIC_SHRINK_POLICY=$SHRINK_POLICY
export DYNAMIC_FORCE_SELECTED_FLOOR=$TARGET_FLOOR
export DYNAMIC_DISABLE_TAIL_GUARD=1
export DYNAMIC_EXPECT_NO_RESPONSE_CAPS=1
export DYNAMIC_SHORT_STEP_CAP_ENABLE=0
export DYNAMIC_FULL_MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH
export DYNAMIC_FULL_MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH
export DYNAMIC_FULL_MAX_RESPONSE_LEN=$MAX_RESPONSE_LENGTH
export DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP=$TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP

export TRAIN_BATCH_SIZE
export ROLLOUT_N
export ROLLOUT_MAX_NUM_SEQS=32
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.9
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export SAVE_CKPT_ENABLE=0
export TRAINER_SAVE_FREQ=-1

export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=$TARGET_FLOOR
export VLLM_ASCEND_SHRINK_AWARE_STAGES=$SHRINK_STAGES
export VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=8,9,10,11,12,13,14,15
export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=$FINAL_GROUP
export VLLM_ASCEND_MODE1_STEP_TIMELINE_LOG=1
export MIN_ADAPTIVE_FLOOR=$TARGET_FLOOR
export SHRINK_AWARE_LOGGING=true
export MAX_RANK_PEAK_TOKENS=$KV_TOKENS
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=$KV_TOKENS
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2=$KV_TOKENS
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=$KV_TOKENS
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=$KV_TOKENS
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16=$KV_TOKENS
export FLOOR_KV_CAPS="2:$KV_TOKENS,4:$KV_TOKENS,8:$KV_TOKENS,16:$KV_TOKENS"
export PHYSICAL_FLOOR_KV_CAPS=$FLOOR_KV_CAPS
export VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1
export VLLM_ASCEND_MODE1_COLD_INIT_KV_TOKENS=$KV_TOKENS
export VLLM_ASCEND_MODE1_BOOTSTRAP_KV_TOKENS=$KV_TOKENS
export VLLM_ASCEND_MODE1_USE_EXPLICIT_KV_CAP_FOR_INIT=1
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_FAIL_ON_UNMET_TARGET=1
export VLLM_ASCEND_MODE1_PARITY_MC2_WARMUP_ROUTE=global

echo "[DeepSeek AdaFloor smoke] root=$RUN_ROOT baseline=$BASELINE_DIR"
echo "[DeepSeek AdaFloor smoke] policy=$SHRINK_POLICY topology=EP16 stages=$SHRINK_STAGES target_floor=$TARGET_FLOOR kv_tokens=$KV_TOKENS smoke_only=1"
echo "[DeepSeek AdaFloor smoke] steps=$TRAINING_STEPS n=$ROLLOUT_N rows_per_step=$EXPECTED_ROWS caps_by_step=$TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP"
echo "[DeepSeek AdaFloor smoke] task_queue=$TASK_QUEUE_LEVEL recompute=$RECOMPUTE_METHOD/$RECOMPUTE_NUM_LAYERS moe_overlap=$MOE_ALLTOALL_OVERLAP/$MOE_SHARED_EXPERT_OVERLAP deallocate_pipeline_outputs=$DEALLOCATE_PIPELINE_OUTPUTS"
if [[ "$SHRINK_POLICY" == planned ]]; then
    echo "[DeepSeek AdaFloor smoke] planned_training_min_free_mib=$PLANNED_MIN_FREE_MIB"
fi

set +e
"$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh" \
    trainer.resume_mode=disable \
    actor_rollout_ref.rollout.load_format=dummy \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override="$((KV_TOKENS / KV_BLOCK_SIZE))" \
    "$@"
run_rc=$?
set -e
if (( run_rc != 0 )); then
    echo "DeepSeek AdaFloor smoke failed with exit_code=$run_rc" >&2
    exit "$run_rc"
fi

EPOCH_DIR="$RUN_ROOT/epoch_001_mode1_${SHRINK_POLICY}"
latest_log=$(find "$EPOCH_DIR/logs" -maxdepth 1 -type f -name '*.txt' \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -z "$latest_log" || ! -f "$latest_log" ]]; then
    echo "DeepSeek AdaFloor smoke produced no training log" >&2
    exit 3
fi

python3 - \
    "$latest_log" \
    "$EPOCH_DIR" \
    "$MODEL_PATH" \
    "$TRAINING_STEPS" \
    "$EXPECTED_ROWS" \
    "$EXPECTED_OUTPUTS_PER_RANK" \
    "$TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP" \
    "$ACTIVE_GROUPS" \
    "$SHRINK_POLICY" \
    "$PLANNED_MIN_FREE_MIB" <<'PY'
from collections import Counter
import json
import math
import re
import statistics
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
epoch_dir = Path(sys.argv[2])
model_path = Path(sys.argv[3])
training_steps = int(sys.argv[4])
expected_rows = int(sys.argv[5])
expected_outputs_per_rank = int(sys.argv[6])
cap_groups = [
    [int(value) for value in group.split(",")]
    for group in sys.argv[7].split(";")
]
active_groups = [
    [int(rank) for rank in group.split(",")]
    for group in sys.argv[8].split(";")
]
shrink_policy = sys.argv[9]
planned_min_free_mib = int(sys.argv[10])
log = log_path.read_text(encoding="utf-8", errors="replace")

model_config = json.loads(
    (model_path / "config.json").read_text(encoding="utf-8")
)
eos_token_ids = model_config.get("eos_token_id", [])
if isinstance(eos_token_ids, int):
    eos_token_ids = [eos_token_ids]
eos_token_ids = {int(token_id) for token_id in eos_token_ids}
if not eos_token_ids:
    raise SystemExit("DeepSeek model config has no eos_token_id")

for active_group in active_groups:
    rendered_group = ", ".join(str(rank) for rank in active_group)
    shrink_ranks = Counter(
        int(rank)
        for rank in re.findall(
            rf"Elastic parallel shrink done: rank=([0-9]+) "
            rf"active_ranks=\[{re.escape(rendered_group)}\]",
            log,
        )
    )
    expected_shrink_ranks = Counter({
        rank: training_steps for rank in active_group
    })
    if shrink_ranks != expected_shrink_ranks:
        raise SystemExit(
            f"incomplete floor{len(active_group)} shrink ranks: "
            f"{dict(shrink_ranks)}"
        )

restore_ranks = Counter(
    int(rank)
    for rank in re.findall(
        r"Elastic parallel restore done: rank=([0-9]+) dp_size=16 ep_size=16",
        log,
    )
)
expected_restore_ranks = Counter({rank: training_steps for rank in range(16)})
if restore_ranks != expected_restore_ranks:
    raise SystemExit(f"incomplete full-world restore ranks: {dict(restore_ranks)}")

steps = [int(value) for value in re.findall(r"training/global_step:([0-9]+)", log)]
expected_steps = list(range(1, training_steps + 1))
if steps != expected_steps:
    raise SystemExit(f"actor update did not finish steps {expected_steps}: {steps}")
aborted = [float(value) for value in re.findall(
    r"response/aborted_ratio:([0-9.eE+-]+)", log)]
if aborted != [0.0] * training_steps:
    raise SystemExit(f"nonzero or missing aborted ratio: {aborted}")
times = [float(value) for value in re.findall(
    r"rollout_output_time_s:\s*([0-9.eE+-]+)", log)]
if len(times) != training_steps or any(
    not math.isfinite(value) or value <= 0 for value in times
):
    raise SystemExit(f"invalid rollout timing: {times}")
for metric in ("update_actor", "reward"):
    values = [float(value) for value in re.findall(
        rf"timing_s/{metric}:([0-9.eE+-]+)", log)]
    if len(values) != training_steps or any(
        not math.isfinite(value) or value <= 0 for value in values
    ):
        raise SystemExit(f"invalid {metric} timing: {values}")
if re.search(r"preempting request|request preempted", log, flags=re.IGNORECASE):
    raise SystemExit("KV preemption found")
if re.search(
    r"NPU out of memory|Memory_Allocation_Failure|"
    r"Failed to allocate[^\r\n]*NPU memory|OutOfMemoryError|"
    r"ACL_ERROR_RT_MEMORY_ALLOCATION",
    log,
    flags=re.IGNORECASE,
):
    raise SystemExit("NPU OOM evidence found")
if "After trainer.fit" not in log or "Training Progress: 100%" not in log:
    raise SystemExit("trainer did not finish")

if shrink_policy == "planned":
    expected_planned_ranks = (
        f"planned_ranks={active_groups} stage_groups={len(active_groups)}"
    )
    if (
        "Mode1 planned floor groups precreated before KV sizing:" not in log
        or expected_planned_ranks not in log
    ):
        raise SystemExit("Planned floor2 residency was not precreated")
    guard_values = [
        int(value)
        for value in re.findall(
            r"Mode1 training memory guard: rank=0 min_free_mib=([0-9]+)",
            log,
        )
    ]
    if guard_values != [planned_min_free_mib] * training_steps:
        raise SystemExit(
            "Planned training HBM guard was not applied at every actor update: "
            f"{guard_values}"
        )
    cleanup_steps = [
        int(step)
        for step in re.findall(
            r"Mode1 training-boundary full-world transient cleanup: "
            r"rank=0 step=([0-9]+)",
            log,
        )
    ]
    if cleanup_steps != list(range(1, training_steps + 1)):
        raise SystemExit(
            "Planned full-world transient cleanup did not precede every actor "
            f"update: {cleanup_steps}"
        )
    if re.search(
        r"Mode1 full-restore transient cleanup:.*canonical_offload_enabled=1",
        log,
    ):
        raise SystemExit("Planned smoke used shape-unsafe canonical weight offload")

def rank_caps(level_caps):
    return {
        **{rank: level_caps[0] for rank in range(0, 8)},
        **{rank: level_caps[1] for rank in range(8, 12)},
        **{rank: level_caps[2] for rank in range(12, 14)},
        14: level_caps[3],
        15: level_caps[4],
    }


if len(cap_groups) != training_steps or any(len(group) != 5 for group in cap_groups):
    raise SystemExit(f"invalid cap groups: {cap_groups}")
observed_starts = re.findall(
    r"rollout_worker_infer_start rank=([0-9]+) step=([0-9]+) "
    r"epoch=0 .*?max_tokens=([0-9]+)",
    log,
)
observed_start_counts = Counter(
    (int(step), int(rank), int(cap)) for rank, step, cap in observed_starts
)
expected_start_counts = Counter(
    (step, rank, cap)
    for step, level_caps in enumerate(cap_groups, start=1)
    for rank, cap in rank_caps(level_caps).items()
)
if observed_start_counts != expected_start_counts:
    raise SystemExit(
        f"unexpected runtime rank caps: {dict(observed_start_counts)}"
    )
observed_done = Counter(
    (int(step), int(rank), int(outputs))
    for rank, step, outputs in re.findall(
        r"rollout_worker_infer_done rank=([0-9]+) step=([0-9]+) "
        r"epoch=0 .*?outputs=([0-9]+)",
        log,
    )
)
expected_done = Counter(
    (step, rank, expected_outputs_per_rank)
    for step in expected_steps
    for rank in range(16)
)
if observed_done != expected_done:
    raise SystemExit(f"incomplete rollout workers: {dict(observed_done)}")
for step in expected_steps:
    generated = Counter(
        int(rank)
        for rank in re.findall(
            rf"megatron_generate_done rank=([0-9]+) step={step} epoch=0",
            log,
        )
    )
    if generated != Counter({rank: 1 for rank in range(16)}):
        raise SystemExit(f"incomplete generation at step {step}: {dict(generated)}")

refreshes = re.findall(
    r"MLA refresh complete: refreshed=([0-9]+) expected=([0-9]+)", log)
if len(refreshes) < 16 * training_steps or any(
    pair != ("27", "27") for pair in refreshes
):
    raise SystemExit(f"incomplete MLA refreshes: count={len(refreshes)}")

for step, level_caps in enumerate(cap_groups, start=1):
    expected_caps = rank_caps(level_caps)
    rollout_path = epoch_dir / "rollout_data" / f"{step}.jsonl"
    length_path = epoch_dir / "rollout_length" / f"length_{step}.txt"
    for path in (rollout_path, length_path):
        if not path.is_file():
            raise SystemExit(f"missing rollout artifact: {path}")
        artifact_rows = sum(1 for line in path.open(
            "r", encoding="utf-8") if line.strip())
        if artifact_rows != expected_rows:
            raise SystemExit(
                f"expected {expected_rows} rows in {path}, got {artifact_rows}"
            )
    rows = [json.loads(line) for line in rollout_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    lengths_by_rank: dict[int, list[int]] = {}
    survivor_ratios = []
    for row in rows:
        rank = int(row["rollout_rank"])
        raw_length = int(sum(row.get("response_mask", [])))
        valid_response_ids = [
            int(token_id) for token_id in row.get("responses", [])[:raw_length]
        ]
        # VERL includes the first EOS in response_mask. DeepSeek uses EOS as
        # the padding fallback, so discount that padding token at a hard cap.
        length = raw_length
        if valid_response_ids and valid_response_ids[-1] in eos_token_ids:
            length -= 1
        lengths_by_rank.setdefault(rank, []).append(length)
        if rank >= 8:
            output = str(row.get("output", ""))
            visible = [char for char in output if not char.isspace()]
            survivor_ratios.append(
                sum(char.isalnum() for char in visible) / max(len(visible), 1)
            )
    if set(lengths_by_rank) != set(range(16)):
        raise SystemExit(
            f"missing rollout ranks at step {step}: {sorted(lengths_by_rank)}"
        )
    for rank, lengths in lengths_by_rank.items():
        if len(lengths) != expected_outputs_per_rank:
            raise SystemExit(
                f"rank {rank} at step {step} has {len(lengths)} outputs"
            )
        if max(lengths) > expected_caps[rank]:
            raise SystemExit(
                f"rank {rank} at step {step} exceeded its "
                f"{expected_caps[rank]}-token cap: {lengths}"
            )
    if max(
        length
        for rank in range(8, 16)
        for length in lengths_by_rank[rank]
    ) <= level_caps[0]:
        raise SystemExit(
            f"no survivor request outlived the donor cohort at step {step}"
        )
    dominated = sum(ratio < 0.10 for ratio in survivor_ratios)
    if dominated > max(2, math.floor(0.25 * len(survivor_ratios))):
        raise SystemExit(
            f"semantic output collapse at step {step}: "
            f"punctuation_dominated={dominated}/{len(survivor_ratios)} "
            f"median_alnum_ratio={statistics.median(survivor_ratios):.4f}"
        )
PY

printf '%s\n' \
    "COMPLETE DeepSeek EP16 $SHRINK_POLICY AdaFloor smoke" \
    "SHRINK_POLICY=$SHRINK_POLICY" \
    "TRANSITION=16-to-$TARGET_FLOOR" \
    "SHRINK_STAGES=$SHRINK_STAGES" \
    "KV_TOKENS=$KV_TOKENS" \
    "KV_POLICY=smoke-only-not-calibrated" \
    "TRAINING_STEPS=$TRAINING_STEPS" \
    "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE" \
    "ROLLOUT_N=$ROLLOUT_N" \
    "EXPECTED_ROWS_PER_STEP=$EXPECTED_ROWS" \
    "MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH" \
    "MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH" \
    "TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP=$TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP" \
    "TASK_QUEUE_ENABLE=$TASK_QUEUE_LEVEL" \
    "RECOMPUTE_METHOD=$RECOMPUTE_METHOD" \
    "RECOMPUTE_NUM_LAYERS=$RECOMPUTE_NUM_LAYERS" \
    "MOE_ALLTOALL_OVERLAP_COMM=$MOE_ALLTOALL_OVERLAP" \
    "MOE_SHARED_EXPERT_OVERLAP=$MOE_SHARED_EXPERT_OVERLAP" \
    "DEALLOCATE_PIPELINE_OUTPUTS=$DEALLOCATE_PIPELINE_OUTPUTS" \
    "PLANNED_TRAINING_MIN_FREE_MIB=$PLANNED_MIN_FREE_MIB" \
    "ALLOW_INFEASIBLE_PLAN=0" \
    "BASELINE_DIR=$BASELINE_DIR" \
    "LOG=$latest_log" \
    > "$RUN_ROOT/COMPLETE"
rm -f "$RUN_ROOT/INCOMPLETE"
echo "[DeepSeek AdaFloor smoke] complete root=$RUN_ROOT log=$latest_log"

exit=0
