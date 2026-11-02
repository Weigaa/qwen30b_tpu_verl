#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT=${DEEPSEEK_ACTOR_PROBE_OUTPUT_ROOT:-/data/adafloor_shared_state/deepseek_v2_lite}
TASK_QUEUE_LEVEL=${DEEPSEEK_ACTOR_PROBE_TASK_QUEUE_ENABLE:-2}
RECOMPUTE_METHOD=${DEEPSEEK_ACTOR_PROBE_RECOMPUTE_METHOD:-uniform}
RECOMPUTE_NUM_LAYERS=${DEEPSEEK_ACTOR_PROBE_RECOMPUTE_NUM_LAYERS:-1}
ACTOR_TOKEN_CAP=${DEEPSEEK_ACTOR_PROBE_ACTOR_TOKEN_CAP:-17408}
LOG_PROB_TOKEN_CAP=${DEEPSEEK_ACTOR_PROBE_LOG_PROB_TOKEN_CAP:-17408}
TRAINING_STEPS=${DEEPSEEK_ACTOR_PROBE_TRAINING_STEPS:-1}
TRAIN_BATCH_SIZE=${DEEPSEEK_ACTOR_PROBE_TRAIN_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${DEEPSEEK_ACTOR_PROBE_MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${DEEPSEEK_ACTOR_PROBE_MAX_RESPONSE_LENGTH:-16384}
TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP=${DEEPSEEK_ACTOR_PROBE_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP:-}
ROLLOUT_N=${DEEPSEEK_ACTOR_PROBE_ROLLOUT_N:-16}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${DEEPSEEK_ACTOR_PROBE_MAX_NUM_BATCHED_TOKENS:-17408}
ROLLOUT_MAX_NUM_SEQS=${DEEPSEEK_ACTOR_PROBE_MAX_NUM_SEQS:-32}
REQUIRE_SEMANTIC_OUTPUT=${DEEPSEEK_ACTOR_PROBE_REQUIRE_SEMANTIC_OUTPUT:-1}
ROLLOUT_LOAD_FORMAT=${DEEPSEEK_ACTOR_PROBE_ROLLOUT_LOAD_FORMAT:-dummy}
PRESERVE_INITIAL_HF_WEIGHTS=${DEEPSEEK_ACTOR_PROBE_PRESERVE_INITIAL_HF_WEIGHTS:-0}
COMPARE_ONLINE_SYNC_TO_HF=${DEEPSEEK_ACTOR_PROBE_COMPARE_ONLINE_SYNC_TO_HF:-0}
MOE_ALLTOALL_OVERLAP=${DEEPSEEK_ACTOR_PROBE_MOE_ALLTOALL_OVERLAP_COMM:-True}
MOE_SHARED_EXPERT_OVERLAP=${DEEPSEEK_ACTOR_PROBE_MOE_SHARED_EXPERT_OVERLAP:-True}
DEALLOCATE_PIPELINE_OUTPUTS=${DEEPSEEK_ACTOR_PROBE_DEALLOCATE_PIPELINE_OUTPUTS:-False}
RUN_NAME=${DEEPSEEK_ACTOR_PROBE_RUN_NAME:-actor_update_probe_${TRAINING_STEPS}step_tq${TASK_QUEUE_LEVEL}_${RECOMPUTE_METHOD}_tok${ACTOR_TOKEN_CAP}_$timestamp}
RUN_ROOT="$OUTPUT_ROOT/$RUN_NAME"
EPOCH_DIR="$RUN_ROOT/epoch_000_mode0_probe"
KV_TOKENS_PER_RANK=${DEEPSEEK_ACTOR_PROBE_KV_TOKENS_PER_RANK:-621056}
KV_BLOCK_SIZE=${VLLM_KV_BLOCK_SIZE:-128}
MODEL_PATH=${MODEL_PATH:-/data/DeepSeek-V2-Lite-Chat}
MODEL_ID=${MODEL_ID:-deepseek-ai/DeepSeek-V2-Lite-Chat}
MODEL_REVISION=${MODEL_REVISION:-85864749cd611b4353ce1decdb286193298f64c7}
DISTCP_PATH=${DISTCP_PATH:-/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4}

if ! [[ "$KV_TOKENS_PER_RANK" =~ ^[1-9][0-9]*$ ]] \
   || (( KV_BLOCK_SIZE <= 0 || KV_TOKENS_PER_RANK % KV_BLOCK_SIZE != 0 )); then
    echo "DeepSeek actor probe KV capacity must be a positive block multiple" >&2
    exit 2
fi
if [[ "$TASK_QUEUE_LEVEL" != 1 && "$TASK_QUEUE_LEVEL" != 2 ]]; then
    echo "DeepSeek actor probe TASK_QUEUE_ENABLE must be 1 or 2" >&2
    exit 2
fi
if [[ "$RECOMPUTE_METHOD" != block && "$RECOMPUTE_METHOD" != uniform ]]; then
    echo "DeepSeek actor probe recompute method must be block or uniform" >&2
    exit 2
fi
for token_cap in \
    "$ACTOR_TOKEN_CAP" \
    "$LOG_PROB_TOKEN_CAP" \
    "$RECOMPUTE_NUM_LAYERS" \
    "$TRAINING_STEPS" \
    "$TRAIN_BATCH_SIZE" \
    "$MAX_PROMPT_LENGTH" \
    "$MAX_RESPONSE_LENGTH" \
    "$ROLLOUT_N" \
    "$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    "$ROLLOUT_MAX_NUM_SEQS"; do
    if ! [[ "$token_cap" =~ ^[1-9][0-9]*$ ]]; then
        echo "DeepSeek actor probe token caps and recompute layers must be positive integers" >&2
        exit 2
    fi
done
if [[ "$REQUIRE_SEMANTIC_OUTPUT" != 0 && "$REQUIRE_SEMANTIC_OUTPUT" != 1 ]]; then
    echo "DeepSeek actor probe semantic-output flag must be 0 or 1" >&2
    exit 2
fi
if [[ "$PRESERVE_INITIAL_HF_WEIGHTS" != 0 && "$PRESERVE_INITIAL_HF_WEIGHTS" != 1 ]]; then
    echo "DeepSeek actor probe HF-weight preservation flag must be 0 or 1" >&2
    exit 2
fi
if [[ "$COMPARE_ONLINE_SYNC_TO_HF" != 0 && "$COMPARE_ONLINE_SYNC_TO_HF" != 1 ]]; then
    echo "DeepSeek actor probe online-sync comparison flag must be 0 or 1" >&2
    exit 2
fi
if [[ "$PRESERVE_INITIAL_HF_WEIGHTS" == 1 && "$COMPARE_ONLINE_SYNC_TO_HF" == 1 ]]; then
    echo "DeepSeek HF preservation and online-sync comparison are mutually exclusive" >&2
    exit 2
fi
if [[ "$PRESERVE_INITIAL_HF_WEIGHTS" == 1 && "$ROLLOUT_LOAD_FORMAT" == dummy* ]]; then
    echo "DeepSeek HF-weight preservation requires a non-dummy rollout load format" >&2
    exit 2
fi
if [[ "$COMPARE_ONLINE_SYNC_TO_HF" == 1 && "$ROLLOUT_LOAD_FORMAT" == dummy* ]]; then
    echo "DeepSeek online-sync comparison requires a non-dummy rollout load format" >&2
    exit 2
fi
if (( ROLLOUT_MAX_NUM_BATCHED_TOKENS < MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH )); then
    echo "DeepSeek actor probe batched-token limit is smaller than one full request" >&2
    exit 2
fi
KV_BLOCKS=$((KV_TOKENS_PER_RANK / KV_BLOCK_SIZE))
EXPECTED_ROWS=$((TRAIN_BATCH_SIZE * ROLLOUT_N))

tail_env_args=(-u VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS)
if [[ -n "$TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP" ]]; then
    IFS=';' read -r -a step_cap_groups <<< "$TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP"
    if (( ${#step_cap_groups[@]} != TRAINING_STEPS )); then
        echo "DeepSeek actor probe requires one threshold group per training step" >&2
        exit 2
    fi
    for cap_group in "${step_cap_groups[@]}"; do
        IFS=',' read -r -a caps <<< "$cap_group"
        if (( ${#caps[@]} != 5 )); then
            echo "DeepSeek actor probe requires five EP16 threshold caps: $cap_group" >&2
            exit 2
        fi
        for cap in "${caps[@]}"; do
            if ! [[ "$cap" =~ ^[1-9][0-9]*$ ]] || (( cap > MAX_RESPONSE_LENGTH )); then
                echo "invalid actor-probe threshold $cap for max response $MAX_RESPONSE_LENGTH" >&2
                exit 2
            fi
        done
    done
    if (( EXPECTED_ROWS % 16 != 0 )); then
        echo "threshold actor-probe outputs must divide evenly across EP16" >&2
        exit 2
    fi
    tail_env_args+=(
        "VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP=$TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP"
    )
else
    tail_env_args+=(-u VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP)
fi

if [[ -e "$RUN_ROOT" ]]; then
    echo "refusing to overwrite DeepSeek actor probe: $RUN_ROOT" >&2
    exit 2
fi

mkdir -p "$EPOCH_DIR"
printf '%s\n' "INCOMPLETE DeepSeek actor update probe" > "$RUN_ROOT/INCOMPLETE"

set +e
env \
    "${tail_env_args[@]}" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    OUTPUT_SUBDIR="$RUN_NAME/epoch_000_mode0_probe" \
    RECORD_DIR="$EPOCH_DIR" \
    MODEL_PATH="$MODEL_PATH" \
    MODEL_REVISION="$MODEL_REVISION" \
    DISTCP_PATH="$DISTCP_PATH" \
    BASELINE_LAUNCHER="$SCRIPT_DIR/internal/wj_train_grpo_deepseek_v2_lite_16die_adafloor.sh" \
    CHECKPOINT_MODEL_DIR_NAME=deepseek_v2_lite \
    TRAIN_LOG_PREFIX=deepseek-v2-lite-actor-probe \
    TASK_QUEUE_ENABLE="$TASK_QUEUE_LEVEL" \
    DEEPSEEK_ACTOR_RECOMPUTE_METHOD="$RECOMPUTE_METHOD" \
    DEEPSEEK_ACTOR_RECOMPUTE_NUM_LAYERS="$RECOMPUTE_NUM_LAYERS" \
    DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM="$MOE_ALLTOALL_OVERLAP" \
    DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP="$MOE_SHARED_EXPERT_OVERLAP" \
    DEEPSEEK_DEALLOCATE_PIPELINE_OUTPUTS="$DEALLOCATE_PIPELINE_OUTPUTS" \
    VLLM_ASCEND_DEEPSEEK_PRESERVE_INITIAL_HF_WEIGHTS="$PRESERVE_INITIAL_HF_WEIGHTS" \
    VLLM_ASCEND_MODE1_WEIGHT_LOADER_DIAG_COMPARE_HF="$COMPARE_ONLINE_SYNC_TO_HF" \
    ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="$ACTOR_TOKEN_CAP" \
    ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="$LOG_PROB_TOKEN_CAP" \
    VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1 \
    VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB=0 \
    VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=0 \
    TRAINER_TOTAL_EPOCHS=1 \
    DATASET_FRACTION=0.005 \
    DATA_SHUFFLE=False \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    ROLLOUT_MAX_NUM_SEQS="$ROLLOUT_MAX_NUM_SEQS" \
    ROLLOUT_N="$ROLLOUT_N" \
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 \
    MODE0_SAVE_ROLLOUT_ARTIFACTS=1 \
    SAVE_CKPT_ENABLE=0 \
    "$SCRIPT_DIR/run_mode0_no_shrink_baseline.sh" \
        trainer.total_training_steps="$TRAINING_STEPS" \
        trainer.resume_mode=disable \
        actor_rollout_ref.rollout.load_format="$ROLLOUT_LOAD_FORMAT" \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override="$KV_BLOCKS" \
        actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method="$RECOMPUTE_METHOD" \
        actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers="$RECOMPUTE_NUM_LAYERS" \
        "$@"
run_rc=$?
set -e

if (( run_rc != 0 )); then
    echo "DeepSeek actor update probe failed with exit_code=$run_rc" >&2
    exit "$run_rc"
fi

latest_log=$(find "$EPOCH_DIR/logs" -maxdepth 1 -type f -name '*.txt' \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -z "$latest_log" || ! -f "$latest_log" ]]; then
    echo "DeepSeek actor update probe produced no training log" >&2
    exit 3
fi

python3 - \
    "$latest_log" \
    "$EPOCH_DIR" \
    "$TRAINING_STEPS" \
    "$EXPECTED_ROWS" \
    "$REQUIRE_SEMANTIC_OUTPUT" \
    "$PRESERVE_INITIAL_HF_WEIGHTS" \
    "$COMPARE_ONLINE_SYNC_TO_HF" \
    "$MODEL_PATH" \
    "$TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP" \
    "$MAX_RESPONSE_LENGTH" <<'PY'
from collections import Counter
import json
import math
import re
import statistics
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
epoch_dir = Path(sys.argv[2])
training_steps = int(sys.argv[3])
expected_rows = int(sys.argv[4])
require_semantic_output = bool(int(sys.argv[5]))
preserve_initial_hf_weights = bool(int(sys.argv[6]))
compare_online_sync_to_hf = bool(int(sys.argv[7]))
model_path = Path(sys.argv[8])
tail_caps_by_step = sys.argv[9]
expected_max_response_length = int(sys.argv[10])
log = log_path.read_text(encoding="utf-8", errors="replace")
steps = [int(value) for value in re.findall(r"training/global_step:([0-9]+)", log)]
expected_steps = list(range(1, training_steps + 1))
if steps != expected_steps:
    raise SystemExit(f"actor update did not complete expected steps {expected_steps}: {steps}")
if training_steps > 1:
    first_update = log.find("training/global_step:1")
    second_generation = log.find("rollout_worker_infer_start rank=0 step=2 epoch=0")
    if first_update < 0 or second_generation <= first_update:
        raise SystemExit("step 2 generation did not follow the first actor update")
times = [float(value) for value in re.findall(r"rollout_output_time_s:\s*([0-9.eE+-]+)", log)]
if len(times) != training_steps or any(not math.isfinite(value) or value <= 0 for value in times):
    raise SystemExit(f"invalid rollout timing: {times}")
aborted = [float(value) for value in re.findall(r"response/aborted_ratio:([0-9.eE+-]+)", log)]
if aborted != [0.0] * training_steps:
    raise SystemExit(f"invalid aborted ratio: {aborted}")
for metric in ("update_actor", "reward"):
    values = [float(value) for value in re.findall(
        rf"timing_s/{metric}:([0-9.eE+-]+)", log)]
    if len(values) != training_steps or any(
        not math.isfinite(value) or value <= 0 for value in values
    ):
        raise SystemExit(f"invalid {metric} timing: {values}")
if re.search(
    r"preempting request|request preempted",
    log,
    flags=re.IGNORECASE,
):
    raise SystemExit("KV preemption found")
for step in expected_steps:
    generate_done = re.findall(rf"megatron_generate_done rank=([0-9]+) step={step} epoch=0", log)
    if sorted(int(rank) for rank in generate_done) != list(range(16)):
        raise SystemExit(f"incomplete generation ranks at step {step}: {generate_done}")
if "After trainer.fit" not in log or "Training Progress: 100%" not in log:
    raise SystemExit("trainer did not finish")
oom = re.findall(
    r"NPU out of memory|Memory_Allocation_Failure|"
    r"Failed to allocate[^\r\n]*NPU memory|OutOfMemoryError|"
    r"ACL_ERROR_RT_MEMORY_ALLOCATION",
    log,
    flags=re.IGNORECASE,
)
if oom:
    raise SystemExit(f"NPU OOM evidence found: {oom[:5]}")
refreshes = re.findall(
    r"MLA refresh complete: refreshed=([0-9]+) expected=([0-9]+) "
    r"impl=([^\r\n]+)",
    log,
)
if preserve_initial_hf_weights:
    preserved = log.count(
        "Preserving initial DeepSeek HF rollout weights for EP forward diagnostics")
    skipped_sync = log.count(
        "Skipping online MCore rollout-weight sync for DeepSeek HF EP forward diagnostics")
    if preserved < 16 or skipped_sync < 16:
        raise SystemExit(
            f"incomplete HF EP diagnostic path: preserved={preserved}, "
            f"skipped_sync={skipped_sync}"
        )
else:
    expected_refreshes = 16 * training_steps
    if len(refreshes) < expected_refreshes:
        raise SystemExit(
            f"expected at least {expected_refreshes} complete MLA refreshes, "
            f"got {len(refreshes)}"
        )
    if any(current != "27" or expected != "27" for current, expected, _ in refreshes):
        raise SystemExit(f"invalid MLA refresh records: {refreshes[:5]}")
if compare_online_sync_to_hf:
    comparison_passes = [
        (int(rank), int(total), int(routed_streams))
        for rank, total, routed_streams in re.findall(
            r"DeepSeek online-sync HF comparison PASS: rank=([0-9]+) "
            r"total=([0-9]+) routed_streams=([0-9]+)",
            log,
        )
    ]
    if sorted(rank for rank, _, _ in comparison_passes) != list(range(16)):
        raise SystemExit(
            f"incomplete strict HF weight comparisons: {comparison_passes}")
    if any(total <= 0 or routed_streams != 8 for _, total, routed_streams in comparison_passes):
        raise SystemExit(
            f"invalid strict HF weight comparison contract: {comparison_passes}")

cap_groups = []
if tail_caps_by_step:
    cap_groups = [
        [int(value) for value in group.split(",")]
        for group in tail_caps_by_step.split(";")
    ]
    if len(cap_groups) != training_steps or any(
        len(group) != 5 for group in cap_groups
    ):
        raise SystemExit(f"invalid threshold groups: {cap_groups}")


def rank_caps(level_caps):
    return {
        **{rank: level_caps[0] for rank in range(0, 8)},
        **{rank: level_caps[1] for rank in range(8, 12)},
        **{rank: level_caps[2] for rank in range(12, 14)},
        14: level_caps[3],
        15: level_caps[4],
    }


outputs_per_rank = expected_rows // 16
observed_starts = Counter(
    (int(step), int(rank), int(cap))
    for rank, step, cap in re.findall(
        r"rollout_worker_infer_start rank=([0-9]+) step=([0-9]+) "
        r"epoch=0 .*?max_tokens=([0-9]+)",
        log,
    )
)
if cap_groups:
    expected_starts = Counter(
        (step, rank, cap)
        for step, level_caps in enumerate(cap_groups, start=1)
        for rank, cap in rank_caps(level_caps).items()
    )
else:
    expected_starts = Counter(
        (step, rank, expected_max_response_length)
        for step in expected_steps
        for rank in range(16)
    )
if observed_starts != expected_starts:
    raise SystemExit(f"unexpected runtime thresholds: {dict(observed_starts)}")
observed_done = Counter(
    (int(step), int(rank), int(outputs))
    for rank, step, outputs in re.findall(
        r"rollout_worker_infer_done rank=([0-9]+) step=([0-9]+) "
        r"epoch=0 .*?outputs=([0-9]+)",
        log,
    )
)
expected_done = Counter(
    (step, rank, outputs_per_rank)
    for step in expected_steps
    for rank in range(16)
)
if observed_done != expected_done:
    raise SystemExit(f"incomplete rollout workers: {dict(observed_done)}")

model_config = json.loads(
    (model_path / "config.json").read_text(encoding="utf-8")
)
eos_token_ids = model_config.get("eos_token_id", [])
if isinstance(eos_token_ids, int):
    eos_token_ids = [eos_token_ids]
eos_token_ids = {int(token_id) for token_id in eos_token_ids}
for step in expected_steps:
    rollout_path = epoch_dir / "rollout_data" / f"{step}.jsonl"
    for path in (
        rollout_path,
        epoch_dir / "rollout_length" / f"length_{step}.txt",
    ):
        if not path.is_file():
            raise SystemExit(f"missing rollout artifact: {path}")
        rows = sum(1 for _ in path.open("r", encoding="utf-8"))
        if rows != expected_rows:
            raise SystemExit(
                f"expected {expected_rows} rows in {path}, got {rows}")
    if require_semantic_output:
        ratios = []
        with rollout_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if cap_groups and int(row["rollout_rank"]) < 8:
                    continue
                output = str(row.get("output", ""))
                visible = [char for char in output if not char.isspace()]
                ratios.append(
                    sum(char.isalnum() for char in visible) / max(len(visible), 1)
                )
        dominated = sum(ratio < 0.10 for ratio in ratios)
        if dominated > max(2, math.floor(0.25 * len(ratios))):
            raise SystemExit(
                f"semantic output collapse at step {step}: "
                f"punctuation_dominated={dominated}/{len(ratios)} "
                f"median_alnum_ratio={statistics.median(ratios):.4f}"
            )
    if cap_groups:
        expected_caps = rank_caps(cap_groups[step - 1])
        lengths_by_rank = {rank: [] for rank in range(16)}
        with rollout_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rank = int(row["rollout_rank"])
                raw_length = int(sum(row.get("response_mask", [])))
                response_ids = [
                    int(token_id)
                    for token_id in row.get("responses", [])[:raw_length]
                ]
                length = raw_length
                if response_ids and response_ids[-1] in eos_token_ids:
                    length -= 1
                lengths_by_rank[rank].append(length)
        for rank, lengths in lengths_by_rank.items():
            if len(lengths) != expected_rows // 16:
                raise SystemExit(
                    f"rank {rank} at step {step} has {len(lengths)} responses"
                )
            if max(lengths) > expected_caps[rank]:
                raise SystemExit(
                    f"rank {rank} at step {step} exceeded threshold "
                    f"{expected_caps[rank]}: {lengths}"
                )
PY

printf '%s\n' \
    "COMPLETE DeepSeek actor update probe" \
    "TASK_QUEUE_ENABLE=$TASK_QUEUE_LEVEL" \
    "RECOMPUTE_METHOD=$RECOMPUTE_METHOD" \
    "RECOMPUTE_NUM_LAYERS=$RECOMPUTE_NUM_LAYERS" \
    "TRAINING_STEPS=$TRAINING_STEPS" \
    "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE" \
    "MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH" \
    "MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH" \
    "TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP=${TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP:-<unset>}" \
    "ROLLOUT_N=$ROLLOUT_N" \
    "ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    "ROLLOUT_MAX_NUM_SEQS=$ROLLOUT_MAX_NUM_SEQS" \
    "EXPECTED_ROWS=$EXPECTED_ROWS" \
    "REQUIRE_SEMANTIC_OUTPUT=$REQUIRE_SEMANTIC_OUTPUT" \
    "ROLLOUT_LOAD_FORMAT=$ROLLOUT_LOAD_FORMAT" \
    "PRESERVE_INITIAL_HF_WEIGHTS=$PRESERVE_INITIAL_HF_WEIGHTS" \
    "COMPARE_ONLINE_SYNC_TO_HF=$COMPARE_ONLINE_SYNC_TO_HF" \
    "MODEL_ID=$MODEL_ID" \
    "MODEL_REVISION=$MODEL_REVISION" \
    "MODEL_PATH=$MODEL_PATH" \
    "DISTCP_PATH=$DISTCP_PATH" \
    "MOE_ALLTOALL_OVERLAP_COMM=$MOE_ALLTOALL_OVERLAP" \
    "MOE_SHARED_EXPERT_OVERLAP=$MOE_SHARED_EXPERT_OVERLAP" \
    "DEALLOCATE_PIPELINE_OUTPUTS=$DEALLOCATE_PIPELINE_OUTPUTS" \
    "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=$ACTOR_TOKEN_CAP" \
    "ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$LOG_PROB_TOKEN_CAP" \
    "KV_TOKENS_PER_RANK=$KV_TOKENS_PER_RANK" \
    > "$RUN_ROOT/COMPLETE"
rm -f "$RUN_ROOT/INCOMPLETE"
echo "[DeepSeek actor probe] complete root=$RUN_ROOT"
