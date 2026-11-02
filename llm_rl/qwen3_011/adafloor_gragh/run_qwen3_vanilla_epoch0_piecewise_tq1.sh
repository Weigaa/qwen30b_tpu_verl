#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ACTION="${1:-dry-run}"
case "$ACTION" in
    run|dry-run|summarize) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_vanilla_epoch0_piecewise_tq1.sh dry-run
  ./run_qwen3_vanilla_epoch0_piecewise_tq1.sh run
  ./run_qwen3_vanilla_epoch0_piecewise_tq1.sh summarize

Runs the vLLM/vLLM-Ascend 0.11 Vanilla Full16 epoch0 workload with native
PIECEWISE ACLGraph and TASK_QUEUE_ENABLE=1. Attention is an eager FX split
boundary, while fixed-topology MoE, HCCL, and dense decode are captured.

The workload and initial model state match the historical AdaFloor eager
epoch0: seed 0, five steps, 32 prompts/step, n=16, max response length 16384,
2975 KV blocks (380800 tokens) per rank, and actor learning rate 1e-6.

Set PIECEWISE_EPOCH0_ROOT to choose a fresh output root. The run action never
overwrites an existing epoch directory and retains its global_step_5 checkpoint.
EOF
        exit 0
        ;;
    *)
        echo "unknown action: $ACTION" >&2
        exit 2
        ;;
esac

EXPERIMENT_ROOT="${PIECEWISE_EPOCH0_ROOT:-/workspace/adafloor_graph_results/qwen3_vanilla_epoch0_piecewise_tq1_seed0}"
RUN_NAME="${PIECEWISE_EPOCH0_RUN_NAME:-common_epoch0_piecewise_tq1_seed0}"
RUN_ROOT="$EXPERIMENT_ROOT/$RUN_NAME"
EPOCH_DIR="$RUN_ROOT/epoch_000_mode0_probe"
EAGER_REFERENCE_ROOT="${PIECEWISE_EAGER_REFERENCE_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
EAGER_EPOCH_DIR="$EAGER_REFERENCE_ROOT/epoch_000_mode0_probe"

GRAPH_WRAPPER="$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh"
COMMON_RUNNER="$SCRIPT_DIR/run_common_epoch0_probe_gpu09_kv380800_permanent.sh"
ASCEND_EXTENSION="${VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION:-/workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so}"

MODEL_PATH="${MODEL_PATH:-/data/Qwen3-30B-A3B}"
DISTCP_PATH="${DISTCP_PATH:-/data/Qwen3-30B-A3B_megatron}"
TRAIN_FILE="${TRAIN_FILE:-/data/deepscaler/train.parquet}"
TEST_FILE="${TEST_FILE:-/data/deepscaler/test.parquet}"

SEED=0
STEPS=5
PROMPTS_PER_STEP=32
ROLLOUT_N=16
EXPECTED_RESPONSES_PER_STEP=512
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=16384
MAX_NUM_BATCHED_TOKENS=17408
MAX_NUM_SEQS=32
KV_TOKENS_PER_RANK=380800
KV_BLOCK_SIZE=128
KV_BLOCKS=2975
GPU_MEMORY_UTILIZATION=0.9
ACTOR_LR=1e-6
CAPTURE_SIZES='[1,2,4,8,16,32]'

PROTOCOL="$EXPERIMENT_ROOT/protocol.env"
CODE_MANIFEST="$EXPERIMENT_ROOT/code_sha256.txt"
SUMMARY_JSON="$EXPERIMENT_ROOT/piecewise_epoch0_summary.json"
SUMMARY_MD="$EXPERIMENT_ROOT/piecewise_epoch0_summary.md"

CODE_PATHS=(
    "$SCRIPT_DIR/run_qwen3_vanilla_epoch0_piecewise_tq1.sh"
    "$GRAPH_WRAPPER"
    "$COMMON_RUNNER"
    "$SCRIPT_DIR/run_mode0_no_shrink_baseline.sh"
    "$SCRIPT_DIR/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager_baseline_util.sh"
    "$SCRIPT_DIR/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    "$SCRIPT_DIR/vllm_ascend/envs.py"
    "$SCRIPT_DIR/vllm_ascend/platform.py"
    "$SCRIPT_DIR/vllm_ascend/models/qwen3_moe.py"
    "$SCRIPT_DIR/vllm_ascend/ops/fused_moe.py"
    "$SCRIPT_DIR/vllm_ascend/compilation/acl_graph.py"
    "$SCRIPT_DIR/vllm_ascend/worker/model_runner_v1.py"
    "$SCRIPT_DIR/vllm/compilation/backends.py"
)

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

require_inputs() {
    local path
    for path in \
        "$GRAPH_WRAPPER" "$COMMON_RUNNER" "$ASCEND_EXTENSION" \
        "$MODEL_PATH" "$DISTCP_PATH" "$TRAIN_FILE" "$TEST_FILE" \
        "$EAGER_EPOCH_DIR"; do
        [[ -e "$path" ]] || {
            echo "missing required input: $path" >&2
            exit 2
        }
    done
    (( KV_TOKENS_PER_RANK / KV_BLOCK_SIZE == KV_BLOCKS )) || {
        echo "KV contract mismatch" >&2
        exit 2
    }
}

write_or_verify_contracts() {
    mkdir -p "$EXPERIMENT_ROOT"
    local code_tmp protocol_tmp eager_log eager_log_sha created
    code_tmp=$(mktemp "$EXPERIMENT_ROOT/.code.XXXXXX")
    protocol_tmp=$(mktemp "$EXPERIMENT_ROOT/.protocol.XXXXXX")
    sha256sum "${CODE_PATHS[@]}" > "$code_tmp"
    mapfile -t eager_logs < <(find "$EAGER_EPOCH_DIR/logs" -maxdepth 1 -type f -name '*.txt' -print)
    (( ${#eager_logs[@]} == 1 )) || {
        echo "expected exactly one eager reference log, found ${#eager_logs[@]}" >&2
        rm -f "$code_tmp" "$protocol_tmp"
        exit 2
    }
    eager_log="${eager_logs[0]}"
    eager_log_sha=$(sha256_file "$eager_log")
    if [[ -f "$PROTOCOL" ]]; then
        created=$(sed -n 's/^created_at_utc=//p' "$PROTOCOL")
    else
        created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    fi
    {
        printf 'schema_version=1\n'
        printf 'created_at_utc=%s\n' "$created"
        printf 'experiment=qwen3_vanilla_epoch0_piecewise_tq1\n'
        printf 'implementation_stack=vllm-0.11_vllm-ascend-0.11rc0\n'
        printf 'output_root=%s\n' "$EXPERIMENT_ROOT"
        printf 'run_name=%s\n' "$RUN_NAME"
        printf 'initial_model_path=%s\n' "$(realpath "$MODEL_PATH")"
        printf 'initial_distcp_path=%s\n' "$(realpath "$DISTCP_PATH")"
        printf 'train_file=%s\n' "$(realpath "$TRAIN_FILE")"
        printf 'train_file_sha256=%s\n' "$(sha256_file "$TRAIN_FILE")"
        printf 'test_file=%s\n' "$(realpath "$TEST_FILE")"
        printf 'test_file_sha256=%s\n' "$(sha256_file "$TEST_FILE")"
        printf 'eager_reference_root=%s\n' "$(realpath "$EAGER_REFERENCE_ROOT")"
        printf 'eager_reference_log=%s\n' "$(realpath "$eager_log")"
        printf 'eager_reference_log_sha256=%s\n' "$eager_log_sha"
        printf 'seed=%s\n' "$SEED"
        printf 'actor_megatron_seed=42\n'
        printf 'actor_lr=%s\n' "$ACTOR_LR"
        printf 'steps=%s\n' "$STEPS"
        printf 'prompts_per_step=%s\n' "$PROMPTS_PER_STEP"
        printf 'rollout_n=%s\n' "$ROLLOUT_N"
        printf 'responses_per_step=%s\n' "$EXPECTED_RESPONSES_PER_STEP"
        printf 'max_prompt_length=%s\n' "$MAX_PROMPT_LENGTH"
        printf 'max_response_length=%s\n' "$MAX_RESPONSE_LENGTH"
        printf 'max_num_batched_tokens=%s\n' "$MAX_NUM_BATCHED_TOKENS"
        printf 'max_num_seqs=%s\n' "$MAX_NUM_SEQS"
        printf 'gpu_memory_utilization=%s\n' "$GPU_MEMORY_UTILIZATION"
        printf 'kv_tokens_per_rank=%s\n' "$KV_TOKENS_PER_RANK"
        printf 'kv_block_size=%s\n' "$KV_BLOCK_SIZE"
        printf 'kv_blocks=%s\n' "$KV_BLOCKS"
        printf 'task_queue_enable=1\n'
        printf 'cudagraph_mode=PIECEWISE\n'
        printf 'capture_sizes=%s\n' "$CAPTURE_SIZES"
        printf 'attention_execution=eager_split_boundary\n'
        printf 'moe_execution=piecewise_aclgraph\n'
        printf 'torchair=false\n'
        printf 'tail_guard=false\n'
        printf 'shrink=false\n'
        printf 'sidecar=false\n'
        printf 'data_shuffle=false\n'
        printf 'temperature=0.9\n'
        printf 'top_p=0.9\n'
        printf 'top_k=50\n'
    } > "$protocol_tmp"

    for pair in "$code_tmp:$CODE_MANIFEST" "$protocol_tmp:$PROTOCOL"; do
        local candidate=${pair%%:*}
        local destination=${pair#*:}
        if [[ -e "$destination" ]]; then
            if [[ ! -f "$destination" || -L "$destination" ]] \
               || ! cmp -s "$candidate" "$destination"; then
                echo "immutable experiment contract changed: $destination" >&2
                rm -f "$code_tmp" "$protocol_tmp"
                exit 2
            fi
            rm -f "$candidate"
        else
            mv "$candidate" "$destination"
            chmod 0444 "$destination"
        fi
    done
}

print_contract() {
    cat <<EOF
[piecewise epoch0] action=$ACTION
[piecewise epoch0] output=$RUN_ROOT
[piecewise epoch0] eager_reference=$EAGER_REFERENCE_ROOT
[piecewise epoch0] initial_distcp=$DISTCP_PATH
[piecewise epoch0] seed=$SEED steps=$STEPS batch=$PROMPTS_PER_STEP n=$ROLLOUT_N max_response=$MAX_RESPONSE_LENGTH
[piecewise epoch0] kv_tokens=$KV_TOKENS_PER_RANK blocks=$KV_BLOCKS max_num_seqs=$MAX_NUM_SEQS
[piecewise epoch0] graph=PIECEWISE task_queue=1 attention=eager_split_boundary moe=aclgraph capture_sizes=$CAPTURE_SIZES
EOF
}

summarize() {
    [[ -f "$RUN_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]] || {
        echo "completed piecewise epoch0 marker is missing: $RUN_ROOT" >&2
        exit 3
    }
    python3 - "$EAGER_EPOCH_DIR" "$EPOCH_DIR" "$SUMMARY_JSON" "$SUMMARY_MD" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

eager_dir, graph_dir, output_json, output_md = map(Path, sys.argv[1:])


def unique_log(root: Path) -> Path:
    logs = list((root / "logs").glob("*.txt"))
    if len(logs) != 1:
        raise SystemExit(f"expected one log under {root}, got {len(logs)}")
    return logs[0]


def arm(root: Path, graph: bool) -> dict:
    log_path = unique_log(root)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    rollout = [float(x) for x in re.findall(r"rollout_output_time_s:\s*([0-9.eE+-]+)", log)]
    step_time = [float(x) for x in re.findall(r"timing_s/step:([0-9.eE+-]+)", log)]
    rewards = [float(x) for x in re.findall(r"critic/score/mean:([0-9.eE+-]+)", log)]
    aborted = [float(x) for x in re.findall(r"response/aborted_ratio:([0-9.eE+-]+)", log)]
    if not all(len(values) == 5 for values in (rollout, step_time, rewards, aborted)):
        raise SystemExit(f"incomplete five-step metrics in {log_path}")
    if any(not math.isfinite(x) or x <= 0 for x in rollout + step_time):
        raise SystemExit(f"invalid timing in {log_path}")
    if any(x != 0 for x in aborted):
        raise SystemExit(f"nonzero abort ratio in {log_path}")
    tokens = []
    for step in range(1, 6):
        length_path = root / "rollout_length" / f"length_{step}.txt"
        values = [int(x) for x in length_path.read_text().split()]
        if len(values) != 512 or any(x <= 0 or x > 16384 for x in values):
            raise SystemExit(f"invalid response lengths in {length_path}")
        rollout_path = root / "rollout_data" / f"{step}.jsonl"
        if sum(1 for _ in rollout_path.open("rb")) != 512:
            raise SystemExit(f"invalid rollout rows in {rollout_path}")
        tokens.append(sum(values))
    lower = log.lower()
    if re.search(r"preempting request|request preempted", lower):
        raise SystemExit(f"preemption evidence in {log_path}")
    if re.search(r"npu out of memory|memory_allocation_failure|outofmemoryerror", lower):
        raise SystemExit(f"OOM evidence in {log_path}")
    if graph:
        required = (
            "'TASK_QUEUE_ENABLE': '1'",
            "PIECEWISE compilation enabled on NPU",
            "vllm.unified_ascend_attention_with_output",
            "Elastic ACLGraph MoE capture enabled",
            "Replaying aclgraph",
        )
        missing = [marker for marker in required if marker not in log]
        if missing:
            raise SystemExit(f"missing graph evidence {missing} in {log_path}")
        forbidden = (
            "Attention ACLGraph metadata update active",
            "FULL_DECODE_ONLY Attention maximum workspace",
        )
        present = [marker for marker in forbidden if marker in log]
        if present:
            raise SystemExit(f"Attention unexpectedly entered graph: {present}")
    total_tokens = sum(tokens)
    total_rollout = sum(rollout)
    return {
        "log": str(log_path.resolve()),
        "tokens_by_step": tokens,
        "rollout_seconds_by_step": rollout,
        "step_seconds_by_step": step_time,
        "reward_by_step": rewards,
        "generated_tokens": total_tokens,
        "rollout_seconds": total_rollout,
        "response_tokens_per_second": total_tokens / total_rollout,
        "step_seconds": sum(step_time),
        "mean_reward": sum(rewards) / len(rewards),
        "aborted_responses": 0,
        "preemptions": 0,
        "oom": False,
    }


eager = arm(eager_dir, False)
piecewise = arm(graph_dir, True)
throughput_delta = (
    piecewise["response_tokens_per_second"] / eager["response_tokens_per_second"] - 1
) * 100
rollout_delta = (piecewise["rollout_seconds"] / eager["rollout_seconds"] - 1) * 100
work_delta = (piecewise["generated_tokens"] / eager["generated_tokens"] - 1) * 100
payload = {
    "schema_version": 1,
    "status": "PASS",
    "experiment": "qwen3_vanilla_epoch0_piecewise_tq1",
    "comparison_is_diagnostic": True,
    "comparison_limit": (
        "same workload and seed, but the eager reference predates the current source snapshot"
    ),
    "eager": eager,
    "piecewise_tq1": piecewise,
    "delta": {
        "response_tokens_per_second_percent": throughput_delta,
        "rollout_seconds_percent": rollout_delta,
        "generated_work_percent": work_delta,
    },
}
output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
output_md.write_text(
    "# Qwen3 Vanilla Epoch0 PIECEWISE TQ1\n\n"
    "Status: PASS\n\n"
    "| Arm | Response tokens | Rollout seconds | Response tok/s |\n"
    "|---|---:|---:|---:|\n"
    f"| Historical eager | {eager['generated_tokens']} | {eager['rollout_seconds']:.3f} | {eager['response_tokens_per_second']:.3f} |\n"
    f"| 0.11 PIECEWISE TQ1 | {piecewise['generated_tokens']} | {piecewise['rollout_seconds']:.3f} | {piecewise['response_tokens_per_second']:.3f} |\n\n"
    f"PIECEWISE throughput delta: {throughput_delta:+.2f}%.  "
    f"Rollout-time delta: {rollout_delta:+.2f}%.  "
    f"Generated-work delta: {work_delta:+.2f}%.\n\n"
    "This is a same-workload, same-seed diagnostic. The historical eager run was produced by an earlier source snapshot, so the delta is not a final causal paper result.\n"
)
print(output_md.read_text(), end="")
PY
}

require_inputs
write_or_verify_contracts
print_contract

if [[ "$ACTION" == dry-run ]]; then
    echo "[piecewise epoch0] dry run only; Ray and NPU were not started"
    exit 0
fi
if [[ "$ACTION" == summarize ]]; then
    summarize
    exit 0
fi

if [[ -e "$RUN_ROOT" ]]; then
    echo "refusing to overwrite existing run root: $RUN_ROOT" >&2
    echo "set PIECEWISE_EPOCH0_ROOT to a fresh path" >&2
    exit 2
fi

CACHE_ROOT="$EXPERIMENT_ROOT/cache"
RAY_TMPDIR_VALUE="${RAY_TMPDIR:-/tmp/qwen3_pw_tq1_${$}}"
mkdir -p "$CACHE_ROOT"/{xdg,hf,triton,torchair,ascend,work} "$RAY_TMPDIR_VALUE"

env -u VERL_PAIRED_REQUEST_SAMPLING_SEEDS \
    COMMON_EPOCH0_OUTPUT_ROOT="$EXPERIMENT_ROOT" \
    COMMON_EPOCH0_RUN_NAME="$RUN_NAME" \
    COMMON_EPOCH0_TRAIN_STEPS="$STEPS" \
    COMMON_EPOCH0_DATASET_FRACTION=0.005 \
    COMMON_EPOCH0_TRAIN_BATCH_SIZE="$PROMPTS_PER_STEP" \
    COMMON_EPOCH0_ROLLOUT_N="$ROLLOUT_N" \
    COMMON_EPOCH0_MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    COMMON_EPOCH0_MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    COMMON_EPOCH0_MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
    COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
    COMMON_EPOCH0_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    COMMON_EPOCH0_PROMPTS_TOTAL=$((STEPS * PROMPTS_PER_STEP)) \
    COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP="$EXPECTED_RESPONSES_PER_STEP" \
    COMMON_EPOCH0_KV_TOKENS_PER_RANK="$KV_TOKENS_PER_RANK" \
    COMMON_EPOCH0_PREEMPTION_POLICY=forbid \
    COMMON_EPOCH0_WORKLOAD_PROFILE_ID=qwen3_vanilla_epoch0_seed0_bs32_n16_len16384 \
    COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256="$(sha256_file "$PROTOCOL")" \
    COMMON_EPOCH0_EXECUTION_PROFILE=piecewise_tq1_attention_boundary_moe_graph \
    COMMON_EPOCH0_ORIGINAL_EXECUTION_CODE_SHA256="$(sha256_file "$CODE_MANIFEST")" \
    MODEL_PATH="$MODEL_PATH" \
    DISTCP_PATH="$DISTCP_PATH" \
    TRAIN_FILE="$TRAIN_FILE" \
    TEST_FILE="$TEST_FILE" \
    ADAFLOOR_GRAPH_BASE_RUNNER="$COMMON_RUNNER" \
    ADAFLOOR_ACLGRAPH_MODE=PIECEWISE \
    ADAFLOOR_GRAPH_CAPTURE_SIZES="$CAPTURE_SIZES" \
    VLLM_ASCEND_ELASTIC_ACLGRAPH=1 \
    VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=0 \
    VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1 \
    VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="$KV_TOKENS_PER_RANK" \
    VLLM_ENABLE_GRAPH_MODE=0 \
    TASK_QUEUE_ENABLE=1 \
    ROLLOUT_ENFORCE_EAGER=False \
    VERL_SIDECAR_ENABLE=0 \
    VERL_HCCL_IF_BASE_PORT_START=12000 \
    VERL_MASTER_PORT_START=28416 \
    RAY_TMPDIR="$RAY_TMPDIR_VALUE" \
    XDG_CACHE_HOME="$CACHE_ROOT/xdg" \
    HF_HOME="$CACHE_ROOT/hf" \
    TRITON_CACHE_DIR="$CACHE_ROOT/triton" \
    TORCHAIR_CACHE_HOME="$CACHE_ROOT/torchair" \
    ASCEND_CACHE_PATH="$CACHE_ROOT/ascend" \
    ASCEND_WORK_PATH="$CACHE_ROOT/work" \
    "$GRAPH_WRAPPER" \
        actor_rollout_ref.rollout.seed="$SEED" \
        actor_rollout_ref.rollout.temperature=0.9 \
        actor_rollout_ref.rollout.top_p=0.9 \
        actor_rollout_ref.rollout.top_k=50 \
        actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
        actor_rollout_ref.actor.megatron.seed=42 \
        actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False

summarize
