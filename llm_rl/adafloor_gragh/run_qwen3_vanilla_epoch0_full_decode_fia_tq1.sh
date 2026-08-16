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
  ./run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh dry-run
  ./run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh run
  ./run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh summarize

Runs the vLLM/vLLM-Ascend 0.11 Qwen3-30B-A3B Vanilla Full16 epoch0
workload with native FULL_DECODE_ONLY ACLGraph. Prefill remains eager. Decode
captures the full model, including KV write, FIA Attention, MoE/HCCL, and dense
layers. FIA reserves one maximum workspace per capture shape and updates only
host sequence metadata before replay, following the 0.14 full-graph protocol.

The default capture list is [32], so each rank keeps one full-decode graph and
pads smaller live decode batches to that graph. Set FULL_DECODE_CAPTURE_SIZES
only for a deliberate memory/performance study.

Set FULL_DECODE_EPOCH0_ROOT to a fresh output root. The run action retains the
global_step_5 checkpoint and never overwrites an existing run.
EOF
        exit 0
        ;;
    *)
        echo "unknown action: $ACTION" >&2
        exit 2
        ;;
esac

EXPERIMENT_ROOT="${FULL_DECODE_EPOCH0_ROOT:-/workspace/adafloor_graph_results/qwen3_vanilla_epoch0_full_decode_fia_tq1_seed0}"
RUN_NAME="${FULL_DECODE_EPOCH0_RUN_NAME:-common_epoch0_full_decode_fia_tq1_seed0}"
RUN_ROOT="$EXPERIMENT_ROOT/$RUN_NAME"
EPOCH_DIR="$RUN_ROOT/epoch_000_mode0_probe"
EAGER_REFERENCE_ROOT="${FULL_DECODE_EAGER_REFERENCE_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
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
RESPONSES_PER_STEP=512
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=16384
MAX_NUM_BATCHED_TOKENS=17408
MAX_NUM_SEQS=32
KV_TOKENS_PER_RANK=380800
KV_BLOCK_SIZE=128
KV_BLOCKS=2975
GPU_MEMORY_UTILIZATION=0.9
ACTOR_LR=1e-6
CAPTURE_SIZES="${FULL_DECODE_CAPTURE_SIZES:-[32]}"
OPTIMIZATION_PROFILE="${FULL_DECODE_OPTIMIZATION_PROFILE:-baseline}"
case "$OPTIMIZATION_PROFILE" in
    baseline|v014_runtime_port) ;;
    *)
        echo "unsupported FULL_DECODE_OPTIMIZATION_PROFILE=$OPTIMIZATION_PROFILE" >&2
        exit 2
        ;;
esac

PROTOCOL="$EXPERIMENT_ROOT/protocol.env"
CODE_MANIFEST="$EXPERIMENT_ROOT/code_sha256.txt"
SUMMARY_JSON="$EXPERIMENT_ROOT/full_decode_epoch0_summary.json"
SUMMARY_MD="$EXPERIMENT_ROOT/full_decode_epoch0_summary.md"

CODE_PATHS=(
    "$SCRIPT_DIR/run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh"
    "$GRAPH_WRAPPER"
    "$COMMON_RUNNER"
    "$SCRIPT_DIR/run_mode0_no_shrink_baseline.sh"
    "$SCRIPT_DIR/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager_baseline_util.sh"
    "$SCRIPT_DIR/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"
    "$SCRIPT_DIR/verl/single_controller/ray/base.py"
    "$SCRIPT_DIR/verl/trainer/constants_ppo.py"
    "$SCRIPT_DIR/verl/trainer/config/ppo_megatron_trainer.yaml"
    "$SCRIPT_DIR/verl/trainer/config/_generated_ppo_megatron_trainer.yaml"
    "$SCRIPT_DIR/verl/trainer/config/rollout/rollout.yaml"
    "$SCRIPT_DIR/verl/workers/config/rollout.py"
    "$SCRIPT_DIR/verl/workers/megatron_workers.py"
    "$SCRIPT_DIR/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    "$SCRIPT_DIR/vllm_ascend/attention/attention_v1.py"
    "$SCRIPT_DIR/vllm_ascend/compilation/acl_graph.py"
    "$SCRIPT_DIR/vllm_ascend/envs.py"
    "$SCRIPT_DIR/vllm_ascend/platform.py"
    "$SCRIPT_DIR/vllm_ascend/worker/model_runner_v1.py"
)
if [[ -n "${FULL_DECODE_EXTRA_CODE_PATH:-}" ]]; then
    CODE_PATHS+=("$FULL_DECODE_EXTRA_CODE_PATH")
fi

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

require_inputs() {
    local path
    for path in "$GRAPH_WRAPPER" "$COMMON_RUNNER" "$ASCEND_EXTENSION" \
        "$MODEL_PATH" "$DISTCP_PATH" "$TRAIN_FILE" "$TEST_FILE" \
        "$EAGER_EPOCH_DIR"; do
        [[ -e "$path" ]] || {
            echo "missing required input: $path" >&2
            exit 2
        }
    done
    [[ "$CAPTURE_SIZES" =~ ^\[[[:space:]]*[1-9][0-9]*([[:space:]]*,[[:space:]]*[1-9][0-9]*)*[[:space:]]*\]$ ]] || {
        echo "invalid FULL_DECODE_CAPTURE_SIZES=$CAPTURE_SIZES" >&2
        exit 2
    }
    (( KV_TOKENS_PER_RANK / KV_BLOCK_SIZE == KV_BLOCKS )) || {
        echo "KV contract mismatch" >&2
        exit 2
    }
}

write_or_verify_contracts() {
    mkdir -p "$EXPERIMENT_ROOT"
    local code_tmp protocol_tmp
    code_tmp=$(mktemp "$EXPERIMENT_ROOT/.code.XXXXXX")
    protocol_tmp=$(mktemp "$EXPERIMENT_ROOT/.protocol.XXXXXX")
    sha256sum "${CODE_PATHS[@]}" > "$code_tmp"
    {
        echo 'schema_version=1'
        echo 'experiment=qwen3_vanilla_epoch0_full_decode_fia_tq1'
        echo "optimization_profile=$OPTIMIZATION_PROFILE"
        echo 'implementation_stack=vllm-0.11_vllm-ascend-0.11rc0'
        echo "output_root=$EXPERIMENT_ROOT"
        echo "run_name=$RUN_NAME"
        echo "model_path=$(realpath "$MODEL_PATH")"
        echo "distcp_path=$(realpath "$DISTCP_PATH")"
        echo "train_file=$(realpath "$TRAIN_FILE")"
        echo "train_file_sha256=$(sha256_file "$TRAIN_FILE")"
        echo "test_file=$(realpath "$TEST_FILE")"
        echo "test_file_sha256=$(sha256_file "$TEST_FILE")"
        echo "seed=$SEED"
        echo "steps=$STEPS"
        echo "prompts_per_step=$PROMPTS_PER_STEP"
        echo "rollout_n=$ROLLOUT_N"
        echo "responses_per_step=$RESPONSES_PER_STEP"
        echo "max_prompt_length=$MAX_PROMPT_LENGTH"
        echo "max_response_length=$MAX_RESPONSE_LENGTH"
        echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
        echo "max_num_seqs=$MAX_NUM_SEQS"
        echo "kv_tokens_per_rank=$KV_TOKENS_PER_RANK"
        echo "kv_blocks=$KV_BLOCKS"
        echo 'task_queue_enable=1'
        echo 'cudagraph_mode=FULL_DECODE_ONLY'
        echo "capture_sizes=$CAPTURE_SIZES"
        echo 'attention_backend=fia_max_workspace'
        echo 'cudagraph_copy_inputs=false'
        echo 'prefill_execution=eager'
        echo 'decode_execution=full_aclgraph'
        echo 'torchair=false'
        echo 'graphex=false'
        echo 'tail_guard=false'
        echo 'shrink=false'
        echo 'sidecar=false'
        echo "native_sleep_mode=${VLLM_ROLLOUT_NATIVE_SLEEP_MODE:-0}"
        echo "native_sleep_level=${VLLM_ROLLOUT_SLEEP_LEVEL:-unset}"
        echo "reuse_aclgraph_after_weight_update=${VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE:-0}"
        echo "async_scheduling=${VLLM_ROLLOUT_ASYNC_SCHEDULING:-false}"
        echo "prefix_caching=${VLLM_ROLLOUT_ENABLE_PREFIX_CACHING:-false}"
        echo "chunked_prefill=${VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL:-true}"
        echo "filtered_custom_opp=${VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP:-0}"
        echo "filtered_custom_opp_path=${VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH:-unset}"
        echo "filtered_custom_opp_bundle_sha256=${VLLM_ASCEND_FILTERED_CUSTOM_OPP_BUNDLE_SHA256:-unset}"
    } > "$protocol_tmp"
    local candidate destination
    for candidate in "$code_tmp" "$protocol_tmp"; do
        if [[ "$candidate" == "$code_tmp" ]]; then
            destination="$CODE_MANIFEST"
        else
            destination="$PROTOCOL"
        fi
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

summarize() {
    [[ -f "$RUN_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" ]] || {
        echo "completed common epoch0 marker is missing: $RUN_ROOT" >&2
        exit 3
    }
    python3 - "$EPOCH_DIR" "$EAGER_EPOCH_DIR" "$SUMMARY_JSON" "$SUMMARY_MD" <<'PY'
import json
import re
import sys
from pathlib import Path

graph_dir, eager_dir, json_path, md_path = map(Path, sys.argv[1:])

def read_arm(root: Path, graph: bool):
    logs = list((root / "logs").glob("*.txt"))
    if len(logs) != 1:
        raise SystemExit(f"expected one log under {root}, got {len(logs)}")
    text = logs[0].read_text(errors="replace")
    rollout = [float(x) for x in re.findall(r"rollout_output_time_s:\s*([0-9.eE+-]+)", text)]
    rewards = [float(x) for x in re.findall(r"critic/score/mean:([0-9.eE+-]+)", text)]
    aborted = [float(x) for x in re.findall(r"response/aborted_ratio:([0-9.eE+-]+)", text)]
    if not (len(rollout) == len(rewards) == len(aborted) == 5):
        raise SystemExit(f"incomplete five-step metrics in {logs[0]}")
    if any(aborted):
        raise SystemExit(f"nonzero abort ratio in {logs[0]}")
    tokens = []
    for step in range(1, 6):
        lengths = [int(x) for x in (root / "rollout_length" / f"length_{step}.txt").read_text().split()]
        if len(lengths) != 512 or any(x <= 0 or x > 16384 for x in lengths):
            raise SystemExit(f"invalid response lengths at step {step}")
        if sum(1 for _ in (root / "rollout_data" / f"{step}.jsonl").open("rb")) != 512:
            raise SystemExit(f"invalid rollout rows at step {step}")
        tokens.append(sum(lengths))
    if graph:
        markers = (
            "FULL_DECODE_ONLY compilation enabled on NPU",
            "attention_backend=fia_max_workspace cudagraph_copy_inputs=false",
            "FULL_DECODE_ONLY FIA max-workspace Attention capture active",
            "Replaying aclgraph",
        )
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise SystemExit(f"missing FULL graph evidence: {missing}")
    lower = text.lower()
    if re.search(r"preempting request|request preempted", lower):
        raise SystemExit("preemption evidence found")
    if re.search(r"npu out of memory|memory_allocation_failure|outofmemoryerror", lower):
        raise SystemExit("OOM evidence found")
    seconds = sum(rollout)
    return {
        "log": str(logs[0].resolve()),
        "tokens_by_step": tokens,
        "rollout_seconds_by_step": rollout,
        "reward_by_step": rewards,
        "generated_tokens": sum(tokens),
        "rollout_seconds": seconds,
        "response_tokens_per_second": sum(tokens) / seconds,
        "mean_reward": sum(rewards) / len(rewards),
    }

graph = read_arm(graph_dir, True)
eager = read_arm(eager_dir, False)
throughput_delta = (graph["response_tokens_per_second"] / eager["response_tokens_per_second"] - 1) * 100
time_delta = (graph["rollout_seconds"] / eager["rollout_seconds"] - 1) * 100
work_delta = (graph["generated_tokens"] / eager["generated_tokens"] - 1) * 100
payload = {
    "schema_version": 1,
    "status": "PASS",
    "experiment": "qwen3_vanilla_epoch0_full_decode_fia_tq1",
    "comparison_is_diagnostic": True,
    "comparison_limit": "same seed and workload, but the eager reference predates this source snapshot",
    "eager": eager,
    "full_decode_fia_tq1": graph,
    "delta": {
        "response_tokens_per_second_percent": throughput_delta,
        "rollout_seconds_percent": time_delta,
        "generated_work_percent": work_delta,
    },
}
json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
md_path.write_text(
    "# Qwen3 Vanilla Epoch0 FULL_DECODE_ONLY FIA TQ1\n\n"
    "Status: PASS\n\n"
    "| Arm | Response tokens | Rollout seconds | Response tok/s |\n"
    "|---|---:|---:|---:|\n"
    f"| Historical eager | {eager['generated_tokens']} | {eager['rollout_seconds']:.3f} | {eager['response_tokens_per_second']:.3f} |\n"
    f"| 0.11 FULL/FIA TQ1 | {graph['generated_tokens']} | {graph['rollout_seconds']:.3f} | {graph['response_tokens_per_second']:.3f} |\n\n"
    f"Throughput delta: {throughput_delta:+.2f}%. Rollout-time delta: {time_delta:+.2f}%. Generated-work delta: {work_delta:+.2f}%.\n\n"
    "The historical eager arm is not a same-source-snapshot causal control.\n"
)
print(md_path.read_text(), end="")
PY
}

require_inputs
write_or_verify_contracts
echo "[full decode epoch0] action=$ACTION output=$RUN_ROOT"
echo "[full decode epoch0] seed=$SEED steps=$STEPS prompts=$PROMPTS_PER_STEP n=$ROLLOUT_N max_response=$MAX_RESPONSE_LENGTH"
echo "[full decode epoch0] mode=FULL_DECODE_ONLY backend=fia_max_workspace capture_sizes=$CAPTURE_SIZES task_queue=1"

if [[ "$ACTION" == dry-run ]]; then
    echo "[full decode epoch0] dry run only; Ray and NPU were not started"
    exit 0
fi
if [[ "$ACTION" == summarize ]]; then
    summarize
    exit 0
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "refusing to overwrite existing run root: $RUN_ROOT" >&2
    exit 2
fi

CACHE_ROOT="$EXPERIMENT_ROOT/cache"
RAY_TMPDIR_VALUE="${RAY_TMPDIR:-/tmp/qwen3_full_fia_tq1_${$}}"
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
    COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP="$RESPONSES_PER_STEP" \
    COMMON_EPOCH0_KV_TOKENS_PER_RANK="$KV_TOKENS_PER_RANK" \
    COMMON_EPOCH0_PREEMPTION_POLICY=forbid \
    COMMON_EPOCH0_WORKLOAD_PROFILE_ID=qwen3_vanilla_epoch0_seed0_bs32_n16_len16384 \
    COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256="$(sha256_file "$PROTOCOL")" \
    COMMON_EPOCH0_EXECUTION_PROFILE="full_decode_fia_tq1_${OPTIMIZATION_PROFILE}" \
    COMMON_EPOCH0_ORIGINAL_EXECUTION_CODE_SHA256="$(sha256_file "$CODE_MANIFEST")" \
    MODEL_PATH="$MODEL_PATH" DISTCP_PATH="$DISTCP_PATH" \
    TRAIN_FILE="$TRAIN_FILE" TEST_FILE="$TEST_FILE" \
    ADAFLOOR_GRAPH_BASE_RUNNER="$COMMON_RUNNER" \
    ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY \
    ADAFLOOR_GRAPH_CAPTURE_SIZES="$CAPTURE_SIZES" \
    VLLM_ASCEND_ELASTIC_ACLGRAPH=1 \
    VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1 \
    VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1 \
    VLLM_ASCEND_FULL_DECODE_ATTENTION_BACKEND=fia_max_workspace \
    VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="$KV_TOKENS_PER_RANK" \
    VLLM_ENABLE_GRAPH_MODE=0 TASK_QUEUE_ENABLE=1 \
    ROLLOUT_ENFORCE_EAGER=False VERL_SIDECAR_ENABLE=0 \
    VERL_HCCL_IF_BASE_PORT_START=12000 VERL_MASTER_PORT_START=28416 \
    RAY_TMPDIR="$RAY_TMPDIR_VALUE" XDG_CACHE_HOME="$CACHE_ROOT/xdg" \
    HF_HOME="$CACHE_ROOT/hf" TRITON_CACHE_DIR="$CACHE_ROOT/triton" \
    TORCHAIR_CACHE_HOME="$CACHE_ROOT/torchair" \
    ASCEND_CACHE_PATH="$CACHE_ROOT/ascend" ASCEND_WORK_PATH="$CACHE_ROOT/work" \
    "$GRAPH_WRAPPER" \
        actor_rollout_ref.rollout.seed="$SEED" \
        actor_rollout_ref.rollout.temperature=0.9 \
        actor_rollout_ref.rollout.top_p=0.9 \
        actor_rollout_ref.rollout.top_k=50 \
        actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
        actor_rollout_ref.actor.megatron.seed=42 \
        actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False

summarize
