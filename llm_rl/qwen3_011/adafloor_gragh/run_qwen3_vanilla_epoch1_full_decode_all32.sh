#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-dry-run}

case "$ACTION" in
    dry-run|run) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_vanilla_epoch1_full_decode_all32.sh dry-run
  ./run_qwen3_vanilla_epoch1_full_decode_all32.sh run

Runs one updated-policy Vanilla Full16 epoch from the preserved common epoch0
checkpoint. FULL_DECODE_ONLY captures every integer batch shape from 1 through
32. A 5.6 GiB/NPU graph reserve is subtracted from the verified eager KV cap.
EOF
        exit 0
        ;;
    *)
        echo "unknown action: $ACTION" >&2
        exit 2
        ;;
esac

COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}
RESULT_ROOT=${FULL32_RESULT_ROOT:-/workspace/adafloor_graph_results/qwen3_vanilla_epoch1_full_decode_all32_seed0_5p6g_v1}
RUN_NAME=${FULL32_RUN_NAME:-qwen3_graph_vanilla_all32_seed0}
FAIR_RUNNER="$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh"
FILTERED_OPP=${VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH:-$SCRIPT_DIR/../qwen3/vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend}
EXTENSION=${VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION:-/workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so}
SAMPLER="$SCRIPT_DIR/internal/sample_npu_util.py"

CAPTURE_SIZES='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]'
KV_BLOCK_SIZE=128
KV_BYTES_PER_TOKEN=98304
KV_BLOCK_BYTES=$((KV_BLOCK_SIZE * KV_BYTES_PER_TOKEN))
EAGER_FULL16_TOKENS=380800
# Keep the reserve integral in shell arithmetic. The block-ceiling below turns
# 5.6 GiB into 478 12-MiB KV blocks, or 5.6015625 GiB of effective reserve.
GRAPH_RESERVE_BYTES=$((56 * 1024 * 1024 * 1024 / 10))
GRAPH_RESERVE_BLOCKS=$(((GRAPH_RESERVE_BYTES + KV_BLOCK_BYTES - 1) / KV_BLOCK_BYTES))
GRAPH_RESERVE_TOKENS=$((GRAPH_RESERVE_BLOCKS * KV_BLOCK_SIZE))
FULL16_KV_TOKENS=$((EAGER_FULL16_TOKENS - GRAPH_RESERVE_TOKENS))

require_file() {
    [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 2; }
}

require_dir() {
    [[ -d "$1" ]] || { echo "missing required directory: $1" >&2; exit 2; }
}

require_file "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT"
require_file "$COMMON_EPOCH0_ROOT/reuse.env"
require_dir "$COMMON_EPOCH0_ROOT/epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5"
require_file "$FAIR_RUNNER"
require_file "$EXTENSION"
require_file "$SAMPLER"
require_dir "$FILTERED_OPP"

FILTERED_OPP=$(realpath "$FILTERED_OPP")
FILTERED_OPP_BUNDLE_SHA256=$(find "$FILTERED_OPP" -type f -print0 \
    | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
EXTENSION=$(realpath "$EXTENSION")
EXTENSION_SHA256=$(sha256sum "$EXTENSION" | awk '{print $1}')

echo "root=$RESULT_ROOT"
echo "run_name=$RUN_NAME"
echo "common_epoch0=$COMMON_EPOCH0_ROOT"
echo "capture_sizes=$CAPTURE_SIZES"
echo "graph_reserve_bytes=$GRAPH_RESERVE_BYTES"
echo "graph_reserve_blocks=$GRAPH_RESERVE_BLOCKS"
echo "graph_reserve_tokens=$GRAPH_RESERVE_TOKENS"
echo "full16_kv_tokens=$FULL16_KV_TOKENS"
echo "epochs=1 steps=5 seed=0 actor_lr=1e-6"

if [[ "$ACTION" == dry-run ]]; then
    exit 0
fi

if [[ -e "$RESULT_ROOT/$RUN_NAME" ]]; then
    echo "refusing to overwrite existing run: $RESULT_ROOT/$RUN_NAME" >&2
    exit 2
fi

mkdir -p "$RESULT_ROOT"
PROTOCOL="$RESULT_ROOT/protocol.env"
if [[ -e "$PROTOCOL" ]]; then
    echo "refusing to reuse result root with an existing protocol: $PROTOCOL" >&2
    exit 2
fi

{
    printf 'schema_version=1\n'
    printf 'experiment=qwen3_vanilla_epoch1_full_decode_all32\n'
    printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
    printf 'seed=0\n'
    printf 'epochs=1\n'
    printf 'steps=5\n'
    printf 'actor_lr=1e-6\n'
    printf 'capture_sizes=%s\n' "$CAPTURE_SIZES"
    printf 'graph_reserve_bytes_per_npu=%s\n' "$GRAPH_RESERVE_BYTES"
    printf 'graph_reserve_blocks_per_npu=%s\n' "$GRAPH_RESERVE_BLOCKS"
    printf 'graph_reserve_tokens_per_npu=%s\n' "$GRAPH_RESERVE_TOKENS"
    printf 'full16_kv_tokens_per_rank=%s\n' "$FULL16_KV_TOKENS"
    printf 'kv_block_size_tokens=%s\n' "$KV_BLOCK_SIZE"
    printf 'kv_bytes_per_token=%s\n' "$KV_BYTES_PER_TOKEN"
    printf 'task_queue_enable=1\n'
    printf 'cudagraph_mode=FULL_DECODE_ONLY\n'
    printf 'native_sleep_mode=0\n'
    printf 'async_scheduling=false\n'
    printf 'filtered_opp_bundle_sha256=%s\n' "$FILTERED_OPP_BUNDLE_SHA256"
    printf 'ascend_extension_sha256=%s\n' "$EXTENSION_SHA256"
} > "$PROTOCOL"

CACHE_ROOT="$RESULT_ROOT/cache"
# Ray appends a long session/socket suffix, so this base must stay well below
# the 107-byte AF_UNIX limit regardless of the experiment result path.
RAY_TMPDIR=${FULL32_RAY_TMPDIR:-/tmp/af_g32_s0_ray}
mkdir -p \
    "$CACHE_ROOT/xdg" \
    "$CACHE_ROOT/hf" \
    "$CACHE_ROOT/triton" \
    "$CACHE_ROOT/torchair" \
    "$CACHE_ROOT/ascend" \
    "$CACHE_ROOT/work" \
    "$RAY_TMPDIR"
MEMORY_CSV="$RESULT_ROOT/npu_memory.csv"
python3 "$SAMPLER" --output "$MEMORY_CSV" --interval 1 &
sampler_pid=$!

stop_sampler() {
    if kill -0 "$sampler_pid" 2>/dev/null; then
        kill -INT "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
    fi
}
trap stop_sampler EXIT INT TERM

set +e
env -u FLOOR_KV_CAPS \
    COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    FAIR_OUTPUT_ROOT="$RESULT_ROOT" \
    FAIR_RUN_NAME="$RUN_NAME" \
    DYNAMIC_RUN_NAME="$RUN_NAME" \
    FAIR_START_EPOCH=1 \
    FAIR_TOTAL_EPOCHS=2 \
    FAIR_TRAIN_STEPS=5 \
    FAIR_TRAIN_BATCH_SIZE=32 \
    FAIR_ROLLOUT_N=16 \
    FAIR_MAX_RESPONSE_LENGTH=16384 \
    FAIR_PROMPTS_PER_EPOCH=160 \
    FAIR_FREEZE_ACTOR=0 \
    FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
    VERL_PAIRED_REQUEST_SAMPLING_SEEDS=0 \
    VANILLA_KV_TOKENS_PER_RANK="$FULL16_KV_TOKENS" \
    ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY \
    ADAFLOOR_GRAPH_CAPTURE_SIZES="$CAPTURE_SIZES" \
    ROLLOUT_ENFORCE_EAGER=False \
    VLLM_ENABLE_GRAPH_MODE=0 \
    VLLM_ASCEND_ELASTIC_ACLGRAPH=1 \
    VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1 \
    VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1 \
    VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION="$EXTENSION" \
    VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256="$EXTENSION_SHA256" \
    VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1 \
    VLLM_ROLLOUT_NATIVE_SLEEP_MODE=0 \
    VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=0 \
    VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1 \
    VLLM_ROLLOUT_ASYNC_SCHEDULING=false \
    VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true \
    VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true \
    VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1 \
    VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH="$FILTERED_OPP" \
    VLLM_ASCEND_FILTERED_CUSTOM_OPP_BUNDLE_SHA256="$FILTERED_OPP_BUNDLE_SHA256" \
    VLLM_ASCEND_FORCE_ALLTOALL_MOE=0 \
    VLLM_ASCEND_MC2_MIN_EP_SIZE=2 \
    VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1 \
    VLLM_ENABLE_MC2=1 \
    TASK_QUEUE_ENABLE=1 \
    VERL_SIDECAR_ENABLE=0 \
    PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
    VERL_HCCL_IF_BASE_PORT_START=12000 \
    VERL_MASTER_PORT_START=28416 \
    RAY_TMPDIR="$RAY_TMPDIR" \
    XDG_CACHE_HOME="$CACHE_ROOT/xdg" \
    HF_HOME="$CACHE_ROOT/hf" \
    TRITON_CACHE_DIR="$CACHE_ROOT/triton" \
    TORCHAIR_CACHE_HOME="$CACHE_ROOT/torchair" \
    ASCEND_CACHE_PATH="$CACHE_ROOT/ascend" \
    ASCEND_WORK_PATH="$CACHE_ROOT/work" \
    "$FAIR_RUNNER" vanilla \
    actor_rollout_ref.rollout.seed=0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY \
    "actor_rollout_ref.rollout.cudagraph_capture_sizes=$CAPTURE_SIZES" \
    actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
    2>&1 | tee "$RESULT_ROOT/runner.log"
run_rc=${PIPESTATUS[0]}
set -e

stop_sampler
trap - EXIT INT TERM
printf 'exit_code=%s\n' "$run_rc" > "$RESULT_ROOT/RUN_STATUS.txt"
exit "$run_rc"
