#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_mode2_perf_clean_test.sh --floor {2|4|8}

Environment overrides:
  MODE2_RESIDENT_EXPERT_SLOTS=8
  MODE2_SINGLE_DISPATCH=1
  MODE2_UNIFORM_MC2_ACTIVE_MASK=1
  MODE2_TRIM_PADDING_MLP=1
  VLLM_ASCEND_MODE2_ENABLE_GENERIC_HEADROOM=0
  VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=2147483648
  VLLM_ASCEND_MODE2_MOE_DISPATCH_HEADROOM_BYTES=6442450944
  VLLM_ASCEND_MODE2_MAX_KV_TOKENS=326656
  PERF_CLEAN_TRAINING_STEPS=1
  REPO_ROOT=/path/to/log/output/root
  PATCH_TREE=/path/to/code/tree
  HCCL_IF_BASE_PORT=47441
  MASTER_PORT=26440
  VERL_HCCL_IF_BASE_PORT_START=47441
  HCCL_HOST_SOCKET_PORT_RANGE=auto
  HCCL_NPU_SOCKET_PORT_RANGE=auto
  CHECK_NPU_IDLE=1

This reproduces the workload configuration recorded by the corresponding
pure-highkv-mode1-{2,4,8}.txt run while selecting mode=2 (CPU/NPU hybrid).
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
PATCH_TREE="${PATCH_TREE:-$REPO_ROOT}"
LAUNCHER="$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"

floor=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--floor)
            [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; usage >&2; exit 2; }
            floor="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$floor" ]]; then
    echo "--floor is required for an isolated mode2 comparison run" >&2
    usage >&2
    exit 2
fi

case "$floor" in
    2|4|8) ;;
    *)
        echo "mode2 pure-highkv comparison requires floor 2, 4, or 8; got floor=$floor" >&2
        exit 2
        ;;
esac
if [[ ! -x "$LAUNCHER" ]]; then
    echo "missing executable launcher: $LAUNCHER" >&2
    exit 1
fi

# Ports are namespace-local, so they cannot detect a job running in another
# container. Refuse to start whenever the device process table is non-empty.
if [[ "${CHECK_NPU_IDLE:-1}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "CHECK_NPU_IDLE is enabled but npu-smi is unavailable" >&2
        exit 1
    fi
    npu_process_rows=$(
        npu-smi info | awk '
            /Process id/ { in_process_table = 1; next }
            in_process_table && /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]*[0-9]+[[:space:]]+/ { print }
        '
    )
    if [[ -n "$npu_process_rows" ]]; then
        echo "NPU devices are already in use; refusing to start:" >&2
        printf '%s\n' "$npu_process_rows" >&2
        exit 1
    fi
fi

# The current Ray WorkerGroup allocator starts at 20000 and applies 4096-port
# phase offsets for actor/rollout/ref. Detect stale or concurrent jobs before
# model loading instead of reporting their bind collision as a mode=2 failure.
worker_group_ports=(20000 24096 28192)
for port in "${worker_group_ports[@]}"; do
    if ss -lnt "sport = :$port" | tail -n +2 | grep -q .; then
        echo "HCCL WorkerGroup port is already in use: $port" >&2
        exit 1
    fi
done

cd "$PATCH_TREE"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
tee_log="$REPO_ROOT/mode2_floor${floor}_perf_clean_${stamp}.log"

# Mode=2 keeps a fixed NPU-resident expert prefix and serves overflow experts
# through the CPU/NPU hybrid wave path after shrinking.
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=2
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="$floor"
export VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS="${MODE2_RESIDENT_EXPERT_SLOTS:-8}"

# Default mode=2 path: keep one synchronized MC2 dispatch/combine sequence on
# every rank, plan expert waves from real tokens only, and skip padding rows at
# the MLP boundary. Override these switches only for controlled diagnostics.
export VLLM_ASCEND_MODE2_SINGLE_DISPATCH="${MODE2_SINGLE_DISPATCH:-1}"
export VLLM_ASCEND_MODE2_UNIFORM_MC2_ACTIVE_MASK="${MODE2_UNIFORM_MC2_ACTIVE_MASK:-1}"
export VLLM_ASCEND_MODE2_TRIM_PADDING_MLP="${MODE2_TRIM_PADDING_MLP:-1}"

# Fixed-slot weights are already represented by profile_run(). Skip the broad
# additive elastic budgets and retain one mode=2 MC2 dispatch peak. The resident
# expert capacity is floor-invariant, so floor 2/4/8 use the same margins. Set
# the generic switch to 1 to restore the broad conservative budgets for diagnostics.
export VLLM_ASCEND_MODE2_ENABLE_GENERIC_HEADROOM="${VLLM_ASCEND_MODE2_ENABLE_GENERIC_HEADROOM:-0}"
export VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES="${VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES:-2147483648}"
export VLLM_ASCEND_MODE2_MOE_DISPATCH_HEADROOM_BYTES="${VLLM_ASCEND_MODE2_MOE_DISPATCH_HEADROOM_BYTES:-6442450944}"
export VLLM_ASCEND_MODE2_MAX_KV_TOKENS="${VLLM_ASCEND_MODE2_MAX_KV_TOKENS:-326656}"

# Keep diagnostic overhead out of the performance comparison.
export VLLM_ASCEND_CUSTOM_MODE1_DEBUG=0
export VLLM_ASCEND_CUSTOM_MODE1_TIMING_EVENTS=0
export VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG=0
export VLLM_ASCEND_MODE3_TRANSFER_LOG=0
export VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG=0
export VLLM_ASCEND_MODE3_TIMING_LOG=0
export VLLM_ASCEND_MODE3_TIMING_SYNC=0
export VLLM_ASCEND_MODE3_TIMING_EVERY=1000000
export VLLM_ASCEND_MODE3_TIMING_FIRST_N=0
export VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS=0
export VLLM_ASCEND_BUCKET_OP_PROFILE=0
export VLLM_ASCEND_BUCKET_OP_PROFILE_BY_STAGE=0
export VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS=""
export VLLM_ASCEND_DUMMY_WASTE_TIMING=0
export VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=0
export VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE=0
export VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS=0

source "$SCRIPT_DIR/perf_clean_common_env.sh"

export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-47441}"
export MASTER_PORT="${MASTER_PORT:-26440}"
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-47441}"
# Actor, rollout, and reference workers can create more than one HCCL process
# on the same NPU. Without ranges, HCCL falls back to fixed socket ports (for
# example 16666) and the second process can fail with Bind_Failed(EJ0003).
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"

export HOME="$REPO_ROOT"
export CONFIG_DIR="$PATCH_TREE/verl/trainer/config"
export PYTHONPATH="$PATCH_TREE${PYTHONPATH:+:$PYTHONPATH}"

printf '[mode2 perf clean] runtime_cwd=%s\n' "$PATCH_TREE"
printf '[mode2 perf clean] tee_log=%s\n' "$tee_log"
printf '[mode2 perf clean] floor=%s resident_slots=%s single_dispatch=%s\n' \
    "$floor" \
    "$VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS" \
    "$VLLM_ASCEND_MODE2_SINGLE_DISPATCH"
printf '[mode2 perf clean] uniform_mc2_mask=%s trim_padding_mlp=%s\n' \
    "$VLLM_ASCEND_MODE2_UNIFORM_MC2_ACTIVE_MASK" \
    "$VLLM_ASCEND_MODE2_TRIM_PADDING_MLP"
printf '[mode2 perf clean] lightweight_generic_headroom=%s kv_init_headroom=%s shared_dispatch_headroom=%s max_kv_tokens=%s\n' \
    "$VLLM_ASCEND_MODE2_ENABLE_GENERIC_HEADROOM" \
    "$VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES" \
    "$VLLM_ASCEND_MODE2_MOE_DISPATCH_HEADROOM_BYTES" \
    "$VLLM_ASCEND_MODE2_MAX_KV_TOKENS"
printf '[mode2 perf clean] max_model_len=%s max_batched_tokens=%s gpu_memory_utilization=%s epochs=%s steps=%s\n' \
    "$ROLLOUT_MAX_MODEL_LEN" \
    "$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    "$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    "$TRAINER_TOTAL_EPOCHS" \
    "$PERF_CLEAN_TRAINING_STEPS"
printf '[mode2 perf clean] ports HCCL_IF_BASE_PORT=%s MASTER_PORT=%s VERL_HCCL_IF_BASE_PORT_START=%s\n' \
    "$HCCL_IF_BASE_PORT" "$MASTER_PORT" "$VERL_HCCL_IF_BASE_PORT_START"
printf '[mode2 perf clean] socket_ranges host=%s npu=%s\n' \
    "$HCCL_HOST_SOCKET_PORT_RANGE" "$HCCL_NPU_SOCKET_PORT_RANGE"
printf '[mode2 perf clean] worker_group_phase_ports=%s\n' "${worker_group_ports[*]}"
print_perf_clean_common_config

bash "$LAUNCHER" "${PERF_CLEAN_LAUNCHER_ARGS[@]}" 2>&1 | tee "$tee_log"
