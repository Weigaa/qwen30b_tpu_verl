#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HF=/workspace/adafloor_revision_results/frontier_common_actor_hf_v2
WORLD=${REVISION_PREFLIGHT_WORLD:-8}
OUT=${REVISION_PREFLIGHT_ROOT:-/workspace/adafloor_revision_results/frontier_ep${WORLD}_primary_preflight_v1}
ACTION=${1:-dry-run}
[[ "$ACTION" == dry-run || "$ACTION" == run ]] || exit 2
[[ "$WORLD" == 8 || "$WORLD" == 16 ]]
[[ -s "$HF/model/config.json" && -s "$HF/frontier_workload.json" ]]
[[ $(realpath -m "$OUT") == /workspace/adafloor_revision_results/* && ! -e "$OUT" ]]
printf 'purpose=structural_EP%s_preflight_not_frontier_result\nmodel=%s/model\nmax_tokens=16\nrequest_count=256\neager=true\nTQ=2\n' "$WORLD" "$HF"
[[ "$ACTION" == run ]] || exit 0
[[ -s "$HF/tensor_parity_audit.json" ]] || { echo 'weight parity audit not finished' >&2; exit 2; }
[[ -z $(pgrep -f '^python3 -m verl.trainer.main_ppo' || true) ]] || exit 3
mkdir -p "$OUT"
cd "$REPO"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=$([[ "$WORLD" == 8 ]] && echo 0,1,2,3,4,5,6,7 || echo 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
export TASK_QUEUE_ENABLE=2 VLLM_ENABLE_GRAPH_MODE=0 VLLM_ENABLE_EXPERT_PARALLEL=1
export VLLM_ENABLE_MC2=1 USE_ALLTOALL_OVERLAP=1 ALL_TO_ALL_RESHARD=1 VLLM_ENABLE_FIX_ROUTE=0
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=800 HCCL_IF_BASE_PORT=12000
export VLLM_DP_SIZE="$WORLD" RAY_DEDUP_LOGS=0 OMP_NUM_THREADS=4
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_EVENT_ENABLE=0 ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE
python3 -m torch.distributed.run --standalone --nproc_per_node="$WORLD" \
    "$REPO/tools/run_revision_frontier_primary_preflight.py" --world-size "$WORLD" \
    --model "$HF/model" --workload "$HF/frontier_workload.json" \
    --weight-audit "$HF/tensor_parity_audit.json" --output-dir "$OUT"
