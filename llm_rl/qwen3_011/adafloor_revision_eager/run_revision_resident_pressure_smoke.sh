#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HF=/workspace/adafloor_revision_results/frontier_common_actor_hf_v2
OUT=${REVISION_PRESSURE_ROOT:-/workspace/adafloor_revision_results/resident_pressure_npu_v1}
ACTION=${1:-dry-run}
[[ "$ACTION" == dry-run || "$ACTION" == run ]] || exit 2
[[ -s "$HF/tensor_parity_audit.json" && $(realpath -m "$OUT") == /workspace/adafloor_revision_results/* && ! -e "$OUT" ]]
printf 'experiment=real_resident_pressure_functional_smoke\nworld=16\nextra_gib_per_rank=14\nadmission_hard_cap=8192\nactual_greedy_test_cap=16\nmodel=%s/model\nTQ=2\ngraph=false\n' "$HF"
[[ "$ACTION" == run ]] || exit 0
[[ -z $(pgrep -f '^python3 -m (verl.trainer.main_ppo|torch.distributed.run)' || true) ]] || exit 3
mkdir -p "$OUT"
cd "$REPO"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TASK_QUEUE_ENABLE=2 VLLM_ENABLE_GRAPH_MODE=0 VLLM_ENABLE_EXPERT_PARALLEL=1 VLLM_ENABLE_MC2=1
export USE_ALLTOALL_OVERLAP=1 ALL_TO_ALL_RESHARD=1 VLLM_ENABLE_FIX_ROUTE=0 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=800
export HCCL_IF_BASE_PORT=12000 OMP_NUM_THREADS=4 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE
sha256sum "$0" "$REPO/tools/run_revision_resident_pressure_smoke.py" "$REPO/verl/utils/revision_kv_guard.py" > "$OUT/code_sha256.txt"
set +e
python3 -m torch.distributed.run --standalone --nproc_per_node=16 "$REPO/tools/run_revision_resident_pressure_smoke.py" \
    --hf-root "$HF" --output-dir "$OUT" > "$OUT/primary.log" 2>&1
rc=$?
set -e
printf 'exit_code=%s\nextra_gib_per_rank=14\nadmission_hard_cap=8192\nactual_test_tokens=16\ngraph=false\nTQ=2\n' "$rc" > "$OUT/protocol.env"
exit "$rc"
