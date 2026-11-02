#!/usr/bin/env bash
# One controlled-length pair. No checkpoints created/deleted; no sidecar/graph.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${REVISION_QUOTA_ROOT:-/workspace/adafloor_revision_results/quota_controlled_pair_v1}
HF=/workspace/adafloor_revision_results/frontier_common_actor_hf_v2
WORKLOAD=${REVISION_QUOTA_WORKLOAD:-/workspace/adafloor_revision_results/quota_controlled_prepared_v2/workload.json}
ACTION=${1:-dry-run}
ORDER=${REVISION_QUOTA_ORDER:-min_skew release_area}
[[ "$ACTION" == dry-run || "$ACTION" == run ]] || exit 2
[[ "$ORDER" == 'min_skew release_area' || "$ORDER" == 'release_area min_skew' ]] || exit 2
[[ -s "$WORKLOAD" && -s "$HF/tensor_parity_audit.json" && -s "$HF/model/model.safetensors.index.json" ]]
[[ $(realpath -m "$ROOT") == /workspace/adafloor_revision_results/* ]]
printf 'workload=%s\ncommon_actor=%s/model\npolicy_pair=min_skew,release_area\nresponses=512\nfloor=4\nKV_tokens_per_rank=133120\nTQ=2\ngraph=false\nforced_length=true\nactor_updates=false\nsidecar=false\noutput=%s\n' "$WORKLOAD" "$HF" "$ROOT"
printf 'execution_order=%s\n' "$ORDER"
[[ "$ACTION" != dry-run ]] || exit 0
[[ ! -e "$ROOT" ]] || { echo "refusing existing run:$ROOT" >&2; exit 2; }
[[ -z $(pgrep -f '^python3 -m (verl.trainer.main_ppo|torch.distributed.run)' || true) ]] || exit 3
mkdir -p "$ROOT"
sha256sum "$0" "$REPO/tools/run_revision_quota_controlled.py" "$REPO/analysis_eval/audit_revision_quota_controlled.py" > "$ROOT/code_sha256.txt"
sha256sum "$REPO/vllm/v1/engine/llm_engine.py" "$REPO/vllm/v1/engine/core.py" \
    "$REPO/vllm_ascend/ops/fused_moe.py" "$REPO/vllm_ascend/worker/model_runner_v1.py" \
    "$REPO/vllm_ascend/envs.py" "$REPO/vllm_ascend/ops/moe/token_dispatcher.py" \
    "$REPO/r1_ascend/vllm_parallel_state.py" >> "$ROOT/code_sha256.txt"
sha256sum "$WORKLOAD" "$HF/tensor_parity_audit.json" > "$ROOT/input_sha256.txt"
cd "$REPO"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TASK_QUEUE_ENABLE=2 VLLM_ENABLE_GRAPH_MODE=0 VLLM_ENABLE_EXPERT_PARALLEL=1 VLLM_ENABLE_MC2=1
export USE_ALLTOALL_OVERLAP=1 ALL_TO_ALL_RESHARD=1 VLLM_ENABLE_FIX_ROUTE=0 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=800
export HCCL_IF_BASE_PORT=12000 VLLM_DP_SIZE=16 RAY_DEDUP_LOGS=0 OMP_NUM_THREADS=4
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_EVENT_ENABLE=0 ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
export VERL_SIDECAR_ENABLE=0
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE
for policy in $ORDER; do
    OUT="$ROOT/$policy"
    mkdir -p "$OUT/primary"
    printf 'policy=%s\nactor_frozen=true\nresponses=512\nfloor=4\nKV_tokens=133120\nTQ=2\ngraph=false\nignore_eos=true\nstartup_excluded=true\nrestore_reported_separately=true\n' "$policy" > "$OUT/protocol.env"
    printf 'execution_order=%s\n' "$ORDER" >> "$OUT/protocol.env"
    sha256sum -c "$ROOT/code_sha256.txt"
    sha256sum -c "$ROOT/input_sha256.txt"
    set +e
    python3 -m torch.distributed.run --standalone --nproc_per_node=16 \
        "$REPO/tools/run_revision_quota_controlled.py" --hf-root "$HF" --workload "$WORKLOAD" \
        --policy "$policy" --output-dir "$OUT/primary" > "$OUT/primary.log" 2>&1
    rc=$?
    set -e
    printf 'exit_code=%s\n' "$rc" >> "$OUT/protocol.env"
    if (( rc != 0 )); then echo "arm failed:$policy rc=$rc; evidence retained" >&2; exit "$rc"; fi
done
sha256sum -c "$ROOT/code_sha256.txt"
sha256sum -c "$ROOT/input_sha256.txt"
python3 "$REPO/analysis_eval/audit_revision_quota_controlled.py" \
    --root "$ROOT" --workload "$WORKLOAD" --output-dir "$ROOT/audit"
