#!/usr/bin/env bash
# Three matched finite-batch operating points, not an unlimited-demand frontier.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${REVISION_FRONTIER_ROOT:-/workspace/adafloor_revision_results/frontier_windows_v1}
HF=/workspace/adafloor_revision_results/frontier_common_actor_hf_v2
BUDGET=${REVISION_FRONTIER_BUDGET:-300}
ARM=${1:-dry-run}
[[ "$ARM" == dry-run || "$ARM" == primary_only || "$ARM" == static8 || "$ARM" == dynamic4 ]] || exit 2
[[ -s "$HF/tensor_parity_audit.json" && -s "$HF/ep8_ep16_preflight_audit.json" ]]
[[ $(realpath -m "$ROOT") == /workspace/adafloor_revision_results/* ]]
printf 'model=%s/model\npolicy=common_epoch0_global_step5_frozen\nrequests=256\nresponse_limit=4608\nrequest_seeds=paired\nTQ=2\ngraph=false\nwindow_seconds=%s\nwindow_start=prepared_primary_cold_sidecar\nprimary_demand=one_closed_batch\nlate_completions=uncredited\n' "$HF" "$BUDGET"
[[ "$ARM" != dry-run ]] || exit 0
OUT="$ROOT/$ARM"
[[ ! -e "$OUT" ]] || { echo "refusing existing arm:$OUT" >&2; exit 2; }
[[ -z $(pgrep -f '^python3 -m (verl.trainer.main_ppo|torch.distributed.run)' || true) ]] || exit 3
mkdir -p "$OUT/primary" "$OUT/sidecar"
printf 'arm=%s\npolicy=common_epoch0_global_step5_frozen\nprimary_requests=256\nresponse_limit=4608\nTQ=2\ngraph=false\nwindow_seconds=%s\nwindow_start=prepared_primary_cold_sidecar\nfinite_batch=true\nlate_completions_uncredited=true\nprimary_deadline_abort=false\n' "$ARM" "$BUDGET" > "$OUT/protocol.env"
sha256sum "$0" "$REPO/tools/run_revision_frontier_primary_window.py" \
    "$REPO/internal/watch_elastic_shrink_and_run_sidecar.sh" "$REPO/internal/run_elastic_sidecar_infer.sh" \
    "$REPO/verl/utils/sidecar_restore_handshake.py" > "$OUT/code_sha256.txt"
sha256sum "$HF/frontier_workload.json" "$HF/tensor_parity_audit.json" "$HF/ep8_ep16_preflight_audit.json" > "$OUT/input_sha256.txt"
cd "$REPO"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
WORLD=$([[ "$ARM" == static8 ]] && echo 8 || echo 16)
export ASCEND_RT_VISIBLE_DEVICES=$([[ "$WORLD" == 8 ]] && echo 0,1,2,3,4,5,6,7 || echo 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
export TASK_QUEUE_ENABLE=2 VLLM_ENABLE_GRAPH_MODE=0 VLLM_ENABLE_EXPERT_PARALLEL=1 VLLM_ENABLE_MC2=1
export USE_ALLTOALL_OVERLAP=1 ALL_TO_ALL_RESHARD=1 VLLM_ENABLE_FIX_ROUTE=0 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=800
export HCCL_IF_BASE_PORT=12000 VLLM_DP_SIZE="$WORLD" RAY_DEDUP_LOGS=0 OMP_NUM_THREADS=4
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_EVENT_ENABLE=0 ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
export VERL_SIDECAR_ENABLE=$([[ "$ARM" == primary_only ]] && echo 0 || echo 1)
export VERL_SIDECAR_LOG_DIR="$OUT/sidecar" VERL_SIDECAR_STATE_DIR="$OUT/sidecar/state"
export VERL_SIDECAR_RESTORE_HANDSHAKE_DIR="$OUT/sidecar/restore_handshake"
export VERL_SIDECAR_LEASE_LOG="$OUT/sidecar/lease.log" VERL_SIDECAR_LOG_FILE="$OUT/sidecar/infer.log"
export VERL_SIDECAR_OUTPUT_FILE="$OUT/sidecar/outputs.jsonl"
export VERL_SIDECAR_MODEL_PATH=/data/Qwen2.5-1.5B-Instruct VERL_SIDECAR_PROMPTS_FILE=/data/gsm8k
export VERL_SIDECAR_DATA_SPLIT=train VERL_SIDECAR_PARALLEL_MODE=dp VERL_SIDECAR_TENSOR_PARALLEL_SIZE=1
export VERL_SIDECAR_DATA_PARALLEL_SIZE=1 VERL_SIDECAR_ENABLE_EXPERT_PARALLEL=0 VERL_SIDECAR_ENFORCE_EAGER=1
export VERL_SIDECAR_N=1 VERL_SIDECAR_TEMPERATURE=0.0 VERL_SIDECAR_TOP_P=1.0 VERL_SIDECAR_MAX_MODEL_LEN=6144
export VERL_SIDECAR_GPU_MEMORY_UTILIZATION=.80 VERL_SIDECAR_MAX_TOKENS=4096 VERL_SIDECAR_MAX_NUM_SEQS=32
export VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=8192 VERL_SIDECAR_MAX_PROMPTS=1024 VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA=128
export VERL_SIDECAR_GENERATE_CHUNK_SIZE=32 VERL_SIDECAR_REPEAT_UNTIL_KILLED=1 VERL_SIDECAR_STREAM_CHECKPOINT=1
export VERL_SIDECAR_START_ONCE=1 VERL_SIDECAR_WORLD_SIZE=16 VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE=1
if [[ "$ARM" == dynamic4 ]]; then
    # A legal forecast miss can leave the primary at Full16 or only reach8.
    # The final-floor4 lease may never form; idle release is valid and credits
    # ZERO sidecar work, never a fabricated running-lease safety certificate.
    export VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE=0
    sha256sum "$HF/dynamic_planner_layout.json" >> "$OUT/input_sha256.txt"
fi
export VERL_SIDECAR_STOP_ACK_TIMEOUT_SECONDS=60 VERL_SIDECAR_WATCH_POLL_INTERVAL=.1
unset VERL_SIDECAR_NPU_DEVICES RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE
WATCHER=""
cleanup() {
    if [[ -n "$WATCHER" ]] && kill -0 "$WATCHER" 2>/dev/null; then
        kill "$WATCHER" 2>/dev/null || true
        wait "$WATCHER" 2>/dev/null || true
    fi
}
trap cleanup EXIT
if [[ "$ARM" != primary_only ]]; then
    if [[ "$ARM" == static8 ]]; then
        export VERL_SIDECAR_START_TRIGGER=static_ready VERL_SIDECAR_EXPECTED_ACTIVE_RANKS=8 VERL_SIDECAR_REPLICA_COUNT=8
        export VERL_SIDECAR_REQUIRE_SHRINK_QUORUM=0 VERL_SIDECAR_STATIC_PRIMARY_DIR="$OUT/primary"
    else
        export VERL_SIDECAR_START_TRIGGER=shrink_done VERL_SIDECAR_EXPECTED_ACTIVE_RANKS=4 VERL_SIDECAR_REPLICA_COUNT=12
        # Second transition is8->4; all8 participants must report. The audit
        # additionally verifies the preceding16-report16->8 transition.
        export VERL_SIDECAR_REQUIRE_SHRINK_QUORUM=1 VERL_SIDECAR_SHRINK_QUORUM_SIZE=8
    fi
    touch "$OUT/primary.log"
    bash "$REPO/internal/watch_elastic_shrink_and_run_sidecar.sh" "$OUT/primary.log" > "$OUT/sidecar/monitor.log" 2>&1 &
    WATCHER=$!
fi
set +e
python3 -m torch.distributed.run --standalone --nproc_per_node="$WORLD" \
    "$REPO/tools/run_revision_frontier_primary_window.py" --arm "$ARM" --hf-root "$HF" \
    --output-dir "$OUT/primary" --budget-seconds "$BUDGET" > "$OUT/primary.log" 2>&1
rc=$?
set -e
printf 'exit_code=%s\n' "$rc" >> "$OUT/protocol.env"
sha256sum -c "$OUT/code_sha256.txt"
sha256sum -c "$OUT/input_sha256.txt"
exit "$rc"
