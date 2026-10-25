#!/usr/bin/env bash
# Formal v0.14 native-sleep level-2 + AdaFloor + FULL_DECODE_ONLY experiment.
# This is an execution entry point, not a smoke probe. It restores the public
# epoch0 checkpoint and runs the configured policy set through full epochs.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

ACTION=${1:-dry-run}
case "$ACTION" in
    dry-run|smoke|run|verify) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_v014_full_decode_sleep2_adafloor.sh dry-run
  ./run_qwen3_v014_full_decode_sleep2_adafloor.sh smoke
  ./run_qwen3_v014_full_decode_sleep2_adafloor.sh run
  ./run_qwen3_v014_full_decode_sleep2_adafloor.sh verify

The default formal profile is `full`: Vanilla, LengthSort, LengthSort+Guard,
MinSkew, Natural-F4, Natural-F2, and Planned-F4, using seeds 0/2/3 and epochs
1/2. Set V014_SLEEP2_PROFILE=core3 for the three-policy paper core (Vanilla,
Natural-F2, Planned-F4). Set V014_SLEEP2_SEEDS and V014_SLEEP2_ONLY_ARM to
select a bounded run without changing the protocol defaults.
EOF
        exit 0
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_ROOT=${V014_SLEEP2_ROOT:-/workspace/adafloor_graph_results/qwen3_v014_full_decode_sleep2_adafloor_${STAMP}}
PROFILE=${V014_SLEEP2_PROFILE:-full}
SEEDS=${V014_SLEEP2_SEEDS:-0,2,3}
ONLY_ARM=${V014_SLEEP2_ONLY_ARM:-}
POLICY=${V014_SLEEP2_POLICY:-}

case "$PROFILE" in full|core3) ;; *) echo "V014_SLEEP2_PROFILE must be full or core3" >&2; exit 2 ;; esac

export GRAPH_MAIN_ROOT="$RESULT_ROOT"
export GRAPH_MAIN_POLICY_PROFILE="$PROFILE"
export GRAPH_MAIN_CAPTURE_SIZES="${GRAPH_MAIN_CAPTURE_SIZES:-[16,32]}"
export GRAPH_MAIN_ACLGRAPH_BUCKET_COUNT=8
export GRAPH_MAIN_SEEDS_CSV="$SEEDS"
export GRAPH_MAIN_ONLY_ARM="$ONLY_ARM"
export GRAPH_MAIN_SLEEP_LEVEL=2
export GRAPH_MAIN_REUSE_SEED0_VANILLA=0
export GRAPH_MAIN_PROTOCOL_WRAPPER="$ROOT/run_qwen3_v014_full_decode_sleep2_adafloor.sh"

# Formal protocol invariants. The runner passes these into every Ray/NPU worker.
export VLLM_ROLLOUT_SLEEP_LEVEL=2
export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
export VLLM_ROLLOUT_VERIFY_WEIGHT_COVERAGE=1
export VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE="${VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE:-0}"
export VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE="${VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE:-0}"
export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE="${VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE:-0}"
if [[ "$VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE" == "1" ]]; then
    export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
    export VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
fi
export VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD=0
export VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD=0
export VLLM_ROLLOUT_DEFER_KV_CACHE_UNTIL_WEIGHT_LOAD=0
export VLLM_ROLLOUT_ASYNC_SCHEDULING=false
export VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM=0
export VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC=1
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=20480
export ACTOR_RECOMPUTE_NUM_LAYERS=13
export PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

if [[ "$ACTION" == dry-run ]]; then
    printf '%s\n' \
        "result_root=$RESULT_ROOT" \
        "profile=$PROFILE" \
        "single_policy=${POLICY:-all_core3_policies}" \
        "seeds=$SEEDS" \
        "only_arm=${ONLY_ARM:-all}" \
        "graph_mode=FULL_DECODE_ONLY" \
        "capture_sizes=[32]" \
        "bucket_count=8" \
        "sleep_level=2" \
        "manual_free_cache_engine=0" \
        "camem_weight_reload=1" \
        "native_sleep_graph_lifecycle=$VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE" \
        "reuse_aclgraph_after_weight_update=1" \
        "actor_max_token_len=20480" \
        "actor_recompute_num_layers=13"
    exec bash "$ROOT/run_qwen3_paper_graph_main_from_common_epoch0.sh" dry-run
fi

if [[ -n "$POLICY" && "$ACTION" != verify ]]; then
    case "$POLICY" in natural_f2|planned_f4) ;; *)
        echo "V014_SLEEP2_POLICY must be natural_f2 or planned_f4" >&2
        exit 2
        ;;
    esac
    IFS=',' read -r -a POLICY_SEEDS <<< "$SEEDS"
    for policy_seed in "${POLICY_SEEDS[@]}"; do
        GRAPH_MAIN_ROOT="$RESULT_ROOT" \
        GRAPH_MAIN_ONLY_ARM="${policy_seed}:$POLICY" \
            bash "$ROOT/run_qwen3_paper_graph_main_from_common_epoch0.sh" "$ACTION"
    done
    exit 0
fi

exec bash "$ROOT/run_qwen3_paper_graph_main_from_common_epoch0.sh" "$ACTION"
