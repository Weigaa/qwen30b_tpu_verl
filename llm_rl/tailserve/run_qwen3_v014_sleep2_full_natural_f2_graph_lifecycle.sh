#!/usr/bin/env bash
# First-stage v0.14 native sleep lifecycle validation.
# It invalidates FULL ACLGraph before sleep(level=2), reloads weights while
# graph/KV replay state is inactive, wakes KV, then recaptures FULL graph.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-dry-run}
case "$ACTION" in
    dry-run|smoke|run|verify) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_v014_sleep2_full_natural_f2_graph_lifecycle.sh dry-run
  ./run_qwen3_v014_sleep2_full_natural_f2_graph_lifecycle.sh run

This is a correctness and memory-lifecycle validation. It uses native
sleep(level=2), FULL_DECODE_ONLY, Natural-F2, and forces graph invalidation
before sleep plus graph recapture after weights and KV are resumed.
Set V014_SLEEP2_ROOT to a writable result directory.
EOF
        exit 0
        ;;
    *) echo "usage: $0 {dry-run|smoke|run|verify}" >&2; exit 2 ;;
esac

export V014_SLEEP2_PROFILE=core3
export V014_SLEEP2_POLICY=natural_f2
export V014_SLEEP2_SEEDS=${V014_SLEEP2_SEEDS:-0}
export V014_SLEEP2_ROOT=${V014_SLEEP2_ROOT:-$ROOT/qwen3_v014_sleep2_natural_f2_graph_lifecycle}
export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE=1
export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_RENEW_POOL=0
# Keep the captured FULL graph/task pool across native sleep. KV and weight
# virtual addresses are remapped in place; rebuilding the graph every RL step
# leaks workspaces and invalidates dynamic attention metadata.
export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_RETAIN="${VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_RETAIN:-0}"
# FULL graph task groups are topology-bound. After natural/planned shrink,
# force eager execution until the world is restored instead of replaying a
# graph captured for a different DP/EP/MC2 communicator.
export VLLM_ASCEND_ELASTIC_ACLGRAPH_DYNAMIC_DISABLE="${VLLM_ASCEND_ELASTIC_ACLGRAPH_DYNAMIC_DISABLE:-0}"
# Restore the large KV allocation before mapping/reloading many MoE weight
# shards. This avoids physical-page fragmentation during native level-2 wake.
export VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_KV_FIRST=1
# FULL_DECODE_ONLY Attention updates dynamic KV metadata on every decode
# iteration.  The validated v0.14 graph protocol uses device-side metadata
# synchronization; leaving this at the generic eager default (0) replays the
# previous step's metadata and produces cap-length garbage responses.
export VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE="${VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE:-1}"
# Keep the validated v0.14 FULL graph protocol's MoE reduction semantics.
export VLLM_QWEN3_MOE_REDUCE_RESULTS="${VLLM_QWEN3_MOE_REDUCE_RESULTS:-1}"
export VLLM_ROLLOUT_PHASE_MEMORY_LOG=1
export VLLM_ROLLOUT_STAGE_TIMING=1

exec "$ROOT/run_qwen3_v014_sleep2_full_natural_f2.sh" "$ACTION"
