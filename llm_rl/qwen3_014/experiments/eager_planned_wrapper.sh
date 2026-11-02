#!/usr/bin/env bash
# Same dynamic Planned pipeline as the graph study, but no graph capture.
set -euo pipefail
: "${ADAFLOOR_GRAPH_BASE_RUNNER:?missing original planned base runner}"
export TASK_QUEUE_ENABLE=2 VLLM_ROLLOUT_TASK_QUEUE_ENABLE=2
export VLLM_ENABLE_GRAPH_MODE=0 ROLLOUT_ENFORCE_EAGER=True
export ADAFLOOR_ACLGRAPH_MODE=NONE VLLM_ASCEND_ELASTIC_ACLGRAPH=0
export VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=0 VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=0
export VERL_HCCL_IF_BASE_PORT_START="${VERL_HCCL_IF_BASE_PORT_START:-12000}"
export VERL_MASTER_PORT_START="${VERL_MASTER_PORT_START:-28416}"
exec bash "$ADAFLOOR_GRAPH_BASE_RUNNER" "$@" actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.cudagraph_mode=NONE
