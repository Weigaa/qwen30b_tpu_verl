#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 1GiB-headroom variant of the floor2 -> floor4 KV recovery probe.
#
# The previous floor2 cap=131072 run failed after rollout restore during actor
# MoE all-to-all workspace allocation, with only ~118MiB free and a ~214MiB
# allocation request.  Qwen3-30B-A3B KV cache costs about 24KiB/token/rank, so
# lowering the cap by 45056 tokens releases about 1.03GiB per rank.

export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2:-86016}"
export DYNAMIC_RUN_NAME="${DYNAMIC_RUN_NAME:-mode1_dynamic_floor2_to_floor4_kv_probe_floor2cap86016}"

exec "$SCRIPT_DIR/run_mode1_dynamic_floor2_to_floor4_kv_probe.sh" "$@"
