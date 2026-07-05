#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Compare real vllm_ascend mode=1 MoE dispatcher backends at a high token count.
# This is not a toy torch.distributed benchmark: it reuses the same
# dispatcher-only warmup path used by the rollout worker after elastic restore.
: "${MOE_BENCH_TOKENS:=1664}"
: "${KV_TOKENS:=240000}"

export KV_TOKENS
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP=1
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP_TOKENS="$MOE_BENCH_TOKENS"
export VLLM_ASCEND_MODE1_PARITY_FORCE_MC2_DISPATCHER_WARMUP=1
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP=1
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP_TOKENS="$MOE_BENCH_TOKENS"
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP_MAX_TOKENS="$MOE_BENCH_TOKENS"

# Keep the benchmark readable: this wrapper only needs the explicit benchmark
# lines, not per-dispatch digest samples unless the caller asks for them.
export VLLM_ASCEND_MOE_DISPATCH_SEQ_DIGEST_LOG="${VLLM_ASCEND_MOE_DISPATCH_SEQ_DIGEST_LOG:-0}"
export VLLM_ASCEND_GROUP_SEQ_DIGEST_LOG="${VLLM_ASCEND_GROUP_SEQ_DIGEST_LOG:-0}"
export VLLM_ASCEND_ELASTIC_HCCL_SEQ_DIGEST_LOG="${VLLM_ASCEND_ELASTIC_HCCL_SEQ_DIGEST_LOG:-0}"
export VLLM_ASCEND_ELASTIC_HCCL_SEQ_SUMMARY_LOG="${VLLM_ASCEND_ELASTIC_HCCL_SEQ_SUMMARY_LOG:-0}"
export VLLM_ASCEND_MODE1_SEQ_CHECKPOINT_LOG="${VLLM_ASCEND_MODE1_SEQ_CHECKPOINT_LOG:-0}"

printf '[mode1 moe backend compare] tokens=%s kv_tokens=%s launcher=%s\n' \
  "$MOE_BENCH_TOKENS" "$KV_TOKENS" "${LOCAL_TEST_LAUNCHER:-default}"
printf '[mode1 moe backend compare] grep: Mode1 parity dispatcher-only benchmark\n'

exec "$SCRIPT_DIR/run_mode1_local_baseline_random_batch_threshold_multistep.sh" "$@"
