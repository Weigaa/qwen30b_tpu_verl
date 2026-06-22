# Mode/Floor Headroom Tuning Notes

This note records locally validated KV-cache-init headroom for Qwen3-30B-A3B elastic modes. It is intentionally empirical: values here are not universal constants. Re-check them when CANN/driver/torch_npu/HCCL, allocator behavior, double-buffer policy, MC2/DispatchV2 behavior, or group-cache policy changes.

## Scope

- Repo: `qwen3_true_mode5_a3cfdc2`
- Success-log convention: `pure-highkv-mode{mode}-{floor}.txt`
- Workload shape used by the success logs:
  - `train_batch_size=32`
  - `max_num_seqs=32`
  - `max_num_batched_tokens=17408`
  - `max_prompt_length=1024`
  - `max_response_length=16384`
  - `rollout_n=16`
  - `gpu_memory_utilization=0.85`
  - `total_epochs=1`
- Primary sizing knob:
  - `VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES`
- Important confounders:
  - `VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE`
  - `VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC`
  - `VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE`
  - `VLLM_ASCEND_MODE5_RUNTIME_MIN_COMPUTE_GROUP_SIZE`
  - mode3 CPU-shadow / dispatch-domain knobs
  - diagnostic logging that calls NPU memory APIs

## Validated Minimums So Far

These are the smallest values currently kept as completed one-step `pure-highkv-*` runs. Treat them as local baselines, not theoretical minima.

| mode | floor | headroom bytes | headroom GiB | success log | KV tokens | rollout s | epoch s |
|---:|---:|---:|---:|---|---:|---:|---:|
| 3 | 1 | `7516192768` | 7 | `pure-highkv-mode3-1.txt` | 330,880 | 376.367 | 526.64 |
| 3 | 2 | `2147483648` | 2 | `pure-highkv-mode3-2_new.txt` | 397,824 | 265.878 | TBD |
| 3 | 4 | `1073741824` | 1 | `pure-highkv-mode3-4.txt` | 397,824 | 199.721 | 345.05 |
| 3 | 8 | `0` | 0 | `pure-highkv-mode3-8.txt` | 431,872 | 197.986 | 345.99 |
| 4 | 1 | `5368709120` | 5 | `pure-highkv-mode4-1.txt` | 352,768 | 224.202 | 374.64 |
| 4 | 2 | `2147483648` | 2 | `pure-highkv-mode4-2.txt` | 397,824 | 193.432 | 340.53 |
| 4 | 4 | `1073741824` | 1 | `pure-highkv-mode4-4.txt` | 414,848 | 164.353 | 312.06 |
| 4 | 8 | `0` | 0 | `pure-highkv-mode4-8.txt` | 431,872 | 144.911 | 288.81 |
| 5 | 1 | `5368709120` | 5 | `pure-highkv-mode5-1.txt` | 352,768 | 244.794 | 403.01 |
| 5 | 2 | `2147483648` | 2 | `pure-highkv-mode5-2.txt` | 397,824 | 201.579 | 351.39 |
| 5 | 4 | `1073741824` | 1 | `pure-highkv-mode5-4.txt` | 414,848 | 185.696 | 336.96 |
| 5 | 8 | `0` | 0 | `pure-highkv-mode5-8.txt` | 431,872 | 151.439 | 303.44 |

## Bracketing Failures

These failures explain why the validated minimums above are where they are.

| log | mode | floor | headroom bytes | observed failure |
|---|---:|---:|---:|---|
| `wjeager...20260614225026...txt` | 3 | 4 | `0` | OOM / insufficient low-floor runtime workspace. |
| `wjeager...20260615141631...txt` | 3 | 4 | `1073741824` | OOM in an earlier code state; later `pure-highkv-mode3-4.txt` passed with the same headroom after mode3 memory-path changes. |
| `wjeager...20260615223447...txt` | 3 | 1 | `6442450944` | `aclnnMoeDistributeDispatchV2` failed while allocating HCCL/MC2 resource `size:1678770176` bytes. |
| `wjeager...20260622230633...txt` | 3 | 2 | `2147483648` | Pre-fix post-shrink DP `HcclAllreduce` warmup allocated `size:1678770176` bytes and OOMed. After disabling mode3 DP warmup by default, `pure-highkv-mode3-2_new.txt` passed with the same 2GiB headroom. |
| `wjeager...20260615160759...txt` | 4 | 2 | `1073741824` | DispatchV2 requested about 1.56GiB after shrink; 1GiB was not enough. |
| `wjeager...20260615174346...txt` | 4 | 1 | `2147483648` | OOM after shrink; 2GiB was not enough. |
| `wjeager...20260615180144...txt` | 4 | 1 | `3221225472` | OOM after shrink; 3GiB was not enough. |
| `wjeager...20260615195344...txt` | 4 | 1 | `4294967296` | Stalled/implicit memory pressure; 5GiB later completed. |
| `wjeager...20260615163144...txt` | 5 | 2 | `2147483648` | Pre-fix logic failure: missing remote NPU source. After dual-source CPU-shadow fallback fix, 2GiB passed. |

## Current Interpretation

- `mode=3` still has higher CPU-source refresh/import overhead than `mode=4/5`, but the latest floor=2 OOM was not an unavoidable CPU-source memory requirement. It was caused by an unnecessary post-shrink DP `HcclAllreduce` warmup that forced an extra ~1.56GiB HCCL workspace allocation. Keep `VLLM_ASCEND_MODE3_ENABLE_POST_SHRINK_DP_WARMUP=0` by default.
- `mode=4/5` remote-cache paths are now much cleaner: high floors can use `0`, floor=4 uses `1GiB`, floor=2 uses `2GiB`, and floor=1 uses `5GiB`.
- `mode=3 floor=4` can now complete with `1GiB`; `mode=3 floor=2` can now complete with `2GiB` after skipping post-shrink DP all_reduce warmup. Keep the success log with the code revision when comparing.
- A single headroom number is not enough to reproduce a result. Record at least mode, floor, runtime floor, post-shrink release flags, KV headroom, low-floor workspace headroom, diagnostic-log flags, and whether the run came from a `pure-highkv-*` success log.

## Recommended Validation Commands

Use the `pure-highkv-*` logs above as the current completed-run reference set. To reproduce a cell, run the corresponding mode/floor script with the listed headroom value, for example:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_true_mode5_a3cfdc2

VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=7516192768 bash run_mode3_perf_clean_test.sh 1
VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=2147483648 bash run_mode3_perf_clean_test.sh 2
VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=1073741824 bash run_mode3_perf_clean_test.sh 4
VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=0 bash run_mode3_perf_clean_test.sh 8

VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=5368709120 bash run_mode4_perf_clean_test.sh 1
VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=2147483648 bash run_mode4_perf_clean_test.sh 2
VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=1073741824 bash run_mode4_perf_clean_test.sh 4
VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=0 bash run_mode4_perf_clean_test.sh 8

VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=5368709120 bash run_mode5_perf_clean_test.sh 1
VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=2147483648 bash run_mode5_perf_clean_test.sh 2
VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=1073741824 bash run_mode5_perf_clean_test.sh 4
VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=0 bash run_mode5_perf_clean_test.sh 8
```

## Open Questions

- Whether `mode=3 floor=1` can be brought down toward `mode=4/5` memory usage. `mode=3 floor=2` is now at the same 2GiB headroom as mode=4/5, but rollout is still slower due to CPU-source refresh/import overhead.
- Whether `mode=4/5 floor=1` can safely run below `5GiB` after more group-cache and low-floor DispatchV2 cleanup.
- Whether the successful `mode=3 floor=4` 1GiB result is stable across repeated runs and after pushing to the remote machine.
