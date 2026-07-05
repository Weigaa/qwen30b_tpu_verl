# Mode1 regroup 通信时延异常：HCCL/new_group 序列诊断日志

日期：2026-06-29

## 背景

当前 mode=1 floor=4 regroup baseline 已经能跑完 5 个 rollouts，但从第二个 rollout 开始 full-restore / MC2 group lifecycle 经常出现 100s 级通信等待。多轮尝试表明：

- 单独 defer MC2 destroy 会把等待从 `destroy_ms` 转移到 `rebuild_ms`，不能消除问题。
- 单独保留 floor4 MC2 cache 也不能消除等待，且有额外 non-torch/HCCL 资源占用风险。
- 最新日志显示 rank 12-15 在 floor4 `(12,13,14,15)` MC2 group 上出现约 106s 的 release/rebuild 等待。

因此当前更高优先级假设是：不同 rank 因为 active/non-member、cache hit/skip、shrink/restore 次数不同，导致 `torch.distributed.new_group()` / HCCL communicator 创建序列不一致。后续 HCCL/Gloo rendezvous 可能等待不同内部序列或 communicator key，表现为高通信时延、超时或 destroy/rebuild 长等待。

## 本次改动

仅加入诊断日志，不改变通信组创建/销毁/cache 策略。

### worker_v1.py

新增 `Elastic group seq digest` 短日志，覆盖：

- active rank rebuild：`rebuild_enter` / `rebuild_cache_hit` / `rebuild_create_start` / `rebuild_create_done` / `rebuild_signature_done`
- non-member placeholder：`non_member_*_new_group_start/done` / `non_member_destroy_*` / `non_member_signature_done`
- restore catch-up：`restore_reconcile_start` / `restore_reconcile_observed` / `restore_reconcile_done`
- stash/release：`stash_enter` / `stash_cache_keep` / `stash_release_start` / `stash_release_done`

日志只读取本地字段，不调用 `get_dp_group()`、`get_ep_group()` 等可能触发异常状态访问的全局 parallel_state。

### 测试脚本

`run_mode1_local_baseline_random_batch_threshold_multistep.sh` 默认开启：

```bash
VLLM_ASCEND_GROUP_SEQ_DIGEST_LOG=1
```

可通过 `VLLM_ASCEND_GROUP_SEQ_DIGEST_LOG=0` 关闭。

### 解析工具

新增：

```bash
tools/analyze_group_seq_digest.py
```

运行方式：

```bash
python3 tools/analyze_group_seq_digest.py \
  mode1_baseline_random_batch_floor4_threshold_multistep/logs/<latest-log>.txt
```

重点看：

- `seq mismatch candidates`：同一个 rank-local `seq` 对应不同 event/ranks，提示 group lifecycle 分叉。
- `slow digest events`：直接列出携带 >1s 或 >100s 耗时的 group lifecycle 事件。
- `tail by rank`：重点比较 rank 12-15 在第二个 rollout 后、floor4 restore 附近的事件链。

## 静态验证

已通过：

```bash
python3 -m py_compile vllm_ascend/worker/worker_v1.py tools/analyze_group_seq_digest.py
bash -n run_mode1_local_baseline_random_batch_threshold_multistep.sh
```

## 2026-06-29 parser usability update

`tools/analyze_group_seq_digest.py` was updated to make the next run easier to inspect:

- `log` path is now optional.
- If omitted, the parser automatically picks the newest `*.txt` under `mode1_baseline_random_batch_floor4_threshold_multistep/logs/`.
- If a log does not contain `Elastic group seq digest` lines, it prints a clear message and exits with status 1. This prevents mistaking an old/non-instrumented log for a clean result.

Default usage after a new run:

```bash
python3 tools/analyze_group_seq_digest.py
```

Explicit usage remains supported:

```bash
python3 tools/analyze_group_seq_digest.py path/to/log.txt --tail 20 --around-slow 1000
```

## 2026-06-29 diagnostic result: restore latency is pinned to floor4 MC2 destroy

Fresh instrumented log:

```bash
mode1_baseline_random_batch_floor4_threshold_multistep/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260629010546.txt
```

Parser command:

```bash
python3 tools/analyze_group_seq_digest.py \
  mode1_baseline_random_batch_floor4_threshold_multistep/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260629010546.txt \
  --tail 4 --around-slow 1000
```

Key observation:

- The 8-rank and 4-rank shrink rebuilds are not the 100s+ bottleneck.
- The slow path consistently appears during full restore, when ranks 12-15 release the previous floor4 MC2 group `(12, 13, 14, 15)`.
- The slow event is exactly:
  - `event=stash_release_done`
  - `kind=mc2 attr=_MC2`
  - `reason=mode1_full_restore`
  - `destroy_ms ~= 109s / 130s / 128s`

Representative clusters from the parser:

```text
cluster=2 ranks=[12, 13, 14, 15] group_ranks=['(12, 13, 14, 15)'] destroy_ms_min=109157.01 destroy_ms_max=109217.47
cluster=3 ranks=[12, 13, 14, 15] group_ranks=['(12, 13, 14, 15)'] destroy_ms_min=130567.14 destroy_ms_max=130589.18
cluster=4 ranks=[12, 13, 14, 15] group_ranks=['(12, 13, 14, 15)'] destroy_ms_min=128526.62 destroy_ms_max=128553.23
```

Interpretation:

- The user's HCCL internal sequence/state mismatch hypothesis remains plausible as the lower-level mechanism, especially because the first rollout restore is cheap while later restores degrade sharply.
- But the actionable observed failure point is more specific: hot-path `GroupCoordinator.destroy()` / device ProcessGroup teardown for the old floor4 MC2 communicator during `mode1_full_restore`.
- KV cache, DispatchV2, model forward, and launcher parity are lower-probability explanations for this specific 100s+ latency, because the digest pins the wait to MC2 release before returning from restore.

Important nuance:

- The existing defer path is not a pure "do not destroy device HCCL PG" path by default. `_retire_group_wrapper_defer_pg_destroy()` still uses `VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE=1` by default, so a previous defer experiment could merely move the same device PG teardown cost from `destroy_ms` to `rebuild_ms`.
- A safer next experiment must distinguish:
  1. Python wrapper/GroupCoordinator destroy cost.
  2. Raw device ProcessGroup destroy cost.
  3. Resource accumulation / non_torch growth if device PG teardown is deferred too aggressively.

Next fix direction:

- Do not repeat broad KV-cache or warmup changes.
- Add a very narrow MC2 full-restore retire mode that can separately time wrapper detach, optional NPU sync, optional device PG destroy, and deferred PG count.
- If testing `DESTROY_DEVICE_PG_ON_RETIRE=0`, it must be paired with explicit non_torch/deferred-count logging and bounded cleanup, otherwise it can silently convert restore latency into HCCL/non_torch accumulation and later OOM.

## 2026-06-29 fix: align mode=1 full-restore stale group handling with successful repo

Additional comparison against the successful 5-step repo `qwen3_true_mode5_a3cfdc2` showed a concrete semantic drift:

- Successful repo logs use:
  - `Elastic mode1 full-restore defers stale live floor group destroy`
  - `Elastic parallel group destroy deferred ... reason=mode1_full_restore ... device_pg_destroy_ms≈0.03 destroy_device_pg=1`
  - `Elastic parallel stashed group released ... destroy_ms=0.00`
- Current staged logs used direct release/destroy:
  - `Elastic mode1 full-restore releases stale live floor group`
  - `stash_release_done ... reason=mode1_full_restore destroy_ms≈109s/130s`

Root cause of the latest regression:

- `run_mode1_local_baseline_random_batch_threshold_multistep.sh` forced:
  - `VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE=0`
- `worker_v1.py` also had a per-reason gate whose default for `mode1_full_restore` was `0`.
- Therefore stale floor4 MC2 groups were forced down the direct `GroupCoordinator.destroy()` path, exactly where the new digest showed 100s+ latency.

Fix applied:

- `run_mode1_local_baseline_random_batch_threshold_multistep.sh`
  - default `VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE=1`
- `vllm_ascend/worker/worker_v1.py`
  - default full-restore defer gate changed to `1`
  - full-restore log wording restored to `defers stale live floor group destroy`
  - MC2 memory tag restored to `before_mode1_full_restore_mc2_defer`

Important nuance:

- This is not the unsafe “keep stale device PG forever” path. The retire path still defaults to:
  - `VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE=1`
- So the expected behavior is to avoid slow `GroupCoordinator.destroy()` while still destroying raw device PG quickly, matching the successful repo evidence.

Expected validation signal in the next run:

- `mode1_full_restore` `_MC2 old_ranks=(12, 13, 14, 15)` should log `destroy deferred` / `destroy_ms=0.00`.
- Parser should no longer show 100s+ `mode1 full-restore MC2 releases` clusters.
- Restore `rebuild_ms/total_ms` for ranks 12-15 should not contain the previous 109s/130s floor4 MC2 destroy wait.

Static validation after the fix:

```bash
python3 -m py_compile vllm_ascend/worker/worker_v1.py tools/analyze_group_seq_digest.py
bash -n run_mode1_local_baseline_random_batch_threshold_multistep.sh
```

Both checks passed.

## 2026-06-29 parser update: verify wait is not merely moved

`tools/analyze_group_seq_digest.py` now also parses non-digest log lines:

- `Elastic parallel group destroy deferred`
- `Elastic parallel group retired immediately`
- `Elastic parallel restore done`
- `rollout_output_time_s`
- `Elastic mode1 low-floor stashed group direct destroy fallback`

Reason:

- After aligning full-restore with the successful defer/retire path, `stash_release_done destroy_ms` alone is not enough.
- The next run must prove that the 100s wait did not simply move from `_MC2 destroy_ms` to `restore rebuild_ms/total_ms` or raw `device_pg_destroy_ms`.

Next-run acceptance signals:

1. `mode1 full-restore MC2 releases`
   - no 100s+ `destroy_ms` clusters.
2. `mode1 full-restore MC2 deferred retire`
   - should show ranks 12-15 entering deferred retire for `(12, 13, 14, 15)`.
   - `device_pg_destroy_ms` should remain small, ideally sub-ms to a few ms.
   - `deferred_device_pg_refs` should stay `0` with the current default `DESTROY_DEVICE_PG_ON_RETIRE=1`.
3. `restore and rollout summary`
   - no later restore cluster with `max_rebuild_ms` or `max_total_ms` in the 100s range.
4. `direct_destroy_fallbacks`
   - no `reason=mode1_full_restore` fallback for `group_kind=mc2 group_ranks=(12, 13, 14, 15)`.

Command after the next run:

```bash
python3 tools/analyze_group_seq_digest.py --tail 0 --around-slow 1000
```

## 2026-06-29 group index mismatch hypothesis check

User hypothesis: HCCL communicators may carry internal sequence/id state; if ranks experience different shrink counts or group create/destroy counts, later rebuild/communication can wait abnormally or timeout.

Evidence from `wjqwen30b-a3b-record_graph_save4eagle3_20260629010546.txt` with group seq digest:

- Every rank has the same digest event count: 285.
- Lifecycle count variants split by expected role, not random rank divergence:
  - ranks 0-7: non-member placeholder create/destroy path, 30 device+cpu placeholder groups.
  - ranks 8-11: active for 16->8, then non-member for 8->4, 15 active creates plus 15 placeholder creates.
  - ranks 12-15: active for both 16->8 and 8->4, 30 active creates/releases.
- The repeated 100s+ stalls are concentrated only on ranks 12-15, and only at:
  - `stash_release_done kind=mc2 attr=_MC2 ranks=(12, 13, 14, 15) reason=mode1_full_restore`
  - destroy times: about 109s / 130s / 128s / 132s.

Interpretation:

- The broad group-id mismatch hypothesis remains plausible at the HCCL implementation level, but this log does not show arbitrary same-role rank divergence.
- The immediate, directly measured slow point is narrower: full-restore old floor4 MC2 `GroupCoordinator.destroy()` on active ranks 12-15.
- Therefore the current code fix is to align with the successful repo behavior: defer/retire stale full-restore floor groups and destroy raw device PG refs without calling `GroupCoordinator.destroy()` in the hot path.

Tooling update:

- `tools/analyze_group_seq_digest.py` now prints `group lifecycle sequence summary` and `lifecycle count variants`.
- Fixed `--tail 0` so it really suppresses tail output instead of printing the full log.

Next validation should use a fresh run after the full-restore defer default change. The existing `20260629010546` log predates that change and still shows `defer=False` / direct destroy fallback for `mode1_full_restore`.

## 2026-06-29 baseline entrypoint default alignment

While checking whether the full-restore defer fix could be bypassed, found that:

- `run_mode1_local_baseline_random_batch_threshold_multistep.sh` explicitly exported `VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE=1`.
- But the base script `run_mode1_local_baseline_random_batch.sh` still defaulted the same variable to `0`.

This was a footgun: any direct baseline run, or any wrapper that failed to export the threshold override, would silently return to the old direct `GroupCoordinator.destroy()` path and reproduce the 100s+ full-restore MC2 delay.

Change:

- `run_mode1_local_baseline_random_batch.sh` now defaults `VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE=1` as well.

This does not change pre-rebuild/stash behavior, which remains direct by default for now. The targeted fix is only the measured full-restore old floor4 MC2 slow path.

## Low-level HCCL sequence digest added

To test the HCCL internal group-id mismatch hypothesis directly, the next run logs `Elastic HCCL seq digest` lines for elastic-only group operations:

- active `GroupCoordinator` device/cpu `new_group()` calls;
- detached placeholder device/cpu `new_group()` calls;
- placeholder `destroy_process_group()` calls;
- raw device/cpu PG destroy during deferred retire.

The threshold multistep script enables `VLLM_ASCEND_ELASTIC_HCCL_SEQ_DIGEST_LOG=1` by default. Analyze with:

```bash
python3 tools/analyze_group_seq_digest.py --summary-only --tail 0 --around-slow 1000
```

The key section is `HCCL/Gloo low-level sequence summary`. Same-role rank divergence or `op+seq` variants would support the internal HCCL id/order mismatch theory. If not, continue focusing on the measured full-restore stale floor4 MC2 retire path.

### Shared sequence counter refinement

The HCCL seq digest uses one rank-local counter for both active `GroupCoordinator` new_group calls and detached placeholder new_group calls. The analyzer normalizes event names by low-level action/backend/ranks, so it reports real order/signature divergence instead of wrapper naming differences.

### Instrumentation side-effect guard

`GroupCoordinator.__init__` now avoids all sequence id allocation and timing calls when `VLLM_ASCEND_ELASTIC_HCCL_SEQ_DIGEST_LOG=0`. The HCCL sequence digest is enabled only by the threshold diagnostic wrapper by default.

### Worker-side instrumentation side-effect guard

Worker-side raw PG retire destroy and non-member placeholder new_group/destroy logs also avoid seq allocation and digest calls unless `VLLM_ASCEND_ELASTIC_HCCL_SEQ_DIGEST_LOG=1`.

### Analyzer latest-log selection

The analyzer now recursively selects the newest rollout training log under `mode1_baseline_random_batch_floor4_threshold_multistep/`, covering both flat `logs/*.txt` and timestamped `<RUN_ID>/logs/logs/*.txt` layouts. It also handles SIGPIPE cleanly when piping output to `head`.

### Diagnosis verdict

The analyzer now prints a `diagnosis verdict` section. On the old 20260629010546 log it correctly identifies `FULL_RESTORE_MC2_DIRECT_DESTROY_SLOW`. Fresh logs with low-level HCCL seq instrumentation should distinguish group sequence mismatch from raw PG retire destroy or restore rebuild/refresh slowdowns.

### Quick verdict mode

`tools/analyze_group_seq_digest.py --verdict-only --around-slow 1000` now prints just the selected log path, digest count, verdict, and action. Use this before reading long lifecycle summaries.
