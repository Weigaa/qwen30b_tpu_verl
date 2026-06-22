# mode=3 显存问题分析和总结（潜在优化空间）

## 背景

目标是在 `qwen3_true_mode5_a3cfdc2` 仓库中比较 mode=3、mode=4、mode=5 在 `floor=4`、相同 KV cache 预算下的运行时显存占用，定位为什么 mode=3 在 KV cache 与 mode=4/5 对齐后仍会在 `aclnnMoeDistributeDispatchV2` 处 OOM。

本轮对照均使用：

- `floor=4`
- `GPU KV cache size = 414,848 tokens`
- `kv_cache_init_headroom = 1GiB`
- DispatchV2 domain: `moe_expert_num=128`, `dispatcher_num_experts=128`, `dispatcher_num_experts_local=32`, `expert_token_nums_type=1`

## 对照结果

| 模式 | 运行日志 | source 分布 | free GiB | torch_current GiB | non_torch GiB | 结果 |
|---|---|---:|---:|---:|---:|---|
| mode=3 direct CPU slot | `wjeagerqwen30b-a3b-with_draft_breakdown_20260617111046_elastic.txt` | `8 local + 0 remote + 24 CPU` | `0.641-0.889` | `45.082` | `15.306-15.547` | OOM |
| mode=3 staging CPU slot | `wjeagerqwen30b-a3b-with_draft_breakdown_20260617131312_elastic.txt` | `8 local + 0 remote + 24 CPU` | `0.321-0.569` | `45.363` | `15.345-15.585` | OOM |
| mode=4 remote NPU | `wjeagerqwen30b-a3b-with_draft_breakdown_20260617111450_elastic.txt` | `8 local + 24 remote + 0 CPU` | `1.786-3.642` | `45.363` | `12.272-14.120` | 成功，`162.95s` |
| mode=5 dual source | `wjeagerqwen30b-a3b-with_draft_breakdown_20260617135627_elastic.txt` | `8 local + 18 remote + 6 CPU` | `1.824-3.680` | `45.346` | `12.252-14.100` | 成功，`185.05s` |

## 已排除的假设

### 1. 不是 KV cache 尺寸导致

四组关键对照的 KV cache 都是 `414,848 tokens`，但 mode=3 OOM，mode=4/5 成功。

### 2. 不是 DispatchV2 domain 更大导致

mode=3、mode=4、mode=5 在 floor=4 的 DispatchV2 domain 一致：

```text
moe_expert_num=128
dispatcher_num_experts=128
dispatcher_num_experts_local=32
expert_token_nums_type=1
```

### 3. 不是 bulk CPU direct copy 导致

将 `VLLM_ASCEND_MODE3_BULK_CPU_DIRECT=0` 后，mode=3 的 `free_bytes`、`non_torch` 与 baseline 基本一致，仍然在 DispatchV2 处 OOM。

结论：bulk coalesced CPU copy 不是主因。

### 4. 不是 direct CPU slot 本身导致

将 `VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT=0` 退回 staging CPU slot 后，OOM 仍然存在，而且更差：

- `torch_current` 从约 `45.08GiB` 升到约 `45.36GiB`
- `non_torch` 仍然维持在约 `15.3-15.6GiB`
- `free_bytes` 进一步下降到约 `0.3-0.6GiB`

结论：staging 路径不是解决方向，direct CPU slot 反而是 mode=3 当前更省 torch 显存的实现。

### 5. 不是“只要使用 CPU source 就必然 OOM”

mode=5 也使用 CPU source，但只有 `6/32` rows 来自 CPU，且 non_torch 水位接近 mode=4。真正的问题是 mode=3 的 CPU source 占比过高。

## 当前根因判断

mode=3 在 `floor=4` 时为了保持“只用本地 NPU resident experts + 本地 CPU shadow experts”的语义，每个 active rank/layer 需要：

```text
source_from_local_npu=8
source_from_remote_npu=0
source_from_cpu=24
```

这条 `24 CPU rows` 的 CPU-source runtime 路径会把 CANN/HCCL/runtime 的 non-torch 显存水位推高到约 `15.3-15.6GiB`。

相比之下：

- mode=4 是 `8 local + 24 remote NPU + 0 CPU`，non_torch 约 `12.3-14.1GiB`
- mode=5 是 `8 local + 18 remote NPU + 6 CPU`，non_torch 约 `12.3-14.1GiB`

因此 mode=3 在 first DispatchV2 前只剩约 `0.6-0.9GiB` free memory，无法满足 `HcclAllocComResourceByTiling` 的 workspace 分配，最终报错：

```text
Memory_Allocation_Failure(EL0004): Failed to allocate memory.
Nnopbase fails to invoke the HcclAllocComResourceByTiling function of the hccl module.
current working operator name is aclnnMoeDistributeDispatchV2
```

## 为什么 mode=5 不 OOM

mode=5 并不是和 mode=3 一样“大量靠 CPU”。它是 dual-source：

- 大多数缺失 experts 从 remote NPU 获取
- 少量 fallback experts 从 CPU shadow 获取

本轮 floor=4 对照中，mode=5 的实际 source 分布是：

```text
source_from_local_npu=8
source_from_remote_npu=18
source_from_cpu=6
```

CPU rows 数量只有 mode=3 的 1/4，因此没有触发 mode=3 那种高 non-torch 水位。

## 当前稳定策略

短期要保证 mode=3/floor=4 稳定运行，需要给 mode=3 额外 headroom。之前验证过更大的 headroom 可以让 mode=3/floor=4 跑通。

建议默认策略：

- mode=3/floor=4 保留额外 headroom，不强行追求 mode=4/5 的最大 KV cache
- mode=3 保持 `DIRECT_CPU_SLOT=1`
- 不建议回退到 staging CPU slot，因为 staging 只会增加 torch 显存占用

## 潜在优化空间

### 方向 1：减少 mode=3 的 CPU rows 数量

如果允许 mode=3 引入部分 remote NPU source，它会逐渐变成 mode=5-like。CPU rows 从 `24` 降到 `6` 后，理论上 non_torch 水位应更接近 mode=5。

代价：这会改变 mode=3 “只使用本地 CPU + 本地 NPU”的语义。

### 方向 2：优化 `24 CPU rows` 的 runtime 组织方式

如果必须保持 mode=3 语义，则需要降低 `source_from_cpu=24` 这条路径带来的 non-torch runtime 开销。

可能切入点：

- CPU shadow row 到 runtime slot 的绑定方式
- CPU-source rows 在 DispatchV2 前是否触发额外 CANN runtime resource
- CPU rows 是否可以按更轻量的 compact shadow 元数据组织
- 是否能避免为 CPU-heavy runtime layout 触发大块 HCCL/tiling 资源

### 方向 3：按 floor 自适应 headroom

在 mode=3 语义不变的前提下，最稳妥的是根据 floor 设置不同 headroom：

- floor=8：通常更容易稳定，可尝试更高 KV cache
- floor=4：需要额外 headroom
- floor=2/1：需要更保守 headroom

## 建议的后续验证

1. 保持 `DIRECT_CPU_SLOT=1`，不要使用 staging。
2. 做 mode=3/floor=4 不同 headroom 的最小稳定阈值扫描。
3. 如果要继续优化显存，优先增加 CPU-source runtime 的 memory diag，而不是继续切 bulk/staging 开关。
4. 如果允许改变 mode=3 语义，可测试 mode3-like + 少量 remote NPU source 的混合路径，观察 `source_from_cpu` 从 24 降到 6/12 时 non_torch 是否线性下降。

## 2026-06-20：不要默认启用 mode=3 synthetic MC2 dispatcher warmup

最新 `mode=3/floor=4` 对照里出现过一次类似 300s/350s 的超长耗时。按日志时间空洞和 shrink phase breakdown 定位，直接慢点不是 CPU slot copy、不是 CPU shadow refresh、也不是真实 MoE forward：

```text
Mode3 MC2 dispatcher-only warmup start: ... ep_size=4
Mode3 MC2 dispatcher-only warmup complete: ...
Elastic parallel shrink phase breakdown: ... warmup_ms=301603ms ...
```

同一轮中，真实 forward timing 显示 layer-level MoE apply 多数只有几毫秒：

```text
Mode3 MoE forward timing: ... prepare_ms~=0.004 apply_ms~=2-4 total_ms~=2-4
```

结论：

- `VLLM_ASCEND_MODE3_FORCE_DISPATCHER_WARMUP=1` 会把一个 synthetic MC2/DispatchV2 warmup 算子放进 shrink 热路径。
- 在当前 CANN/HCCL/torch_npu 栈上，这个 synthetic warmup 可能触发 300s 级底层通信 runtime timeout/retry。
- 这类现象和 mode=1 350s 踩坑类似：不是上层 Python 循环慢，而是某个通信算子或 communicator/runtime 状态在同步点暴露出长超时。
- 因此该 warmup 只能作为显式诊断开关，不应默认开启。

当前默认：

```bash
VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP=0
VLLM_ASCEND_MODE3_FORCE_DISPATCHER_WARMUP=0
```

如果未来要重新验证 lazy MC2 初始化，应只在单独 A/B 实验中显式打开，并同时记录：

- `Elastic parallel shrink phase breakdown` 的 `warmup_ms`
- `Mode3 MC2 dispatcher-only warmup start/complete`
- `Mode3 MoE forward timing`
- CANN/HCCL 错误码或超时日志

## 2026-06-20：mode=3/floor=4 300s 不是 warmup 问题，而是首个 MC2 DispatchV2 问题

后续对照修正了上面的判断：关闭 synthetic dispatcher warmup 后，300s 并没有消失，只是从 `warmup_ms` 转移到了真正的 stage=4 decode forward。

观察到两种形态：

- `VLLM_ASCEND_MODE3_FORCE_DISPATCHER_WARMUP=1` 时，长耗时出现在 `Mode3 MC2 dispatcher-only warmup start/complete` 之间，`warmup_ms ~= 301s`。
- 关闭 warmup 后，`Elastic parallel shrink done` 很快结束，但首次 stage=4 `Hybrid MoE effective comm path: ... comm=MC2` 后出现约 300s 空洞，随后 rollout 继续。

因此 warmup 只是移动了第一次 MC2/DispatchV2 runtime 初始化或 timeout/retry 的位置，不是根因修复。正确排查方向应和 mode=1 的 350s 踩坑类似：找出底层通信算子看到的异常/不一致元数据，避免它进入 HCCL/CANN runtime 长重试路径。

当前最可疑的不一致：

- mode=3 会先把逻辑 expert id remap 到 compact runtime domain，例如 floor=4 时 `topk_ids` 变成 `0..31`。
- 旧路径仍把原始逻辑 `expert_map` 传给 MC2 dispatcher/combine 状态，该 map 是 128 长度并包含 `-1` holes。
- 这会形成 `topk_ids` 属于 dense runtime domain、`expert_map` 属于 original logical domain 的混用。

本次代码修复方向：

- 新增 `VLLM_ASCEND_MODE3_DENSE_RUNTIME_EXPERT_MAP=1`，默认开启。
- mode=3 remapped MC2 路径使用 dense identity `expert_map = arange(runtime_num_experts)`。
- synthetic mode=3 dispatcher warmup 也使用同一 dense runtime `expert_map`，避免诊断路径继续测试旧元数据。
- 新增轻量 `MC2 host timing` 日志，只记录 `dispatch_v2/combine_v2` Python 调用耗时，不做 NPU tensor `.item()` 或内存 snapshot，避免引入新的同步扰动。

下一轮验证重点：

- 日志头部应出现 `dense_runtime_expert_map=1`。
- layer0 的 `MC2 dispatch args` 应显示 floor=4 时 `expert_map_shape=(32,)`，而不是原始 `(128,)`。
- 如果仍慢，查 `MC2 host timing: phase=dispatch/combine ... elapsed_ms=...`，确认 300s 卡在 dispatch 还是 combine。
- 如果 `expert_map_shape=(32,)` 且仍出现 300s，则下一步继续对比 mode=5 remote_fraction=0.00 的 DispatchV2 入参：`ep_world_size`、`moe_expert_num`、`expert_token_nums_type`、`mc2_mask`、group name / communicator cache 生命周期。

## 2026-06-23：mode=3/floor=2 OOM 来自 post-shrink DP all_reduce warmup

在 `mode=3/floor=2` 尝试把 KV cache 预算对齐到 mode=4/5 的 `2GiB` headroom 时，旧逻辑直接 OOM：

```text
log: wjeagerqwen30b-a3b-with_draft_breakdown_20260622230633_elastic.txt
mode=3 floor=2 kv_cache_init_headroom=2147483648
GPU KV cache size: 397,824 tokens
current working operator name is HcclAllreduce
Memory_Allocation_Failure(EL0004): Failed to allocate memory
Failed to allocate resource[DeviceMemory] with info [size:1678770176]
exit_code=1
```

栈上直接位置：

```text
rebuild_elastic_ep_group()
  -> _warmup_post_shrink_dp_collectives()
     -> torch.distributed.all_reduce(...)
     -> torch.npu.synchronize()
```

因此这次 OOM 不是 CPU slot copy 本身，也不是 CPU shadow refresh 本身，而是 shrink 后额外执行的 DP metadata/collective warmup 强制 HCCL 为新 DP group 初始化 workspace。该 workspace 申请约 `1.56GiB`，在 high-KV/floor=2 的紧内存状态下越过了 OOM 边界。

### 为什么 floor=4 没改也能跑

`pure-highkv-mode3-4_new.txt` 说明 floor=4 旧逻辑也执行过同一个 post-shrink DP all_reduce warmup：

```text
mode=3 floor=4 kv_cache_init_headroom=1073741824
GPU KV cache size: 414,848 tokens
Elastic post-shrink DP all_reduce warmup done: ... dp_size=8 total_ms=...
Elastic post-shrink DP all_reduce warmup done: ... dp_size=4 total_ms=...
exit_code=0
```

所以 floor=4 成功并不说明 warmup 安全，只说明当时没有跨过失败阈值。floor=2 与 floor=4 的区别是：

- floor=2 最终会 shrink 到 `[14, 15]`，`dp_size=2`；floor=4 最小只到 `[12, 13, 14, 15]`，`dp_size=4`。
- floor=2 下 mode=3 需要补更多 CPU-source runtime rows，shrink 后 refresh/import 压力更高。
- floor=2 high-KV 目标使用 `2GiB` headroom、KV cache 为 `397,824 tokens`，动态 workspace 余量仍然很紧。
- post-shrink DP all_reduce warmup 不是业务必要路径，却会额外触发 HCCL workspace 初始化。

结论：floor=4 没爆是余量/规模刚好没越界，不是这个 warmup 对 mode=3 必须存在。

### 修复策略

将 mode=3 对齐 mode=4/5：默认跳过 post-shrink DP all_reduce warmup。

代码行为：

```text
Elastic post-shrink DP all_reduce warmup skipped: ... reason=mode3_default_disabled
```

控制开关：

```bash
VLLM_ASCEND_MODE3_ENABLE_POST_SHRINK_DP_WARMUP=0  # default
```

如果要做诊断，才显式打开：

```bash
VLLM_ASCEND_MODE3_ENABLE_POST_SHRINK_DP_WARMUP=1
```

打开时必须同时观察：

- `before_post_shrink_dp_all_reduce_warmup`
- `Elastic post-shrink DP all_reduce warmup done`
- `HcclAllreduce`
- `Memory_Allocation_Failure`
- `Failed to allocate resource[DeviceMemory]`

### 修复后验证

成功日志：

```text
log: pure-highkv-mode3-2_new.txt
mode=3 floor=2 kv_cache_init_headroom=2147483648
GPU KV cache size: 397,824 tokens
Elastic post-shrink DP all_reduce warmup skipped: ... reason=mode3_default_disabled
Elastic parallel shrink done: rank=14 active_ranks=[14, 15] ... warmup_ms=2.39 total_ms=13699.74
Elastic parallel shrink done: rank=15 active_ranks=[14, 15] ... warmup_ms=2.41 total_ms=13836.37
rollout_output_time_s: 265.878322
exit_code=0
```

这说明：

- mode=3/floor=2 现在可以在和 mode=4/5/floor=2 相同的 `2GiB` KV headroom 下完成 1 step。
- OOM 和 `HcclAllreduce` workspace failure 消失。
- 灾难性的 300s/350s 通信等待消失。
- 剩余性能差距主要来自 mode=3 CPU-source `refresh/import` 路径，而不是 HCCL all_reduce warmup。

### 对后续优化的约束

1. 不要默认重新打开 `VLLM_ASCEND_MODE3_ENABLE_POST_SHRINK_DP_WARMUP`。
2. 如果某个改动让 `warmup_ms` 从几毫秒回到几百毫秒/秒级，需要先检查是否误开了 DP warmup。
3. floor=2 的稳定性基线应使用 `pure-highkv-mode3-2_new.txt`，不要再用早期 `pure-highkv-mode3-2.txt` 的 6GiB headroom 作为当前下限。
4. 继续优化 mode=3 性能时，应聚焦 CPU-source `refresh_ms/preload_import`，而不是 post-shrink DP collective warmup。
