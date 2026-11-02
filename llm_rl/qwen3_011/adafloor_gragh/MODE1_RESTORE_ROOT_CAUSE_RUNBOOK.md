# Mode1 Restore Root Cause Runbook

Date: 2026-06-04

## 结论

- `mode=1 floor=8` 现在已经恢复到可正常完成 `shrink -> restore -> rollout_output_time_s` 的状态。
- 这次真正的回归点不是 `mode=1` 的算子逻辑本身，而是 `mode=1` 在 `shrink` 和 `restore` 之间对 DP/EP/MC2 communicator cache 的生命周期管理出了问题。
- 之前出现过一次 `mode=1` OOM，但那次不是正确 baseline，而是误跑成了 `floor=1`，不能拿来证明 `mode=1 floor=8` 本身有问题。

## 1. 之前为什么会有一次 Mode1 OOM

那次 OOM 日志是：

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260603165022_elastic.txt`

它的开头配置是：

- `sidecar=1 mode=1 floor=1`
- 证据：`wjeagerqwen30b-a3b-with_draft_breakdown_20260603165022_elastic.txt:2`

所以那次并不是要对比的 `mode=1 floor=8` baseline，而是误用了通用入口脚本的默认配置，跑成了 `floor=1`。

OOM 直接证据：

- `RuntimeError: NPU out of memory. Tried to allocate 138.00 MiB`
- 证据：`wjeagerqwen30b-a3b-with_draft_breakdown_20260603165022_elastic.txt:3027`
- 当时 `NPU 0` 只剩 `295.52 MiB` 左右 free，分配 `138 MiB` 就失败
- 结束码：`wjeagerqwen30b-a3b-with_draft_breakdown_20260603165022_elastic.txt:3134`

因此，这次 OOM 的原因应归类为：

1. 误配置成了 `floor=1`
2. 不是目标 baseline `floor=8`
3. 当时显存余量本来就极小，额外分配 `138 MiB` 直接失败

## 2. 这次真正的问题是什么

真正花时间修的是 `mode=1 floor=8` 的 restore 回归。

坏样本：

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260603235803_elastic.txt`

它的症状是：

1. `restore requested` 在 `00:02:21`
   - `...20260603235803_elastic.txt:4070`
2. `restore done` 拖到 `00:07:26`
   - `...20260603235803_elastic.txt:4314`
3. 单 rank `restore total_ms` 约 `305811 ms`
   - `...20260603235803_elastic.txt:4314`
4. `rollout_output_time_s` 被拖到 `400.759063`
   - `...20260603235803_elastic.txt:4501`

更关键的是，坏样本在 `shrink` 阶段就提前 drop stale cache：

- `before_stale_group_cache_drop_after_shrink`
  - `...20260603235803_elastic.txt:3851`
- `drop_stale_group_cache_ms` 非零，且在多张卡上达到 `600-1000ms`
  - 例如 `...20260603235803_elastic.txt:3933`

这说明坏路径是：

1. `shrink` 时提前清掉了 full-world 的 stale DP/EP/MC2 cache
2. `restore` 时无法复用 full-world communicator
3. `restore` 被迫走慢路径，最终掉进多分钟黑洞

## 3. Root Cause

根因在 `mode=1` 的 communicator cache 生命周期回归：

1. `mode=1` 在 `shrink` 后错误地提前 drop 了 stale full-world cache
2. `restore` 时又把 shrink-stage live group 和 full-world rebuild 纠缠在一起
3. MoE comm setup cache 没有在 shrink/restore refresh 前明确 reset，存在复用旧 group signature 的风险

## 4. 这 4 个小时里主要修了什么

### 4.1 修正 shrink 后 stale cache 的默认行为

文件：

- `vllm_ascend/worker/worker_v1.py`

关键改动：

- `_should_drop_stale_group_cache_after_elastic_shrink()`
- `VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK` 的默认值从旧坏路径恢复为不在 shrink 后默认 drop

代码位置：

- `vllm_ascend/worker/worker_v1.py:6081`
- `vllm_ascend/worker/worker_v1.py:6096`

现在的语义是：

1. `mode=1 zero-headroom`
2. 默认保留 full-world DP/EP/MC2 cache 穿过 shrink
3. 等 full restore 结束后，再清理 floor-N 的 stale MC2

### 4.2 在 full restore 前主动销毁 shrink-stage live group

文件：

- `vllm_ascend/worker/worker_v1.py`

关键改动：

- 新增 `mode=1` full restore helper
- full restore 前不再把当前 live 的 floor-8 DP/EP/MC2 继续 stashed 下去

代码位置：

- `vllm_ascend/worker/worker_v1.py:6453`
- `vllm_ascend/worker/worker_v1.py:6512`

作用：

1. 避免 shrink-stage live group 和 full-world rebuild 打架
2. 避免 stale communicator state 把 restore 拖慢到分钟级

### 4.3 在 shrink / restore refresh 前 reset MoE comm setup cache

文件：

- `vllm_ascend/ops/moe/moe_comm_method.py`
- `vllm_ascend/worker/worker_v1.py`

关键改动：

- 增加 `reset_moe_comm_method_cache()`
- 在 shrink refresh / restore refresh 前显式 reset

代码位置：

- `vllm_ascend/ops/moe/moe_comm_method.py:47`
- `vllm_ascend/ops/moe/moe_comm_method.py:80`
- `vllm_ascend/worker/worker_v1.py:7255`
- `vllm_ascend/worker/worker_v1.py:7466`

作用：

1. 避免沿用旧 group signature
2. 保证新的 DP/EP/MC2 topology 刷新后，MoE comm method 与当前 group 一致

## 5. 修复后的验证结果

### 成功样本 1

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260604003155_elastic.txt`

证据：

1. `restore requested`
   - `...20260604003155_elastic.txt:3968`
2. `restore done` 回到秒级，约 `2.17s - 2.86s`
   - `...20260604003155_elastic.txt:4214`
   - `...20260604003155_elastic.txt:4393`
3. `rollout_output_time_s = 102.028926`
   - `...20260604003155_elastic.txt:4399`
4. `exit_code=0`
   - `...20260604003155_elastic.txt:4550`

### 成功样本 2

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260604004550_elastic.txt`

证据：

1. `restore requested`
   - `...20260604004550_elastic.txt:3966`
2. `restore done` 回到秒级，约 `2.18s - 2.66s`
   - `...20260604004550_elastic.txt:4207`
   - `...20260604004550_elastic.txt:4312`
3. `rollout_output_time_s = 100.079605`
   - `...20260604004550_elastic.txt:4308`
4. `exit_code=0`
   - `...20260604004550_elastic.txt:4532`

### 与坏样本对比

坏样本：

- `...20260603235803_elastic.txt:4501`
- `rollout_output_time_s = 400.759063`

修好后：

- `...20260604003155_elastic.txt:4399`
- `rollout_output_time_s = 102.028926`

- `...20260604004550_elastic.txt:4308`
- `rollout_output_time_s = 100.079605`

## 6. 当前判断

当前可以下的结论是：

1. `mode=1 floor=8` 的 restore 根因已经修掉
2. `rollout_output_time_s` 已经过两轮独立样本验证
3. 那次 OOM 是误跑成 `floor=1` 的配置问题，不是这次 restore 回归的根因
4. 本次修复只针对 `mode=1` 的 communicator cache 生命周期，没有去改 `mode=0/2/4/5` 的模式语义

## 7. 复现实验命令

```bash
export TRAINER_TOTAL_EPOCHS=1
export DRAFT_PROFILE_MODE=breakdown
bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager_shrink8_util.sh
```

期望看到：

1. `mode=1 floor=8`
2. `Elastic parallel restore requested before rollout restore rpc`
3. `Elastic parallel restore done ... total_ms=2xxx`
4. `rollout_output_time_s: ~100s`

## 8. Adaptive KV resize 容量回不来的踩坑记录

### 8.1 现象

后续做 `mode=1` adaptive floor 规划时，step 5 需要从 floor-4 的 KV
容量恢复到 full-world/no-shrink 容量：

- 旧容量：`old_tokens=280576`
- 目标容量：`target_tokens=380800`
- 预期：`new_tokens=380800`

但问题日志里实际只能恢复到约 `286k-291k tokens`，典型表现是：

```text
Mode1 adaptive KV resize phase=plan_new_kv_done target_tokens=380800 new_tokens=286976
GPU KV cache size: 286,976 tokens
```

这会导致 step 5 即使规划上认为 KV-safe，运行时仍然没有足够 KV cache
空间，容易出现 preempting 或 OOM。

### 8.2 一开始容易误判的方向

这个问题很容易被误判为：

1. KV cache 没有释放干净；
2. floor=16 的专家槽位没有恢复成 no-shrink 布局；
3. communicator / HCCL group 没有释放；
4. `mode=1 floor16` 本身比 `mode=0` 多占了不可避免的 non-torch 空间。

但 live tensor scan 证明这些都不是主因。

关键证据是：

```text
known_bytes={'model_runner.kv_caches': 0, ...}
loaded_capacity_hist={8: 48}
weight_shape_hist={'(8, 1536, 2048)/(8, 2048, 768)': 48}
```

这说明：

- 旧 KV cache 已经清掉；
- 当前 live MoE module 已经 compact 到 8-slot 物理权重；
- 但可用空间仍然不足。

### 8.3 真正根因

真正根因是：旧的 32-slot MoE 参数张量已经不再属于当前 module，
但仍然被一个 full-name parameter dict 持有引用。

问题日志里的直接证据：

```text
stale_referrers=
201326592:torch.bfloat16:(32, 1536, 2048):
dict(keys=['model.layers.0.mlp.experts.w13_weight'], owners=[])
```

也就是说，`module.w13_weight / module.w2_weight` 已经被替换成新的
8-slot Parameter，但旧的 32-slot Parameter 仍然存在于类似下面的缓存字典里：

```python
{
    "model.layers.0.mlp.experts.w13_weight": old_32_slot_parameter,
    "model.layers.0.mlp.experts.w2_weight": old_32_slot_parameter,
    ...
}
```

每个 rank 会残留 96 个这样的 full-name expert 参数条目：

- 48 层 `w13_weight`
- 48 层 `w2_weight`

合计大约：

```text
cleared_stale_param_entries=96
cleared_stale_param_bytes=14495514624
```

这约 14.5GB stale NPU storage 会直接挤占 KV cache 的可用空间，所以
KV resize 规划阶段只能拿到约 28GB 可用空间，最终只能建出约 `286k`
tokens，而不是 `380800` tokens。

### 8.4 修复方法

修复点在：

- `vllm_ascend/worker/worker_v1.py`

核心逻辑是在 step-floor KV resize 前，当 `target_floor == world_size`
准备进入 full-world/no-shrink step 时，清理顺序必须保证“旧引用先断开，
新 KV 后申请”：

1. 先 compact 当前 MoE module 的物理权重到 8-slot；
2. 在 floor prepare 阶段先扫描并清一次 stale full-name parameter dict；
3. 清理旧 KV cache；
4. 如果仍怀疑有额外 stale 引用，可设置
   `VLLM_ASCEND_MODE1_CLEAR_STALE_PARAM_DICTS_AFTER_OLD_KV=1`，在旧 KV
   已释放、申请新 KV 之前，再扫描并清一次 stale full-name parameter dict；
5. 扫描 Python GC 中的 dict；
6. 找到 key 形如：
   - `model.layers.*.mlp.experts.w13_weight`
   - `model.layers.*.mlp.experts.w2_weight`
7. 如果 value 是 NPU tensor/Parameter，shape 第一维大于当前 compact 后的
   8-slot，并且不是当前 `model.named_parameters()` 里的 live Parameter，
   就从 dict 中 pop 掉；
8. 再执行 `gc.collect()` / `torch.npu.empty_cache()` / `torch.npu.synchronize()`；
9. 最后才重新规划并分配新 KV cache。

默认快速路径下，关键 resize 阶段日志是：

```text
Mode1 adaptive KV resize phase=clear_old_kv_done
Mode1 adaptive KV resize phase=clear_stale_param_dicts_skipped reason=disabled
Mode1 adaptive KV resize phase=plan_new_kv_start
```

如果打开二次 stale 参数清理，关键日志是：

```text
Mode1 adaptive KV resize phase=clear_old_kv_done
Mode1 adaptive KV resize phase=clear_stale_param_dicts_start
Mode1 adaptive KV resize stale full-name parameter caches cleared:
Mode1 adaptive KV resize phase=clear_stale_param_dicts_done
Mode1 adaptive KV resize phase=plan_new_kv_start
```

也就是说，不再依赖“先保留旧参数、创建新参数、之后再检索冗余并删除”
这种滞后流程。通常在 floor prepare 里就断开旧 32-slot Parameter 缓存引用；
如果需要更保守的诊断路径，可以在旧 KV cache 释放窗口里再清一次，然后才
进入 `plan_new_kv` / `allocate_new_kv`。

性能注意：`shrink_lossless_loaded_weights_to_primary()` 默认不再每层执行
`gc.collect()` / `torch.npu.empty_cache()` / `torch.npu.synchronize()`，而是
依赖 floor prepare 末尾统一释放。若需要回退到逐层强同步清理，可设置：

```bash
export VLLM_ASCEND_MODE1_FULL_WORLD_COMPACT_EMPTY_CACHE_PER_LAYER=1
```

修复后的健康日志应看到：

```text
cleared_stale_param_dicts=1
cleared_stale_param_entries=96
cleared_stale_param_bytes=14495514624
```

然后在 `before_plan_new_kv` 阶段应看到：

```text
known_bytes={'model_runner.kv_caches': 0, ...}
stale_referrers=
```

也就是旧 KV 已清空，旧 32-slot MoE 参数也没有 referrer 了。

### 8.5 修复后的验证结果

成功样本：

- `mode1_length_sorted_e2e_adaptive_floor4_fast15_threshold/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260624215127.txt`

关键证据：

```text
Mode1 step floor preparation restored full-world layout:
cleared_stale_param_dicts=1
cleared_stale_param_entries=96
cleared_stale_param_bytes=14495514624

Mode1 adaptive KV resize phase=plan_new_kv_done
target_tokens=380800 new_tokens=380800

GPU KV cache size: 380,800 tokens
```

16 个 rank 都成功恢复到：

```text
GPU KV cache size: 380,800 tokens
```

并且本次日志没有：

- `Preempting`
- `Memory_Allocation_Failure`
- `OOM`
- `RuntimeError`

最终正常结束：

```text
Epoch 0 completed in 406.10 seconds.
response/aborted_ratio:0.0
```

### 8.6 经验结论

如果 adaptive KV resize 目标容量回不来，不要只看 KV cache 本身。
要同时检查：

1. `known_bytes['model_runner.kv_caches']` 是否已经为 0；
2. 当前 MoE module 的 `weight_shape_hist` 是否已经 compact；
3. live tensor scan 里是否还有 `(32, 1536, 2048)` 或 `(32, 2048, 768)`
   的旧 MoE 权重；
4. `stale_referrers` 是否出现 full-name 参数字典。

这类问题的本质是“旧参数对象仍被 Python 引用”，不是 KV allocator
本身不会释放空间。只有把 stale Parameter 引用断开后，NPU allocator
才能真正把空间还给后续 KV cache 扩容。
