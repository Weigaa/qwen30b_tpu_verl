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

