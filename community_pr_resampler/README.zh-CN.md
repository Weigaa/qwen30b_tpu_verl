# Length-Aware Resampler PR 说明

这个目录把和 resampler 相关的改动单独整理了出来，方便作为一个独立的社区 PR 提交。

目录中包含三个主要文件：

- `0001-length-aware-resampler.patch`: 最小可用补丁
- `README.md`: 英文使用说明
- `README.zh-CN.md`: 中文使用说明
- `run_resampler_example.sh`: 基于你的 regroup 脚本整理出的示例启动脚本

## 包含的功能

这个补丁包含 4 个和 resampler 直接相关的部分：

1. `LengthAwareEpochSampler`
   一个 curriculum sampler，会在每个 epoch 开始时，根据前一轮 rollout 的 response 长度统计，对 prompt 顺序重新排序。

2. 为 dataset 增加稳定 sample id
   给每条样本增加 `dataset_item_idx`，这样 rollout 长度统计就可以在不同 epoch 之间稳定回写到同一条 dataset 样本上。

3. 训练集按完整 batch 对齐
   将 train dataset 的样本数向下截断到 `gen_batch_size` 的整数倍，这样在 `drop_last=True` 时，每个 epoch 使用的是同一批 prompt。

4. 可选的 rollout 长尾保护
   根据 sampler 估计的 response 长度，对某个 batch 的 `max_tokens` 做动态裁剪：
   `cap = min(response_length, max(min_tokens, factor * expected_len))`

## 如何应用补丁

在仓库根目录下执行：

```bash
git apply community_pr_resampler/0001-length-aware-resampler.patch
```

如果你想先检查补丁是否可以应用：

```bash
git apply --check community_pr_resampler/0001-length-aware-resampler.patch
```

## 如何启用 resampler

在 Hydra 配置中指定自定义 sampler：

```bash
data.sampler.class_path=pkg://verl.experimental.dataset.length_bucket_sampler
data.sampler.class_name=LengthAwareEpochSampler
+data.sampler.bucket_size=1024
+data.sampler.ema_decay=0.7
+data.sampler.shuffle_batch_blocks=True
```

推荐同时设置：

- `data.dataloader_num_workers=0`
- `data.shuffle=False`
- 训练 dataloader 保持 `drop_last=True`

之所以推荐 `dataloader_num_workers=0`，是因为 sampler 会在主进程中根据上一轮 rollout 统计信息更新顺序。如果开启多 worker，容易出现 worker 侧拿到旧 sampler 状态的问题，也会让行为更难复现。

## 可选的 rollout 长尾保护

下面这些环境变量用于控制过长 batch 的提前截断：

```bash
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=1
export VLLM_ROLLOUT_EARLY_STOP_FACTOR=2.0
export VLLM_ROLLOUT_EARLY_STOP_MIN_TOKENS=10000
```

默认行为：

- 默认开启
- 只有 dataloader sampler 是 `AbstractCurriculumSampler` 时才会生效
- 如果算出来的 cap 大于等于原始 `response_length`，则不做任何修改

## 预期行为

- 第 1 个 epoch 和 baseline 基本一致，因为这时还没有历史 rollout 长度信息
- 从第 2 个 epoch 开始，会把预估 response 长度接近的 prompt 分到邻近 batch 中
- 这样可以减少同一个 generation batch 内部“超长样本 + 超短样本”混合的问题，通常能提升 rollout 吞吐

## PR 范围说明

这个目录故意只保留了和 resampler 直接相关的改动，不包含以下其他本地实验改动：

- request id 日志增强
- checkpoint 保存逻辑
- draft training / profiling 相关逻辑
- elastic EP 或 MoE 实验逻辑

这样做是为了让社区 PR 的 review 范围更小，更聚焦在 resampler 优化本身。
