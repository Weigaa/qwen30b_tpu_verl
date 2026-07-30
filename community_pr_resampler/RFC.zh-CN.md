# RFC: 基于历史生成长度的 Epoch 级重分组 Resampler

## 1. 背景与动机

在 GRPO + vLLM 的 rollout 场景中，同一个 generation batch 内 prompt 对应的 response 长度差异往往很大。  
当短样本和长样本混在同一个 batch 中时，整个 batch 的结束时间会被少数超长样本拖住，导致以下问题：

- rollout 吞吐下降
- dummy run 或空转等待增多
- 每 step 的生成耗时波动很大
- 在长序列配置下，个别离群样本会显著拉长训练总时间

从实验观察来看，这类问题并不一定来自模型本身退化，很多时候只是 batch 内部长度分布不均衡。  
因此，一个更直接、侵入性更低的优化方向是：在不改变训练目标的前提下，尽量让长度相近的样本进入同一个 generation batch。

本 RFC 提出的方案就是一个面向 epoch 边界的 length-aware resampler。它利用上一轮 rollout 收集到的 response 长度统计，在下一个 epoch 开始前重新组织样本顺序，以降低长短样本混排带来的尾部拖慢问题。

## 2. 核心设计

### 2.1 LengthAwareEpochSampler

新增 `LengthAwareEpochSampler`，继承已有的 `AbstractCurriculumSampler`。

它的基本行为是：

1. 在 rollout 结束后，从 batch 中读取 `response_mask`
2. 统计每个 dataset 样本对应的 response 实际长度
3. 用 EMA 更新每个样本的长度估计
4. 在下一个 epoch 开始时，根据估计长度对样本进行排序
5. 将长度相近的样本尽量放入相邻 batch

这样做的效果是：

- 第一个 epoch 与 baseline 基本一致
- 从第二个 epoch 开始，batch 内长度分布会更集中
- rollout 的平均耗时和尾部耗时都会下降

### 2.2 稳定样本标识

为了让“上一轮 rollout 的长度统计”能够稳定映射回 dataset 样本，需要给每条样本增加一个稳定 id。  
因此在 dataset 返回的数据中新增：

- `dataset_item_idx`

这个字段用于把 rollout 统计和 dataset 行建立稳定关联，避免 shuffle 或 epoch 切换时发生错配。

### 2.3 训练集按完整 batch 对齐

如果 train dataset 的样本总数不是 batch size 的整数倍，在 `drop_last=True` 时，不同 epoch 看到的有效样本集合可能发生变化。  
这会让 resampler 的长度统计和下一轮样本集合之间出现偏移。

因此本方案增加一个简单的对齐策略：

- 将训练集样本数向下截断到 `gen_batch_size` 的整数倍

这样可以保证：

- 每个 epoch 使用的是同一批样本
- resampler 的长度统计具有可复用性
- 第二个 epoch 之后的分桶效果更稳定、更可解释

### 2.4 Rollout 长尾保护

除了重排之外，本方案还提供一个可选的长尾保护机制：

- 根据当前 batch 的历史估计长度，给 rollout 设置一个动态 `max_tokens` 上限

公式为：

`cap = min(response_length, max(min_tokens, factor * expected_len))`

用途是：

- 对明显超过预期很多倍的长尾 batch 进行提前截断
- 防止极少数异常样本严重拖慢整步训练

这个机制默认开启，但只有在 sampler 是 `AbstractCurriculumSampler` 时才会生效。

## 3. 实现效果

根据当前实验观察，这套方案带来的主要收益是：

- 第一个 epoch 基本不变，行为接近 baseline
- 从第二个 epoch 开始，长度接近的样本会被集中到同一 batch
- rollout 生成时间显著下降
- 每 step 的耗时分布更可控
- 对整体收敛性没有观察到明显负面影响

这个优化的重点不是改变模型训练目标，而是提升 rollout 阶段的调度效率。  
因此它更像一个“系统侧优化”，而不是算法本身的改动。

## 4. 文件说明

本目录下当前包含 5 个文件，其中与 PR 交付直接相关的 4 个核心文件如下：

### 4.1 `0001-length-aware-resampler.patch`

这是最小可用补丁，包含以下代码改动：

- 新增 `LengthAwareEpochSampler`
- 为 dataset 增加 `dataset_item_idx`
- 在 trainer 中增加 dataset 对齐逻辑
- 增加 rollout 长尾保护逻辑
- 在 vLLM rollout 中支持 `response_max_tokens_cap`

适合用于：

- 向社区提交代码变更
- 在干净基线仓库中直接应用补丁

### 4.2 `README.md`

英文版使用说明，面向社区开发者。  
主要内容包括：

- 补丁包含哪些能力
- 如何应用补丁
- 如何在 Hydra 中启用 resampler
- 推荐配置项
- rollout 长尾保护的环境变量说明

### 4.3 `README.zh-CN.md`

中文版使用说明，与英文版内容一一对应。  
适合：

- 在国内团队内部同步方案
- 作为中文背景材料附带给评审或协作者

### 4.4 `run_resampler_example.sh`

一个最小示例启动脚本，参考现有 regroup 训练脚本整理而来。  
主要作用是：

- 演示如何在脚本里启用 resampler
- 演示如何传入相关环境变量
- 给社区评审者提供一个可快速复现的入口

## 5. 额外说明

本 RFC 本身是一个设计与说明文档，不属于必须提交的代码补丁。  
它的主要作用是帮助评审者快速理解：

- 为什么需要这个优化
- 这个优化具体解决了什么问题
- 它和现有训练逻辑的关系是什么
- 目录中的几个文件分别做什么

如果要把这套内容提交给社区，建议将：

- `patch`
- `README`
- `example script`
- 本 RFC

一起作为完整材料提供，这样 review 成本会更低。
