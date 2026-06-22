# Elastic Hybrid MoE Decode 项目总结

## 1. 项目目标

本项目的核心目标，是解决 MoE 模型在 decode 尾部阶段的严重负载不均衡问题。

在 baseline 中，一旦某个 rank 提前完成，它不会退出当前并行执行，而是继续通过 `dummy run` 陪伴其余未完成 rank 一直运行到整个 step 结束。这个方案实现简单、语义稳定，但会产生明显的资源浪费：

- 空闲 rank 持续占用设备算力与显存
- decode 尾部存在大量无效执行
- 吞吐、能效和资源利用率都较差

我们的目标不是简单“减少一点浪费”，而是构建一套 **可以在 decode 尾部动态缩容、并保持模型语义不损失的弹性 MoE 运行时机制**。

从研究角度看，这项工作的核心问题可以表述为：

> 当 MoE decode 进入长尾阶段时，如何在不破坏正确性、不改变模型语义的前提下，将完成较早的 rank 从“dummy 陪跑”转变为“弹性收缩”，并进一步通过固定容量的 resident expert 机制支撑更深层次的并行组缩容？

---

## 2. 目前已经完成的工作

### 2.1 从 Dummy-Run 陪跑转向 Elastic Shrink

我们已经把原本“某个 rank 完成后继续 dummy run”的策略，推进成了真实的弹性缩容机制。

当前系统不再默认让空闲 rank 持续陪跑，而是允许在 decode 尾部根据 unfinished requests 的分布进行并行组重构。

这意味着系统现在具备了：

- 从 full-world 执行态进入小规模 active-rank 执行态
- active-rank 之间重新构建 DP/EP 组
- inactive rank 通过 sequence catch-up 与 restore 机制回归 full-world

这一步已经把问题从“被动浪费算力”转成了“主动重构执行形态”。

### 2.2 从单次缩容扩展到 Repeated Shrink

我们没有停留在一次性 `16 -> 8` 缩容，而是继续推进到了 repeated shrink：

- `16 -> 8`
- `8 -> 4`

这非常关键，因为 decode 尾部的不平衡并不是一次 shrink 就能吃干净的。很多情况下，8-rank 组内部仍然会继续出现明显的完成时间偏差。

因此 repeated shrink 的价值在于：

- 继续回收尾部 slack
- 进一步压缩空转设备数
- 把 decode 尾部性能优化从“一次性改进”推进为“多阶段调度能力”

### 2.3 引入 Lossless Hybrid Expert Residency

当执行态进一步缩到 4-rank 时，每个 rank 逻辑上拥有的 expert 数会显著增加。如果还要求所有 owned experts 全部常驻 NPU，会导致显存压力过大。

因此我们实现了 hybrid resident/offload 机制：

- 每个 rank 逻辑上拥有完整的 owned experts
- 其中一部分 resident experts 常驻 NPU
- 其余 experts 保存在 CPU
- 执行时根据 routing 需要，在 CPU 和 NPU resident slots 之间进行交换

这个机制的本质，是把原本“所有权”和“物理驻留”解耦了。

这已经不只是 shrink，而是在做一套 **expert residency management** 运行时。

### 2.4 从 Fresh Allocation 演化到 Fixed-Slot Reuse

在最初的 hybrid 版本中，每次进入 hybrid resident materialization 时，系统会重新在 NPU 上申请新的 runtime expert buffer。这带来了两个明显问题：

- 增加切换延迟
- 在 live KV cache 存在时容易触发 OOM

因此我们进一步把机制推进成 **固定槽位复用**：

- NPU 上最多始终只使用固定数量的 resident slots
- 初始 full-world 阶段使用前 8 个槽位
- 第一次 shrink 后使用前 16 个槽位
- 第二次 shrink 进入 hybrid 阶段后，仍然只使用这 16 个 NPU 槽位
- 额外 owned experts 保留在 CPU，需要时与 NPU resident slots 进行交换

这个改动使系统从“功能正确”变成了“更具系统合理性”的版本。

它带来的直接好处包括：

- 避免重复 `torch.empty`
- 降低切换瞬时显存峰值
- 降低 allocator 干扰
- 缓解 live KV cache 场景下的 OOM 风险
- 降低 expert materialization 延迟

---

## 3. 已经遇到并解决掉的关键挑战

这部分是整个项目最有论文价值的地方之一，因为它体现了这不是一个简单工程 patch，而是一系列真实的系统问题。

### 3.1 通信后端混用问题

在 hybrid path 早期版本中，prepare/apply 已经切到了 `AllToAll`，但 finalize 仍然留在 `MC2`。

这导致系统在同一次 forward 内部混用了不同 MoE backend 的 prepare/finalize 语义，直接引发执行错误。

我们已经修复为：

- 整个 hybrid forward 期间统一强制走 `AllToAll`
- 直到整个 forward 结束后才恢复原上下文

这一步解决的是 **通信后端语义一致性**。

### 3.2 AllToAll 元数据同步与设备侧 gather 问题

在 `AllToAll` token dispatch 里，小元数据 gather 曾经出现 ACL stream synchronize 错误。问题并不在主计算 tensor，而是在小规模 split 元数据同步上先炸掉。

我们做了两步修复：

- 将元数据 gather 切到 host path
- 避免在 NPU 上对小元数据执行易出错的 gather/sync

这一步解决的是 **小元数据同步的鲁棒性**。

### 3.3 多 rank wave 规划不一致

早期 hybrid wave 计划是每个 rank 根据自己的本地 token 单独生成的。这样会导致：

- 不同 rank 的 wave 数不一致
- 不同 rank 的 resident group 切换顺序不一致
- collective 次序分叉
- 最终形成 gloo timeout / hang

我们修复为：

- 用 active ranks 的全局 metadata 共同生成 wave plan
- 所有 rank 基于同一份 wave schedule 执行

这一步解决的是 **distributed wave scheduling correctness**。

### 3.4 Zero-Token Wave 的一致性问题

在某些 wave 中，某个 rank 可能本地没有 token。如果该 rank 直接跳过 wave，collective 序列就会不一致。

我们修复为：

- zero-token wave 也必须参与 collective
- 同时补齐 combine/unpermute 的空 tensor 边界处理

这一步解决的是 **collective participation completeness**。

### 3.5 Dummy Batch 与 Hybrid 状态不兼容

在 delayed shrink 场景下，空闲 rank 会跑 dummy batch 维持同步。但 dummy path 早期没有走 hybrid 路径，结果直接落到普通 remap 分支，最终把一些 logical expert remap 成 `-1`。

我们修复为：

- dummy batch 也走 hybrid path
- 同时修复 zero-token combine 的特殊形状问题

这一步解决的是 **dummy-safe elastic execution**。

### 3.6 Restore 路径与 Full-World 回归

在 repeated shrink 后，系统不只是要“继续往前跑”，还要能回到 full-world。这里涉及：

- non-member ranks 的 group sequence catch-up
- full-world MoE layout reset
- prefix cache reset
- runtime state 清理与恢复

目前 restore 路径已经真实走通到：

- `Elastic parallel restore done`
- `Elastic parallel groups restored`

这说明系统已经具备了 **双向切换能力**，而不是单向 shrink。

### 3.7 Live KV Cache 场景下的 OOM

这是迄今为止最典型的系统级挑战。

OOM 并不是发生在普通 forward 里，而是发生在 `has_unfinished_requests()` 的动态 shrink 阶段。当时：

- KV cache 仍然驻留在设备上
- 系统正在尝试进入 `8 -> 4` hybrid
- hybrid resident materialization 还要额外申请 runtime expert buffer

这导致切换瞬间内存峰值过高，直接 OOM。

这也是最终促成 fixed-slot reuse 设计的重要原因。

---

## 4. 当前最值得强调的创新点

如果以 HPCA 的标准来组织贡献点，目前最有价值的创新可以提炼为以下几项。

### 4.1 Tail-Aware Repeated Shrink for MoE Decode

相较于 baseline 的 dummy-run 陪跑，我们提出的是：

- 面向 decode 尾部不平衡的弹性重构
- 并且支持 repeated shrink，而不是一次性收缩

这是整个系统的入口创新点。

### 4.2 Lossless Hybrid Expert Residency

我们没有简单粗暴地要求所有 owned experts 同时驻留 NPU，而是构建了：

- logical ownership 完整保留
- physical residency 受限于固定 resident capacity
- 通过 CPU/NPU expert swapping 实现更深缩容

这是系统设计层面的核心创新。

### 4.3 Fixed-Slot Resident Reuse

这一步非常像一个“系统论文里很像样的点”：

- resident capacity 固定
- 物理槽位固定
- expert 内容在固定槽位内交换
- 避免重复分配 runtime buffer

它同时改善：

- 显存峰值
- allocator 干扰
- materialization latency
- runtime stability

### 4.4 Elastic Correctness Protocol

为了让 repeated shrink + hybrid swap 真正成立，我们实际上已经做出了一整套 correctness protocol，包括：

- 全局一致的 wave planning
- zero-token wave 参与 collective
- dummy-safe hybrid path
- restore catch-up
- communication backend consistency

这部分如果写清楚，会显著提升论文的系统深度。

---

## 5. 相较于 Baseline Dummy-Run 方案，各方案的优劣势

### 5.1 Baseline：完成后继续 Dummy Run 陪跑

优点：

- 实现最简单
- 正确性风险最低
- 不需要复杂的 group rebuild、expert migration、restore 协议

缺点：

- 空闲 rank 长时间空转
- decode 尾部设备利用率低
- 吞吐和能效都较差
- 没有真正解决 long-tail imbalance，只是把它“陪跑掉”

### 5.2 单次 Shrink：`16 -> 8`

优点：

- 比 baseline 明显更进一步
- 复杂度相对可控
- 不一定需要引入 CPU swap

缺点：

- 收益存在上限
- 8-rank 组内仍然可能继续出现明显尾部不平衡
- 对极长尾场景不够激进

### 5.3 Repeated Shrink：`16 -> 8 -> 4` + Fresh Runtime Buffer

优点：

- 能进一步回收尾部 slack
- 4-rank 阶段收益潜力更大
- 证明了 repeated shrink 机制本身是可行的

缺点：

- hybrid 切换瞬间会重新分配 runtime resident buffer
- live KV cache 下容易 OOM
- allocator 造成额外性能和稳定性风险

### 5.4 Repeated Shrink：`16 -> 8 -> 4` + Fixed-Slot Reuse

优点：

- 这是目前最强、最完整的版本
- 始终只使用固定数量 resident NPU slots
- 避免重复分配 runtime buffer
- 更接近稳定、长期可运行的 runtime 设计
- 更有系统论文风格

缺点：

- 实现复杂度更高
- 需要 stash/restore primary prefix
- 如果 routing locality 很差，CPU/NPU expert swapping 开销可能变大
- 仍然需要更强实验证明收益覆盖了额外 swap 成本

---

## 6. 目前这项工作的论文价值

如果目标是投稿 HPCA，那么目前这项工作已经具备了不错的系统论文雏形。

### 6.1 这不是一个简单 patch，而是一个运行时系统

当前实现已经不只是修补某个 bug，而是在构建一个 runtime：

- tail-aware shrink 决策
- dynamic expert residency
- multi-stage parallel reconfiguration
- distributed correctness maintenance
- restore protocol

这类问题本身就很有系统结构与 runtime co-design 的味道。

### 6.2 论文故事可以如何表述

一个比较自然的论文叙事是：

#### Observation

MoE decode 尾部存在严重 completion skew。传统 dummy-run 方案为保证同步，牺牲了大量设备利用率。

#### Challenge

简单 shrink 不够。真实系统中还会遇到：

- expert placement 变化
- communication backend 不一致
- wave scheduling 分叉
- dummy / restore 兼容性
- live KV cache 下的内存峰值问题

#### Insight

真正需要的是一套：

- repeated shrink
- fixed-capacity resident execution
- correctness-preserving elastic protocol

#### Design

设计一套 elastic hybrid MoE decode runtime：

- repeated shrink
- hybrid expert residency
- fixed-slot resident reuse
- global wave synchronization
- dummy-safe and restore-safe execution

#### Claim

在不改变模型语义的前提下，显著降低 decode 尾部空转，提高吞吐与资源利用率，并控制切换过程中的内存峰值。

---

## 7. 为了达到 HPCA 目标，接下来还需要做什么

目前最大的短板已经不是“有没有机制”，而是“如何把它系统化地证明出来”。

### 7.1 需要更完整的实验设计

至少应当有以下版本做对比：

- baseline dummy-run
- single shrink：`16 -> 8`
- repeated shrink + fresh buffer
- repeated shrink + fixed-slot reuse

### 7.2 需要的核心指标

建议实验中至少报告：

- step latency
- decode 尾部时长
- tokens/s
- 有效设备利用率
- 被消除的 idle-rank 时间
- shrink latency
- restore latency
- NPU allocated / reserved peak
- CPU<->NPU swapped bytes
- resident hit / CPU miss
- 每阶段 wave 数与 wave token 分布

如果条件允许，还可以增加：

- power / energy
- OOM incidence / stability

### 7.3 需要 correctness 验证

必须证明：

- 输出没有语义错误
- 多轮 shrink/restore 后不会积累状态错误
- dummy / hybrid / restore 路径切换不会破坏运行一致性

如果能做到，最好补：

- 输出分布或 logprob 近似一致性
- 采样结果统计性质稳定

### 7.4 需要更系统化的分析模型

HPCA 会更喜欢看到一些“解释为什么有效”的分析，而不只是跑出来的结果。建议补：

- shrink gain model
- swap overhead model
- memory peak model
- 何时值得继续从 8 shrink 到 4 的判定依据

即使这些模型是近似的，也会显著提升论文说服力。

### 7.5 需要更抽象的机制提炼

当前代码里已经有很多有价值的机制，但论文里需要进一步抽象成更少、更清晰的设计原则，比如：

- Tail slack should be converted into elastic compute-group contraction.
- Logical expert ownership and physical expert residency should be decoupled.
- Elastic shrink requires a correctness protocol, not just a scheduler decision.
- Resident expert buffers should be fixed-capacity and reusable to avoid transition-time memory spikes.

这类表述对论文非常重要。

---

## 8. 建议的下一步推进顺序

为了把这项工作真正收敛成可投稿的系统论文，建议按如下顺序推进。

### 第一步：把当前版本稳定跑通

目标：

- 多 step 稳定执行
- repeated shrink 与 restore 稳定
- 不再出现通信路径错误、dummy 不兼容错误、或切换期 OOM

### 第二步：系统化补 instrumentation

需要把当前零散日志升级成可用于论文的数据采样机制。重点记录：

- shrink / restore 事件时间线
- resident hit / CPU miss
- swap bytes
- memory allocated / reserved / peak
- 每阶段 latency breakdown

### 第三步：做完整 ablation

把下列几个版本全部跑出来：

- baseline
- single shrink
- repeated shrink without slot reuse
- repeated shrink with slot reuse

### 第四步：开始写论文骨架

建议尽快固定论文主结构：

- Introduction
- Motivation
- Background and Problem
- Design
- Correctness Protocol
- Implementation
- Evaluation
- Discussion / Limitations
- Related Work
- Conclusion

---

## 9. 当前阶段的总体判断

到目前为止，这项工作已经明显超出了“普通工程优化”的范畴。

它已经具备：

- 清晰的问题定义
- 非平凡的系统机制
- 真实且复杂的分布式运行时挑战
- 已经跑起来的原型证据
- 进一步组织成高水平论文的潜力

从论文成熟度看，目前最缺的不是“idea”，而是：

- 更系统的实验
- 更抽象的机制总结
- 更扎实的 correctness 与 overhead 分析

如果后续实验结果能够支撑 repeated shrink + fixed-slot reuse 在真实 tail decode 场景中显著优于 baseline dummy-run，那么这项工作具备冲击 HPCA 这类系统结构会议的潜力。

---

## 10. 一句话总结

这项工作的核心贡献，不只是“让提前完成的 rank 不再 dummy 陪跑”，而是：

> 构建了一套面向 MoE decode 长尾阶段的弹性执行运行时，将重复缩容、混合 expert 驻留、固定槽位复用与正确性协议结合起来，在不改变模型语义的前提下，尝试系统性地提升尾部阶段的设备利用率、吞吐和内存效率。

