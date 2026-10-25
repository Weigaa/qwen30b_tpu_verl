# TailServe

Working paper title: **TailServe: In-Place Opportunistic Serving in MoE RL Rollout Tails**.

本目录是从 `llm_rl/qwen3` 的本地vLLM **0.14.1**、vLLM-Ascend、VERL代码复制出的独立原型，不修改原AdaFloor仓库。模型权重、数据集和公共epoch0仍复用原有只读路径，没有复制检查点。题目是工作名称，尚未完成全面重名或新颖性检索。

## 已实现的闭环

1. 原RL题目、rank分配、采样参数及更新顺序保持原baseline设置，关闭重排、TailGuard、专家缩容与独立sidecar。EP16不变。
2. 本地RL请求全部完成、其他rank仍有RL请求时，在**同一个vLLM引擎**接入低优先级、同策略快照的普通推理。没有额外LLM实例、模型加载或专家迁移。
3. 默认采用连续推理。本地RL排空后，有请求且KV准入通过的rank每轮都推进真实辅助计算，不做固定间隔节流，不因主任务请求数变化重新暂停校准。分块prefill和每rank token预算控制单轮工作量。旧反馈门控仅供显式选择的对照实验使用，不能提供RL slowdown硬上界。
4. 原始RL internal/public请求ID独立跟踪，不推进原LLM请求计数器，辅助输出不进入RL DataProto、reward或训练样本。
5. **全局RL完成即关闭本轮服务**，不等待辅助长回答跑完。原地abort未完成辅助请求，检查前后端请求均排空并同步NPU，再返回VERL。权重更新/下一代请求不能复用旧KV。未完成结果标为cancelled，不计完成，不声称无损续跑。
6. 每次生成绑定 `RUN_ID / global_steps / generation` 逻辑策略版本。显式旧版本请求拒绝；`current`在准入时绑定。它不是模型权重的密码学hash，也不提供多个历史策略的同时驻留服务。
7. 辅助请求使用独立显式seed。只在本地RL已排空后保存随机数状态，结束时恢复Python、NumPy、torch CPU及当前NPU随机流，避免污染下一步RL。**这不是batch-invariant或逐token等价保证**。

HTTP接口只持久化队列，不接触NPU。辅助计算仍经过整个EP世界，不能把“本地队列空”当作全卡空闲，也不能把其rank-time从RL占用中直接扣除。

## 当前验证边界

已完成32项CPU测试。2026-10-25旧节流配置的真实NPU验证中，DeepScaleR的512条答案与训练更新完成，输出隔离和全局RL结束时取消/排空已验证，但两个GSM8K前缀退化。随后发现活KV保护漏掉实际使用的native `worker.py`，补齐后，16个rank的暂停/不暂停分块推理前64token完全一致。**新的默认连续、多rank辅助推理尚未执行NPU验证，当前不能作为已验证服务或论文性能结果。** 详细证据见`EXPERIMENTS.md`。

首版只支持：单机、TP1/PP1、固定DP/EP、in-process `external_launcher`、同步scheduler、chunked prefill、native sleep2/lifecycle1。支持选择eager/TQ2或FULL_DECODE_ONLY/TQ1启动路径，但真实Ascend混合prefill/decode及图路径仍需NPU smoke验证。MP/async调度、speculative、LoRA、多KV组、动态EPLB和缩容均fail closed，不自动降级为别的执行模式。

服务仅在rollout尾部推进，可排队、可取消、不承诺TTFT/TPOT或连续可用。更新前取消而不是持有旧KV跨权重继续。尚无崩溃后 exactly-once 外部系统承诺，单个spool只能由一个broker及一组RL worker使用。

## 快速检查，不启动NPU

```bash
cd /workspace/cann-recipes-train/llm_rl/tailserve
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_tailserve.py' -v
python tools/verify_tailserve_source.py
bash run_tailserve.sh dry-run
TAILSERVE_MODE=eager bash run_tailserve.sh dry-run
```

## HTTP接口手动运行

先选一个全新spool和结果名称，避免与已有实验串数据。示例三个终端：

终端1，启动不加载模型的队列/API：

```bash
cd /workspace/cann-recipes-train/llm_rl/tailserve
python -m tailserve_runtime.server --spool /data/tailserve_spool/first_trial --port 8099
```

终端2，提交短问题，可在训练启动前排队：

```bash
cd /workspace/cann-recipes-train/llm_rl/tailserve
python tools/tailserve_client.py --prompt 'What is 12 multiplied by 13?' --max-tokens 64
```

客户端仅加载tokenizer。默认按16个rank轮转放入队列，不动态改变RL分配。可用 `--rank 0` 固定辅助目标rank。若该rank迟迟未完成RL，请求会继续排队，这是机会式服务，不是请求丢失。tokenizer必须与实际MODEL_PATH一致。原始接口接受 `prompt_token_ids`，不暗中替用户添加另一个chat模板。

终端3，仅在16个NPU可用且准备好测量时手动执行：

```bash
cd /workspace/cann-recipes-train/llm_rl/tailserve
TAILSERVE_SPOOL=/data/tailserve_spool/first_trial \
TAILSERVE_RUN_ID=first_trial \
TAILSERVE_RL_STEPS=1 \
bash run_tailserve.sh run
```

默认graph/TQ1、native sleep2，一步RL，使用已有公共epoch0（若reuse.env存在）。没有自动tmux或自动占卡。用 `TAILSERVE_MODE=eager` 明确选择eager/TQ2，不是失败后的隐式fallback。可用 `TAILSERVE_ENABLE=0` 运行同路径原baseline对照，其他输入设置保持一致。结果默认写 `/data/tailserve_results/first_trial`，拒绝覆盖。

用返回的请求id查询状态：

```bash
curl http://127.0.0.1:8099/v1/tailserve/requests/REQUEST_ID
python tools/summarize_tailserve.py --spool /data/tailserve_spool/first_trial
```

状态包括queued、admitted、completed、cancelled、rejected_version、rejected_input。cancelled仅保存已观测到的前缀供诊断，不计完成，下一次重试必须作为新请求从prompt重新计算。辅助队列在RL更新阶段不推进。对外绑定非loopback地址必须传 `--auth-token`，生产部署仍需TLS、鉴权、限流和多用户隔离。

## DeepScaleR主任务 + GSM8K推理的首版实验

此入口提前将GSM8K放入同一队列，省去单独启动HTTP网关，不加载第二份模型。主任务保持DeepScaleR原始顺序、batch32 × n16、响应上限16384、公共epoch0、seed0。默认实际执行一个RL step和一次训练更新，不保存额外checkpoint。默认FULL_DECODE_ONLY/TQ1、native sleep2，不进行AdaFloor缩容、重排或侧模型部署。

```bash
cd /workspace/cann-recipes-train/llm_rl/tailserve
bash run_deepscaler_gsm8k.sh dry-run
TAILSERVE_RUN_ID=my_unique_gsm8k_trial bash run_deepscaler_gsm8k.sh run
```

GSM8K固定取`/data/gsm8k/train.parquet`前256行，按原始顺序轮转分配到16个rank，使用同模型chat模板并显式`enable_thinking=False`，贪心解码，输出上限与主任务一样为16384 tokens。Prompt上限同为1024，总上下文上限为17408。长度超限会报错，不会静默截断、跳题或筛选短题。可用`GSM8K_REQUESTS`改变请求数。参考答案只放在离线manifest里，不进入vLLM请求或RL训练batch。

默认只在本地RL请求全部完成后准入辅助，每rank最多一个在途请求。所有符合条件的空闲rank都可参与，每rank每轮调度预算与RL引擎`max_num_batched_tokens`对齐，默认17408，不再人为按8 tokens切分prefill。全局预算默认16 × 17408 = 278528，确保多个辅助rank参与时不会将单rank预算再次分摊到小于引擎上限。这个全局值是各rank预算之和，不是扩大单rank引擎容量。实际调度仍受引擎原有上限、prompt长度和KV容量约束。

入口从`ROLLOUT_MAX_NUM_BATCHED_TOKENS`、`MAX_RESPONSE_LENGTH`和`MAX_PROMPT_LENGTH`推导对应辅助默认值，排队前与运行时保持一致，不再使用旧的8-token/1024-output上限。可显式指定`TAILSERVE_*`做不同配置的对照，但旧spool不可直接复用，必须生成新请求manifest。贪心解码、thinking关闭、每题一个答案暂未改成RL采样方式，不能声称所有生成设置都与RL一致。

Decode通常每轮生成一个token，结果必须收集，不走丢弃输出的dummy分支。新请求重新prefill，不复用先前RL请求的上下文。KV准入按prompt加16384输出的最坏长度检查，不因追求性能而取消容量保护。更大prefill可能减少退出decode图的轮数，但也可能增加某一轮的全局停顿，收益需实测，不能宣称已达到最优。

默认`TAILSERVE_INTERFERENCE_FEEDBACK=0`且`TAILSERVE_AUX_INTERVAL_ROUNDS=1`。不等待8轮校准、不每32轮限流，也不因为主任务请求数变化或时延观测值而暂停辅助。若显式启用旧反馈对照，才允许校准和暂停；暂停时FULL dummy用PAD_SLOT_ID隔离写入，不能覆盖活KV。所有模式仍参与整个EP世界，因此必须做ON/OFF配对才能报告RL slowdown，不能把连续执行理解为零干扰保证。

停止条件**只有全局RL请求数为零**。达到该边界后不再执行模型step，立即取消未完成的GSM8K请求、清理它们的KV，然后交回训练。取消和设备同步存在有限开销，会由`max_stop_cleanup_s`记录，不为等待GSM8K生成完毕而延长rollout。

输出位于`/data/tailserve_results/RUN_ID`和`RUN_ID_spool`：

| 文件 | 内容与统计口径 |
|---|---|
| `RUN_ID/rollout_data/1.jsonl` | 仅DeepScaleR，返回给RL训练的答案，预期512条 |
| `RUN_ID_spool/gsm8k_manifest.jsonl` | GSM8K固定源行号、标签及token上限，推理请求本身不带标签 |
| `RUN_ID_spool/events/rank_*.jsonl` | 准入、首次真实prefill/decode、取消、全局RL结束边界及排空证明 |
| `RUN_ID_spool/gsm8k_completed.jsonl` | 独立导出的GSM8K完成答案，不用于本次RL更新 |
| `RUN_ID_spool/gsm8k_cancelled.jsonl` | 取消前缀，仅诊断，不计完整答案 |
| `RUN_ID_spool/report.json` | 两类答案数、token数、未准入数、完成原因、排空时间与RL产物计数核对 |

`finish_reason=length`表示达到显式输出上限，单独统计，不能声称都是完整解答。GSM8K准确率按`####`答案提取规则计算，不能与其他模型或其他提取规则直接比较。未完成和取消的请求不进入该准确率分母，但必须同时报告完成覆盖率，避免误导。

重新汇总而不启动NPU：

```bash
PYTHONPATH=. python tools/summarize_tailserve.py \
  --spool /data/tailserve_results/my_unique_gsm8k_trial_spool \
  --rl-output /data/tailserve_results/my_unique_gsm8k_trial \
  --save /data/tailserve_results/my_unique_gsm8k_trial_spool/report.json
```

## 默认预算

| 环境变量 | 默认 | 含义 |
|---|---:|---|
| `TAILSERVE_GLOBAL_TOKENS` | 278528 | 默认max_aux_ranks × per_rank_tokens，不额外压缩单rank预算 |
| `TAILSERVE_PER_RANK_TOKENS` | 17408 | 与RL的max_num_batched_tokens对齐，包含prefill |
| `TAILSERVE_MAX_AUX_RANKS` | 16 | 不人为排除16卡场景中的空闲rank |
| `TAILSERVE_INTERFERENCE_FEEDBACK` | 0 | 0为连续执行，1显式启用旧反馈对照 |
| `TAILSERVE_WARMUP_ROUNDS` | 8 | 仅反馈模式使用，连续模式不等待校准 |
| `TAILSERVE_SLOWDOWN_TARGET` | 0.05 | 仅反馈模式使用，不是已证明的端到端上界 |
| `TAILSERVE_VIOLATION_ROUNDS` | 3 | 仅反馈模式使用 |
| `TAILSERVE_AUX_INTERVAL_ROUNDS` | 1 | 默认每轮执行，间隔大于1须显式启用反馈模式 |
| `TAILSERVE_MAX_PROMPT_TOKENS` | 1024 | 与主任务prompt长度上限对齐 |
| `TAILSERVE_MAX_NEW_TOKENS` | 16384 | 与主任务response长度上限对齐 |
| `TAILSERVE_KV_RESERVE_BLOCKS` | 32 | 最坏长度准入后仍保留的KV blocks |

配置应在API和RL worker间一致。所有rank启动时检查配置/版本摘要一致；Token预算通过真正的vLLM scheduler执行，而不仅仅是队列优先级标签。服务网关不提供OpenAI兼容流式接口，首版是可审计的异步作业API。

## 代码位置与下一步验证

- `tailserve_runtime/policy.py`：全局准入与反馈。
- `tailserve_runtime/loop.py`：RL/辅助结束条件及输出隔离。
- `tailserve_runtime/vllm_adapter.py`：真实in-process engine桥接、KV/随机数/版本边界。
- `tailserve_runtime/queue.py` / `server.py`：持久化队列与HTTP API。
- `run_tailserve.sh`：默认host-only dry-run、显式NPU运行。
- `tests/test_tailserve.py`：CPU接口及模拟多rank测试，不能当NPU实验结果。

只修改复制仓库中的四处接入点：`vllm/entrypoints/llm.py`、`vllm/v1/core/sched/scheduler.py`、`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`、`verl/trainer/constants_ppo.py`。其他继承脚本为0.14基础依赖，不建议直接使用旧AdaFloor入口运行本原型。

下一步应先验证：16-rank smoke中的实际辅助完成、固定EP拓扑、所有RL输出不混入辅助数据、更新前取消及下一步恢复、相同checkpoint的质量对照，再测RL-only/混合的slowdown与有效辅助产出。当前不把CPU夹具的数字写成论文实验，也不修改已经完成的AdaFloor论文。
