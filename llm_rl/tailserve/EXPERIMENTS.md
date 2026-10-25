# DeepScaleR + GSM8K initial validation

2026-10-25，vLLM 0.14.1，固定EP16，不进行AdaFloor缩容或重排。

## 固定输入

- 主任务：`/data/deepscaler/train.parquet`，原始顺序前一个训练batch，batch32 × n16，seed0，响应上限16384，一次实际RL更新。
- 初始权重：`/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5`。只读复用，不保存新checkpoint。
- 辅助：`/data/gsm8k/train.parquet`前256行，不按长度筛选，16-rank轮转，greedy、thinking关闭、输出上限1024。manifest记录数据源SHA256、行号和参考答案，标签不进入请求或RL batch。
- FULL_DECODE_ONLY/TQ1，native sleep2/lifecycle1。没有第二份模型。

## 诊断轮，不是有效完成实验

`deepscaler_gsm8k_smoke_20261025T0740Z`：因独立目录漏复制`patches`运行依赖，在模型加载前失败。随后补齐`patches`和`r1_ascend`，原qwen3目录未修改。

`deepscaler_gsm8k_smoke_20261025T0743Z`：正常进入FULL graph rollout，16个rank各32个primary requests。真实scheduler事件证实本地已完成的rank开始执行GSM8K prefill。但反馈及负载改变导致反复取消，尚未完成prefill就放弃，未观察到decode和完整辅助答案。该诊断轮于07:50 UTC主动终止，不作为已完成RL训练或性能结果。

对应修正：保留同版本KV、暂停而不是反复取消；每32轮推进一个有token预算的辅助量子；原有5%目标按量子间隔摊销，不放宽该目标；暂停期间FULL dummy必须隔离live KV写入。全局RL完成和异常退出仍取消所有辅助请求，禁止跨权重恢复KV。

## 当前验证轮

- Run ID：`deepscaler_gsm8k_paced_20261025T0754Z`
- tmux：`tailserve_gsm8k_smoke`
- 外层日志：`/data/tailserve_deepscaler_gsm8k_paced_20261025T0754Z.log`
- RL产物：`/data/tailserve_results/deepscaler_gsm8k_paced_20261025T0754Z/rollout_data/`
- 辅助审计与最终报告：`/data/tailserve_results/deepscaler_gsm8k_paced_20261025T0754Z_spool/`

已结束，训练退出码0。DeepScaleR生成512条、3055741个response tokens，训练JSONL也是512条，计数一致，平均reward为0.9140625。日志rollout耗时980.3187秒、3157.9088 tokens/s，吞吐日志与response-token口径不同，不能混算。没有RL-only配对，不声称零干扰。

GSM8K实际准入2条，两个rank都完成分块prefill并进入decode，但**两个前缀均为退化重复文本**。完整有效答案0条，639个前缀token均在主任务结束时取消，不计完成，剩余254条未准入。这是正确性失败的原型验证，不能作为有效服务或论文性能结果。

16个rank均在global primary归零后排空，最慢清理0.008783秒。停止边界按RL结束工作，没有为等待辅助EOS延长生成。

## 短控制诊断，独立于RL性能实验

脚本`run_tailserve_continuation_probe.sh`使用本地HF基座快照，**不是公共epoch0训练权重**。在同一引擎、同一权重、同一问题、贪心前64token下比较完整prefill、chunk8、chunk8+暂停三个路径。其目的是区分分块prefill与暂停/图重放状态问题，不能替代公共epoch0的最终验证。

正式完成的修复前短对照：`/data/tailserve_results/continuation_probe_20261025T0838Z/`。不暂停的分块推理输出正常，暂停路径在16个rank上均从首token偏离。

定位：fixed-topology模式实际使用native `vllm_ascend/worker/worker.py`，此前保护仅位于`worker_v1.py`，没有在真实dummy路径生效。补齐native worker活KV写入保护后，完成修复后短对照`/data/tailserve_results/continuation_probe_20261025T0843Z/`，16个rank的暂停路径与不暂停的分块路径均逐token一致至64token。两个路径相对完整prefill参考，在同一道题上均于第38token出现分歧，不能声称三种路径全部逐token一致。

## 用户要求暂停新实验后的代码调整

按用户要求，新的默认配置改为连续辅助推理：interval=1、feedback=0、max_aux_ranks=16、global_tokens=128、per_rank_tokens=8。所有符合本地RL排空、队列和KV条件的rank每轮均可执行真实辅助计算。默认没有32轮节流、8轮校准或时延反馈暂停，旧机制仅保留为显式对照选项。

这个修改经过32项CPU测试及host-only dry-run，**没有重新执行NPU实验**。前述NPU结果全部来自旧配置，不能改写为新连续配置的实验结果。

## 连续辅助推理的已完成NPU验证

Run ID：`deepscaler_gsm8k_continuous_20261025T0922Z`。FULL/TQ1、公共epoch0、seed0、DeepScaleR第一batch、interval1/feedback0、每rank8-token调度预算、辅助输出上限1024。训练退出码0，主任务512条答案和3055741个response tokens，与前述paced实验逐答案完全一致。Rollout时间1076.743584秒，日志吞吐2875.110701 tokens/s（含primary prompt）。

辅助完成144条、34143个输出token、12913个prompt token，均正常stop。严格格式86/144，VERL原有flexible数值提取133/144。9个rank各完成其本地16题，112题没有准入，未出现取消；最后一个辅助请求比global primary结束早77.297秒，因此这轮没有实际触发在途辅助取消。16个rank均排空，最慢stop cleanup为0.004952秒。

Prefill在各rank累计1677轮。当前实现将local prefill标志同步为global prefill，导致该轮RL decode也不能使用FULL_DECODE_ONLY图。尚无同配置TailServe-OFF对照及逐轮耗时归因，不能将时间差全部归因于prefill，也不能称聚合吞吐已提升。

## 新的大预算与16K输出配置，仅CPU验证

按用户要求，辅助`max_tokens`改为16384，prompt上限1024，每rank调度预算改为与主任务`max_num_batched_tokens`一致的17408。全局预算为16 × 17408 = 278528，不再把所有rank限制为合计128 tokens。调度预算上限和单请求输出上限是不同参数，均已对齐，实际调度继续受引擎及KV容量约束。launcher会在排队前从主任务环境参数推导辅助默认值。

保留continuous、固定EP16、独立输出、greedy/thinking关闭、每rank一个辅助在途请求及global RL完成即停止的规则。这个修改通过36项CPU测试、shell语法和默认/自定义长度的dry-run，**没有自动启动新的NPU实验**。历史NPU结果仍属于8-token/1024-output配置，不能复用为新配置的性能证据。
