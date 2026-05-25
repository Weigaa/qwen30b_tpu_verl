# PanguMoE Sidecar Execution Notes

本文记录当前 `PanguProMoEForCausalLM` sidecar 在 `tp4dp2` 和
`tp8dp1` 两种配置下能跑通时的自上而下执行逻辑。这里的 `dp2`
指 sidecar runner 启动两个独立 replica，每个 replica 内部是 TP4；
不是训练侧 `VLLM_DP_SIZE=16`，也不是 vLLM 内部跨 replica 的数据并行。

## 入口脚本

主入口是：

`internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager_sidepangu.sh`

它同时启动主训练和 sidecar watcher。

关键环境变量：

- 主训练 Qwen3 rollout 使用 `VLLM_USE_V1=1`、`VLLM_ENABLE_GRAPH_MODE=0`、
  `VLLM_ENABLE_EXPERT_PARALLEL=1`。
- 主训练弹性模式默认 `VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1`，最小计算组
  `VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=8`。也就是说主训练 shrink 到
  8 个 active ranks 后给 sidecar 留出 8 张卡。
- sidecar 默认模型为 `/data/pangu-pro-moe-model`，数据为 `/data/gsm8k`。
- 当前 tp8dp1 默认：
  - `VERL_SIDECAR_TENSOR_PARALLEL_SIZE=8`
  - `VERL_SIDECAR_REPLICA_COUNT=1`
  - `VERL_SIDECAR_ENABLE_EXPERT_PARALLEL=1`
  - `VERL_SIDECAR_MAX_NUM_SEQS=153`
  - `VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=65536`
  - `VERL_SIDECAR_MAX_MODEL_LEN=6144`
  - `VERL_SIDECAR_MAX_TOKENS=4096`
  - `VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA=306`
  - `VERL_SIDECAR_GENERATE_CHUNK_SIZE=306`

要跑 tp4dp2 时，通过环境变量覆盖：

```bash
VERL_SIDECAR_TENSOR_PARALLEL_SIZE=4 \
VERL_SIDECAR_REPLICA_COUNT=2 \
internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager_sidepangu.sh
```

`sidepangu.sh` 会创建：

- 主训练日志：`record_72b_tp${tp}dp${replica}_${time}_elastic.txt`
- sidecar 目录：`sidecar_runs/record_72b_tp${tp}dp${replica}_${time}/`
- watcher 日志：`monitor.log`
- sidecar 推理日志：`infer.log`
- lease/生命周期日志：`lease.log`
- 输出：`outputs.jsonl`
- 停止信号：`outputs.jsonl.stop_requested`

## Shrink Watcher

watcher 文件：

`internal/watch_elastic_shrink_and_run_sidecar.sh`

执行逻辑：

1. tail 主训练日志 `VERL_SIDECAR_TRAIN_LOG`。
2. 等待出现 `Elastic parallel shrink done`。
3. 从日志行里解析 `active_ranks=[...]`。
4. 如果 active rank 数等于 `VERL_SIDECAR_EXPECTED_ACTIVE_RANKS`，默认是 8，
   则认为 shrink window 可用。
5. 若没有手动指定 `VERL_SIDECAR_NPU_DEVICES`，则用
   `0..WORLD_SIZE-1 - active_ranks` 推导 inactive devices。
6. 用 `setsid internal/run_elastic_sidecar_infer.sh &` 启动 sidecar。
7. 当主训练日志出现 restore 或 `rollout_output_time_s` 时，写
   `VERL_SIDECAR_STOP_FILE`，给 sidecar 软停时间。
8. 如果超过 `VERL_SIDECAR_GRACEFUL_KILL_SECONDS` 仍未退出，再 kill 进程组。

当前主训练 restore 前也有一层额外保护：

`verl/trainer/ppo/ray_trainer.py`

`_restore_rollout_elastic_parallel_groups_if_needed()` 会先调用
`_request_sidecar_release_before_restore()`：

1. 主动写 sidecar stop file，避免只依赖 watcher tail 日志。
2. 从 `VERL_SIDECAR_LEASE_LOG` 读取 `sidecar_pid`。
3. 等待 `VERL_SIDECAR_RESTORE_WAIT_SECONDS`，默认 12 秒。
4. sidecar 退出后再 sleep `VERL_SIDECAR_RESTORE_EXTRA_GRACE_SECONDS`，默认 8 秒，
   给 NPU runtime/allocator 释放资源，降低 restore OOM 风险。

## Sidecar Runner

runner 文件：

`internal/run_elastic_sidecar_infer.sh`

它负责把 inactive devices 切成 replica group，然后启动临时 Python 推理脚本。

### 并行规划

输入：

- `VERL_SIDECAR_NPU_DEVICES`
- `VERL_SIDECAR_PARALLEL_MODE`
- `VERL_SIDECAR_TENSOR_PARALLEL_SIZE`
- `VERL_SIDECAR_REPLICA_COUNT`
- `VERL_SIDECAR_DATA_PARALLEL_SIZE`

规划结果：

- `VERL_SIDECAR_DEVICE_GROUPS`
- `VERL_SIDECAR_UNUSED_DEVICES`
- `VERL_SIDECAR_TENSOR_PARALLEL_SIZE`
- `VERL_SIDECAR_DATA_PARALLEL_SIZE`
- `VERL_SIDECAR_REPLICA_COUNT`

tp8dp1 示例：

```text
released devices: 2,3,4,6,8,10,13,15
tp_size=8
replica_count=1
device_groups=2,3,4,6,8,10,13,15
```

tp4dp2 示例：

```text
released devices: 2,3,4,6,8,10,13,15
tp_size=4
replica_count=2
device_groups=2,3,4,6;8,10,13,15
```

tp4dp2 时 runner 会起两个 shard 子进程，每个子进程：

- 设置自己的 `ASCEND_RT_VISIBLE_DEVICES`
- 设置独立 `MASTER_PORT`
- 设置独立 `HCCL_IF_BASE_PORT`
- 设置 `VERL_SIDECAR_SHARD_INDEX`
- 输出到 `outputs.jsonl.shard0` / `outputs.jsonl.shard1`

全部 shard 结束后，runner 合并 shard 输出到总 `outputs.jsonl`。

tp8dp1 时只有一个 shard，直接写 `outputs.jsonl`。

### EP 判定

runner 会读取模型 `config.json`，或使用
`VERL_SIDECAR_MODEL_IS_MOE=1` 覆盖，得到：

- `VERL_SIDECAR_MODEL_IS_MOE=1`
- `VERL_SIDECAR_ENABLE_EXPERT_PARALLEL_EFFECTIVE=1`

随后导出：

```bash
export VLLM_ENABLE_EXPERT_PARALLEL="${VERL_SIDECAR_ENABLE_EXPERT_PARALLEL_EFFECTIVE}"
```

这保证 PanguMoE sidecar 用 EP 初始化。

### vLLM 启动参数

runner 生成的 Python 脚本中核心调用是：

```python
from vllm import LLM, ModelRegistry, SamplingParams

llm = LLM(
    model=VERL_SIDECAR_MODEL_PATH,
    tensor_parallel_size=VERL_SIDECAR_TENSOR_PARALLEL_SIZE,
    enable_expert_parallel=VERL_SIDECAR_ENABLE_EXPERT_PARALLEL_EFFECTIVE,
    gpu_memory_utilization=VERL_SIDECAR_GPU_MEMORY_UTILIZATION,
    max_num_seqs=VERL_SIDECAR_MAX_NUM_SEQS,
    max_num_batched_tokens=VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS,
    max_model_len=VERL_SIDECAR_MAX_MODEL_LEN,
    trust_remote_code=True,
    enforce_eager=True,
)
```

若 `VERL_SIDECAR_DATA_PARALLEL_SIZE > 1`，还会向 `LLM` 传入
`data_parallel_size` 和 `data_parallel_backend`。当前 `tp4dp2` 是两个独立
replica shard，不依赖这个参数。

### 模型注册

runner 内嵌 Python 会执行：

```python
ModelRegistry.register_model(
    "PanguProMoEForCausalLM",
    "vllm_ascend.torchair.models.torchair_pangu_moe:PanguProMoEForCausalLM",
)
```

也就是说 sidecar PanguMoE 模型类来自 vLLM-Ascend 仓库目录：

`vllm_ascend/torchair/models/torchair_pangu_moe.py`

另外，vLLM-Ascend 的 torchair 工具里也注册了同一映射：

`vllm_ascend/torchair/utils.py`

## vLLM / vLLM-Ascend 执行路径

sidecar 启动后进入 vLLM V1 引擎：

```text
internal/run_elastic_sidecar_infer.sh
  -> temporary /tmp/elastic_sidecar_infer.*.py
  -> vllm.LLM(...)
  -> vLLM V1 engine / worker
  -> vllm_ascend worker/model runner
  -> PanguProMoEForCausalLM
```

关键文件：

- `vllm/entrypoints/llm.py`
- `vllm/v1/...`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/ascend_forward_context.py`
- `vllm_ascend/torchair/models/torchair_pangu_moe.py`
- `vllm_ascend/ops/fused_moe.py`
- `vllm_ascend/ops/moe/moe_comm_method.py`
- `vllm_ascend/ops/moe/token_dispatcher.py`

## PanguMoE 模型路径

模型实现：

`vllm_ascend/torchair/models/torchair_pangu_moe.py`

关键类：

- `PanguProMoEForCausalLM`
- `PanguProMoEModel`
- `PanguProMoEDecoderLayer`
- `PanguProMoESparseMoeBlock`
- `PanguProMoEMLP`

MoE block 中：

```python
self.experts = FusedMoE(...)
self.gate = ReplicatedLinear(...)
```

forward 时：

1. shared expert 先算 `shared_output`。
2. gate 算 `router_logits`。
3. 强制：

```python
get_forward_context().moe_comm_method_name = "allgathercommimpl"
```

4. 调用 `self.experts.forward_impl(...)` 进入 vLLM / vLLM-Ascend FusedMoE。
5. 如果有 shared expert，把 `final_hidden_states + shared_output`。

## MoE 通信选择

通信类型选择在：

`vllm_ascend/worker/model_runner_v1.py`

`_select_moe_comm_method()` 一般会按 EP size、prefill/decode、MC2 可用性、
SoC 版本选择 `ALLGATHER`、`MC2` 或 `ALLTOALL`。

但 PanguMoE 这里有硬约束：

```python
if model_type == "PanguProMoE":
    moe_comm_type = MoECommType.ALLGATHER
```

所以 Pangu sidecar 当前不会走 MC2。日志里的 fingerprint 也应该看到：

```text
moe_comm_type=MoECommType.ALLGATHER
```

forward context 设置在：

`vllm_ascend/ascend_forward_context.py`

它会把 `moe_comm_type` 放进 `forward_context`，并调用：

```python
from vllm_ascend.ops.moe.moe_comm_method import get_moe_comm_method
forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)
```

MoE 通信实现注册在：

`vllm_ascend/ops/moe/moe_comm_method.py`

```python
_MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
_MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)
_MoECommMethods[MoECommType.MC2] = MC2CommImpl(moe_config)
_MoECommMethods[MoECommType.NAIVE_MULTICAST] = NaiveMulticastCommImpl(moe_config)
```

PanguMoE 在 `AllGatherCommImpl` 下还有特殊 dispatcher：

```python
if self.model_type == "PanguProMoE":
    return TokenDispatcherWithMoge(...)
else:
    return TokenDispatcherWithAllGather(...)
```

因此 PanguMoE 当前可运行路径可以概括为：

```text
PanguProMoESparseMoeBlock
  -> FusedMoE.forward_impl
  -> forward_context.moe_comm_type = ALLGATHER
  -> AllGatherCommImpl
  -> TokenDispatcherWithMoge
  -> fused/grouped MoE compute
```

## 输出与软停

sidecar Python 使用 GSM8K prompt 文件或 parquet：

- `/data/gsm8k/train.parquet`
- `/data/gsm8k/test.parquet`
- 或对应 `_short.parquet`

输出格式为 JSONL，每行包含：

- `prompt_id`
- `prompt_source`
- `prompt`
- `resume_prompt`
- `outputs`
- `outputs[*].text`
- `outputs[*].finish_reason`
- `outputs[*].token_ids_len`

软停流程：

1. 主训练 restore 前或 watcher 检测到 restore/rollout 完成。
2. 写 `outputs.jsonl.stop_requested`。
3. sidecar 推理循环 `_stop_requested()` 检测到 stop file。
4. 如果当前有未完成请求，写：
   - `state/.../resume.shard${i}.jsonl`
   - `state/.../partials.shard${i}.jsonl`
5. 打印 `sidecar_soft_stop_checkpointed`。
6. abort active request，shutdown vLLM engine，释放缓存。

这使得训练恢复资源时 sidecar 能尽量温和退出，未完成样本下轮从
`resume.shard*.jsonl` 继续。

## tp4dp2 与 tp8dp1 的主要差异

### tp4dp2

```text
tp_size=4
replica_count=2
device_groups=两组，每组 4 张卡
num_shards=2
outputs.jsonl.shard0 / outputs.jsonl.shard1 -> merge 到 outputs.jsonl
```

特点：

- 两个 replica 并发处理不同 prompt shard。
- 每个 replica 的 KV cache 容量比 TP8 小。
- 模型权重/runtime 在两个 replica 间各自占一份。
- 启动/加载有两个 shard 子进程。

### tp8dp1

```text
tp_size=8
replica_count=1
device_group=一组 8 张卡
num_shards=1
直接写 outputs.jsonl
```

特点：

- 单 replica，模型权重只加载一份 TP8 分片。
- TP/EP 更大，每卡 local expert 数更少，KV cache 可用空间显著增加。
- 并发上限可以设置得更高，例如当前 `max_num_seqs=153`。
- restore 时内存交接更紧，所以需要主训练 restore 前等待 sidecar 退出并给资源释放缓冲。

## 当前注意事项

- PanguMoE sidecar 当前是 eager 模式：`VERL_SIDECAR_ENFORCE_EAGER=1`。
- PanguMoE 当前强制 ALLGATHER，不走 MC2；这不是主训练 Qwen3 的 MC2 路径。
- `tp4dp2` 中的 `dp2` 是 runner 的两个 replica shard，不是同一个 vLLM engine
  内部 DP2。
- `tp8dp1` KV cache 容量大，但 restore 资源释放更敏感；保留
  `VERL_SIDECAR_RESTORE_WAIT_SECONDS` 和
  `VERL_SIDECAR_RESTORE_EXTRA_GRACE_SECONDS` 比较稳。
- 如果日志里看到 `sidecar_soft_stop_checkpointed`，一般表示 sidecar 是正常软停，
  不是推理错误。
