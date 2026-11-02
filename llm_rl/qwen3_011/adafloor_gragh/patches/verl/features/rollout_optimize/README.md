# RL On-Policy 推理场景的序列级均衡调度引擎

## 1. 简介

### 1.1 背景
RLHF的Rollout阶段面临典型的“木桶效应”：由于输入Prompt所生成的响应（Response）长度存在长尾分布，少数极长的生成任务会拖慢整个DP组的进展。这使得处理短序列的节点在完成计算后不得不进入长时间的闲置等待，造成算力浪费。长尾问题优化的本质是RL训练系统的负载均衡，对此，我们针对单轮推理的同步场景，优化目标是提升进入长尾状态后的推理效率。


### 1.2 解决方案
本优化的核心目标是在On Policy场景中，针对部分rollout提前结束导致各rank间的负载不均时，对未结束的rollout进行负载均衡的策略分析和重调度，从而提升计算资源的利用率和长尾状态下的推理效率。

前置依赖：
vllm_ascend上在`torchair_graph_config`中提供了`use_cached_graph`和`graph_batch_sizes`的能力，支持提前配置多档位BatchSize的图，并随着剩余Seq减少时自动匹配最小BatchSize的图进行推理。

本方案中包含以下关键功能实现：
1. Rebalance条件检测与调度策略生成；
2. Request(SEQ)级的数据搬迁与恢复（包含对应的kvCache）；
3. Rollout后的结果还原；


### 1.3 实验结果
我们在Atlas A3集群64卡环境上进行了如下实验，发现本方案开启后单轮推理耗时从10200s左右优化到约6100s，性能收益达60%左右。

实验配置如下：
* 模型：Qwen3 235B;
* 数据集：deepscaler;
* data.train_batch_size=512;
* data.max_response_length=32768;
* actor_rollout_ref.rollout.n=16;
* TP=4; DP=32;

性能收益主要来自于单个step的TPOT性能的差距，默认场景下的TPOT会从125ms上升到200ms，而通过使能Rebalance并配合多档位编图，能在1~2K推理长度时就快速将推理档位降低，让单个step的TPOT降低到60ms的量级，在长尾场景下，性能差距被持续放大。

### 1.4 具体实现
以下的详细代码均位于`rollout_rebalance.py`文件中：
#### 全局状态感知
```python
     def get_current_state(self):
        return dict(
            rank=self.rank,
            req_ids=list(self.llm_engine.output_processor.request_states.keys()),
        )
```
```Python
    def sync_group_states(self):
        if self.world_size == 1:
            return [self.get_current_state()]
        group_states = [None for _ in range(self.world_size)]
        dist.all_gather_object(group_states, self.get_current_state(), group=self.dp_group)
        return group_states
```

#### 基于BatchSize档位预设的最大档位最小化均衡算法
```python
    def calc_balancing_tasks(self, rank_state_list):
        """
        计算最优的请求迁移任务清单，以实现负载均衡。

        最优策略定义为三层目标：
        1. 主要目标：将整个DP组所需的最大档位(max_bs)降至最低。
        2. 次要目标：在满足1的前提下，使迁移的请求数量(cost)最少。
        3. 补充目标：在满足1和2的前提下，使得各个rank间的数据搬迁流向尽可能均匀，避免多对一阻塞。
        """

        balancing_tasks = []
        rank_cnt_list = [x['running'] + x['waiting'] for x in rank_state_list]
        max_bs_before = max(self._get_bs(x) for x in rank_cnt_list)
        avg_bs = sum(rank_cnt_list) / len(rank_cnt_list)
        max_bs_next = min(max_bs_before, States.graph_batch_sizes[0], self.scheduler.max_num_running_reqs)
        for target_bs in States.graph_batch_sizes[::-1]:
            if avg_bs <= target_bs < max_bs_next:
                max_bs_next = target_bs
                break

        receivers = []
        waiting_donors = []
        other_donors = []
        for rank_state in rank_state_list:
            if rank_state['waiting']:
                waiting_donors.append(dict(
                    rank=rank_state['rank'],
                    waiting=rank_state['waiting'],
                ))
                continue
            usage = rank_state['block_usage']
            if usage > 0.8:
                continue
            delta = max_bs_next - rank_state['running']
            if delta > 0:
                if usage > 0:
                    capacity = min(delta, int(rank_state['running'] * (1 - usage) / usage))
                else:
                    capacity = delta
                receivers.append(dict(
                    rank=rank_state['rank'],
                    capacity=capacity,
                    running=rank_state['running'],
                ))
            elif delta < 0:
                other_donors.append(dict(
                    rank=rank_state['rank'],
                    surplus=-delta,
                ))

        if waiting_donors and receivers:
            total_tasks_to_distribute = sum(d['waiting'] for d in waiting_donors)
            for r in receivers:
                r['assigned_tasks'] = 0
            for _ in range(total_tasks_to_distribute):
                best_receiver = None
                min_load = float('inf')
                for r in receivers:
                    if r['assigned_tasks'] < r['capacity']:
                        current_load = r['running'] + r['assigned_tasks']
                        if current_load < min_load:
                            min_load = current_load
                            best_receiver = r
                if not best_receiver:
                    break
                best_receiver['assigned_tasks'] += 1
            current_donor_idx = 0

            for receiver in receivers:
                num_to_assign = receiver.get('assigned_tasks', 0)
                if num_to_assign == 0:
                    continue
                while num_to_assign > 0 and current_donor_idx < len(waiting_donors):
                    donor = waiting_donors[current_donor_idx]
                    num_can_move = min(num_to_assign, donor['waiting'])
                    if num_can_move > 0:
                        balancing_tasks.append(dict(
                            from_rank=donor['rank'],
                            to_rank=receiver['rank'],
                            num_to_move=num_can_move,
                            need_kv_cache=False,
                        ))
                        num_to_assign -= num_can_move
                        donor['waiting'] -= num_can_move
                        receiver['capacity'] -= num_can_move
                    if donor['waiting'] == 0:
                        current_donor_idx += 1

        waiting_donors = [x for x in waiting_donors if x['waiting']]

        if waiting_donors:
            return balancing_tasks

        if max_bs_next >= max_bs_before:
            return balancing_tasks

        while other_donors and receivers:
            donor_index = 0
            for receiver in sorted(receivers, key=lambda r: r['capacity'], reverse=True):
                donor = other_donors[donor_index]
                num_to_move = min(donor['surplus'], receiver['capacity'])
                if num_to_move > 0:
                    balancing_tasks.append(dict(
                        from_rank=donor['rank'],
                        to_rank=receiver['rank'],
                        num_to_move=num_to_move,
                        need_kv_cache=True,
                    ))
                    donor['surplus'] -= num_to_move
                    receiver['capacity'] -= num_to_move
                    donor_index = (donor_index + 1) % len(other_donors)
            other_donors = [x for x in other_donors if x['surplus']]
            receivers = [x for x in receivers if x['capacity']]
        return balancing_tasks
```

#### 序列请求的跨Rank发送与接收（含KvCache）
```python
   def all_to_all_v_tasks(self, schedule_tasks):
        objects_to_send = [[] for _ in range(dist.get_world_size())]
        send_tasks = []
        for schedule_task in schedule_tasks:
            if self.rank != schedule_task['from_rank']:
                continue
            for request in [*self.scheduler.running, *self.scheduler.waiting][-schedule_task['num_to_move']:]:
                request_task = RebalanceRequestTask(self.llm_engine).load_by_req_id(request.request_id, 
                                schedule_task['need_kv_cache'])
                send_tasks.append((request_task, schedule_task['to_rank']))
                objects_to_send[schedule_task['to_rank']].append(request_task.get_transfer_dict())
                request_task.trigger_abort()

        tensor_list = [
            torch.frombuffer(pickle.dumps(data), dtype=torch.uint8).to(self.device)
            for data in objects_to_send]
        local_sizes = torch.tensor([len(t) for t in tensor_list], dtype=torch.long).to(self.device)
        remote_sizes = torch.empty_like(local_sizes).to(self.device)
        dist.all_to_all_single(remote_sizes, local_sizes)
        input_tensor = torch.cat(tensor_list).to(self.device)
        output_tensor = torch.empty(remote_sizes.sum().item(), dtype=torch.uint8).to(self.device)
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=remote_sizes.tolist(),
            input_split_sizes=local_sizes.tolist(),
        )
        received_tensor = torch.split(output_tensor, remote_sizes.tolist())
        received_tasks = []
        for rank_data in received_tensor:
            received_tasks += pickle.loads(rank_data.to('cpu').numpy().tobytes())
        self.send_kv_caches(send_tasks)
        self.load_received_tasks(received_tasks)
```
```Python
    def send_kv_caches(self, send_tasks):
        for request_task, to_rank in send_tasks:
            if request_task.need_kv_cache:
                for cache_block_index in range(len(request_task.model_runner.kv_caches[0])):
                    block_kv_cache = torch.stack([layer[cache_block_index][request_task.request_block_table]
                                                  for layer in request_task.model_runner.kv_caches])
                    dist.send(block_kv_cache, dst=to_rank)
                rank_log_info(
                    f'[TaskSendKvCache][ToRank={to_rank}][ReqId={request_task.req_id}]'
                )

```
```Python
    def load_received_tasks(self, received_tasks):
        for request_task_dict in received_tasks:
            send_time = datetime.datetime.strptime(
                request_task_dict['send_time'],
                '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc)
            task_recv_time = datetime.datetime.now(tz=datetime.timezone.utc)
            cost = round((task_recv_time - send_time).total_seconds() * 1000, 1)
            from_rank = request_task_dict["src_rank"]
            rank_log_info(
                f'[ReceivedTask][NeedKV={request_task_dict["need_kv_cache"]}][FromRank={from_rank}]'
                f'[Cost={cost}ms][SendTime={request_task_dict["send_time"]}]',
                force=True)

            layers_kv_cache_blocks = []

            if request_task_dict['need_kv_cache']:
                for kv_cache_block_shape in request_task_dict['layers_kv_cache_shapes']:
                    block = torch.empty(kv_cache_block_shape, dtype=torch.bfloat16).cuda()
                    dist.recv(block, src=from_rank)
                    layers_kv_cache_blocks.append(block)

                cost = round(
                    (datetime.datetime.now(tz=datetime.timezone.utc) - task_recv_time).total_seconds() * 1000, 1)
                rank_log_info(
                    f'[ReceivedKvCache][FromRank={from_rank}][Cost={cost}ms]'
                    f'[KV_CACHE_SHAPE_LIST={request_task_dict["layers_kv_cache_shapes"]}]',
                    force=True)

            start_time = datetime.datetime.now(tz=datetime.timezone.utc)
            request_task = RebalanceRequestTask(self.llm_engine).load_by_transfer_info(
                request_task_dict=request_task_dict,
                layers_kv_cache_blocks=layers_kv_cache_blocks)

            request_task.trigger_load()
            cost = round(
                (datetime.datetime.now(tz=datetime.timezone.utc) - start_time).total_seconds() * 1000, 1)
            rank_log_info(
                f'[ReceivedTaskLoaded][FromRank={from_rank}][Cost={cost}ms]',
                force=True)
```
#### KvCache搬迁与恢复
```python
        # request级的kvCache采集
        self.layers_kv_cache_blocks = []
        self.layers_kv_cache_shapes = []
        if self.need_kv_cache:
            input_batch = self.model_runner.input_batch
            block_table = input_batch.block_table[0].block_table
            batch_index = input_batch.req_id_to_index[req_id]
            self.request_block_table = torch.tensor([x for x in block_table[batch_index].tolist() if x])

            for cache_block_index in range(len(self.model_runner.kv_caches[0])):
                self.layers_kv_cache_shapes.append(
                    [len(self.model_runner.kv_caches), len(self.request_block_table), 
                    *list(self.model_runner.kv_caches[0][cache_block_index].shape)[1:]]
                )
```
```Python
        # kvCache还原
        for layer_index, layer_caches in enumerate(self.model_runner.kv_caches):
            reload_indexes = list(range(len(new_block_ids)))
            for i, cache_block in enumerate(self.layers_kv_cache_blocks):
                layer_caches[i][new_block_ids] = cache_block[layer_index][reload_indexes]
```

## 2. 使用说明
### 2.1 初始化配置
在`verl/workers/megatron_workers.py`文件开头追加了以下代码，通过环境变量`ROLLOUT_REBALANCE_ENABLE=1`使能本特性功能：
```python
from patches.verl.features.rollout_optimize import init_rollout_rebalance
init_rollout_rebalance()
```

### 2.2 配置项介绍
可以直接在`config.py`中修改配置，或将以下配置写入verl启动的yaml中，在2.1节所写位置进行配置提取并传入init_rollout_rebalance方法。
```python
class RolloutRebalanceConfig:
    enable = int(os.environ.get("ROLLOUT_REBALANCE_ENABLE", "0"))  # RolloutRebalance特性总开关
    check_interval = 1000  # 间隔多少个step进行一次rebalance检查

    multi_graph = True  # 是否开启多档位编图，如果关闭，rebalance依然会按预编图的档位做均衡调度，但是不会形成明显的性能收益
    graph_batch_sizes = [64, 32, 16, 8, 4]  # 预编图的档位设置

    profile = True  # 是否打印过程中的性能数据
    profile_interval = 100   # 打印间隔步长
```
