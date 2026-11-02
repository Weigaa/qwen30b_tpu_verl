# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import pickle
import torch
import torch.distributed as dist
import vllm.distributed.parallel_state as ps
from vllm import RequestOutput, CompletionOutput
from vllm.logprobs import Logprob
from vllm.entrypoints.llm import LLM, LLMEngine
from .config import RolloutRebalanceConfig
from .utils import CheckCounter, hook, rank_log_info


class RolloutRebalanceEngine(object):
    """
    Core engine for dynamic request migration and load rebalancing in multi-NPU vLLM inference.

    Features:
    - Periodically detects load imbalance across DP ranks.
    - Computes optimal migration tasks to minimize max graph size and migration cost.
    - Supports KV Cache-aware migration (running requests) and KV Cache-free migration (waiting requests).
    - Seamless integration via method hooking, no need to modify vLLM core codes.
    """
    device = 'npu'

    def __init__(self, llm_engine=None, check_interval=200):
        self.llm_engine = llm_engine
        self.output_processor = self.llm_engine.output_processor
        self.engine_core = self.llm_engine.engine_core.engine_core
        self.scheduler = self.engine_core.scheduler
        self.model_runner = self.engine_core.model_executor.driver_worker.worker.model_runner
        self.rank = dist.get_rank()
        self.rebalance_counter = CheckCounter(check_interval)
        self.dp_group = self.llm_engine.dp_group
        self.world_size = len(ps.get_dp_group().ranks)

        # To allow child seqs to migrate independently, temporarily remove the 
        # parent-child relationship of the request here.
        # The original parent-child relationship will be restored later in self.recover.
        for req_state in self.llm_engine.output_processor.request_states.values():
            req_state.parent_req = None

    @staticmethod
    def _get_bs(size):
        for bs in States.graph_batch_sizes[::-1]:
            if size <= bs:
                return bs
        return size

    @staticmethod
    def _split_rebalance_outputs(outputs):
        """
        Separate normal outputs from migration outputs (marked with 'src_rank_')
        """
        rebalance_outputs = []
        current_outputs = []
        for request_output in outputs:
            if not request_output.request_id.startswith('src_rank'):
                current_outputs.append(request_output)
                continue
            src_rank, src_request_id = request_output.request_id.split('src_rank_', 1)[-1].split('_', 1)
            seq_output = request_output.outputs[0]
            logprobs_cache = dict(
                logprobs=[],
                cumulative_logprob=seq_output.cumulative_logprob,
            )
            for item in seq_output.logprobs:
                key = list(item.keys())[0]
                value = list(item.values())[0]
                logprobs_cache['logprobs'].append([key, value.logprob, value.rank])

            rebalance_outputs.append(dict(
                rank=int(src_rank),
                req_id=src_request_id,
                prompt_token_ids=request_output.prompt_token_ids,
                token_ids=seq_output.token_ids,
                **logprobs_cache,
            ))
        return rebalance_outputs, current_outputs

    @staticmethod
    def _build_rebalance_request_output(rebalance_output):
        '''
        Reconstruct a standard RequestOutput from migration data
        '''
        return RequestOutput(
            request_id=rebalance_output['req_id'],
            prompt=None,
            prompt_token_ids=rebalance_output['prompt_token_ids'],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=rebalance_output['req_id'].split('_', 1)[0],
                    text='',
                    token_ids=rebalance_output['token_ids'],
                    cumulative_logprob=rebalance_output['cumulative_logprob'],
                    logprobs=[{token_id: Logprob(logprob=logprob, rank=rank, decoded_token=None)
                               for token_id, logprob, rank in rebalance_output['logprobs']}],
                    finish_reason='stop'
                )
            ],
            finished=True,
        )

    def calc_balancing_tasks(self, rank_state_list):
        """
        Compute the optimal list of request migration tasks to achieve load balancing across ranks.

        The optimization follows a three-tier object:
        1. Primary objective: 
            Clear the waiting queue while minimizing the maximum graph size required by the entire DP group.
        2. Secondary objective: 
            Under the premise of satisfying the primary objective, minimize the total number of requests migrated.
        3. Tie-breaker objective: 
            When the above two objectives are met, make the data migration flows as evenly distributed as possible 
            across ranks to avoid many-to-one bottlenecks and blocking.
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

        # Prioritize Filling 'waiting'
        # The allocation strategy should aim to distribute tasks as evenly as possible among receivers after allocation.
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

    def check(self):
        """
        Periodic check: decide whether to trigger profiling or rebalance
        """
        need_profile = self.ProfileCache.enable and self.ProfileCache.counter.check()
        need_rebalance_check = self.rebalance_counter.check()
        if not (need_profile or need_rebalance_check):
            return
        start = datetime.datetime.now(tz=datetime.timezone.utc)
        group_states = self.sync_group_states()
        if need_profile:
            self.profile(group_states)
        if not need_rebalance_check:
            return
        schedule_tasks = self.calc_balancing_tasks(group_states)
        if schedule_tasks:
            rank_log_info(f'[RebalanceScheduleTasks][TaskCnt={len(schedule_tasks)}]'
                          f'[SumToMov={sum(x["num_to_move"] for x in schedule_tasks)}]')
            for schedule_task in schedule_tasks:
                rank_log_info(
                    f'[Rebalance]'
                    f'[Src={schedule_task["from_rank"]}]'
                    f'[Dst={schedule_task["to_rank"]}]'
                    f'[NumToMov={schedule_task["num_to_move"]}]')
            self.all_to_all_v_tasks(schedule_tasks)
        cost = round((datetime.datetime.now(tz=datetime.timezone.utc) - start).total_seconds() * 1000, 1)
        rank_log_info(f'[RebalanceSchedule][Cost={cost}ms]')

    def get_rebalance_outputs(self, rebalance_outputs):
        """
        Synchronize via `all_gather` and obtain this rank's portion of rebalance output from the distributed set.
        """
        ret = []
        if self.world_size <= 1:
            return ret
        full_rebalance_outputs = [[] for _ in range(self.world_size)]
        dist.all_gather_object(full_rebalance_outputs, rebalance_outputs, group=self.dp_group)
        for rank_rebalance_outputs in full_rebalance_outputs:
            for rebalance_output in rank_rebalance_outputs:
                if self.rank != rebalance_output['rank']:
                    continue
                ret.append(rebalance_output)
        return ret

    def recover(self, outputs):
        '''
        After rebalance migration, Use `all_gather` to restore outputs to their source ranks.
        '''
        rebalance_outputs, current_outputs = self._split_rebalance_outputs(outputs)
        for rebalance_output in self.get_rebalance_outputs(rebalance_outputs):
            rank_log_info(f'[RecvReqId={rebalance_output["req_id"]}]', force=True)
            current_outputs.append(self._build_rebalance_request_output(rebalance_output))

        current_outputs_map = {}

        for request_output in current_outputs:
            request_id = request_output.request_id
            if '_' in request_id:
                request_id = request_id.split('_', 1)[-1]

            parent_request = current_outputs_map.get(request_id)
            if not parent_request:
                parent_request = RequestOutput(
                    request_id=request_id,
                    prompt=None,
                    prompt_token_ids=request_output.prompt_token_ids,
                    prompt_logprobs=None,
                    outputs=[],
                    finished=True,
                )
                current_outputs_map[request_id] = parent_request
            parent_request.outputs += request_output.outputs

        return current_outputs_map.values()

    def get_current_state(self):
        """
        Get current rank load information
        """
        return dict(
            rank=self.rank,
            running=len(self.scheduler.running),
            waiting=len(self.scheduler.waiting),
            block_usage=self.scheduler.kv_cache_manager.block_pool.get_usage(),
        )

    def sync_group_states(self):
        """
        Synchronize load information of all DP ranks
        """
        if self.world_size == 1:
            return [self.get_current_state()]
        group_states = [None for _ in range(self.world_size)]
        dist.all_gather_object(group_states, self.get_current_state(), group=self.dp_group)
        return group_states

    def all_to_all_v_tasks(self, schedule_tasks):
        """
        Execute request migration according to the scheduling plan.
        """
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

    def send_kv_caches(self, send_tasks):
        """
        Send KV Cache blocks via point-to-point communication
        """
        for request_task, to_rank in send_tasks:
            if request_task.need_kv_cache:
                for cache_block_index in range(len(request_task.model_runner.kv_caches[0])):
                    block_kv_cache = torch.stack([layer[cache_block_index][request_task.request_block_table]
                                                  for layer in request_task.model_runner.kv_caches])
                    dist.send(block_kv_cache, dst=to_rank)
                rank_log_info(
                    f'[TaskSendKvCache][ToRank={to_rank}][ReqId={request_task.req_id}]'
                )


    def load_received_tasks(self, received_tasks):
        """
        Receive migrated requests and resume execution
        """
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

    class ProfileCache:
        enable = False
        last_time = None
        start_time = None
        max_bs = None
        times = 0
        counter = CheckCounter(100)

        @classmethod
        def start(cls, profile_interval=None):
            cls.enable = True
            now = datetime.datetime.now(tz=datetime.timezone.utc)
            cls.last_time = now
            cls.start_time = now
            cls.max_bs = None
            cls.times = 0
            if profile_interval:
                cls.counter.threshold = profile_interval
            cls.counter.reset()

    def profile(self, group_states):
        """
        Print load distribution and performance statistics
        """
        self.ProfileCache.times += 1
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        cost = (now - self.ProfileCache.last_time).total_seconds()
        duration = (now - self.ProfileCache.start_time).total_seconds()
        self.ProfileCache.last_time = now
        bs_list = []
        seq_cnt_list = []
        seq_cnt_info_list = []
        waiting_cnt_list = []
        for state_info in group_states:
            seq_cnt = state_info['waiting'] + state_info['running']
            bs_list.append(self._get_bs(seq_cnt))
            seq_cnt_list.append(seq_cnt)
            seq_cnt_info_list.append(f"{state_info['running']}+{state_info['waiting']}")
            waiting_cnt_list.append(state_info['waiting'])
        max_bs = max(bs_list)
        msgs = [
            f'[ProfileIndex={self.ProfileCache.times}]',
            f'[TotalSeqs={sum(seq_cnt_list)}]',
            f'[Waiting={sum(waiting_cnt_list)}]',
            f'[CurrentMaxBS={max_bs}]',
            f'[ProfileStepCost={round(cost, 1)}s]',
            f'[ProfileDuration={round(duration, 1)}s]',
        ]
        if max_bs != self.ProfileCache.max_bs:
            msgs.append(f'[MaxBSChanged: {self.ProfileCache.max_bs} -> {max_bs}]')
            self.ProfileCache.max_bs = max_bs

        rank_log_info(f'[BSMap]: {bs_list}')
        rank_log_info(f'[SeqCntMap]: {seq_cnt_list}')
        rank_log_info(f'[SeqCntDetailMap]: {seq_cnt_info_list}')
        rank_log_info(''.join(msgs))


class RebalanceRequestTask:
    req_id: str
    prompt_token_ids: list
    output_token_ids: list
    logprobs_processor_cache: dict
    layers_kv_cache_blocks: list
    layers_kv_cache_shapes: list
    src_rank: int = None
    max_tokens: int
    request_block_table = None
    need_kv_cache = False

    def __init__(self, llm_engine: LLMEngine):
        self.llm_engine = llm_engine
        self.output_processor = self.llm_engine.output_processor
        self.engine_core = self.llm_engine.engine_core.engine_core
        self.scheduler = self.engine_core.scheduler
        self.model_runner = self.engine_core.model_executor.driver_worker.worker.model_runner

    def load_by_req_id(self, req_id, need_kv_cache=False):
        """
        Collect request state and (optionally) KV Cache from current rank
        """
        self.req_id = req_id
        self.need_kv_cache = need_kv_cache
        request_state = self.output_processor.request_states[req_id]
        engine_core_request = self.scheduler.requests[req_id]

        self.max_tokens = engine_core_request.max_tokens

        self.prompt_token_ids = request_state.prompt_token_ids
        self.output_token_ids = engine_core_request.output_token_ids

        # Collect the KV Cache at the request level.
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

        self.logprobs_processor_cache = dict(
            logprobs=[],
            cumulative_logprob=request_state.logprobs_processor.cumulative_logprob,
        )

        for item in request_state.logprobs_processor.logprobs:
            key = list(item.keys())[0]
            value = list(item.values())[0]
            self.logprobs_processor_cache['logprobs'].append([key, value.logprob, value.rank])

        return self

    def get_transfer_dict(self):
        return dict(
            req_id=self.req_id,
            src_rank=States.rebalance_engine.rank,
            prompt_token_ids=self.prompt_token_ids,
            output_token_ids=self.output_token_ids,
            max_tokens=self.max_tokens,
            logprobs_processor_cache=self.logprobs_processor_cache,
            layers_kv_cache_shapes=self.layers_kv_cache_shapes,
            send_time=datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f"),
            need_kv_cache=self.need_kv_cache,
        )

    def load_by_transfer_info(self,
                              request_task_dict: dict,
                              layers_kv_cache_blocks: list):
        self.src_rank = request_task_dict['src_rank']
        self.req_id = request_task_dict['req_id']
        self.max_tokens = request_task_dict['max_tokens']
        self.prompt_token_ids = request_task_dict['prompt_token_ids']
        self.output_token_ids = request_task_dict['output_token_ids']
        self.logprobs_processor_cache = request_task_dict['logprobs_processor_cache']
        self.need_kv_cache = request_task_dict['need_kv_cache']
        self.layers_kv_cache_blocks = layers_kv_cache_blocks
        return self

    def trigger_abort(self):
        """
        Aborts the execution of the current request within this rank's worker.
        """
        request_state = self.output_processor.request_states[self.req_id]
        self.llm_engine.abort_request([self.req_id])
        self.llm_engine.engine_core.abort_requests([self.req_id])
        return request_state

    def recover_request_state(self, new_req_id):
        request_state = self.output_processor.request_states[new_req_id]
        request_state.is_prefilling = False
        request_state.logprobs_processor.cumulative_logprob = \
            self.logprobs_processor_cache['cumulative_logprob']
        request_state.detokenizer.token_ids.extend(self.output_token_ids)
        request_state.logprobs_processor.logprobs = [
            {token_id: Logprob(logprob=logprob, rank=rank, decoded_token=None)}
            for token_id, logprob, rank in self.logprobs_processor_cache['logprobs']
        ]

    def recover_scheduler_request(self, new_req_id, sampling_params):
        scheduler_request = self.scheduler.requests[new_req_id]
        scheduler_request.max_tokens = self.max_tokens
        scheduler_request.append_output_token_ids(self.output_token_ids)
        scheduler_request.sampling_params = sampling_params
        from vllm.v1.request import RequestStatus
        if self.need_kv_cache:
            scheduler_request.num_computed_tokens = scheduler_request.num_tokens - 1
            scheduler_request.status = RequestStatus.RUNNING
            self.scheduler.waiting.remove(scheduler_request)
            self.scheduler.running.append(scheduler_request)
        else:
            scheduler_request.num_computed_tokens = 0
            scheduler_request.status = RequestStatus.PREEMPTED
        return scheduler_request

    def recover_model_runner(self, scheduler_request):
        if self.need_kv_cache:
            new_blocks = self.scheduler.kv_cache_manager.allocate_slots(scheduler_request, 1)
            new_block_ids = new_blocks.get_block_ids()
        else:
            new_block_ids = []
        sampling_params = scheduler_request.sampling_params
        from vllm.sampling_params import SamplingType
        if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.model_runner.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        from vllm_ascend.worker.npu_input_batch import CachedRequestState
        from vllm_ascend.utils import vllm_version_is

        model_runner_req_state = CachedRequestState(
            req_id=scheduler_request.request_id,
            prompt_token_ids=scheduler_request.prompt_token_ids,
            mm_features=scheduler_request.mm_features,
            sampling_params=scheduler_request.sampling_params,
            pooling_params=scheduler_request.pooling_params,
            generator=generator,
            block_ids=new_block_ids,
            num_computed_tokens=scheduler_request.num_computed_tokens,
            output_token_ids=list(scheduler_request.output_token_ids),
            lora_request=scheduler_request.lora_request,
            **({
                "mm_hashes": scheduler_request.mm_hashes
            } if not (vllm_version_is("0.10.1.1")
                        or vllm_version_is("0.10.1")) else {
                "mm_hashes": None
            }),
        )
        self.model_runner.requests[scheduler_request.request_id] = model_runner_req_state

        if self.need_kv_cache:
            self.model_runner.input_batch.add_request(model_runner_req_state, )
            self.model_runner.input_batch.condense()
            self.model_runner.input_batch.refresh_metadata()
            # Restore the KV Cache
            for layer_index, layer_caches in enumerate(self.model_runner.kv_caches):
                reload_indexes = list(range(len(new_block_ids)))
                for i, cache_block in enumerate(self.layers_kv_cache_blocks):
                    layer_caches[i][new_block_ids] = cache_block[layer_index][reload_indexes]

    def trigger_load(self):
        """
        Add the task to the llm engine on the current rank's worker and resumes scheduling.
        """
        # Copy sampling parameters inherited from other requests.
        import copy
        sampling_params = copy.copy(States.sampling_params)
        new_req_id = self.req_id if self.req_id.startswith('src_rank') \
            else f'src_rank_{self.src_rank}_{self.req_id}'
        self.llm_engine.add_request(
            request_id=new_req_id,
            prompt=dict(
                prompt_token_ids=self.prompt_token_ids,
            ),
            params=sampling_params
        )
        # Restore request state in output processor.
        self.recover_request_state(new_req_id)
        # Refresh scheduler request values and internal state.
        scheduler_request = self.recover_scheduler_request(new_req_id, sampling_params)
        # Restore model runner state including KV Cache.
        self.recover_model_runner(scheduler_request)


class States:
    config = RolloutRebalanceConfig
    rebalance_engine: RolloutRebalanceEngine
    outputs_cache = []
    graph_batch_sizes = [64, 32, 16, 8, 4]
    sampling_params = None


def init_rollout_rebalance(config=States.config):
    """
    Entry function to enable the entire rebalance feature
    """
    States.config = config

    if not config.enable:
        return

    @hook.before(LLM, '__init__')
    def _(self: LLM, *args, **kwargs) -> None:
        additional_config = kwargs.get('additional_config')
        graph_batch_sizes = additional_config['torchair_graph_config'].get('graph_batch_sizes')
        if graph_batch_sizes:
            max_batch_size = graph_batch_sizes[0]

            # Filter and sort batch sizes: keep max_batch_size as primary,
            # then include smaller sizes in descending order.
            graph_batch_sizes = [max_batch_size]
            for bs in sorted(set(config.graph_batch_sizes), reverse=True):
                if bs < max_batch_size:
                    graph_batch_sizes.append(bs)
            States.graph_batch_sizes = graph_batch_sizes
            if config.multi_graph:
                additional_config['torchair_graph_config'].update(
                    graph_batch_sizes_init=False,
                    use_cached_graph=True,
                    graph_batch_sizes=graph_batch_sizes,
                )

    @hook.before(LLM, '_run_engine')
    def _(self: LLM, *args, **kwargs) -> None:
        States.rebalance_engine = RolloutRebalanceEngine(self.llm_engine,
                                                         check_interval=config.check_interval)
        States.outputs_cache = []
        if config.profile:
            States.rebalance_engine.ProfileCache.start(config.profile_interval)
        request_list = list(self.llm_engine.engine_core.engine_core.scheduler.requests.values())
        States.sampling_params = request_list[0].sampling_params

    @hook.after(LLMEngine, 'step')
    def _(self: LLMEngine, step_outputs, *args, **kwargs) -> None:
        States.rebalance_engine.check()
        if step_outputs:
            for request_output in step_outputs[:]:
                if request_output.request_id.startswith('src_rank'):
                    States.outputs_cache.append(request_output)
                    step_outputs.remove(request_output)
    
    @hook.after(LLM, '_run_engine')
    def _(self: LLM, outputs, *args, **kwargs):
        outputs += States.outputs_cache
        States.outputs_cache = []
        outputs = States.rebalance_engine.recover(outputs)
        return sorted(outputs, key=lambda x: int(x.request_id))
    
    if config.profile:
        cache = dict(
            msg='',
            step=0,
            start_time=datetime.datetime.now(tz=datetime.timezone.utc),
            is_prefill=False,
        )

        from vllm.v1.engine.core import EngineCore, SchedulerOutput
        
        @hook.before(EngineCore, 'execute_model_with_error_logging')
        def _(self, model_fn, scheduler_output: SchedulerOutput, *args, **kwargs):
            cache['msg'] = ''
            cache['start_time'] = datetime.datetime.now(tz=datetime.timezone.utc)
            cache['step'] += 1
            cache['is_prefill'] = False
            if scheduler_output.scheduled_new_reqs:
                cache['is_prefill'] = True
                cache['msg'] += f"[Prefill][Cnt={len(scheduler_output.scheduled_new_reqs)}]"
            elif scheduler_output.scheduled_cached_reqs:
                length_list = [len(x._output_token_ids) for x in self.scheduler.running] + [0]
                cache['msg'] += (f'[Decode][Cnt={len(scheduler_output.scheduled_cached_reqs.req_ids)}]'
                                 f'[Step={cache["step"]}][MaxLength={max(length_list)}][MinLength={min(length_list)}]')
        
        @hook.after(EngineCore, 'execute_model_with_error_logging')
        def _(self, output, *args, **kwargs):
            delta = (datetime.datetime.now(tz=datetime.timezone.utc) - cache['start_time']).total_seconds() * 1000
            msg = cache['msg']
            rank_log_info(f"[Cost={int(delta)}ms] {msg}", force=cache['is_prefill'])

    rank_log_info('[RolloutRebalance][Enabled]')
