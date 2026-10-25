"""Narrow vLLM 0.14 in-process / external-launcher bridge. No model creation."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random

from .loop import run
from .policy import Config, Observation
from .queue import Queue


def enabled():
    value = os.getenv('TAILSERVE_ENABLE', '0')
    if value not in ('0', '1'):
        raise ValueError('TAILSERVE_ENABLE must be 0 or 1')
    return value == '1'


@contextmanager
def generation_scope(llm, metadata, validation=False):
    if not enabled() or validation:
        yield
        return
    engine = llm.llm_engine
    if getattr(engine, '_tailserve_policy_version', None) is not None:
        raise RuntimeError('nested/concurrent TailServe generation is unsupported')
    run_id = os.environ.get('TAILSERVE_RUN_ID')
    step = metadata.get('global_steps')
    if not run_id or step is None or int(step) < 0:
        raise RuntimeError('TailServe requires RUN_ID and actual global_steps metadata')
    generation = getattr(engine, '_tailserve_generation_counter', 0) + 1
    engine._tailserve_generation_counter = generation
    engine._tailserve_policy_version = f'{run_id}/step-{int(step)}/generation-{generation}'
    try:
        yield
    finally:
        engine._tailserve_policy_version = None


class Collective:
    def __init__(self, group):
        import torch
        self.torch = torch
        self.group = group
        self.rank = torch.distributed.get_rank(group=group)
        self.size = torch.distributed.get_world_size(group=group)

    def objects(self, value):
        result = [None] * self.size
        self.torch.distributed.all_gather_object(result, value, group=self.group)
        return result

    def consensus_error(self, error):
        flag = self.torch.tensor([int(error is not None)], dtype=self.torch.int64)
        self.torch.distributed.all_reduce(flag, op=self.torch.distributed.ReduceOp.MAX, group=self.group)
        if flag.item():
            raise RuntimeError(f'TailServe coordinated failure: {self.objects(error)}')

    def gather(self, observation):
        # Numeric CPU collectives avoid pickling on every token round.
        torch = self.torch
        value = torch.tensor([observation.primary, observation.auxiliary,
                              observation.queued, observation.elapsed], dtype=torch.float64)
        result = [torch.empty_like(value) for _ in range(self.size)]
        torch.distributed.all_gather(result, value, group=self.group)
        return [Observation(int(v[0]), int(v[1]), int(v[2]), float(v[3])) for v in result]


class Adapter:
    def __init__(self, llm, collective):
        self.llm, self.engine, self.collective = llm, llm.llm_engine, collective
        self.rank, self.world_size = collective.rank, collective.size
        self.core = getattr(self.engine.engine_core, 'engine_core', None)
        if self.core is None or not self.engine.external_launcher_dp:
            raise RuntimeError('TailServe v1 requires in-process external_launcher DP, not async/server MP mode')
        config = self.engine.vllm_config
        if config.scheduler_config.async_scheduling or self.engine.backend_dp_dummy_flow:
            raise RuntimeError('TailServe v1 requires synchronous scheduling, for both eager and FULL')
        if getattr(self.core, 'batch_queue', None) is not None:
            raise RuntimeError('TailServe v1 requires no outstanding asynchronous batch queue')
        if self.engine.elastic_ep_no_dummy or getattr(self.engine, 'elastic_ep_scaled_once', False):
            raise RuntimeError('TailServe must keep the original EP topology, disable elastic shrink')
        if getattr(config, 'speculative_config', None) is not None:
            raise RuntimeError('speculative decoding is not supported by TailServe v1')
        for key, value in [('VLLM_ROLLOUT_NATIVE_SLEEP_MODE','1'), ('VLLM_ROLLOUT_SLEEP_LEVEL','2'),
                           ('VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE','1')]:
            if os.environ.get(key) != value:
                raise RuntimeError(f'TailServe v1 requires {key}={value}')
        if os.environ.get('VLLM_ENABLE_EPLB', '0') != '0':
            raise RuntimeError('dynamic expert balancing must remain disabled')
        if config.parallel_config.tensor_parallel_size != 1 or config.parallel_config.pipeline_parallel_size != 1:
            raise RuntimeError('TailServe v1 validates TP1/PP1 with a fixed DP/EP world only')
        if not config.parallel_config.enable_expert_parallel or not config.model_config.is_moe:
            raise RuntimeError('TailServe v1 targets the existing MoE expert-parallel world')
        self.scheduler = self.core.scheduler
        if not config.scheduler_config.enable_chunked_prefill:
            raise RuntimeError('bounded auxiliary prefills require chunked_prefill=True')
        if len(self.scheduler.kv_cache_config.kv_cache_groups) != 1:
            raise RuntimeError('TailServe v1 requires one uniform KV cache group')
        states = self.engine.output_processor.request_states
        if any(s.lora_request is not None for s in states.values()):
            raise RuntimeError('TailServe v1 does not share adapters/LoRA')
        self.primary_ids = set(states)
        self.primary_external = {s.parent_req.external_req_id if s.parent_req else s.external_req_id
                                 for s in states.values()}
        self.outputs = {}
        self.schedule_stats = {}
        self.progress_tokens = int(os.getenv('TAILSERVE_PROGRESS_TOKENS', '64'))
        if self.progress_tokens < 0:
            raise ValueError('TAILSERVE_PROGRESS_TOKENS must be nonnegative')
        self.progress_emitted = {}
        self.rng = None
        self.torch = collective.torch

    def check_collective_contract(self, config, version):
        contract = dict(config=asdict(config), version=version, world_size=self.world_size,
                        spool=str(Path(os.environ['TAILSERVE_SPOOL']).resolve()))
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        if len(set(self.collective.objects(digest))) != 1:
            raise RuntimeError('TailServe rank configuration or policy-version mismatch')

    def consensus_error(self, error):
        self.collective.consensus_error(error)

    def gather(self, observation):
        return self.collective.gather(observation)

    def primary_count(self):
        return len(self.primary_ids.intersection(self.engine.output_processor.request_states))

    def validate_candidate(self, job):
        model = self.engine.model_config
        vocab = int(model.get_vocab_size())
        if any(t >= vocab for t in job['prompt_token_ids']):
            return 'token ID outside model vocabulary'
        if len(job['prompt_token_ids']) + job['max_tokens'] > model.max_model_len:
            return 'request exceeds current model context length'
        return None

    def can_admit(self, job, config):
        group = self.scheduler.kv_cache_config.kv_cache_groups[0]
        needed = math.ceil((len(job['prompt_token_ids']) + job['max_tokens']) / group.kv_cache_spec.block_size)
        return self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() >= needed + config.kv_reserve_blocks

    def save_primary_rng(self):
        if self.rng is not None:
            return
        import numpy as np
        torch = self.torch
        npu = getattr(torch, 'npu', None)
        self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                    npu.get_rng_state() if npu is not None else None)

    def restore_primary_rng(self):
        if self.rng is None:
            return
        import numpy as np
        py, numpy, cpu, npu = self.rng
        random.setstate(py)
        np.random.set_state(numpy)
        self.torch.set_rng_state(cpu)
        if npu is not None:
            self.torch.npu.set_rng_state(npu)
        self.rng = None

    def admit(self, job, version):
        from vllm.sampling_params import SamplingParams, RequestOutputKind
        if self.primary_count():
            raise RuntimeError('auxiliary admission must not displace local primary work')
        self.save_primary_rng()
        request_id = f'tailserve.{self.rank}.{job["id"]}'
        seed = int.from_bytes(hashlib.sha256((version + '/' + job['id']).encode()).digest()[:4], 'big')
        params = SamplingParams(n=1, temperature=0, max_tokens=job['max_tokens'],
                                seed=seed, output_kind=RequestOutputKind.CUMULATIVE)
        # Do not advance LLM.request_counter: later RL request IDs stay unchanged.
        self.engine.add_request(request_id, {'prompt_token_ids': job['prompt_token_ids']},
                                params, priority=1_000_000)
        self.schedule_stats[request_id] = dict(prefill_scheduled_tokens=0, decode_scheduled_tokens=0,
                                              prefill_rounds=0, decode_rounds=0)
        return request_id

    def observe_schedule(self, scheduler_output):
        # Called before scheduler._update_after_schedule advances computed tokens.
        if self.primary_count():
            raise RuntimeError('auxiliary scheduling overlaps local RL requests')
        for internal_id, count in scheduler_output.num_scheduled_tokens.items():
            state = self.engine.output_processor.request_states[internal_id]
            request_id = state.external_req_id
            if request_id not in self.schedule_stats:
                raise RuntimeError('unclassified request in auxiliary scheduling')
            request = self.scheduler.requests[internal_id]
            prompt_left = max(0, request.num_prompt_tokens - request.num_computed_tokens)
            prefill = min(count, prompt_left)
            stats = self.schedule_stats[request_id]
            for phase, tokens in [('prefill', prefill), ('decode', count-prefill)]:
                if tokens:
                    first = stats[phase+'_rounds'] == 0
                    stats[phase+'_scheduled_tokens'] += tokens
                    stats[phase+'_rounds'] += 1
                    if first:
                        self.audit_queue.emit('auxiliary_phase', request_id=request_id,
                            policy_version=self.engine._tailserve_policy_version,
                            phase=phase, scheduled_tokens=tokens, local_primary_requests=0)

    def set_auxiliary_token_budget(self, budget):
        self.scheduler._tailserve_token_budget = budget
        self.scheduler._tailserve_schedule_observer = self.observe_schedule if budget else None

    def step(self):
        # All ranks execute this check even if a local engine is empty; normal
        # EP dummy participation still serves the other ranks' real tokens.
        if not self.engine.has_unfinished_requests():
            raise RuntimeError('global primary is live but vLLM world stopped')
        if getattr(self.scheduler, '_tailserve_token_budget', None) == 0:
            if self.primary_count():
                raise RuntimeError('must not pause RL work')
            # Retain scheduler/request/KV state, but do not call schedule() for
            # this local auxiliary request. All ranks still participate in EP.
            self.engine.should_execute_dummy_batch = True
        return self.engine.step()

    def is_primary(self, output):
        return output.request_id in self.primary_external

    def note_auxiliary_output(self, output):
        self.outputs[output.request_id] = output
        count = len(output.outputs[0].token_ids)
        if self.progress_tokens and count >= self.progress_emitted.get(output.request_id, 0) + self.progress_tokens:
            self.progress_emitted[output.request_id] = count
            self.audit_queue.emit('auxiliary_progress', request_id=output.request_id,
                policy_version=self.engine._tailserve_policy_version,
                generated_tokens=count, text_tail=output.outputs[0].text[-512:],
                token_ids_tail=list(output.outputs[0].token_ids[-self.progress_tokens:]))

    def serialize_output(self, output):
        if output.finished:
            self.outputs.pop(output.request_id, None)
        answer = output.outputs[0]
        return dict(text=answer.text, token_ids=list(answer.token_ids),
                    finish_reason=answer.finish_reason,
                    **self.schedule_stats.get(output.request_id, {}))

    def cancel_auxiliary(self, active, queue, version, reason):
        if not active:
            return
        self.engine.abort_request(list(active))
        for request_id, job in active.items():
            output = self.outputs.pop(request_id, None)
            fields = self.serialize_output(output) if output else dict(self.schedule_stats.get(request_id, {}))
            fields.pop('finish_reason', None)
            queue.emit('cancelled', job_id=job['id'], policy_version=version, reason=reason,
                       counted_as_completion=False, **fields)

    def verify_drained_collectively(self):
        npu = getattr(self.torch, 'npu', None)
        if npu is not None:
            npu.synchronize()
        error = None
        if self.engine.output_processor.request_states or self.scheduler.get_num_unfinished_requests():
            error = 'front-end or backend request survived the primary boundary'
        self.collective.consensus_error(error)


def run_vllm(llm):
    engine = llm.llm_engine
    group = getattr(engine, 'dp_group', None)
    if group is None:
        raise RuntimeError('TailServe requires the fixed DP CPU group')
    collective = Collective(group)
    error = None
    try:
        version = getattr(engine, '_tailserve_policy_version', None)
        if not version:
            raise RuntimeError('use the VERL generation scope or explicitly bind a policy version')
        config = Config.from_env()
        adapter = Adapter(llm, collective)
        spool = os.environ['TAILSERVE_SPOOL']
        queue = getattr(engine, '_tailserve_queue', None)
        if queue is None:
            queue = Queue(spool, adapter.rank, adapter.world_size, config)
            engine._tailserve_queue = queue
        adapter.audit_queue = queue
    except Exception as exc:
        error = str(exc)
    collective.consensus_error(error)
    return run(adapter, queue, config, version)
