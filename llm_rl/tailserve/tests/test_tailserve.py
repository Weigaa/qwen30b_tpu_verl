"""CPU-only policy, real queue/API, and multi-rank loop contracts. Not NPU evidence."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import subprocess
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

from tailserve_runtime.loop import run
from tailserve_runtime.policy import Config, Observation as O, Policy
from tailserve_runtime.queue import Queue, validate_job
from tailserve_runtime.server import Store, make_handler
from tailserve_runtime.vllm_adapter import generation_scope, Adapter
from tools.prepare_gsm8k_queue import prepare
from tools.summarize_tailserve import summarize

ROOT = Path(__file__).resolve().parents[1]


class PolicyTests(unittest.TestCase):
    def test_zero_primary_stops_even_with_infinite_auxiliary(self):
        d = Policy(Config()).decide([O(0, 10, 1), O(0, 20, 1)])
        self.assertTrue(d.stop)

    def test_primary_rank_never_gets_auxiliary(self):
        p = Policy(Config(warmup_rounds=1, aux_interval_rounds=1))
        d = p.decide([O(0, 0, 1, 1), O(5, 0, 1, 1), O(0, 0, 1, 1)])
        self.assertEqual(d.ranks, (0, 2))
        self.assertLessEqual(len(d.ranks) * d.tokens_per_rank, p.config.global_tokens)

    def test_rank_quota_and_token_quota(self):
        p = Policy(Config(warmup_rounds=1, global_tokens=3, max_aux_ranks=2, aux_interval_rounds=1))
        d = p.decide([O(0, 0, 1, 1)] * 15 + [O(1, 0, 0, 1)])
        self.assertEqual(len(d.ranks), 2)
        self.assertEqual(d.tokens_per_rank, 1)

    def test_calibration_and_load_change(self):
        p = Policy(Config(warmup_rounds=2, aux_interval_rounds=1, interference_feedback=1))
        self.assertEqual(p.decide([O(0, 0, 1, 1), O(3, 0, 0, 1)]).ranks, ())
        self.assertTrue(p.decide([O(0, 0, 1, 1), O(3, 0, 0, 1)]).ranks)
        self.assertEqual(p.decide([O(0, 1, 1, 1), O(2, 0, 0, 1)]).ranks, ())

    def test_slowdown_backoff(self):
        p = Policy(Config(warmup_rounds=1, violation_rounds=2, aux_interval_rounds=1, interference_feedback=1))
        p.decide([O(0, 0, 1, 1), O(1, 0, 0, 1)])
        p.decide([O(0, 1, 1, 2), O(1, 0, 0, 2)])
        d = p.decide([O(0, 1, 1, 2), O(1, 0, 0, 2)])
        self.assertEqual(d.reason, 'feedback_backoff')
        self.assertEqual(d.ranks, ())

    def test_bad_config_fails(self):
        for values in [dict(global_tokens=0), dict(slowdown_target=float('nan')),
                       dict(slowdown_target=0), dict(max_aux_ranks=-1),
                       dict(interference_feedback=2), dict(aux_interval_rounds=32)]:
            with self.assertRaises(ValueError):
                Config(**values)

    def test_default_all_idle_ranks_run_on_every_round(self):
        p=Policy(Config())
        for _ in range(40):
            d=p.decide([O(0,0,1)]*15+[O(32,0,1)])
            self.assertEqual(d.ranks,tuple(range(15)))
            self.assertEqual(d.tokens_per_rank,17408)
            self.assertFalse(d.stop)

    def test_continuous_has_no_recalibration_or_latency_backoff(self):
        p=Policy(Config())
        for remaining in range(32,0,-1):
            d=p.decide([O(0,1,1,1000),O(remaining,0,1,1000)])
            self.assertEqual(d.ranks,(0,))
            self.assertEqual(d.reason,'continuous_auxiliary')
            self.assertGreater(d.tokens_per_rank,0)

    def test_primary_stop_overrides_continuous_inference(self):
        p=Policy(Config())
        self.assertTrue(p.decide([O(0,1,1)]*16).stop)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name, 2)
        self.queue = Queue(self.tmp.name, 0, 2, self.store.config)

    def job(self, **kwargs):
        return dict(id='job', rank=0, prompt_token_ids=[1, 2], **kwargs)

    def test_durable_roundtrip(self):
        self.store.submit(self.job())
        self.queue.poll()
        job = self.queue.take('policy1')
        self.assertEqual(job['id'], 'job')
        self.queue.emit('completed', job_id='job', policy_version='policy1', text='answer')
        self.assertEqual(self.store.status('job')['text'], 'answer')
        self.assertEqual(self.store.status('job')['policy_version'], 'policy1')

    def test_version_pinning(self):
        self.store.submit(self.job(policy_version='old-policy'))
        self.queue.poll()
        self.assertIsNone(self.queue.take('new-policy'))
        self.assertEqual(self.store.status('job')['status'], 'rejected_version')

    def test_partial_queue_line_not_consumed(self):
        self.queue.path.parent.mkdir(parents=True)
        payload=json.dumps(validate_job(self.job(), self.store.config, 2)).encode()
        self.queue.path.write_bytes(payload[:5])
        self.queue.poll()
        self.assertEqual(self.queue.offset, 0)
        self.queue.path.write_bytes(payload + b'\n')
        self.queue.poll()
        self.assertEqual(self.queue.take('v')['id'], 'job')

    def test_restart_cancels_orphan_without_reusing_kv(self):
        self.store.submit(self.job())
        self.queue.poll();self.queue.take('v')
        self.queue.emit('admitted', job_id='job', policy_version='v')
        new = Queue(self.tmp.name, 0, 2, self.store.config)
        new.poll()
        self.assertIsNone(new.take('new'))
        self.assertEqual(self.store.status('job')['reason'], 'worker_restart_no_kv_reuse')

    def test_duplicate_and_invalid_input(self):
        self.store.submit(self.job())
        with self.assertRaises(ValueError):self.store.submit(self.job())
        for job in [dict(self.job(), prompt_token_ids=[True]), dict(self.job(), rank=3),
                    dict(self.job(), max_tokens=100000), dict(self.job(), id='../bad')]:
            with self.assertRaises(ValueError):validate_job(job, self.store.config, 2)

    def test_http_post_get_auth_and_no_model(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.store, 'secret'))
        thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join()))
        url = f'http://127.0.0.1:{server.server_port}'
        with self.assertRaises(HTTPError):urlopen(url + '/health')
        req = Request(url + '/v1/tailserve/requests', data=json.dumps(self.job()).encode(),
                      headers={'Authorization':'Bearer secret','Content-Type':'application/json'})
        with urlopen(req) as response:self.assertEqual(response.status, 202)
        with urlopen(Request(url+'/v1/tailserve/requests/job', headers={'Authorization':'Bearer secret'})) as response:
            self.assertEqual(json.load(response)['status'], 'queued')


class ScopeTests(unittest.TestCase):
    def test_scope_versions_change_and_close(self):
        llm = NS(llm_engine=NS())
        with patch.dict(os.environ, {'TAILSERVE_ENABLE':'1','TAILSERVE_RUN_ID':'test'}):
            with generation_scope(llm, {'global_steps':1}):
                first = llm.llm_engine._tailserve_policy_version
                with self.assertRaises(RuntimeError):
                    with generation_scope(llm, {'global_steps':1}):pass
            self.assertIsNone(llm.llm_engine._tailserve_policy_version)
            with generation_scope(llm, {'global_steps':2}):
                self.assertNotEqual(first, llm.llm_engine._tailserve_policy_version)

    def test_missing_step_rejected(self):
        with patch.dict(os.environ, {'TAILSERVE_ENABLE':'1','TAILSERVE_RUN_ID':'test'}):
            with self.assertRaises(RuntimeError):
                with generation_scope(NS(llm_engine=NS()), {}):pass

    def test_validation_scope_bypasses_auxiliary(self):
        llm=NS(llm_engine=NS())
        with patch.dict(os.environ, {'TAILSERVE_ENABLE':'1'}):
            with generation_scope(llm, {}, validation=True):
                self.assertFalse(hasattr(llm.llm_engine, '_tailserve_policy_version'))

    def test_kv_worst_case_reservation(self):
        adapter=Adapter.__new__(Adapter)
        pool=NS(get_num_free_blocks=lambda:34)
        adapter.scheduler=NS(kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=NS(block_size=128))]),
                             kv_cache_manager=NS(block_pool=pool))
        job=dict(prompt_token_ids=[1]*129, max_tokens=128)
        self.assertFalse(adapter.can_admit(job, Config()))
        pool.get_num_free_blocks=lambda:35
        self.assertTrue(adapter.can_admit(job, Config()))

    def test_actual_integration_hooks(self):
        self.assertIn('return run_vllm(self)', (ROOT/'vllm/entrypoints/llm.py').read_text())
        self.assertIn('_tailserve_token_budget', (ROOT/'vllm/v1/core/sched/scheduler.py').read_text())
        self.assertIn('generation_scope(self.inference_engine', (ROOT/'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py').read_text())
        self.assertIn('"TAILSERVE_"', (ROOT/'verl/trainer/constants_ppo.py').read_text())
        self.assertIn('_isolate_live_kv_during_aclgraph_capture(quiet=True)',
                      (ROOT/'vllm_ascend/worker/worker_v1.py').read_text())
        self.assertIn('_isolate_live_kv_during_aclgraph_capture(quiet=True)',
                      (ROOT/'vllm_ascend/worker/worker.py').read_text())

    def test_generation_failure_still_restores_rng_and_budget(self):
        adapter=NS(check_collective_contract=lambda *_:None,
                   consensus_error=lambda _:None,primary_count=lambda:1,
                   gather=lambda _:(_ for _ in ()).throw(RuntimeError('diagnostic failure')),
                   cancel_auxiliary=lambda *_:None,
                   set_auxiliary_token_budget=lambda b:setattr(adapter,'budget',b),
                   restore_primary_rng=lambda:setattr(adapter,'restored',True))
        queue=NS(emit=lambda *_,**__:None,poll=lambda:None,candidate=lambda _:None)
        with self.assertRaisesRegex(RuntimeError,'diagnostic failure'):
            run(adapter,queue,Config(),'v')
        self.assertTrue(adapter.restored)
        self.assertIsNone(adapter.budget)

    def test_rng_restoration_after_auxiliary_scope(self):
        import random
        import numpy as np
        adapter=Adapter.__new__(Adapter)
        state={'torch':b'primary-stream'}
        adapter.torch=NS(get_rng_state=lambda:state['torch'],
                         set_rng_state=lambda x:state.update(torch=x))
        adapter.rng=None
        random.seed(123);np.random.seed(456)
        adapter.save_primary_rng()
        expected=(random.random(),float(np.random.rand()))
        state['torch']=b'auxiliary-stream'
        random.random();np.random.rand()
        adapter.restore_primary_rng()
        self.assertEqual((random.random(),float(np.random.rand())),expected)
        self.assertEqual(state['torch'],b'primary-stream')

    def test_async_backend_rejected_before_model_use(self):
        engine=NS(engine_core=NS(engine_core=NS()),external_launcher_dp=True,
                  vllm_config=NS(scheduler_config=NS(async_scheduling=True)),backend_dp_dummy_flow=True)
        with self.assertRaisesRegex(RuntimeError,'synchronous scheduling'):
            Adapter(NS(llm_engine=engine),NS(rank=0,size=16,torch=None))

    def test_elastic_backend_rejected(self):
        engine=NS(engine_core=NS(engine_core=NS()),external_launcher_dp=True,
                  vllm_config=NS(scheduler_config=NS(async_scheduling=False)),
                  backend_dp_dummy_flow=False,elastic_ep_no_dummy=True)
        with self.assertRaisesRegex(RuntimeError,'original EP topology'):
            Adapter(NS(llm_engine=engine),NS(rank=0,size=16,torch=None))


class World:
    def __init__(self, n):
        self.barrier=threading.Barrier(n, timeout=10)
        self.values=[None]*n

    def gather(self, rank, value):
        self.values[rank]=value;self.barrier.wait()
        result=list(self.values);self.barrier.wait()
        return result


class FakeAdapter:
    def __init__(self, rank, world, primary_ticks, aux_ticks=2):
        self.rank,self.world=rank,world
        self.remaining=primary_ticks;self.aux={};self.length=aux_ticks
        self.restored=False;self.drained=False;self.budget=None

    def check_collective_contract(self, *_):self.world.barrier.wait()
    def consensus_error(self, error):
        errors=self.world.gather(self.rank,error)
        if any(errors):raise RuntimeError(str(errors))
    def gather(self, obs):return self.world.gather(self.rank,obs)
    def primary_count(self):return int(self.remaining>0)
    def validate_candidate(self, job):return None
    def can_admit(self, *_):return True
    def admit(self, job, version):
        self.aux[job['id']]=self.length
        return job['id']
    def set_auxiliary_token_budget(self, budget):self.budget=budget
    def is_primary(self, output):return output.request_id==str(self.rank)
    def note_auxiliary_output(self, output):pass
    def serialize_output(self, output):return dict(text='auxiliary',token_ids=[7])
    def cancel_auxiliary(self, active, queue, version, reason):
        for key,job in active.items():
            self.aux.pop(key,None)
            queue.emit('cancelled',job_id=job['id'],policy_version=version,reason=reason,counted_as_completion=False)
    def restore_primary_rng(self):self.restored=True
    def verify_drained_collectively(self):
        assert not self.aux and self.remaining==0
        self.world.barrier.wait();self.drained=True
    def step(self):
        self.world.barrier.wait();time.sleep(.001)
        outputs=[]
        if self.remaining:
            self.remaining-=1
            if not self.remaining:outputs.append(NS(request_id=str(self.rank),finished=True))
        for key in list(self.aux):
            if not self.budget:continue
            self.aux[key]-=1
            finished=self.aux[key]==0
            outputs.append(NS(request_id=key,finished=finished))
            if finished:del self.aux[key]
        return outputs


class MultiRankTests(unittest.TestCase):
    def execute(self, aux_ticks):
        with tempfile.TemporaryDirectory() as root:
            config=Config(warmup_rounds=1, slowdown_target=.99,violation_rounds=100, aux_interval_rounds=1)
            store=Store(root,2,config)
            for i in range(8):store.submit(dict(id=f'aux{i}',rank=0,prompt_token_ids=[1]))
            world=World(2)
            adapters=[FakeAdapter(0,world,1,aux_ticks),FakeAdapter(1,world,20,aux_ticks)]
            queues=[Queue(root,i,2,config) for i in range(2)]
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures=[executor.submit(run,a,q,config,'v1') for a,q in zip(adapters,queues)]
                results=[future.result(timeout=15) for future in futures]
            self.assertEqual([[o.request_id for o in out] for out in results],[['0'],['1']])
            self.assertTrue(all(a.drained and a.restored and a.budget is None for a in adapters))
            return [store.status(f'aux{i}') for i in range(8)]

    def test_primary_outputs_isolated_and_real_aux_completes(self):
        states=self.execute(2)
        self.assertTrue(any(s['status']=='completed' for s in states))

    def test_auxiliary_never_extends_primary_for_completion(self):
        states=self.execute(1000)
        self.assertTrue(any(s['status']=='cancelled' and s['reason']=='primary_complete' for s in states))
        self.assertFalse(any(s['status']=='completed' for s in states))


class WorkloadTests(unittest.TestCase):
    def test_default_queue_uses_16k_output_cap_without_prompt_splitting_budget(self):
        config=Config()
        self.assertEqual(config.max_new_tokens,16384)
        self.assertEqual(config.max_prompt_tokens,1024)
        self.assertEqual(config.per_rank_tokens,17408)
        self.assertEqual(config.global_tokens,16*17408)
        row=dict(prompt=[],reward_model=dict(ground_truth='5'))
        with tempfile.TemporaryDirectory() as root:
            spool=Path(root)/'spool'
            prepare([row],NS(apply_chat_template=lambda *_,**__:[1]*90),spool,config)
            job=json.loads((spool/'queues/rank_0.jsonl').read_text())
            self.assertEqual(job['max_tokens'],16384)
            self.assertEqual(len(job['prompt_token_ids']),90)
            self.assertGreaterEqual(config.per_rank_tokens,len(job['prompt_token_ids']))

    def test_16k_response_plus_1k_prompt_fits_shared_context(self):
        adapter=Adapter.__new__(Adapter)
        adapter.engine=NS(model_config=NS(get_vocab_size=lambda:100,max_model_len=17408))
        job=dict(prompt_token_ids=[1]*1024,max_tokens=16384)
        self.assertIsNone(adapter.validate_candidate(job))
        job['max_tokens']=16385
        self.assertEqual(adapter.validate_candidate(job),'request exceeds current model context length')

    def test_launcher_derives_auxiliary_limits_from_primary_environment(self):
        env={k:v for k,v in os.environ.items() if not k.startswith('TAILSERVE_')}
        env.update(PYTHONDONTWRITEBYTECODE='1',MAX_PROMPT_LENGTH='512',MAX_RESPONSE_LENGTH='8192',
                   ROLLOUT_MAX_NUM_BATCHED_TOKENS='8704')
        output=subprocess.check_output(['bash',str(ROOT/'run_deepscaler_gsm8k.sh'),'dry-run'],
                                       env=env,text=True,cwd=ROOT)
        start=output.index('{')
        record,_=json.JSONDecoder().raw_decode(output[start:])
        self.assertEqual(record['primary_limits'],dict(max_prompt_length=512,
                         max_response_length=8192,max_num_batched_tokens=8704))
        self.assertEqual(record['config']['max_prompt_tokens'],512)
        self.assertEqual(record['config']['max_new_tokens'],8192)
        self.assertEqual(record['config']['per_rank_tokens'],8704)
        self.assertEqual(record['config']['global_tokens'],16*8704)

    def test_pacing_preserves_live_auxiliary_during_clean_recalibration(self):
        p=Policy(Config(warmup_rounds=1,aux_interval_rounds=4,interference_feedback=1))
        decisions=[p.decide([O(0,1,1,1),O(3,0,0,1)]) for _ in range(8)]
        self.assertEqual([i+1 for i,d in enumerate(decisions) if d.ranks],[4,8])
        self.assertTrue(all(d.tokens_per_rank==0 for i,d in enumerate(decisions) if (i+1)%4))

    def test_paused_adapter_uses_dummy_without_advancing_local_scheduler(self):
        adapter=Adapter.__new__(Adapter)
        engine=NS(has_unfinished_requests=lambda:True,should_execute_dummy_batch=False)
        engine.step=lambda:engine.should_execute_dummy_batch
        adapter.engine=engine
        adapter.scheduler=NS(_tailserve_token_budget=0)
        adapter.primary_count=lambda:0
        self.assertTrue(adapter.step())
        adapter.primary_count=lambda:1
        with self.assertRaisesRegex(RuntimeError,'must not pause RL'):adapter.step()

    def test_fixed_gsm8k_prefix_has_separate_labels_and_no_overwrite(self):
        rows=[dict(uid=f'u{i}', prompt=[dict(role='user',content=str(i))],
                   reward_model=dict(ground_truth=str(i))) for i in range(4)]
        tokenizer=NS(apply_chat_template=lambda *_,**__:[1,2,3])
        with tempfile.TemporaryDirectory() as root:
            spool=Path(root)/'spool'
            self.assertEqual(prepare(rows,tokenizer,spool,Config(),2,128),4)
            jobs=[json.loads(s) for s in (spool/'queues/rank_0.jsonl').read_text().splitlines()]
            self.assertEqual([j['id'] for j in jobs],['gsm8k_00000','gsm8k_00002'])
            self.assertNotIn('ground_truth',jobs[0])
            self.assertEqual(len((spool/'gsm8k_manifest.jsonl').read_text().splitlines()),4)
            with self.assertRaises(FileExistsError):prepare(rows,tokenizer,spool,Config(),2,128)

    def test_length_failure_does_not_silently_skip_or_create_partial_spool(self):
        row=dict(prompt=[],reward_model=dict(ground_truth='1'))
        with tempfile.TemporaryDirectory() as root:
            spool=Path(root)/'spool'
            with self.assertRaises(ValueError):
                prepare([row],NS(apply_chat_template=lambda *_,**__:[1]*1025),spool,Config(),2,128)
            self.assertFalse(spool.exists())

    def test_scheduler_phase_audit_records_real_chunked_prefill_and_decode(self):
        adapter=Adapter.__new__(Adapter)
        req=NS(num_prompt_tokens=10,num_computed_tokens=0)
        adapter.engine=NS(output_processor=NS(request_states={'internal':NS(external_req_id='aux')}),
                          _tailserve_policy_version='v')
        adapter.scheduler=NS(requests={'internal':req})
        adapter.primary_count=lambda:0
        events=[]
        adapter.audit_queue=NS(emit=lambda *a,**kw:events.append((a,kw)))
        adapter.schedule_stats={'aux':dict(prefill_scheduled_tokens=0,decode_scheduled_tokens=0,
                                           prefill_rounds=0,decode_rounds=0)}
        out=NS(num_scheduled_tokens={'internal':8})
        adapter.observe_schedule(out)
        req.num_computed_tokens=8;out.num_scheduled_tokens={'internal':2};adapter.observe_schedule(out)
        req.num_computed_tokens=10;out.num_scheduled_tokens={'internal':1};adapter.observe_schedule(out)
        self.assertEqual(adapter.schedule_stats['aux']['prefill_scheduled_tokens'],10)
        self.assertEqual(adapter.schedule_stats['aux']['decode_scheduled_tokens'],1)
        self.assertEqual([e[1]['phase'] for e in events],['prefill','decode'])
        adapter.primary_count=lambda:1
        with self.assertRaises(RuntimeError):adapter.observe_schedule(out)

    def test_reporting_does_not_count_cancelled_prefix_as_gsm8k_completion(self):
        with tempfile.TemporaryDirectory() as root:
            q=Queue(root,0,1,Config())
            q.emit('cancelled',job_id='incomplete',token_ids=[1]*10)
            q.emit('completed',job_id='done',token_ids=[2]*3,policy_version='v',text='#### 72',finish_reason='stop')
            result=summarize(root)
            self.assertEqual(result['completed_jobs'],1)
            self.assertEqual(result['completed_tokens'],3)
            self.assertEqual(result['cancelled_prefix_tokens'],10)

    def test_reporting_separates_format_misses_from_numeric_answers(self):
        with tempfile.TemporaryDirectory() as root:
            q=Queue(root,0,1,Config())
            from tailserve_runtime.queue import append_record
            for key,text in [('boxed',r'Final answer: \\boxed{5}'),('dollar','#### $5'),
                             ('strict','#### 5'),('wrong','#### 6')]:
                append_record(Path(root)/'gsm8k_manifest.jsonl',dict(id=key,ground_truth='5'),timestamp=False)
                q.emit('completed',job_id=key,token_ids=[1],policy_version='v',text=text,finish_reason='stop')
            result=summarize(root)
            self.assertEqual(result['completed_jobs'],4)
            self.assertEqual(result['gsm8k_correct_completed'],1)
            self.assertEqual(result['gsm8k_flexible_correct_completed'],3)
            self.assertEqual(result['gsm8k_format_only_misses'],2)
            self.assertEqual(result['gsm8k_flexible_accuracy'],.75)


if __name__ == '__main__':unittest.main()
