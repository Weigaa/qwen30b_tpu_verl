#!/usr/bin/env python3
"""One frozen controlled-length rank, using v0.14 native FULL_DECODE_ONLY.

No RL updates or sidecar. Observes existing completion checks/rebuild RPCs;
does not add token-level collectives or synchronizations.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType


def uniform_kv_capacity(configs):
    if not configs or any(len(cfg.kv_cache_groups)!=1 for cfg in configs):
        raise RuntimeError('Controlled Qwen requires one uniform KV group')
    caps={int(cfg.num_blocks)*int(cfg.kv_cache_groups[0].kv_cache_spec.block_size) for cfg in configs}
    if len(caps)!=1:raise RuntimeError('Worker KV capacities differ')
    return caps.pop()


def configure_layout():
    # The RL rollout explicitly disables NZ before constructing its engine.
    # This standalone entry must do the same before Ascend env/model imports.
    if os.environ.get('VLLM_ASCEND_ENABLE_NZ','0')!='0':
        raise RuntimeError('Controlled frozen RL actor requires NZ disabled')
    os.environ['VLLM_ASCEND_ENABLE_NZ']='0'


def initialize_parallel_with_config(init_fn,config_context,config):
    # Ascend calculates HCCL DP buffer options even while constructing the
    # initial world group. r1's own context starts later, at model-parallel
    # setup, so it is not sufficient for this standalone import ordering.
    with config_context(config):
        init_fn(1)


def validate_initial_residency(modules):
    """Physical capacity is not the EP16 primary-only valid-row count."""
    counts=[];capacity=[]
    for layer in modules:
        if (int(layer.elastic_original_num_experts)!=128 or int(layer.elastic_original_ep_size)!=16
                or int(layer.loaded_local_num_experts)!=8 or int(layer.active_local_num_experts)!=8):
            raise RuntimeError('Not the standard initial EP16 primary-only logical layout')
        if (int(layer.loaded_weight_capacity)!=32 or int(layer.w13_weight.shape[0])!=32
                or int(layer.w2_weight.shape[0])!=32):
            raise RuntimeError('Fixed-F4 physical expert capacity is not32 rows')
        counts.append(8);capacity.append(32)
    if len(counts)!=48:raise RuntimeError('Not48 fixed-F4 MoE layers')
    return counts,capacity


def install_observers(engine,rank,runner):
    trace=dict(local_done=None,seen_requests=False,events=[])
    original_has=engine.has_unfinished_requests_dp
    def observed_has(has_unfinished):
        if has_unfinished:trace['seen_requests']=True
        elif trace['seen_requests'] and trace['local_done'] is None:trace['local_done']=time.time()
        return original_has(has_unfinished)
    engine.has_unfinished_requests_dp=observed_has
    original_rpc=engine.engine_core.collective_rpc
    def observed_rpc(method,*args,**kwargs):
        begin=time.time();result=original_rpc(method,*args,**kwargs);end=time.time()
        if method=='rebuild_elastic_ep_group':
            target=(kwargs.get('args') or args[0])[0]
            trace['events'].append(dict(reporter=rank,active_ranks=list(target),start=begin,done=end,
                full_iterations_after=int((runner._mode1_forward_audit or {}).get('full_iterations',0))))
        return result
    engine.engine_core.collective_rpc=observed_rpc
    return trace


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--hf-root',type=Path,required=True);p.add_argument('--workload',type=Path,required=True)
    p.add_argument('--policy',choices=('min_skew','release_area'),required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.workload.read_text());c=data['comparison']
    if data['status']!='V014_CONTROLLED_WORKLOAD_PREPARED_NOT_NPU_RESULT' or c['floor']!=4:
        raise RuntimeError('Not the prespecified v0.14 fixed-F4 workload')
    if int(os.environ['WORLD_SIZE'])!=16 or c['kv_cap_tokens_per_rank']!=133120:
        raise RuntimeError('Wrong world or KV capacity')
    for path,wanted in data['input_sha256'].items():
        with Path(path).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=wanted:raise RuntimeError('Prepared workload provenance changed: '+path)
    audit=json.loads((args.hf_root/'tensor_parity_audit.json').read_text())
    if audit['status']!='ALL_EXPORTED_TENSORS_EQUAL_ORIGINAL_DCP':raise RuntimeError('Unaudited frozen actor')
    configure_layout()
    import torch
    import torch_npu
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local)
    import vllm
    import vllm_ascend
    from vllm_ascend.patch import platform,worker
    from r1_ascend.vllm_parallel_state import init_parallel_state
    from vllm import LLM,SamplingParams
    from vllm.config import VllmConfig,set_current_vllm_config
    from vllm.distributed.parallel_state import get_ep_group
    from vllm_ascend import envs as ascend_envs
    from vllm_ascend.ops.fused_moe import AscendFusedMoE
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
        _invalidate_rollout_aclgraphs,_recapture_rollout_aclgraphs,
    )
    if not str(vllm.__version__).startswith('0.14'):raise RuntimeError('Not vLLM0.14')
    if Path(vllm.__file__).resolve()!=(args.source/'vllm/__init__.py').resolve():
        raise RuntimeError('Unexpected vLLM source tree')
    torch.distributed.init_process_group('hccl')
    initialize_parallel_with_config(init_parallel_state,set_current_vllm_config,VllmConfig())
    control=torch.distributed.new_group(backend='gloo')
    selected=c['policies'][args.policy]
    owned=set(selected['rank_to_prompt_ids'][str(rank)])
    rows=[r for r in data['requests'] if r['id'] in owned]
    if len(rows)!=32:raise RuntimeError('Rank must own32 response requests')
    started=time.time();torch._dynamo.config.log_compilation_metrics=False
    llm=LLM(model=str(args.hf_root/'model'),load_format='safetensors',dtype='bfloat16',
        tensor_parallel_size=1,enable_expert_parallel=True,distributed_executor_backend='external_launcher',
        worker_cls='vllm_ascend.worker.worker_v1.NPUWorker',enable_sleep_mode=True,
        enforce_eager=False,gpu_memory_utilization=.9,disable_custom_all_reduce=True,
        max_model_len=17408,max_num_seqs=32,max_num_batched_tokens=17408,block_size=128,
        num_gpu_blocks_override=1040,async_scheduling=False,enable_prefix_caching=False,
        enable_chunked_prefill=False,trust_remote_code=False,disable_log_stats=True,seed=0,
        compilation_config=dict(cudagraph_mode='FULL_DECODE_ONLY',
            cudagraph_capture_sizes=[1,2,4,8,12,16,24,32]),
        additional_config=dict(ascend_scheduler_config=dict(enabled=True,enable_chunked_prefill=False),
            npugraph_ex_config=dict(enable=True,enable_static_kernel=False),refresh=True,
            elastic_execution_mode=1,elastic_moe_mode='lossless',
            init_redundancy_expert=ascend_envs.compute_elastic_init_redundancy_expert(128,16,0),
            eplb_config=dict(dynamic_eplb=False)))
    engine=llm.llm_engine;core=engine.engine_core.engine_core
    runner=engine.model_executor.driver_worker.worker.model_runner
    model=runner.get_model();original_logits=model.compute_logits;vocab=len(llm.get_tokenizer())
    resident_modules=[layer for layer in model.modules() if isinstance(layer,AscendFusedMoE)]
    initial_counts,physical_slots=validate_initial_residency(resident_modules)
    value_audits={id(a):a for module in model.modules()
        if (a:=getattr(module,'_reload_weight_value_audit',None)) is not None}
    if len(value_audits)!=1:raise RuntimeError('Missing unique real HF weight-value audit')
    value_audit=next(iter(value_audits.values()));value_report=value_audit.check('controlled_post_init')
    primary_ids=sorted({key[0] for key in value_audit.expected})
    if value_report['checked']!=24 or value_report['failed']!=0 or len(primary_ids)!=8:
        raise RuntimeError('Not8 real primary experts x3 checked projection tensors')
    def masked_logits(self,*a,**kw):
        logits=original_logits(*a,**kw);logits[...,vocab:]=float('-inf');return logits
    model.compute_logits=MethodType(masked_logits,model)
    # This one-window serving arm already allocates exactly1040*128 tokens.
    # Native per-step resize requires a sleeping engine and is unnecessary
    # here. Sleeping at setup with level2 would discard the frozen HF weights.
    actual=uniform_kv_capacity(core.kv_cache_configs)
    worker_config=runner.kv_cache_config
    if worker_config.num_blocks!=core.kv_cache_configs[0].num_blocks:
        raise RuntimeError('Scheduler/worker KV capacities disagree')
    with set_current_vllm_config(runner.vllm_config):
        _invalidate_rollout_aclgraphs(runner,'controlled_static_role_and_vocab_ready')
        _recapture_rollout_aclgraphs(runner,'controlled_static_role_and_vocab_ready',force=True)
    if list(get_ep_group().ranks)!=list(range(16)):raise RuntimeError('Not initial EP16')
    if actual!=133120:raise RuntimeError('Allocated KV differs from declared fixed graph')
    runner._mode1_forward_audit={};preempt=int(getattr(core.scheduler,'_adafloor_preemption_count',0))
    trace=install_observers(engine,rank,runner)
    params=[]
    for row in rows:
        sample=SamplingParams(n=1,temperature=.9,top_k=50,top_p=.9,
            max_tokens=row['max_tokens'],ignore_eos=True,seed=row['request_seed'],detokenize=False)
        sample.__dict__.pop('sampling_type',None);params.append(sample)
    torch.npu.synchronize();torch.distributed.barrier(group=control)
    clock=[time.time() if rank==0 else None];torch.distributed.broadcast_object_list(clock,src=0,group=control)
    begin=clock[0]
    outputs=llm.generate([dict(prompt_token_ids=row['prompt_token_ids']) for row in rows],
                        sampling_params=params,use_tqdm=False)
    end=time.time();forward=dict(runner._mode1_forward_audit);runner._mode1_forward_audit=None
    if len(outputs)!=32 or trace['local_done'] is None:raise RuntimeError('Missing completions/local-done event')
    if forward.get('full_iterations',0)<=0:raise RuntimeError('No actual FULL replay')
    results=[]
    for row,out in zip(rows,outputs):
        if len(out.outputs)!=1:raise RuntimeError('Wrong sample count')
        answer=out.outputs[0]
        if len(answer.token_ids)!=row['max_tokens'] or answer.finish_reason!='length':
            raise RuntimeError('Declared forced response length not reached')
        results.append(dict(id=row['id'],sample_index=row['sample_index'],prompt_hash=row['prompt_hash'],
            request_seed=row['request_seed'],generated_tokens=len(answer.token_ids),finish_reason=answer.finish_reason))
    if int(getattr(core.scheduler,'_adafloor_preemption_count',0))!=preempt:raise RuntimeError('Preemption')
    # Match the primary native-sleep lifecycle at local generation return.
    # A detached donor can release before the surviving ranks finish. Group
    # detach alone is never credited as physical resource release.
    sleep_start=time.time();free_before=int(torch.npu.mem_get_info()[0])
    _invalidate_rollout_aclgraphs(runner,'controlled_before_native_sleep2')
    llm.sleep(level=2)
    torch.npu.synchronize();gc.collect();torch.npu.empty_cache()
    sleep_done=time.time();free_after=int(torch.npu.mem_get_info()[0])
    torch.distributed.barrier(group=control);restore_start=time.time()
    engine.restore_elastic_parallel_groups_if_needed();restore_end=time.time()
    if list(get_ep_group().ranks)!=list(range(16)):raise RuntimeError('Full16 restore failed')
    finish=torch.tensor([restore_end],dtype=torch.float64)
    torch.distributed.all_reduce(finish,op=torch.distributed.ReduceOp.MAX,group=control)
    window_end=float(finish.item())
    args.output.mkdir(parents=True,exist_ok=True);path=args.output/f'rank_{rank:02d}.json'
    if path.exists():raise RuntimeError('Refusing existing result')
    report=dict(status='V014_FULL_CONTROLLED_RANK_COMPLETE',policy=args.policy,rank=rank,pid=os.getpid(),start=begin,
        local_done=trace['local_done'],generation_end=end,restore_start=restore_start,restore_end=restore_end,
        sleep_start=sleep_start,sleep_done=sleep_done,native_sleep_level=2,
        effective_weight_layout_NZ=int(os.environ['VLLM_ASCEND_ENABLE_NZ']),
        initial_loaded_experts_per_layer=initial_counts,initial_physical_expert_slots=physical_slots,
        initial_weight_value_audit=value_report,initial_primary_expert_ids=primary_ids,
        free_bytes_before_sleep=free_before,free_bytes_after_sleep=free_after,
        measured_free_bytes_change=free_after-free_before,window_end=window_end,
        startup_seconds=begin-started,events=trace['events'],forward_audit=forward,requests=results,
        kv_tokens=actual,feasible_graph_sha256=selected['feasible_graph_sha256'],
        source_vllm=str(vllm.__file__),source_ascend=str(vllm_ascend.__file__),
        versions=dict(vllm=vllm.__version__,torch=str(torch.__version__),torch_npu=str(torch_npu.__version__)),
        workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        weights_audit_sha256=hashlib.sha256((args.hf_root/'tensor_parity_audit.json').read_bytes()).hexdigest(),
        scope='Frozen standalone FULL/TQ1, constant Planned F4 residency, native sleep2/lifecycle1 after local generation and full-world restore. Q/E window includes generation, cleanup/sleep and restore but excludes initial checkpoint load/capture. Completion/RPC observation only, no added token collectives. Ignore-EOS synthetic lengths: no reward/quality claim.')
    path.write_text(json.dumps(report,indent=2)+'\n')
    torch.npu.synchronize();torch.distributed.barrier(group=control);torch.distributed.destroy_process_group()


if __name__=='__main__':main()
