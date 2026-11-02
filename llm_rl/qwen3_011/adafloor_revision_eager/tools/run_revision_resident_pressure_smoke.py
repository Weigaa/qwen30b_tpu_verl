#!/usr/bin/env python3
"""Actual extra-HBM allocation and conservative Full16 admission diagnostic.

Dummy resident bytes model capacity pressure only, not a larger model's
compute, accuracy or communication costs. Short greedy output is functional
verification, not a performance measurement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hf-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--extra-gib',type=int,default=14)
    args=ap.parse_args()
    if args.extra_gib<=0:raise ValueError('positive pressure required')
    if json.loads((args.hf_root/'tensor_parity_audit.json').read_text())['status']!='ALL_EXPORTED_TENSORS_EQUAL_ORIGINAL_DCP':raise ValueError('weights unverified')
    os.environ.update(VLLM_DP_SIZE='16',VLLM_ENABLE_EXPERT_PARALLEL='1',TASK_QUEUE_ENABLE='2',VLLM_USE_V1='1',
        VLLM_ENABLE_GRAPH_MODE='0',VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK='0',VLLM_ASCEND_ELASTIC_EXECUTION_MODE='0',
        VLLM_ASCEND_REGISTER_CUSTOM_MODELS='1',VLLM_ASCEND_USE_NATIVE_QWEN3_MOE='0')
    import torch
    import torch_npu  # noqa:F401
    from vllm_ascend.patch import platform,worker  # noqa:F401
    from r1_ascend.vllm_parallel_state import init_parallel_state
    from vllm import LLM,SamplingParams
    from vllm.distributed.parallel_state import get_ep_group
    from verl.utils.revision_kv_guard import admit
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local);torch.distributed.init_process_group('hccl');init_parallel_state(1)
    requests=json.loads((args.hf_root/'frontier_workload.json').read_text())['requests']
    config=json.loads((args.hf_root/'model/config.json').read_text())
    kv_bytes=2*config['num_hidden_layers']*config['num_key_value_heads']*config['head_dim']*2
    extra=args.extra_gib*2**30
    base={2:0,4:133120,8:262656,16:380800}
    caps={floor:(max(0,tokens*kv_bytes-extra)//(kv_bytes*128))*128 for floor,tokens in base.items()}
    plan={'selected_floor':8,'shrink_stages':[8],'stage_survivor_ranks':[list(range(8,16))]}
    _,_,_,admission=admit(plan,{},{},[len(row['prompt_token_ids']) for row in requests],8192,caps)
    if admission['selected_floor']!=16:raise ValueError('fixture does not exercise Full16 fallback')
    target=caps[16]
    llm=LLM(model=str(args.hf_root/'model'),load_format='safetensors',dtype='bfloat16',tensor_parallel_size=1,
        enable_expert_parallel=True,distributed_executor_backend='external_launcher',enable_sleep_mode=False,
        enforce_eager=True,gpu_memory_utilization=.9,disable_custom_all_reduce=True,max_model_len=9216,
        max_num_seqs=32,max_num_batched_tokens=9216,block_size=128,num_gpu_blocks_override=target//128,
        enable_prefix_caching=False,enable_chunked_prefill=False,
        additional_config={'torchair_graph_config':{'enabled':False},'ascend_scheduler_config':{'enabled':True,'enable_chunked_prefill':False}})
    if list(get_ep_group().ranks)!=list(range(16)):raise ValueError('not Full16')
    core=llm.llm_engine.engine_core.engine_core
    actual=min(core._kv_cache_token_capacity(config) for config in core.kv_cache_configs)
    if actual<max(admission['rank_upper_tokens']) or actual!=target:raise ValueError('actual KV capacity differs from admission')
    torch.npu.synchronize();torch.npu.empty_cache();torch.npu.synchronize()
    before_free,_=torch.npu.mem_get_info();before_allocated=torch.npu.memory_allocated()
    # Allocate only after profiling/initialization, whose free-memory gate
    # assumes no other resident job. Keep these tensors throughout generation.
    pressure=[torch.empty(1<<30,dtype=torch.uint8,device='npu') for _ in range(args.extra_gib)]
    torch.npu.synchronize();after_free,_=torch.npu.mem_get_info();after_allocated=torch.npu.memory_allocated()
    if after_allocated-before_allocated<extra or before_free-after_free<extra:
        raise ValueError('pressure was not actually resident on NPU')
    width=len(requests)//16;rows=requests[rank*width:(rank+1)*width]
    outputs=llm.generate([{'prompt_token_ids':row['prompt_token_ids']} for row in rows],
                        sampling_params=[SamplingParams(n=1,temperature=0,max_tokens=16) for _ in rows],use_tqdm=False)
    if len(outputs)!=len(rows) or any(out.outputs[0].finish_reason not in ('stop','length') for out in outputs):raise ValueError('failed functional generation')
    result=[{'prompt_hash':row['prompt_hash'],'sample_index':row['sample_index'],'token_ids':list(out.outputs[0].token_ids)} for row,out in zip(rows,outputs)]
    payload={'status':'REAL_RESIDENT_PRESSURE_FULL16_SMOKE_FINISHED','rank':rank,'extra_resident_bytes':extra,
        'free_memory_drop_bytes':before_free-after_free,'torch_allocated_delta_bytes':after_allocated-before_allocated,
        'effective_caps_tokens':caps,'actual_kv_override_tokens':target,'actual_kv_capacity_tokens':actual,'admission':admission,'requests':result,
        'pressure_tensor_count_during_generation':len(pressure),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Real extra-HBM allocation held across short Full16 decode, with budget-derived admission escalation. Not actual larger-model execution, long stochastic-generation safety or speedup evidence.'}
    args.output_dir.mkdir(parents=True,exist_ok=True);(args.output_dir/f'rank_{rank:02d}.json').write_text(json.dumps(payload,sort_keys=True)+'\n')
    torch.distributed.barrier();del pressure;torch.distributed.destroy_process_group()


if __name__=='__main__':main()
