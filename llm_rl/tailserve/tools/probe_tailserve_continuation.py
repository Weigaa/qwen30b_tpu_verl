#!/usr/bin/env python3
"""Diagnostic only: same-weight full/chunked/paused greedy prefixes, no RL update.

Uses the local HF snapshot, NOT the resumed RL checkpoint. This isolates backend
continuation correctness; it is neither an RL performance result nor a replacement
for the DeepScaleR validation.
"""
import argparse
import json
import os
from pathlib import Path


def first_difference(a, b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),
                None if len(a)==len(b) else min(len(a),len(b)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--model',default='/data/Qwen3-30B-A3B')
    p.add_argument('--tokens',type=int,default=64)
    args=p.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    from r1_ascend.vllm_parallel_state import init_parallel_state
    from vllm import LLM,SamplingParams
    from vllm.config import CompilationConfig,CUDAGraphMode
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    rank=int(os.environ['RANK'])
    # Same setting as VERL's native FULL graph constructor.
    torch._dynamo.config.log_compilation_metrics=False
    # CUDA TMA is not available on an Ascend NPU. transfer_to_npu makes
    # cuda.is_available() true but its capability is None; Dynamo's unguarded
    # CUDA-only feature discovery must not interpret this as a CUDA device.
    import torch.utils._triton as triton_discovery
    triton_discovery._device_supports_tma=lambda:False
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group('hccl')
    init_parallel_state(1)
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    row_index=2 if rank%2==0 else 7
    row=pq.read_table('/data/gsm8k/train.parquet').slice(row_index,1).to_pylist()[0]
    tokens=list(tokenizer.apply_chat_template(row['prompt'],tokenize=True,
                add_generation_prompt=True,enable_thinking=False))
    llm=LLM(model=args.model,tensor_parallel_size=1,
        distributed_executor_backend='external_launcher',enable_expert_parallel=True,
        dtype='bfloat16',enforce_eager=False,enable_sleep_mode=True,
        max_model_len=2048,max_num_seqs=32,max_num_batched_tokens=2048,
        kv_cache_memory_bytes=512*1024**2,enable_prefix_caching=False,
        enable_chunked_prefill=True,async_scheduling=False,
        disable_custom_all_reduce=True,trust_remote_code=True,
        additional_config={'ascend_scheduler_config':{'enabled':True,'enable_chunked_prefill':True},
                           'refresh':True,'dynamic_eplb':False,
                           'npugraph_ex_config':{'enable':True,'enable_static_kernel':False}},
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                                             cudagraph_capture_sizes=[32]))
    engine=llm.llm_engine
    worker=engine.model_executor.driver_worker.worker
    print('TAILSERVE_PROBE_WORKER',rank,type(worker).__module__,type(worker).__name__,flush=True)
    scheduler=engine.engine_core.engine_core.scheduler
    prompt={'prompt_token_ids':tokens}
    params=SamplingParams(temperature=0,max_tokens=args.tokens,seed=0)
    arms=[]
    for name,budget,pause in [('full_prefill',None,False),('chunk8',8,False),('chunk8_pause',8,True)]:
        scheduler._tailserve_token_budget=budget
        if not pause:
            outputs=llm.generate([prompt],params,use_tqdm=False)
        else:
            engine.add_request(f'probe.{rank}',prompt,params)
            outputs=[];rounds=0
            while engine.has_unfinished_requests():
                rounds+=1
                if rounds%2==0:
                    engine.should_execute_dummy_batch=True
                outputs.extend(o for o in engine.step() if o.finished)
        if len(outputs)!=1:
            raise RuntimeError(f'{name}: wrong final output count {len(outputs)}')
        answer=outputs[0].outputs[0]
        record=dict(arm=name,token_ids=list(answer.token_ids),text=answer.text,
                    finish_reason=answer.finish_reason)
        arms.append(record)
        print('TAILSERVE_PROBE',rank,json.dumps(record),flush=True)
    reference=arms[0]['token_ids']
    result=dict(rank=rank,source_row=row_index,model=args.model,
        checkpoint_kind='HF base snapshot, diagnostic only',prompt_tokens=len(tokens),arms=arms,
        first_difference={a['arm']:first_difference(reference,a['token_ids']) for a in arms[1:]})
    target=Path(args.output)
    target.mkdir(parents=True,exist_ok=True)
    with (target/f'rank_{rank}.json').open('w') as f:json.dump(result,f,indent=2)
    torch.distributed.barrier()
    if rank==0:
        results=[json.loads((target/f'rank_{r}.json').read_text()) for r in range(16)]
        print('TAILSERVE_PROBE_DIFFERENCES',json.dumps([r['first_difference'] for r in results]),flush=True)
    torch.distributed.destroy_process_group()


if __name__=='__main__':main()
