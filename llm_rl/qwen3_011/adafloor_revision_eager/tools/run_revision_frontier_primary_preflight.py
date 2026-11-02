#!/usr/bin/env python3
"""Real external-launcher EP8/EP16 inference preflight, not a frontier result."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--workload",type=Path,required=True)
    parser.add_argument("--weight-audit",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--world-size",type=int,choices=(8,16),required=True)
    parser.add_argument("--max-tokens",type=int,default=16)
    args=parser.parse_args()
    audit=json.loads(args.weight_audit.read_text())
    if audit.get("status")!="ALL_EXPORTED_TENSORS_EQUAL_ORIGINAL_DCP":
        raise ValueError("uncertified derived weights must not enter static controls")
    for path in (args.model/"config.json",args.model/"model.safetensors.index.json"):
        expected=audit["input_sha256"].get(str(path))
        if expected is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError("HF config/index changed after weight audit")
    if int(os.environ["WORLD_SIZE"])!=args.world_size:raise ValueError("launch world differs from control")
    os.environ.update(VLLM_DP_SIZE=str(args.world_size),VLLM_ENABLE_EXPERT_PARALLEL="1",
        VLLM_ENABLE_GRAPH_MODE="0",VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK="0",
        VLLM_ASCEND_ELASTIC_EXECUTION_MODE="0",VLLM_ASCEND_REGISTER_CUSTOM_MODELS="1",
        VLLM_ASCEND_USE_NATIVE_QWEN3_MOE="0",VLLM_USE_V1="1",TASK_QUEUE_ENABLE="2")
    import torch
    import torch_npu  # noqa: F401
    from vllm_ascend.patch import platform,worker  # noqa: F401
    from r1_ascend.vllm_parallel_state import init_parallel_state
    from vllm import LLM,SamplingParams
    from vllm.distributed.parallel_state import get_ep_group
    rank,local_rank=int(os.environ["RANK"]),int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    torch.distributed.init_process_group("hccl")
    init_parallel_state(1)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir/f"rank_{rank:02d}.json"
    if output.exists():raise ValueError("refusing existing preflight rank result")
    data=json.loads(args.workload.read_text());rows=data["requests"]
    if len(rows)%args.world_size:raise ValueError("nonuniform ownership")
    width=len(rows)//args.world_size
    rows=rows[rank*width:(rank+1)*width]
    tokens=262656 if args.world_size==8 else 380800
    start=time.perf_counter()
    llm=LLM(model=str(args.model),load_format="safetensors",dtype="bfloat16",
        tensor_parallel_size=1,enable_expert_parallel=True,
        distributed_executor_backend="external_launcher",enable_sleep_mode=False,
        enforce_eager=True,gpu_memory_utilization=.9,disable_custom_all_reduce=True,
        max_model_len=5632,max_num_seqs=32,max_num_batched_tokens=5632,
        block_size=128,num_gpu_blocks_override=tokens//128,
        enable_prefix_caching=False,enable_chunked_prefill=False,
        additional_config={"torchair_graph_config":{"enabled":False},
                           "ascend_scheduler_config":{"enabled":True,"enable_chunked_prefill":False}})
    load_s=time.perf_counter()-start
    ep_ranks=list(get_ep_group().ranks)
    if ep_ranks!=list(range(args.world_size)):raise ValueError(f"not a single genuine EP world:{ep_ranks}")
    params=[SamplingParams(n=1,temperature=0,max_tokens=args.max_tokens,seed=row["request_seed"]) for row in rows]
    generation_start=time.perf_counter()
    outputs=llm.generate([{"prompt_token_ids":row["prompt_token_ids"]} for row in rows],sampling_params=params,use_tqdm=False)
    if len(outputs)!=len(rows):raise ValueError("missing preflight outputs")
    result=[]
    for row,out in zip(rows,outputs):
        if len(out.outputs)!=1 or out.outputs[0].finish_reason not in ("stop","length"):
            raise ValueError("failed primary completion")
        result.append({"prompt_hash":row["prompt_hash"],"sample_index":row["sample_index"],
                       "token_ids":list(out.outputs[0].token_ids),"finish_reason":out.outputs[0].finish_reason})
    payload={"status":"HF_EP_PREFLIGHT_FINISHED","rank":rank,"ep_ranks":ep_ranks,
             "world_size":args.world_size,"load_seconds":load_s,"generation_seconds":time.perf_counter()-generation_start,
             "kv_tokens_override":tokens,"requests":result,"graph":False,"task_queue_enable":2,
             "weights_audit_sha256":hashlib.sha256(args.weight_audit.read_bytes()).hexdigest(),
             "workload_sha256":hashlib.sha256(args.workload.read_bytes()).hexdigest(),
             "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "scope":"Greedy16-token structural preflight on the exact common actor. Not a fixed-time goodput, training quality or sidecar result."}
    output.write_text(json.dumps(payload,sort_keys=True)+"\n")
    torch.distributed.barrier()
    print(f"FRONTIER_PRIMARY_PREFLIGHT_DONE rank={rank} ep_world={args.world_size} requests={len(result)}",flush=True)
    torch.distributed.destroy_process_group()


if __name__=="__main__":main()
