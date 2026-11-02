#!/usr/bin/env python3
"""One finite frozen-policy batch, observed under a shared wall-time budget.

No per-token consensus or deadline-induced primary abort. Completions after
the window are retained but uncredited; ranks finish the batch safely before
restore. This is an inference-phase operating point, not PPO convergence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",choices=("primary_only","static8","dynamic4"),required=True)
    ap.add_argument("--hf-root",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--budget-seconds",type=float,default=300)
    args=ap.parse_args()
    if args.budget_seconds<=0:raise ValueError("positive common window required")
    world=8 if args.arm=="static8" else 16;dynamic=args.arm=="dynamic4"
    if int(os.environ["WORLD_SIZE"])!=world:raise ValueError("incorrect static partition")
    audit=args.hf_root/"tensor_parity_audit.json"
    if json.loads(audit.read_text())["status"]!="ALL_EXPORTED_TENSORS_EQUAL_ORIGINAL_DCP":raise ValueError("weight parity missing")
    if json.loads((args.hf_root/"ep8_ep16_preflight_audit.json").read_text())["token_mismatches"]!=0:raise ValueError("EP preflight missing")
    os.environ.update(VLLM_DP_SIZE=str(world),VLLM_ENABLE_EXPERT_PARALLEL="1",TASK_QUEUE_ENABLE="2",
        VLLM_ENABLE_GRAPH_MODE="0",VLLM_ASCEND_REGISTER_CUSTOM_MODELS="1",VLLM_ASCEND_USE_NATIVE_QWEN3_MOE="0",
        VLLM_ASCEND_ELASTIC_EXECUTION_MODE="1" if dynamic else "0",
        VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK="1" if dynamic else "0",VLLM_USE_V1="1")
    if dynamic:
        os.environ.update(VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="4",VLLM_ASCEND_SHRINK_AWARE_ENABLE="1",
            VLLM_ASCEND_SHRINK_AWARE_MODE="staged",VLLM_ASCEND_SHRINK_AWARE_STAGES="8,4",
            VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY="manual",VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY="planned",
            VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS="8,9,10,11,12,13,14,15",
            VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS="12,13,14,15",
            VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS="[[8,9,10,11,12,13,14,15],[12,13,14,15]]",
            VLLM_ASCEND_MODE1_PARITY_CURRENT_FLOOR="4",VLLM_ASCEND_MODE1_PARITY_CURRENT_KV_TOKENS="133120",
            VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS="1",
            VLLM_ASCEND_MODE1_PARITY_PREFILL_PLANNED_EXPERT_SLOTS="1",
            VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE="1",
            VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP="1",
            VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE="1",
            VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP="0",
            VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP_KIND="mc2_dispatcher_only",
            VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP="0",
            VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE="1")
    import torch
    import torch_npu  # noqa:F401
    from vllm_ascend.patch import platform,worker  # noqa:F401
    from r1_ascend.vllm_parallel_state import init_parallel_state
    from vllm import LLM,SamplingParams
    from vllm.distributed.parallel_state import get_ep_group
    from transformers import AutoTokenizer
    from deepscaler import compute_score
    rank,local=int(os.environ["RANK"]),int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local);torch.distributed.init_process_group("hccl")
    init_parallel_state(1)
    control=torch.distributed.new_group(backend="gloo")
    data=json.loads((args.hf_root/"frontier_workload.json").read_text())
    width=len(data["requests"])//world;rows=data["requests"][rank*width:(rank+1)*width]
    if dynamic:
        layout=json.loads((args.hf_root/"dynamic_planner_layout.json").read_text())
        if layout['status']!='HISTORY_ONLY_PAPER_PLANNER_LAYOUT' or layout['actual_results_used']:
            raise ValueError('dynamic mapping must come from historical paper planner')
        rows=layout['rank_requests'][str(rank)]
    cap=4608;target=133120 if dynamic else 262656 if world==8 else 380800
    upper=sum(((len(row["prompt_token_ids"])+cap+127)//128)*128 for row in rows)
    if upper>target:raise ValueError("independent prompt+hard-cap bound exceeds chosen capacity")
    extra={"torchair_graph_config":{"enabled":False},"ascend_scheduler_config":{"enabled":True,"enable_chunked_prefill":False}}
    if dynamic:
        from vllm_ascend import envs as ascend_envs
        extra.update(refresh=True,elastic_moe_mode=ascend_envs.VLLM_ASCEND_ELASTIC_MOE_MODE,
            init_redundancy_expert=ascend_envs.compute_elastic_init_redundancy_expert(128,16,0),
            dynamic_eplb=False)
    startup=time.time()
    llm=LLM(model=str(args.hf_root/"model"),load_format="safetensors",dtype="bfloat16",
        tensor_parallel_size=1,enable_expert_parallel=True,distributed_executor_backend="external_launcher",
        enable_sleep_mode=False,enforce_eager=True,gpu_memory_utilization=.9,disable_custom_all_reduce=True,
        max_model_len=5632,max_num_seqs=32,max_num_batched_tokens=5632,block_size=128,
        num_gpu_blocks_override=target//128,enable_prefix_caching=False,enable_chunked_prefill=False,
        additional_config=extra)
    if list(get_ep_group().ranks)!=list(range(world)):raise ValueError("wrong initial EP world")
    if dynamic:
        llm.llm_engine.engine_core.engine_core.resize_kv_cache_for_mode1_step(target,4)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/f"ready_{rank:02d}").write_text("primary_model_ready\n")
    # One shared host-side boundary, not any additional token consensus.
    torch.distributed.barrier(group=control)
    if rank==0:(args.output_dir/"window_start.json").write_text(json.dumps({"time":time.time()+2,"budget_seconds":args.budget_seconds})+"\n")
    torch.distributed.barrier(group=control)
    boundary=json.loads((args.output_dir/"window_start.json").read_text());t0=boundary["time"]
    while time.time()<t0:time.sleep(.01)
    finished={};original_step=llm.llm_engine.step
    def observed_step(*a,**kw):
        outputs=original_step(*a,**kw);now=time.time()
        for output in outputs:
            if output.finished:finished[str(output.request_id)]=now
        return outputs
    llm.llm_engine.step=observed_step
    params=[SamplingParams(n=1,temperature=.9,top_k=50,top_p=.9,max_tokens=cap,seed=row["request_seed"]) for row in rows]
    outputs=llm.generate([{"prompt_token_ids":row["prompt_token_ids"]} for row in rows],sampling_params=params,use_tqdm=False)
    generation_end=time.time()
    tokenizer=AutoTokenizer.from_pretrained(str(args.hf_root/"model"))
    if len(outputs)!=len(rows):raise ValueError("missing primary responses")
    result=[]
    for row,out in zip(rows,outputs):
        if len(out.outputs)!=1 or out.outputs[0].finish_reason not in ("stop","length"):raise ValueError("failed primary request")
        sample=out.outputs[0];timestamp=finished.get(str(out.request_id))
        if timestamp is None:raise ValueError("completion timestamp was not observed")
        text=tokenizer.decode(sample.token_ids,skip_special_tokens=True)
        result.append(dict(prompt_hash=row["prompt_hash"],sample_index=row["sample_index"],request_seed=row["request_seed"],
            completion_time=timestamp,completion_seconds=timestamp-t0,credited=timestamp<=t0+args.budget_seconds,
            generated_tokens=len(sample.token_ids),finish_reason=sample.finish_reason,
            reward=compute_score("deepscaler",text,row["ground_truth"]),token_ids=list(sample.token_ids)))
    while time.time()<t0+args.budget_seconds:time.sleep(.1)
    torch.distributed.barrier(group=control)
    if rank==0:
        if args.arm!="primary_only":
            from verl.utils.sidecar_restore_handshake import request_sidecar_stop_before_restore
            ack=request_sidecar_stop_before_restore()
            print(f"Frontier release acknowledged: request_id={ack['request_id']} lease_id={ack['lease_id']} time={time.time():.9f}",flush=True)
    torch.distributed.barrier(group=control)
    if dynamic:
        print(f"Frontier restore RPC start: rank={rank} time={time.time():.9f}",flush=True)
        llm.llm_engine.restore_elastic_parallel_groups_if_needed()
    payload={"status":"FRONTIER_PRIMARY_WINDOW_FINISHED","arm":args.arm,"rank":rank,"initial_world":world,
        "window_start":t0,"budget_seconds":args.budget_seconds,"prepared_primary_startup_seconds":t0-startup,
        "generation_end_seconds":generation_end-t0,"post_window_uncredited_tail_seconds":max(0,generation_end-t0-args.budget_seconds),
        "admission_upper_tokens":upper,"kv_target_tokens":target,"requests":result,
        "weight_audit_sha256":hashlib.sha256(audit.read_bytes()).hexdigest(),
        "workload_sha256":hashlib.sha256((args.hf_root/"frontier_workload.json").read_bytes()).hexdigest(),
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope":"Finite frozen-policy inference batch; prepared primary startup excluded, cold sidecar startup included. Shared16-device fixed-time prefix; late primary results uncredited, no primary deadline abort. Not steady-state throughput or PPO training progress."}
    (args.output_dir/f"rank_{rank:02d}.json").write_text(json.dumps(payload,sort_keys=True)+"\n")
    torch.distributed.barrier(group=control);torch.distributed.destroy_process_group()


if __name__=="__main__":main()
