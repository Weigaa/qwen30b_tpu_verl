#!/usr/bin/env python3
"""Matched frozen inference with per-response forced length caps, eager/TQ2.

Uses the successful external-launcher Planned-F4 setup, not a new Ray trainer.
Times the EXISTING rebuild RPC; adds no per-token collective or sidecar.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf-root', type=Path, required=True)
    ap.add_argument('--workload', type=Path, required=True)
    ap.add_argument('--policy', choices=('min_skew', 'release_area'), required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    if int(os.environ['WORLD_SIZE']) != 16:
        raise ValueError('controlled case requires16 actual ranks')
    data = json.loads(args.workload.read_text())
    if data['status'] != 'CONTROLLED_LENGTH_RUNTIME_WORKLOAD_PREPARED':
        raise ValueError('unprepared controlled workload')
    c = data['comparison']; target = c['kv_cap_tokens_per_rank']
    if c['floor'] != 4 or target != 133120 or c['responses'] != 512:
        raise ValueError('not the declared fixed paper Planned-F4 protocol')
    if not c['policies'][args.policy]['feasible']:
        raise ValueError('infeasible selected policy')
    for path, digest in data['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('workload/planner provenance changed')
    audit_path = args.hf_root / 'tensor_parity_audit.json'
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'ALL_EXPORTED_TENSORS_EQUAL_ORIGINAL_DCP':
        raise ValueError('common actor HF export not verified')
    for name in ('config.json', 'model.safetensors.index.json'):
        path = args.hf_root / 'model' / name
        if audit['input_sha256'].get(str(path)) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('model config/index differs from export audit')
    os.environ.update(VLLM_DP_SIZE='16', VLLM_ENABLE_EXPERT_PARALLEL='1', TASK_QUEUE_ENABLE='2',
        VLLM_ENABLE_GRAPH_MODE='0', VLLM_ASCEND_REGISTER_CUSTOM_MODELS='1', VLLM_ASCEND_USE_NATIVE_QWEN3_MOE='0',
        VLLM_ASCEND_ELASTIC_EXECUTION_MODE='1', VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK='1', VLLM_USE_V1='1',
        VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE='4', VLLM_ASCEND_SHRINK_AWARE_ENABLE='1',
        VLLM_ASCEND_SHRINK_AWARE_MODE='staged', VLLM_ASCEND_SHRINK_AWARE_STAGES='8,4',
        VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY='manual', VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY='planned',
        VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS='8,9,10,11,12,13,14,15',
        VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS='12,13,14,15',
        VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS='[[8,9,10,11,12,13,14,15],[12,13,14,15]]',
        VLLM_ASCEND_MODE1_PARITY_CURRENT_FLOOR='4', VLLM_ASCEND_MODE1_PARITY_CURRENT_KV_TOKENS=str(target),
        VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS='1',
        VLLM_ASCEND_MODE1_PARITY_PREFILL_PLANNED_EXPERT_SLOTS='1', VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE='1',
        VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP='1',
        VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE='1', VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP='0',
        VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP_KIND='mc2_dispatcher_only',
        VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP='0', VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE='1')
    import torch
    import torch_npu  # noqa:F401
    import vllm
    import vllm_ascend
    from vllm_ascend.patch import platform, worker  # noqa:F401
    from r1_ascend.vllm_parallel_state import init_parallel_state
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import get_ep_group
    from vllm_ascend import envs as ascend_envs
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local); torch.distributed.init_process_group('hccl'); init_parallel_state(1)
    control = torch.distributed.new_group(backend='gloo')
    owned = c['policies'][args.policy]['rank_to_prompt_ids'][str(rank)]
    rows = [r for r in data['requests'] if r['id'] in owned]
    if len(rows) != 32:
        raise ValueError('each rank must own two real prompts and32 responses')
    max_context = math.ceil(max(len(r['prompt_token_ids']) + r['max_tokens'] for r in data['requests']) / 128) * 128
    started = time.time()
    llm = LLM(model=str(args.hf_root / 'model'), load_format='safetensors', dtype='bfloat16',
        tensor_parallel_size=1, enable_expert_parallel=True, distributed_executor_backend='external_launcher',
        enable_sleep_mode=False, enforce_eager=True, gpu_memory_utilization=.9, disable_custom_all_reduce=True,
        max_model_len=max_context, max_num_seqs=32, max_num_batched_tokens=max_context, block_size=128,
        num_gpu_blocks_override=target // 128, enable_prefix_caching=False, enable_chunked_prefill=False,
        additional_config=dict(torchair_graph_config={'enabled': False},
            ascend_scheduler_config={'enabled': True, 'enable_chunked_prefill': False},
            refresh=True, elastic_moe_mode=ascend_envs.VLLM_ASCEND_ELASTIC_MOE_MODE,
            init_redundancy_expert=ascend_envs.compute_elastic_init_redundancy_expert(128, 16, 0), dynamic_eplb=False))
    if list(get_ep_group().ranks) != list(range(16)):
        raise ValueError('wrong initial EP world')
    llm.llm_engine.engine_core.engine_core.resize_kv_cache_for_mode1_step(target, 4)
    core = llm.llm_engine.engine_core.engine_core
    actual = min(int(core._kv_cache_token_capacity(config)) for config in core.kv_cache_configs)
    if actual < target:
        raise ValueError('actual allocated KV smaller than declared budget')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outpath = args.output_dir / f'rank_{rank:02d}.json'
    if outpath.exists():
        raise ValueError('refusing existing rank result')
    torch.distributed.barrier(group=control)
    clock = [time.time() if rank == 0 else None]
    torch.distributed.broadcast_object_list(clock, src=0, group=control)
    t0 = clock[0]
    events = []; original_rpc = llm.llm_engine.engine_core.collective_rpc
    def observed_rpc(method, *a, **kw):
        begin = time.time(); result = original_rpc(method, *a, **kw); end = time.time()
        if method == 'rebuild_elastic_ep_group':
            target_ranks = (kw.get('args') or a[0])[0]
            events.append(dict(reporter=rank, active_ranks=list(target_ranks), start=begin, done=end))
        return result
    llm.llm_engine.engine_core.collective_rpc = observed_rpc
    params = [SamplingParams(n=1, temperature=.9, top_k=50, top_p=.9,
                            max_tokens=r['max_tokens'], ignore_eos=True, seed=r['request_seed']) for r in rows]
    outputs = llm.generate([{'prompt_token_ids': r['prompt_token_ids']} for r in rows], sampling_params=params, use_tqdm=False)
    ended = time.time(); results = []
    if len(outputs) != len(rows):
        raise ValueError('missing response artifacts')
    for row, out in zip(rows, outputs):
        if len(out.outputs) != 1 or len(out.outputs[0].token_ids) != row['max_tokens']:
            raise ValueError('forced response length did not match declared work')
        sample = out.outputs[0]
        if sample.finish_reason != 'length':
            raise ValueError('unexpected non-cap termination')
        results.append(dict(id=row['id'], prompt_hash=row['prompt_hash'], sample_index=row['sample_index'],
                            request_seed=row['request_seed'], generated_tokens=len(sample.token_ids),
                            finish_reason=sample.finish_reason))
    torch.distributed.barrier(group=control)
    restore_start = time.time(); llm.llm_engine.restore_elastic_parallel_groups_if_needed(); restore_end = time.time()
    if list(get_ep_group().ranks) != list(range(16)):
        raise ValueError('Full16 restore not completed')
    payload = dict(status='CONTROLLED_LENGTH_ARM_FINISHED', policy=args.policy, rank=rank,
                   start=t0, generation_end=ended, prepared_startup_s=t0-started,
                   restore_start=restore_start, restore_end=restore_end, events=events,
                   kv_target_tokens=target, kv_actual_tokens=actual, requests=results,
                   workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
                   weight_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                   code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   runtime_versions=dict(python=sys.version, torch=str(torch.__version__),
                       torch_npu=str(getattr(torch_npu, '__version__', 'unknown')),
                       vllm=str(getattr(vllm, '__version__', 'unknown')),
                       vllm_source=str(Path(vllm.__file__).resolve()),
                       ascend_source=str(Path(vllm_ascend.__file__).resolve())),
                   scope='Controlled length inference only; no actor updates, rewards, sidecar or population claim. Startup excluded from generation; restore recorded separately.')
    outpath.write_text(json.dumps(payload, sort_keys=True) + '\n')
    torch.distributed.barrier(group=control); torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
