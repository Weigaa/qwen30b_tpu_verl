"""Host-only launcher checks: does not import torch, vLLM, or initialize NPUs."""
from dataclasses import asdict
import json
import os
from pathlib import Path
from .policy import Config


def main():
    root=Path(__file__).resolve().parents[1]
    assert "'0.14.1'" in (root/'vllm/_version.py').read_text()
    flags={'VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK':'0',
           'VLLM_ASCEND_ELASTIC_EXECUTION_MODE':'0','VLLM_EPOCH_LENGTH_REGROUP_ENABLE':'0',
           'VLLM_ROLLOUT_EARLY_STOP_ENABLE':'0','VERL_SIDECAR_ENABLE':'0',
           'VLLM_ROLLOUT_ASYNC_SCHEDULING':'false','VLLM_ROLLOUT_SLEEP_LEVEL':'2',
           'VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE':'0'}
    for k,v in flags.items():
        if os.getenv(k)!=v:raise ValueError(f'{k} must be {v}')
    if os.getenv('TAILSERVE_ENABLE') not in ('0','1'):raise ValueError('invalid enable flag')
    if not os.getenv('TAILSERVE_SPOOL') or not os.getenv('TAILSERVE_RUN_ID'):
        raise ValueError('spool and run identity are required')
    print(json.dumps(dict(title='TailServe: In-Place Opportunistic Serving in MoE RL Rollout Tails',
        root=str(root),vllm='0.14.1',mode=os.environ['TAILSERVE_MODE'],config=asdict(Config.from_env()),
        primary_limits=dict(max_prompt_length=int(os.environ['MAX_PROMPT_LENGTH']),
            max_response_length=int(os.environ['MAX_RESPONSE_LENGTH']),
            max_num_batched_tokens=int(os.environ['ROLLOUT_MAX_NUM_BATCHED_TOKENS'])),
        native_sleep=2,topology='fixed EP16',serving='shared current model and engine',
        spool=os.environ['TAILSERVE_SPOOL'],run_id=os.environ['TAILSERVE_RUN_ID'],
        npu_test_status='NOT_RUN',async_scheduler_supported=False),indent=2))


if __name__=='__main__':main()
