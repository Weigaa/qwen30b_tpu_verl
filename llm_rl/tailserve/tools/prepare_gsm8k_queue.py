#!/usr/bin/env python3
"""Queue a fixed prefix of GSM8K, without loading model weights or selecting by length."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from tailserve_runtime.policy import Config
from tailserve_runtime.queue import append_record, validate_job


def prepare(rows, tokenizer, root, config, world_size=16, max_tokens=16384):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f'refusing to mix requests into existing spool: {root}')
    # Validate every job before writing anything. Never truncate or silently skip.
    prepared = []
    for index, row in enumerate(rows):
        tokens = list(tokenizer.apply_chat_template(row['prompt'], tokenize=True,
                      add_generation_prompt=True, enable_thinking=False))
        job = validate_job(dict(id=f'gsm8k_{index:05d}', rank=index % world_size,
            prompt_token_ids=tokens, max_tokens=max_tokens, policy_version='current'), config, world_size)
        prepared.append((job, dict(id=job['id'], source='openai/gsm8k', source_row=index,
            uid=row.get('uid'), ground_truth=row['reward_model']['ground_truth'],
            prompt_tokens=len(tokens), rank=job['rank'], max_tokens=max_tokens,
            chat_template_enable_thinking=False)))
    for job, manifest in prepared:
        append_record(root/'queues'/f'rank_{job["rank"]}.jsonl', job, timestamp=False)
        append_record(root/'gsm8k_manifest.jsonl', manifest, timestamp=False)
    return len(prepared)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spool', required=True)
    p.add_argument('--dataset', default='/data/gsm8k/train.parquet')
    p.add_argument('--tokenizer', default='/data/Qwen3-30B-A3B')
    p.add_argument('--count', type=int, default=256)
    p.add_argument('--max-tokens', type=int, default=Config.from_env().max_new_tokens)
    args = p.parse_args()
    if args.count <= 0:
        p.error('count must be positive')
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    table = pq.read_table(args.dataset)
    if args.count > len(table):
        p.error('count exceeds dataset size')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    count = prepare(table.slice(0, args.count).to_pylist(), tokenizer, args.spool,
                    Config.from_env(), max_tokens=args.max_tokens)
    digest = hashlib.sha256()
    with open(args.dataset, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    root=Path(__file__).resolve().parents[1]
    source_files=list((root/'tailserve_runtime').glob('*.py')) + [root/p for p in [
        'vllm/v1/core/sched/scheduler.py','vllm/entrypoints/llm.py',
        'vllm_ascend/worker/worker.py','vllm_ascend/worker/worker_v1.py',
        'vllm_ascend/worker/model_runner_v1.py',
        'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
        'run_tailserve.sh','run_deepscaler_gsm8k.sh']]
    implementation={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    append_record(Path(args.spool)/'source.jsonl', dict(dataset=str(Path(args.dataset).resolve()),
        sha256=digest.hexdigest(), selection=f'first {count} rows in original order',
        tokenizer=args.tokenizer, train_use=False, max_tokens=args.max_tokens,
        config=asdict(Config.from_env()), implementation_sha256=implementation), timestamp=False)
    print(f'Queued {count} GSM8K requests in {args.spool}; inference only, no RL labels mixed in')


if __name__ == '__main__':
    main()
