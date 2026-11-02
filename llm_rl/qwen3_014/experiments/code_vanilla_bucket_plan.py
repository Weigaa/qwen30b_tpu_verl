#!/usr/bin/env python3
"""Supply the real chat tokenizer only to Vanilla's offline history mapper.

The immutable planner/DP, dataset, history, assignment and timed model path
are unchanged. Multi-message code prompts cannot use the math-only fallback
string representation to match decoded epoch0 history keys.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

SOURCE = Path('/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main/tools/build_vanilla_aclgraph_bucket_plan.py')


def attach_tokenizer(module, tokenizer):
    original = module._map_stats_to_dataset
    def mapped(*args, **kwargs):
        if kwargs.get('prompt_tokenizer') is not None:
            raise RuntimeError('Unexpected second tokenizer in vanilla mapper')
        return original(*args, **dict(kwargs,prompt_tokenizer=tokenizer))
    module._map_stats_to_dataset = mapped


def main():
    if Path(sys.argv[1]).resolve() != SOURCE.resolve():
        raise RuntimeError('Unexpected vanilla planner interception')
    model = os.environ.get('FAIR_CODE_CHAT_TOKENIZER')
    if model != '/data/Qwen3-30B-A3B':
        raise RuntimeError('Explicit original code-task tokenizer required')
    sys.path.insert(0,str(SOURCE.parent))
    spec = importlib.util.spec_from_file_location('code_vanilla_original_bucket_planner',SOURCE)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model,trust_remote_code=True)
    attach_tokenizer(module,tokenizer)
    print(json.dumps(dict(event='offline_vanilla_multiturn_history_mapping',tokenizer=model,
        original=str(SOURCE),original_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        dataset_and_DP_unchanged=True)),flush=True)
    sys.argv = [str(SOURCE),*sys.argv[2:]]
    module.main()
    output = Path(sys.argv[sys.argv.index('--output')+1])
    train = Path(sys.argv[sys.argv.index('--train-file')+1])
    plan = json.loads(output.read_text())
    ids = [i for row in plan for rank in range(16) for i in row['rank_to_dataset_item_idx'][str(rank)]]
    if (len(plan) != 5 or [row['step'] for row in plan] != list(range(1,6))
            or ids != list(range(160)) or any(row['aclgraph_bucket_count'] != 8 or
                len(row['aclgraph_capture_sizes']) != 8 for row in plan)):
        raise RuntimeError('Vanilla fixed160 input order/5steps/K8 bucket contract changed')
    receipt = dict(status='ALL160_VANILLA_CHAT_HISTORY_BUCKETS_AUDITED_CPU_ONLY',
        plan_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
        tokenizer=model,original_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unchanged_prompt_order=True,steps=5,prompts=160,scope='Offline CPU history-key normalization and unchanged K8 bucket DP; not NPU performance or quality.')
    output.with_suffix('.audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':main()
