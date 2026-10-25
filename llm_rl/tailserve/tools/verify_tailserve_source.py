#!/usr/bin/env python3
"""Read-only provenance check against the current local v0.14 base tree."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'qwen3'
EXPECTED={
    'vllm/entrypoints/llm.py',
    'vllm/v1/core/sched/scheduler.py',
    'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
    'verl/trainer/constants_ppo.py',
    'vllm_ascend/worker/worker_v1.py',
    'vllm_ascend/worker/worker.py',
    'vllm_ascend/worker/model_runner_v1.py',
}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if not BASE.is_dir():raise SystemExit('local original qwen3 tree is required for this provenance check')
    changed=[];checked=0;missing=[]
    for tree in ['vllm','vllm_ascend','verl','megatron','mindspeed','patches','r1_ascend','internal','tools']:
        for source in (BASE/tree).rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts or source.suffix=='.pyc':continue
            rel=source.relative_to(BASE);target=ROOT/rel
            if not target.is_file():missing.append(str(rel));continue
            checked+=1
            if sha(source)!=sha(target):
                changed.append(dict(file=str(rel),base_sha256=sha(source),tailserve_sha256=sha(target)))
    if missing or {r['file'] for r in changed}!=EXPECTED:
        raise SystemExit(json.dumps(dict(missing=missing,unexpected_changes=changed),indent=2))
    print(json.dumps(dict(status='PASS',base=str(BASE),target=str(ROOT),vllm='0.14.1',
        inherited_files_checked=checked,modified_integration_files=changed,
        original_tree_not_written_by_this_check=True,
        npu_experiment_status='not assessed by source comparison'),indent=2))


if __name__=='__main__':main()
