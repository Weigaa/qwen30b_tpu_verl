#!/usr/bin/env python3
"""Prespecify256 valid disjoint math questions before looking at model scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def plain(value):
    if hasattr(value,'tolist'):return plain(value.tolist())
    if isinstance(value,list):return [plain(x) for x in value]
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    return value


def canonical(prompt):
    return json.dumps([{**m,'content':' '.join(m['content'].split())} for m in plain(prompt)],
                      sort_keys=True,ensure_ascii=False,separators=(',',':'))


def choose(candidates,n=256):
    if n%4 or len(candidates)<n:raise ValueError('Need enough candidates and four balanced strata')
    ordered=sorted(candidates,key=lambda r:(r['prompt_tokens'],r['prompt_sha256']))
    selected=[]
    for i in range(4):
        group=ordered[i*len(ordered)//4:(i+1)*len(ordered)//4]
        ranked=sorted(group,key=lambda r:hashlib.sha256(('mlsys-heldout-seed0:'+r['prompt_sha256']).encode()).hexdigest())
        selected.extend(dict(r,stratum=i) for r in ranked[:n//4])
    if len(selected)!=n or len({r['prompt_sha256'] for r in selected})!=n:raise RuntimeError('Bad held-out sample')
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise RuntimeError('Refusing to overwrite prespecified sample')
    import pandas as pd
    from transformers import AutoTokenizer
    train=Path('/data/deepscaler/train.parquet');test=Path('/data/deepscaler/test.parquet')
    tokenizer_path=Path('/data/Qwen3-30B-A3B')
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=False,local_files_only=True)
    training=pd.read_parquet(train,columns=['prompt'])
    train_keys={canonical(x) for x in training['prompt']}
    frame=pd.read_parquet(test);seen=set();reasons=Counter();candidates=[]
    for i,row in frame.iterrows():
        truth=row['reward_model'].get('ground_truth')
        if truth is None or str(truth).strip().lower() in ('','nan','none'):
            reasons['empty_ground_truth']+=1;continue
        key=canonical(row['prompt'])
        if key in train_keys:reasons['overlap_with_local_RL_train']+=1;continue
        if key in seen:reasons['duplicate_test_prompt']+=1;continue
        seen.add(key);messages=plain(row['prompt'])
        tokens=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        if len(tokens)>1024:reasons['prompt_over_1024_tokens']+=1;continue
        candidates.append(dict(source_index=int(i),prompt_sha256=hashlib.sha256(key.encode()).hexdigest(),
            prompt_tokens=len(tokens),expected_decoded_input=tokenizer.decode(tokens,skip_special_tokens=True),
            ground_truth=plain(truth),data_source=str(row['data_source'])))
    selected=choose(candidates);output=frame.iloc[[r['source_index'] for r in selected]].copy()
    output['uid']=[r['prompt_sha256'] for r in selected]
    args.output.mkdir(parents=True)
    parquet=args.output/'heldout256.parquet';output.to_parquet(parquet,index=False)
    def sha(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    manifest=dict(status='HELDOUT_SUBSET_PRESPECIFIED_NOT_EVALUATED',questions=256,seed=0,
        raw_test_rows=len(frame),eligible_candidates=len(candidates),excluded=dict(reasons),
        sampling='Four prompt-token-length quartiles,64 fixed-hash-selected questions each; not task-name strata',
        data_source_counts=dict(Counter(r['data_source'] for r in selected)),
        disjoint_scope='Normalized exact prompts excluded against the entire local RL train split, not a pretraining-contamination certificate',
        inference_contract=dict(n=1,do_sample=False,temperature=0,max_response_length=16384,
            max_prompt_length=1024,validation_shuffle=False,serving='same nonelastic eager/TQ2 backend for all endpoints'),
        examples=selected,input_sha256={str(x):sha(x) for x in (train,test,parquet,
            tokenizer_path/'tokenizer_config.json',Path(__file__))},
        claim_limits='Single-seed short-training endpoint quality on this256-question subset; not full-benchmark accuracy, training-seed variance, equivalence or final convergence.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('status','questions','eligible_candidates','excluded','data_source_counts')}))


if __name__=='__main__':main()
