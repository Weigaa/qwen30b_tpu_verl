#!/usr/bin/env python3
"""Interim same-checkpoint first-step evidence, never relabelled a full epoch."""
import hashlib
import json
import re
import statistics
from eager_controls import ROOT,GRAPH,records,suite


def main():
    actual=next((ROOT/'eager_controls/seed_0/vanilla/fig6').glob('epoch_001_*'))
    reference=next((GRAPH/'seed_0/vanilla/fig6').glob('epoch_001_*'))
    a=list(records(actual/'rollout_data/1.jsonl'));b=list(records(reference/'rollout_data/1.jsonl'))
    if len(a)!=512 or len(b)!=512 or [r['input'] for r in a]!=[r['input'] for r in b]:
        raise RuntimeError('First-step actual inputs/count differ')
    if [r['gts'] for r in a]!=[r['gts'] for r in b]:raise RuntimeError('Ground truths changed')
    output={};vectors=[]
    for label,epoch,rows in [('eager',actual,a),('graph',reference,b)]:
        vector=list(map(int,(epoch/'rollout_length/length_1.txt').read_text().split()))
        if len(vector)!=512 or any(n<0 or n>16384 for n in vector):raise RuntimeError('Invalid response lengths')
        vectors.append(vector)
        log=next((epoch/'logs').glob('*.txt'));text=log.read_text(errors='replace')
        metric=next(line for line in text.splitlines() if re.search(r'step:1 - ',line))
        fields=dict(re.findall(r'([\w/]+):([^ ]+)',metric));seconds=float(fields['rollout/rollout_output_time_s'])
        audits=[json.loads(s) for s in re.findall(r'MODE1_STEP_RUNTIME_AUDIT (\{[^\n]+\})',text)]
        audits=[r for r in audits if r['step']==1]
        if (len(audits)!=16 or {r['rank'] for r in audits}!=set(range(16)) or
                any(not r['complete'] or r['kv_tokens']!=380800 or r['preemptions'] for r in audits)):
            raise RuntimeError('Missing actual16-rank first-step/budget evidence')
        if label=='eager':
            if any(r.get('full_iterations',0)!=0 or r.get('eager_iterations',0)<=0 for r in audits):
                raise RuntimeError('Not actual eager')
        elif any(r.get('full_iterations',0)<=0 for r in audits):raise RuntimeError('Not actual FULL')
        output[label]=dict(epoch_dir=str(epoch),seconds=seconds,response_tokens=sum(vector),
            response_tokens_s=sum(vector)/seconds,reward=statistics.mean(float(r['score']) for r in rows),
            lr=fields['actor/lr'],grad_norm=fields['actor/grad_norm'],actual_rank_audits=16)
    report=dict(status='INTERIM_FIRST_STEP_ONLY_NOT_FULL_EPOCH',seed=0,
        source_execution_sha256=suite().execution_digest(),actual_inputs_and_GTs_equal=True,
        identical_output_text=sum(r['output']==s['output'] for r,s in zip(a,b)),
        response_length_vectors_equal=vectors[0]==vectors[1],primary=output,
        graph_throughput_gain_pct=100*(output['graph']['response_tokens_s']/output['eager']['response_tokens_s']-1),
        scope='Same common actor before any new update, seed0/epoch1/step1, exact historical source, KV380800, CPU metadata0 in both. Eager/TQ2 versus FULL/TQ1 configured paths, not pure graph-only toggle. No full-epoch result/CI/convergence claim. Native sleep2/lifecycle1 in both.')
    destination=ROOT/'eager_controls/seed_0/vanilla/INTERIM_FIRST_STEP_RESULT.json'
    destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
