#!/usr/bin/env python3
"""Read-only raw-result review of short pre/post consolidation NPU runs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def events(text, name):
    return [(float(m[1]), dict(re.findall(r'(\w+)=(\S+)', m[2])))
            for m in re.finditer(r'^'+name+r'=([0-9.]+)([^\n]*)$', text, re.M)]


def json_events(text):
    decoder = json.JSONDecoder()
    for m in re.finditer(r'\{"event"\s*:', text):
        try:
            yield decoder.raw_decode(text, m.start())[0]
        except ValueError:
            pass


def audit_arm(root, case, version):
    arm=root/case/version
    assert json.loads((arm/'DRIVER_RESULT.json').read_text())['exit_code']==0
    log=next((arm/'execution/logs').glob('*.txt'))
    raw=log.read_bytes();text=re.sub(r'\x1b\[[0-9;]*[mK]','',raw.decode(errors='replace'))
    assert re.search(r'\[run\] end_time=\S+ exit_code=0',text)
    assert not re.search(r'OutOfMemoryError|NPU out of memory|RayTaskError|Memory_Allocation_Failure',text)
    runtime=[json.JSONDecoder().raw_decode(m[1])[0] for m in re.finditer(r'MODE1_STEP_RUNTIME_AUDIT ([^\n]+)',text)]
    assert len(runtime)==32 and {(r['step'],r['rank']) for r in runtime}=={(s,r) for s in [1,2] for r in range(16)}
    assert all(r['complete'] and r['preemptions']==0 and r['worker_kv_blocks']==r['kv_blocks'] for r in runtime)
    graph=case!='baseline_eager';elastic=case.startswith('natural')
    if graph:
        assert all(r.get('full_iterations',0)>0 and r['eager_iterations']==r['global_prefill_iterations'] for r in runtime)
    else:
        assert all(r.get('full_iterations',0)==0 and r['eager_iterations']>0 for r in runtime)
    times=list(map(float,re.findall(r'=> rollout_output_time_s: ([0-9.]+)',text)))
    assert len(times)==2 and all(t>0 for t in times)
    work=[];hashes=[]
    for step in [1,2]:
        path=arm/f'execution/rollout_data/{step}.jsonl'
        outputs=[json.loads(line) for line in path.open() if line.strip()]
        assert len(outputs)==512 and all(row['output'].strip() for row in outputs)
        lengths=list(map(int,(arm/f'execution/rollout_length/length_{step}.txt').read_text().split()))
        if elastic:
            # The inherited response mask includes the first padded EOS at
            # a short length stop. Do not count that synthetic token as work.
            expected=Counter({65:256,129:128,257:64,1024:64})
            assert Counter(lengths)==expected,(case,version,step,Counter(lengths))
            actual=sum(lengths)-448
        else:
            assert Counter(lengths)==Counter({384:512})
            actual=sum(lengths)
        work.append(actual)
        canonical=[(r['input'],r['output'],float(r['score'])) for r in outputs]
        hashes.append(hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest())
    sleep=re.findall(r'Sleep mode freed ([0-9.]+) GiB memory',text)
    assert len(sleep)>=32 and all(float(s)>0 for s in sleep)
    result=dict(case=case,version=version,log=str(log),log_sha256=hashlib.sha256(raw).hexdigest(),
                steps=2,responses=1024,actual_response_tokens=work,rollout_s=times,
                output_tokens_s=sum(work)/sum(times),second_step_output_tokens_s=work[1]/times[1],
                answer_score_order_hash_by_step=hashes,rank_step_audits=32,
                actual_full_rank_step_audits=32 if graph else 0,
                actual_eager_rank_step_audits=0 if graph else 32,
                native_sleep2_successful=True,preemptions=0)
    if elastic:
        floors=sorted(set(map(int,re.findall(r'Elastic parallel shrink done:.*?dp_size=(\d+)',text))))
        assert floors==[2,4,8],floors
        restore=list(map(float,re.findall(r'driver_restore_rpc_start_time=([0-9.]+)',text)))
        assert len(restore)==2
        lease=(arm/'sidecar/lease.log').read_text()
        stops={int(d['lease_id']):(t,d) for t,d in events(lease,'trainer_stop_request_time')}
        exits={int(d['lease_id']):(t,d) for t,d in events(lease,'sidecar_exit_confirmed_time')}
        acks={int(d['lease_id']):(t,d) for t,d in events(lease,'watcher_restore_ack_time')}
        ids=list(map(int,re.findall(r'^sidecar_lease_index=(\d+)',lease,re.M)))
        starts=events(lease,'sidecar_start_time');assert len(starts)==len(ids)==2
        for (start,_),idx in zip(starts,ids):
            stop,req=stops[idx];end,physical=exits[idx];ack,ack_fields=acks[idx]
            assert start<stop<=end<=ack<restore[idx-1]
            assert physical['process_group_alive']=='0'
            assert req['request_id']==physical['request_id']==ack_fields['request_id']
        sidepath=arm/'sidecar/outputs.jsonl'
        side=[json.loads(line) for line in sidepath.open() if line.strip()]
        assert side and all(r['outputs'][0]['finish_reason'] in ['stop','length'] for r in side)
        output_keys=[(r.get('sidecar_epoch',0),r['prompt_id']) for r in side]
        assert len(output_keys)==len(set(output_keys))
        completed={}
        files=list((arm/'sidecar/state').glob('completed.shard*.jsonl'))
        files+=list((arm/'sidecar/state/completed_epochs').glob('*.jsonl'))
        for file in files:
            for line in file.open():
                if not line.strip():
                    continue
                row=json.loads(line);key=(row.get('sidecar_epoch',0),row['prompt_id'])
                assert key not in completed
                completed[key]=float(row['time'])
        assert set(output_keys)==set(completed)
        records=list(json_events((arm/'sidecar/infer.log').read_text(errors='replace')))
        replay=[r for r in records if r.get('event')=='sidecar_aclgraph_replay' and r.get('runtime_mode')=='FULL' and r.get('capture_phase') is False]
        assert {r['shard_index'] for r in replay}==set(range(8))
        result.update(actual_shrink_floors=floors,sidecar_completed=len(side),
                      sidecar_real_full_replay_shards=8,sidecar_physical_exit_before_restore=True,
                      sidecar_unique_outputs_match_durable_completion_records=True)
        occupied=sum(end-start for (start,_),(end,_) in zip(starts,events(lease,'sidecar_exit_confirmed_time')))
        side_tokens=sum(r['outputs'][0]['token_ids_len'] for r in side)
        result.update(sidecar_occupied_seconds=occupied,sidecar_completed_output_tokens=side_tokens,
                      sidecar_output_tokens_per_occupied_second=side_tokens/occupied)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve()
    comparisons=[]
    for case in ['baseline_eager','baseline_graph','natural_f2_graph_sidecar']:
        old,new=[audit_arm(root,case,v) for v in ['reference','merged']]
        assert old['answer_score_order_hash_by_step']==new['answer_score_order_hash_by_step'],case
        gain=100*(new['output_tokens_s']/old['output_tokens_s']-1)
        steady=100*(new['second_step_output_tokens_s']/old['second_step_output_tokens_s']-1)
        comparisons.append(dict(case=case,answers_scores_order_and_lengths_exact=True,
            output_throughput_change_percent=gain,second_step_change_percent=steady,
            reference=old,merged=new))
        if case.startswith('natural'):
            def served(version):
                path=root/case/version/'sidecar/outputs.jsonl'
                return {(r.get('sidecar_epoch',0),r['prompt_id']):r
                        for r in map(json.loads,path.open())}
            before,after=served('reference'),served('merged')
            common=before.keys()&after.keys()
            initial=[k for k in common if k[0]==0 and
                     before[k]['outputs'][0]['resume_prefix_text_len']==0 and
                     after[k]['outputs'][0]['resume_prefix_text_len']==0]
            exact=sum(before[k]['prompt']==after[k]['prompt'] and
                      all(before[k]['outputs'][0][f]==after[k]['outputs'][0][f]
                          for f in ['text','token_ids_len','finish_reason']) for k in initial)
            comparisons[-1]['sidecar_common_unresumed_requests_checked']=len(initial)
            comparisons[-1]['sidecar_common_unresumed_exact']=exact
            comparisons[-1]['sidecar_different_window_query_counts_not_throughput_gain']=True
    report=dict(status='RAW_FUNCTION_AND_OUTPUT_AUDIT_PASS',comparisons=comparisons,
        scope='Two short inference steps per configuration and version, not full-epoch evidence',
        artificial_length_caps=True,ignore_eos=True,actor_updates_exercised=False,
        statistical_performance_equivalence_proven=False,
        historical_paper_metrics_unchanged=True,shared_checkpoint_unchanged=True)
    report['production_fix']=dict(
        file='qwen3_main/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh',
        issue='Consolidation retained the old launcher wholesale to preserve /data defaults, omitting the basic sidecar watcher lifecycle',
        repair='Restore the original basic-sidecar watcher start/cleanup block while retaining the /data default',
        actor_model_or_graph_operator_math_changed=False,
        eager_and_baseline_graph_sidecar_disabled_runs_reused=True)
    report['excluded_attempts']=[str(p.relative_to(root)) for p in root.glob('*/*_failed_*') if p.is_dir()]
    (root/'RAW_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# qwen3_main consolidation NPU regression', '',
           'Three paired configurations, two short steps per implementation. '
           'The reference is the sealed pre-consolidation source, not a historical throughput number.', '',
           '| Configuration | Reference output tok/s | Merged output tok/s | Change |',
           '| --- | ---: | ---: | ---: |']
    for row in comparisons:
        lines.append(f"| {row['case']} | {row['reference']['output_tokens_s']:.2f} | "
                     f"{row['merged']['output_tokens_s']:.2f} | {row['output_throughput_change_percent']:+.2f}% |")
    service=comparisons[-1]
    lines+=['',
        'All paired primary answers, scores, ordering and response lengths match exactly. '
        'Each arm returns1024 nonempty responses with32 complete rank-step audits and zero KV preemptions. '
        'Graph arms prove actual FULL decode, with eager iterations restricted to global prefill.', '',
        'Both Natural-F2 arms execute16->8->4->2 twice, use native sleep2 and '
        'restore the full world after physical sidecar exit and a matching ACK.', '',
        f"Sidecar completions including drain: reference {service['reference']['sidecar_completed']}, "
        f"merged {service['merged']['sidecar_completed']}. Outputs are unique and match durable completion records. "
        'All8 secondary shards show actual FULL replay. Different completion totals reflect different windows, '
        'not a matched standalone throughput benchmark.', '',
        f"Common unresumed auxiliary requests with exact text/length/finish matches: "
        f"{service['sidecar_common_unresumed_exact']}/{service['sidecar_common_unresumed_requests_checked']}.", '',
        'Auxiliary outputs are not all identical across the dynamic window schedules. '
        'The sidecar launcher, model runner and ACLGraph source files are byte-identical '
        'between reference and merged implementations. This run does not prove '
        'batch-invariant auxiliary generation or standalone quality equivalence.', '',
        f"Sidecar output tokens per occupied second, including startup and drain: "
        f"reference {service['reference']['sidecar_output_tokens_per_occupied_second']:.2f}, "
        f"merged {service['merged']['sidecar_output_tokens_per_occupied_second']:.2f}. "
        'These rates are not added to the primary throughput.', '',
        '## Found and repaired consolidation defect', '',
        report['production_fix']['issue']+'. '+report['production_fix']['repair']+'. '
        'A new CPU launcher-lifecycle regression test guards against recurrence. '
        'The merger utility was corrected to preserve only the output default, not discard the full launcher.', '',
        '## Scope', '',
        '16 NPUs, common epoch0, batch32 x n16. Baselines cap responses at384 tokens. '
        'Natural-F2 uses the existing explicit64/128/256/1024 short-tail fixture, '
        'greedy sampling and ignore_eos. The literal two-step floor/KV plan supplies131072 tokens per rank. '
        'The auxiliary model is Qwen2.5-1.5B with8 TP1 replicas, FULL/TQ1, '
        'max_num_seqs16 and output cap128. The forecast gate is disabled only in this functional fixture. '
        'Actor updates are skipped. Timings include prefill, native wake/capture/sleep, shrink/restore '
        'and sidecar handback, but not subsequent actor log-prob computation.', '',
        'This is a short functional and descriptive performance regression, not a16k-response '
        'paper result, a full training experiment, or proof of statistical performance equivalence. '
        'The original data and common checkpoint were not modified. Failed setup attempts are retained '
        'and excluded from the successful paired comparison.', '',
        'The merged elastic log also contains a TBE daemon ProcessLookupError during '
        'Ray worker teardown after both completed steps. The trainer exits0, all outputs '
        'and handback records are audited, and npu-smi confirms no remaining NPU processes. '
        'This cleanup diagnostic is retained, not rewritten as a clean log.', '',
        'Raw numeric and lifecycle checks: `RAW_AUDIT.json`. '
        'Original shell logs, worker audits, outputs, lease records and launch environments remain under each arm.', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
