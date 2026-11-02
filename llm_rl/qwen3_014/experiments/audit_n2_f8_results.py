#!/usr/bin/env python3
"""Recompute the new-budget campaign evidence; partial state is never PASS.

No inference or grading retries. Reads original logs, saved outputs and the
one-time judge results. Writes only a separate experiment report, not TeX.
"""
import argparse
import json
import math
from pathlib import Path
import re
from statistics import mean,stdev

import run_n2_f8_campaign as campaign
from rank_occupancy import parse_epoch as occupancy


def require(condition,message):
    if not condition:raise RuntimeError(message)


def launch_budget(path):
    value=json.loads(path.read_text());env=value['env']
    require(env['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8']=='349952','Wrong new F8 runtime budget')
    require(env.get('FLOOR_KV_CAPS')==campaign.CAPS,'Missing/changed full floor capacity map')
    for key,wanted in [('FLOOR2','131072'),('FLOOR4','280576'),('FLOOR16','380800')]:
        require(env['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_'+key]==wanted,'Wrong '+key+' capacity')
    require(env['TASK_QUEUE_ENABLE']=='1' and env['VLLM_ROLLOUT_SLEEP_LEVEL']=='2' and
            env['VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE']=='1' and
            env['ADAFLOOR_ACLGRAPH_MODE']=='FULL_DECODE_ONLY','Execution path changed')
    return value


def epoch_occupancy(suite,epoch):
    caps=suite.tailguard_caps(epoch);work={}
    for step in range(1,6):
        values=list(map(int,(epoch/f'rollout_length/length_{step}.txt').read_text().split()))
        work[step]=suite.validate_response_lengths(values,caps[step],True)[0]
    return occupancy(epoch,work,'FULL')


def actual_kv_budget(suite,epoch):
    plans=json.loads((epoch/'oracle/length_sorted_rank_plan.json').read_text())
    budgets={int(k):int(v) for k,v in (pair.split(':') for pair in campaign.CAPS.split(','))}
    require(len(plans)==5 and {int(p['step']) for p in plans}==set(range(1,6)),
            'Not five actual new-budget plans')
    caps={int(p['step']):int(p['kv_cap']) for p in plans}
    require(all(int(p['kv_cap'])==budgets[int(p['selected_floor'])] for p in plans),
            'Planned KV is not the new N2 capacity family')
    audits=suite.json_records(epoch,'MODE1_STEP_RUNTIME_AUDIT ')
    require(len(audits)==80 and {(a['step'],a['rank']) for a in audits}==
            {(step,rank) for step in range(1,6) for rank in range(16)},
            'Actual KV budget lacks five-step/16-rank coverage')
    require(all(a['kv_tokens']==caps[a['step']] for a in audits),
            'Runtime/planned KV differs')


def combined_efficiency(epochs):
    """Paper convention: pool W and H across epochs BEFORE taking W/H."""
    work=sum(e['response_tokens'] for e in epochs)
    held=[sum(e['occupied_rank_s_bounds'][i] for e in epochs) for i in (0,1)]
    require(work>0 and 0<held[0]<=held[1],'Invalid pooled response/rank-time evidence')
    return [work/held[1],work/held[0]]


def worker_shrink_groups(text):
    """Actual worker completions, not the optional driver's RPC log prefix."""
    groups={}
    for match in re.finditer(r'Elastic parallel shrink done: rank=(\d+) active_ranks=(\[[^\]]+\]) dp_size=(\d+)',text):
        rank=int(match[1]);active=tuple(json.loads(match[2]));width=int(match[3])
        require(len(active)==width and len(set(active))==width and rank in active,
                'Worker shrink completion has inconsistent active topology')
        groups.setdefault(active,set()).add(rank)
    return [dict(active_ranks=list(active),floor=len(active),
                 worker_completion_ranks=sorted(ranks))
            for active,ranks in sorted(groups.items(),key=lambda row:(-len(row[0]),row[0]))]


def math_recovery_evidence(root,arm,result):
    driver=json.loads((arm/'DRIVER_RESULT.json').read_text())
    if driver['exit_code']==0:return None
    require(result['seed']==3,'Unexpected failed math child outside diagnosed seed3 incident')
    from recover_n2_seed3_epoch2 import require_preexecution_failure,resume_script,natural_resume_script
    stage=arm/'ray_startup_recovery_v2'
    proof=json.loads((stage/'RECOVERY_PROTOCOL.json').read_text())
    receipt=json.loads((stage/'RECOVERY_PASS.json').read_text())
    require(receipt['status']=='ORIGINAL_EPOCH1_AND_RECOVERED_EPOCH2_VALIDATED',
            'Failed original driver lacks validated epoch2 recovery')
    require(json.loads((stage/'DRIVER_RESULT.json').read_text())['exit_code']==0,
            'Recovered fair runner failed')
    require(proof['failed_driver_receipt']==driver,'Original failed receipt was modified')
    logs=list((Path(proof['original_failed_epoch'])/'logs').glob('*.txt'))
    require(len(logs)==1 and campaign.sha(logs[0])==proof['failed_log_sha256'],
            'Original failure log no longer preserved')
    require_preexecution_failure(logs[0].read_text())
    require(result['epochs'][0]==proof['original_epoch1'],'Recovery altered completed epoch1')
    old_plan=Path(proof['original_failed_epoch'])/'oracle'
    new_plan=arm/'fig6/epoch_002_mode1_natural/oracle'
    for name in ('length_sorted_rank_plan.json','length_sorted_rank_plan_summary.json',
                 'length_sorted_length_oracle.json'):
        require(json.loads((old_plan/name).read_text())==json.loads((new_plan/name).read_text()),
                'Recovered epoch2 plan differs from original pre-execution plan: '+name)
    import pyarrow.parquet as pq
    require(pq.read_table(old_plan/'length_sorted_train.parquet').to_pylist()==
            pq.read_table(new_plan/'length_sorted_train.parquet').to_pylist(),
            'Recovered epoch2 ordered prompt inputs differ')
    launch_budget(stage/'LAUNCH.json')
    original=campaign.MAIN/'run_paper_fair_epoch1_2_from_common_epoch0.sh'
    storage=Path('/workspace/adafloor_graph_results')/(root.name+'_epoch2_math_seed3')
    script=stage/'resume_fair_natural_epoch2.sh'
    require(campaign.sha(original)==proof['original_fair_runner_sha256'] and
            campaign.sha(script)==proof['host_resume_snapshot_sha256'] and
            script.read_text()==resume_script(original.read_text(),campaign.MAIN,storage,arm/'fig6'),
            'Recovered fair runner changed more than audited startup adapter')
    natural=campaign.MAIN/'run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh'
    snapshot=stage/'natural_wrapper_colon_history.sh'
    require(campaign.sha(natural)==proof['original_natural_wrapper_sha256'] and
            campaign.sha(snapshot)==proof['host_natural_snapshot_sha256'] and
            snapshot.read_text()==natural_resume_script(natural.read_text(),campaign.MAIN),
            'Natural resume wrapper changed runtime instead of only history preflight')
    return dict(status=receipt['status'],original_failure_preserved=True,generation_repeated=False,
                exact_original_epoch2_plan_and_ordered_inputs=True)


def math_result(root,seed):
    suite=campaign.main_suite();arm=root/f'primary/seed_{seed}/natural_f2'
    launch_budget(arm/'LAUNCH.json')
    result=suite.parse_arm(root/'primary',seed,'natural_f2')
    require(result==json.loads((arm/'ARM_PASS.json').read_text()),'Math arm receipt differs from raw evidence')
    recovery=math_recovery_evidence(root,arm,result)
    baseline=suite.parse_arm(campaign.OLD,seed,'vanilla')
    own=[];base=[]
    for number,(row,reference) in enumerate(zip(result['epochs'],baseline['epochs']),1):
        epoch=Path(row['epoch_dir'])
        require(row['prompt_hash']==reference['prompt_hash'],'Math prompt multiset differs from Full16 control')
        logs=list((epoch/'logs').glob('*.txt'));text=logs[0].read_text(errors='replace')
        expected=(suite.COMMON/'epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5/actor'
                  if number==1 else arm/'fig6/epoch_001_mode1_natural/checkpoints/qwen3moe_for_eagle3/global_step_5/actor')
        loaded=re.findall(r'\[Rank (\d+)\] Loaded sharded model checkpoint from '+re.escape(str(expected)),text)
        require(len(loaded)==16 and set(map(int,loaded))==set(range(16)),
                'Actual common/own checkpoint load not proved on all16 ranks')
        actual_kv_budget(suite,epoch)
        own.append(epoch_occupancy(suite,epoch))
        old_epoch=Path(reference['epoch_dir'])
        work={}
        for step in range(1,6):
            lengths=list(map(int,(old_epoch/f'rollout_length/length_{step}.txt').read_text().split()))
            work[step]=suite.validate_response_lengths(lengths,16384,True)[0]
        base.append(occupancy(old_epoch,work,'FULL'))
    require(own==json.loads((arm/'RANK_OCCUPANCY.json').read_text())['epochs'],
            'Stored resource accounting differs from recomputation')
    return dict(seed=seed,epochs=result['epochs'],Q=result['response_tokens_s'],reward=result['reward'],
        startup_recovery=recovery,
        E_bounds=combined_efficiency(own),
        baseline_Q=baseline['response_tokens_s'],
        baseline_E_bounds=combined_efficiency(base))


def code_result(root):
    import run_code_natural_f2 as code
    result=json.loads((root/'CODE_N2_RESULT.json').read_text())
    destination=Path('/workspace/adafloor_graph_results')/(root.name+'_code_n2_seed401')
    require(result==json.loads((destination/'CODE_N2_PAIR_RESULT.json').read_text()),'Code result copies disagree')
    launch_budget(destination/'seed_401/natural_f2/LAUNCH.json')
    actual,outputs,selected=code.audit_natural(destination)
    require(json.loads((destination/'PROTOCOL.json').read_text())['floor_kv_caps']==campaign.CAPS,
            'Code planning protocol used wrong capacity family')
    actual_kv_budget(code.old.suite(),destination/'seed_401/natural_f2/fig6/epoch_001_mode1_natural')
    for key,value in actual.items():
        require(result['natural_f2'][key]==value,'Code raw performance mismatch: '+key)
    _,reference,_=code.old.audit_arm('vanilla')
    require(set(outputs)==set(reference),'Code request identity/sampling seed pair differs')
    for key,row in outputs.items():
        require((row['input'],row['gts'])==(reference[key]['input'],reference[key]['gts']),
                'Code prompt/test values changed')
    grades=[json.loads(l) for l in (destination/'natural_f2_quality.jsonl').read_text().splitlines() if l.strip()]
    require(len(grades)==160 and {r['prompt_hash'] for r in grades}==set(selected),'Code fixed160 grading panel changed')
    import hashlib
    for row in grades:
        require(row['sample_index']==0 and row['output_sha256']==hashlib.sha256(
            selected[row['prompt_hash']]['output'].encode()).hexdigest(),'Code grade references wrong answer')
        if row['passed']:
            require(row.get('tests',0)>0 and row.get('returned_tests')==row['tests'] and
                    row.get('test_passes')==row['tests'],'Code full-suite success evidence incomplete')
    require(sum(row['passed'] for row in grades)==result['natural_f2']['correct'],'Code accuracy tally mismatch')
    # Preserve the original completed result verbatim. Add independently
    # derived evidence to this audit report only: the old driver counted a
    # literal "Shrink RPC result:" prefix which this runtime does not emit.
    epoch=destination/'seed_401/natural_f2/fig6/epoch_001_mode1_natural'
    text='\n'.join(p.read_text(errors='replace') for p in (epoch/'logs').glob('*.txt'))
    result['worker_confirmed_shrink_groups']=worker_shrink_groups(text)
    footprint=json.loads((destination/'NATURAL_OCCUPANCY.json').read_text())
    result['actual_reusable_detached_rank_counts_by_step']={str(s['step']):sum(
        r['detached'] and r['reusable_interval_s_bounds'][0]>0 for r in s['ranks'])
        for s in footprint['steps']}
    result['shrink_evidence_note']='Original actual_shrink_rpc_count counts only a literal optional driver log prefix. '
    result['shrink_evidence_note']+='Worker-completed topologies and audited detach/sleep/restore intervals establish actual shrink and resource reuse.'
    return result


def sidecar_result(root):
    helper=campaign.module('audit_new_n2_sidecar',campaign.SIDECAR/'tools/run_v014_sidecar_training.py')
    side=root/'graph_sidecar';report=json.loads((side/'TRAINING_SIDECAR_RESULT.json').read_text())
    metadata=root/'postrun_metadata_recovery/MANIFEST_REPAIR.json'
    if metadata.exists():
        repair=json.loads(metadata.read_text())
        original=Path(repair['original_manifest'])
        before=json.loads(original.read_text())
        after=json.loads((side/'v014_launch.json').read_text())
        contract=json.loads((root/'PROTOCOL.json').read_text())
        pinned=contract['execution_sha256']['qwen3_graph_sidecar_v014']
        require(campaign.sha(original)==repair['original_manifest_sha256'] and
                campaign.sha(side/'v014_launch.json')==repair['repaired_manifest_sha256'],
                'Sidecar metadata repair original/current manifest identity changed')
        require('execution_sha256' not in before and after==dict(before,execution_sha256=pinned)
                and repair['changed_fields']==['execution_sha256']
                and repair['source_protocol_sha256']==campaign.sha(root/'PROTOCOL.json')
                and repair['training_repeated'] is False,
                'Post-run metadata fix changed experiment configuration or lacks original pinned provenance')
        child=json.loads((side/'DRIVER_RESULT.json').read_text())
        require(child==repair['child_result'] and child['exit_code']==0 and child['retained_endpoint'],
                'Sidecar successful child receipt changed during metadata recovery')
    launch=launch_budget(side/'LAUNCH.json');env=launch['env']
    require(env['VERL_SIDECAR_ENFORCE_EAGER']=='0' and env['VERL_SIDECAR_TASK_QUEUE_ENABLE']=='1',
            'Sidecar is not the requested graph path')
    require(float(env['VERL_SIDECAR_WINDOW_GATE_MIN_SECONDS'])==60 and
            env['VERL_SIDECAR_WINDOW_GATE_STRICT_MINIMUM']=='1','Sidecar gate changed')
    primary=helper.suite.parse_arm(side,0,'natural_f2')
    require(primary==report['primary'],'Sidecar primary raw audit differs')
    for row in primary['epochs']:actual_kv_budget(helper.suite,Path(row['epoch_dir']))
    actual=[helper.sidecar_summary(Path(e['epoch_dir']),'graph',60.0,True) for e in primary['epochs']]
    require(actual==report['sidecar'],'Sidecar raw query/lifecycle/graph audit differs')
    reference=helper.suite.parse_arm(root/'primary',0,'natural_f2')
    require(reference==report['baseline'],'Sidecar uses wrong-budget OFF reference')
    return report


def endpoint_result(root):
    import eager_controls as quality
    output=root/'heldout256';endpoint=root/'heldout_checkpoint/global_step_5'
    result=quality.parse_evaluation(output,endpoint)
    require(result==json.loads((output/'RESULT.json').read_text()),'Endpoint raw evaluation differs')
    proof=json.loads((root/'ENDPOINT_RETENTION_AUDIT.json').read_text())
    require(proof['status']=='NEW_N2_EPOCH1_ENDPOINT_INODE_IDENTITY_OBSERVED',
            'Missing actual new endpoint retention identity evidence')
    receipt=proof['receipt'];expected=root/'graph_sidecar/seed_0/natural_f2/fig6/epoch_001_mode1_natural/checkpoints/qwen3moe_for_eagle3/global_step_5'
    require(receipt['source']==str(expected) and receipt['endpoint']==str(endpoint) and
            receipt['additional_actor_updates_from_public_epoch0']==5 and receipt['actor_frozen'] is False,
            'Evaluated endpoint does not match new N2 epoch1')
    require(len([x for x in receipt['files'] if x['path'].endswith('.distcp')])==32,
            'Endpoint provenance lacks32 trained checkpoint shards')
    require((root/'ENDPOINT_CLEANUP.json').is_file() and not endpoint.exists(),
            'Validated private endpoint was not cleaned')
    return result


def audit(root,strict=False):
    campaign.check_sources(root)
    pending=[];results={};trials=[]
    if (root/'CAPACITY_GATE_PASS.json').exists():results['capacity']=campaign.audit_calibration(root)
    else:pending.append('capacity_gate')
    for seed in campaign.SEEDS:
        if (root/f'primary/seed_{seed}/natural_f2/ARM_PASS.json').exists():trials.append(math_result(root,seed))
        else:pending.append('math_seed'+str(seed)+'_epochs1_2')
    if len(trials)==3:
        Q=mean(t['Q'] for t in trials);bQ=mean(t['baseline_Q'] for t in trials)
        E=[mean(t['E_bounds'][i] for t in trials) for i in (0,1)]
        bE=[mean(t['baseline_E_bounds'][i] for t in trials) for i in (0,1)]
        results['math']=dict(trials=trials,Q_mean=Q,Q_sample_sd=stdev(t['Q'] for t in trials),
            baseline_Q_mean=bQ,Q_gain_percent=100*(Q/bQ-1),E_bounds=E,baseline_E_bounds=bE,
            E_gain_percent_bounds=[100*(E[0]/bE[1]-1),100*(E[1]/bE[0]-1)],
            aggregation='Pool W/T and W/H across epochs1/2 within seed,then mean and sampleSD across0/2/3. '
                        'E bounds are recorded-clock uncertainty,not a statistical confidence interval.')
    else:results['math_completed_seeds']=[t['seed'] for t in trials]
    for name,receipt,reader in [('code',root/'CODE_N2_RESULT.json',code_result),
        ('graph_sidecar',root/'graph_sidecar/TRAINING_SIDECAR_RESULT.json',sidecar_result),
        ('heldout256',root/'heldout256/RESULT.json',endpoint_result)]:
        if receipt.exists():results[name]=reader(root)
        else:pending.append(name)
    if not (root/'ALL_DEPENDENCIES_COMPLETE.json').exists():pending.append('final_dependency_queue_receipt')
    else:
        dependency=json.loads((root/'ALL_DEPENDENCIES_COMPLETE.json').read_text())
        require(dependency['status']=='ALL_NEW_F8_DEPENDENCIES_VALIDATED','Dependency queue not validated')
        results['dependency']=dependency
    report=dict(status='ALL_NEW_N2_EXPERIMENTS_RAW_EVIDENCE_AUDITED' if not pending else 'INTERIM_NOT_COMPLETE',
        root=str(root),candidate_floor8=349952,pending=pending,results=results,
        scope='No old/new N2 pooling. Existing unchanged Full16 controls reused. All claims from actual logs/outputs,not launch flags alone.')
    campaign.write(root/'NEW_N2_AUDIT_REPORT.json',report)
    lines=['# New Ada-N2 experiments: F8=349952','',report['status'],'',
           'Pending: '+(', '.join(pending) if pending else 'none')+'.','',
           'Q=W/T. E=W/H, where H sums occupied rank-time, including restore reoccupation. '
           'E bounds reflect recorded-clock resolution, not a statistical confidence interval.','']
    if 'math' in results:
        a=results['math'];lines += [f"Math Q={a['Q_mean']:.2f} ±{a['Q_sample_sd']:.2f} tokens/s, "
            f"gain vs verl baseline={a['Q_gain_percent']:.2f}%.",
            f"E={a['E_bounds']} tokens/(occupied-rank*s), gain={a['E_gain_percent_bounds']}%.",'']
    if 'code' in results:
        a=results['code'];lines += [f"Code Q={a['natural_f2']['response_tokens_s']:.2f} tokens/s, "
            f"gain vs verl baseline={a['Q_gain_pct']:.2f}%.",
            f"Code E={a['natural_f2']['E_response_tokens_per_occupied_rank_s_bounds']} tokens/(occupied-rank*s), "
            f"gain={a['E_gain_pct_bounds']}%.",
            f"Code sampled pass@1={a['natural_f2']['correct']}/160 versus71/160.",
            'Code actual reusable detached rank counts by step: '+
            str(a['actual_reusable_detached_rank_counts_by_step'])+'.',
            'Worker-confirmed shrink groups: '+str([g['floor'] for g in a['worker_confirmed_shrink_groups']])+'. '
            'The original optional RPC-prefix counter is not a physical shrink counter.','']
    if 'graph_sidecar' in results:
        a=results['graph_sidecar'];lines += [f"Graph Sidecar completed-before-stop={a['completed_before_stop']}, "
            f"actual starts={a['actual_sidecar_starts']}.",'']
    if 'heldout256' in results:
        a=results['heldout256'];lines += [f"New N2 epoch1 endpoint: {sum(a['scores_by_prompt_sha256'].values()):.0f}/256 correct.",'']
    (root/'NEW_N2_RESULTS.md').write_text('\n'.join(lines)+'\n')
    if strict:require(not pending,'Campaign not complete: '+', '.join(pending))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    value=audit(args.root.resolve(),args.require_complete)
    print(json.dumps(dict(status=value['status'],pending=value['pending']),ensure_ascii=False))
