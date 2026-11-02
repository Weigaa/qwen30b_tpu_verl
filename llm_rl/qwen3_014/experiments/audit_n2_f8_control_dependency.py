#!/usr/bin/env python3
"""Check MinSkew's shared Natural-F2 capacity dependency before reusing it.

Replan all six historical windows on CPU at the new F8 cap. If any exact
rank/bucket/floor/cap plan changes, rerun that seed's BOTH epochs after the
primary campaign finishes. Otherwise retain a proof of unused-budget parity.
No NPU overlap, no source or original-output edits.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_n2_f8_campaign as campaign


def audit(root):
    result=[]
    for seed in campaign.SEEDS:
        for epoch in (1,2):
            old=list((campaign.OLD/f'seed_{seed}/minskew/fig6').glob(f'epoch_{epoch:03d}_*/oracle'))
            if len(old)!=1:raise RuntimeError('Missing original MinSkew plan')
            old=old[0];summary=json.loads((old/'length_sorted_rank_plan_summary.json').read_text())
            work=root/'control_dependency_audit'/f'seed_{seed}_epoch_{epoch}'
            work.mkdir(parents=True,exist_ok=False)
            args=[]
            for baseline in summary[0]['length_prediction_baseline_dirs']:
                args+=['--baseline-dir',baseline]
            args+=['--require-compact-history','--train-file','/data/deepscaler/train.parquet',
                '--tokenizer-path','/data/Qwen3-30B-A3B','--require-prompt-token-lengths',
                '--steps','5','--batch-size','32','--responses-per-prompt','16','--aclgraph-bucket-count','8',
                '--dataset-fraction','0.005','--max-rank-peak-tokens','380800',
                '--adaptive-floor','--min-adaptive-floor','2','--floor-kv-caps',campaign.CAPS,
                '--rank-matching-policy','min_skew','--kv-block-size-tokens','128',
                '--active-peak-safety-factor','1.16','--max-response-len','16384',
                '--tail-guard-ratio-quantile','0.95','--tail-guard-ratio-window','3',
                '--tail-guard-default-ratio','1.20','--tail-guard-min-cap','4096','--tail-guard-round-to','512',
                '--max-cross-step-repair-swaps','8','--repair-candidate-limit','8']
            for option,name in [('train','length_sorted_train.parquet'),('plan','length_sorted_rank_plan.json'),
                                ('summary','length_sorted_rank_plan_summary.json'),('oracle','length_sorted_length_oracle.json')]:
                args+=['--output-'+option,str(work/name)]
            command=[sys.executable,'-u',str(campaign.MAIN/'tools/build_mode1_length_sorted_e2e_plan.py'),*args]
            campaign.write(work/'LAUNCH.json',dict(command=command,purpose='CPU dependency audit,not a timed replicate'))
            with (work/'planner.log').open('x') as log:
                subprocess.run(command,cwd=campaign.MAIN,stdout=log,stderr=subprocess.STDOUT,check=True)
            import pyarrow.parquet as pq
            equal={name:json.loads((work/name).read_text())==json.loads((old/name).read_text())
                   for name in ('length_sorted_rank_plan.json','length_sorted_rank_plan_summary.json',
                                'length_sorted_length_oracle.json')}
            equal['ordered_parquet_rows']=(pq.read_table(work/'length_sorted_train.parquet').to_pylist()==
                                            pq.read_table(old/'length_sorted_train.parquet').to_pylist())
            result.append(dict(seed=seed,epoch=epoch,all_fields_equal=all(equal.values()),checks=equal,
                               old=str(old),new=str(work)))
    changed=sorted({row['seed'] for row in result if not row['all_fields_equal']})
    proof=dict(status='MINSKEW_NEW_F8_DEPENDENCY_AUDITED',windows=result,affected_seeds=changed,
        reused_only_if='Exact five-step rank assignments,buckets,floors,KV caps,TailGuard and ordered input rows unchanged in BOTH epochs.',
        capacity_change_only='Unselected F8 option288256->349952; no changed selected step budget credited as equivalent.')
    campaign.write(root/'CONTROL_DEPENDENCY_AUDIT.json',proof)
    return changed


def rerun(root,seed):
    suite=campaign.main_suite();base=root/'controls';arm=base/f'seed_{seed}/minskew'
    env=campaign.new_capacity(suite.arm_env(base,seed,'minskew'));command=suite.arm_command(seed,'minskew')
    storage=campaign.split_prepare(root,'minskew_seed'+str(seed));run=arm/'fig6'
    campaign.launch(root,f'REQUIRED_MINSKEW_SEED{seed}',arm,campaign.MAIN,env,command,
                    run_dir=run,storage=storage)
    report=suite.parse_arm(base,seed,'minskew')
    campaign.occupancy_for(suite,report,arm/'RANK_OCCUPANCY.json')
    campaign.write(arm/'ARM_PASS.json',report)
    campaign.split_cleanup(root,'minskew_seed'+str(seed),storage,run)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();root=args.root.resolve()
    campaign.write(root/'DEPENDENCY_AUDIT_PROTOCOL.json',dict(
        controller_sha256=campaign.sha(__file__),possible_extra_control='MinSkew uses the Natural-F2 family',
        policy='CPU check all3seeds x2epochs. Rerun both epochs of any affected seed,only after main queue. '
               'Fixed4 forces4,Full16 controls never select8,N4/P4 budgets unchanged.',
        other_controls_unchanged=True))
    try:
        if args.resume:
            proof=json.loads((root/'CONTROL_DEPENDENCY_AUDIT.json').read_text())
            if proof['status']!='MINSKEW_NEW_F8_DEPENDENCY_AUDITED' or len(proof['windows'])!=6:
                raise RuntimeError('Missing completed dependency audit')
            import pyarrow.parquet as pq
            for row in proof['windows']:
                old=Path(row['old']);work=Path(row['new'])
                checks={name:json.loads((work/name).read_text())==json.loads((old/name).read_text())
                        for name in ('length_sorted_rank_plan.json','length_sorted_rank_plan_summary.json',
                                     'length_sorted_length_oracle.json')}
                checks['ordered_parquet_rows']=(pq.read_table(work/'length_sorted_train.parquet').to_pylist()==
                                                pq.read_table(old/'length_sorted_train.parquet').to_pylist())
                if checks!=row['checks']:raise RuntimeError('Existing dependency evidence changed')
            changed=sorted({row['seed'] for row in proof['windows'] if not row['all_fields_equal']})
            if changed!=proof['affected_seeds']:raise RuntimeError('Affected control set changed')
        else:changed=audit(root)
        campaign.write(root/'DEPENDENCY_QUEUE_STATE.json',dict(state='WAITING_MAIN_CAMPAIGN',affected_seeds=changed))
        while True:
            status=json.loads((root/'STATE.json').read_text())['state']
            if status=='STOPPED_WITH_ERROR':raise RuntimeError('Main campaign failed; no control NPU launch')
            if status=='ALL_AFFECTED_N2_EXPERIMENTS_COMPLETE':break
            time.sleep(15)
        for seed in changed:rerun(root,seed)
        campaign.write(root/'ALL_DEPENDENCIES_COMPLETE.json',dict(status='ALL_NEW_F8_DEPENDENCIES_VALIDATED',
            main_queue_complete=True,minskew_rerun_seeds=changed,
            unused_budget_controls_reused=sorted(set(campaign.SEEDS)-set(changed))))
    except Exception as error:
        campaign.write(root/'DEPENDENCY_QUEUE_STATE.json',dict(state='STOPPED_WITH_ERROR',error=repr(error)))
        raise


if __name__=='__main__':main()
