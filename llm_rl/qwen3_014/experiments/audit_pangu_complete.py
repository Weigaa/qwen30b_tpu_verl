"""Full-arm completion audit, not a substitute for running both RL epochs."""
import argparse
import json
from pathlib import Path
import re
import time

import run_n4_sidecar_models as m


def read(path):
    return json.loads(path.read_text())


def audit(root):
    root=root.resolve();arm=root/'pangu_moe'
    assert read(root/'STATE.json')['state']=='COMPLETE','Formal controller not complete'
    protocol=read(root/'PROTOCOL.json')
    assert protocol['order']==['pangu_moe']
    assert protocol['seed']==0 and protocol['epochs']==[1,2] and protocol['steps_each']==5
    m.check_contract(root)
    proof=m.proof(root,'pangu_moe')
    stop_proof=read(root/'stop_preflight/pangu_moe/RESULT.json')
    assert stop_proof['status']=='PASS' and not stop_proof['sigkill']
    assert read(arm/'execution/DRIVER_RESULT.json')['exit_code']==0
    stored=read(arm/'TRAINING_SIDECAR_RESULT.json')
    assert stored['status']=='PASS' and stored['epochs']==[1,2]
    assert stored['model_profile']==dict(model='/data/pangu-pro-moe-model',tp=1,dp=8,replicas=1,ep=True)
    assert read(arm/'STATE.json')['state']=='COMPLETE'
    comparison=read(root/'COMPARISON.json')
    assert comparison['status']=='ALL_REQUESTED_MODEL_ARMS_COMPLETE'
    assert comparison['requested_models']==['pangu_moe'] and comparison['arms']==[stored]
    launch=read(arm/'v014_launch.json');env=launch['env']
    requirements={
        'TASK_QUEUE_ENABLE':'1','FAIR_START_EPOCH':'1','FAIR_TOTAL_EPOCHS':'3',
        'FAIR_TRAIN_STEPS':'5','FAIR_TRAIN_BATCH_SIZE':'32','FAIR_ROLLOUT_N':'16',
        'FAIR_MAX_RESPONSE_LENGTH':'16384','FAIR_FREEZE_ACTOR':'0',
        'VLLM_ROLLOUT_SLEEP_LEVEL':'2','VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE':'1',
        'VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_KV_FIRST':'1',
        'ADAFLOOR_ACLGRAPH_MODE':'FULL_DECODE_ONLY',
        'FLOOR_KV_CAPS':'4:280576,8:349952,16:380800',
        'VERL_SIDECAR_TENSOR_PARALLEL_SIZE':'1','VERL_SIDECAR_DATA_PARALLEL_SIZE':'8',
        'VERL_SIDECAR_REPLICA_COUNT':'1','VERL_SIDECAR_ENABLE_EXPERT_PARALLEL':'1',
        'VERL_SIDECAR_WINDOW_GATE_MIN_SECONDS':'60.0',
        'VERL_SIDECAR_WINDOW_GATE_STRICT_MINIMUM':'1',
        'VERL_SIDECAR_WINDOW_GATE_CALIBRATION':'prior_step_q95',
        'VERL_SIDECAR_MAX_MODEL_LEN':'6144','VERL_SIDECAR_MAX_TOKENS':'4096',
        'VERL_SIDECAR_MAX_NUM_SEQS':'128','VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS':'65536'}
    assert all(env.get(k)==v for k,v in requirements.items()),'Formal settings changed'
    primary=m.n4.helper().suite.parse_arm(arm,0,'natural_f4')
    assert primary==stored['primary']==read(arm/'seed_0/natural_f4/ARM_PASS.json')
    common=m.n4.helper().suite.COMMON
    assert (common/'DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT').is_file()
    assert str(common/'epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5') in (
        '\n'.join(p.read_text(errors='replace') for p in Path(primary['epochs'][0]['epoch_dir']).glob('logs/*.txt')))
    epoch_audits=[];sides=[];memory_budgets=[]
    helper=m.c.module('pangu_final_sidecar_validator',m.SECONDARY/'tools/run_v014_sidecar_training.py')
    for index,epoch in enumerate(primary['epochs']):
        ep=Path(epoch['epoch_dir']);assert ep.name==f'epoch_{index+1:03d}_mode1_natural'
        text='\n'.join(p.read_text(errors='replace') for p in ep.glob('logs/*.txt'))
        checkpoint=(common/'epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5'
            if index==0 else Path(primary['epochs'][0]['epoch_dir'])/'checkpoints/qwen3moe_for_eagle3/global_step_5')
        loaded=re.findall(r'\[Rank (\d+)\] Loaded sharded model checkpoint from '+re.escape(str(checkpoint)),text)
        assert set(map(int,loaded))==set(range(16)),'Missing actual per-rank checkpoint load receipts'
        plans=read(ep/'oracle/length_sorted_rank_plan_summary.json')
        expected_history=[str(common/'epoch_000_mode0_probe')]
        if index==1:expected_history.append(primary['epochs'][0]['epoch_dir'])
        assert len(plans)==5 and all(p['length_prediction_baseline_dirs']==expected_history for p in plans)
        runtime=m.n4.helper().suite.json_records(ep,'MODE1_STEP_RUNTIME_AUDIT ')
        assert len(runtime)==80 and {(r['step'],r['rank']) for r in runtime}=={
            (s,r) for s in range(1,6) for r in range(16)}
        assert all(r['complete'] and r['full_iterations']>0 for r in runtime)
        answers=[]
        for step in range(1,6):
            rows=[json.loads(l) for l in (ep/f'rollout_data/{step}.jsonl').open() if l.strip()]
            assert len(rows)==512 and all(r['output'].strip() for r in rows)
            answers.extend(rows)
        side=helper.sidecar_summary(ep,'graph',60.0,True,expected_replicas=8)
        assert side==stored['sidecar'][index]
        lease=(ep/'sidecar/lease.log').read_text()
        device_lines=re.findall(r'^sidecar_devices=([^\n]+)$',lease,re.M)
        active_lines=re.findall(r'^sidecar_active_ranks=([^\n]+)$',lease,re.M)
        assert len(device_lines)==len(side['windows'])
        assert len(active_lines)==len(device_lines)
        for devices,active in zip(device_lines,active_lines):
            ranks=list(map(int,devices.split(',')))
            assert len(ranks)==len(set(ranks))==8
            assert set(ranks)==set(range(16))-set(map(int,active.split(',')))
        events=m.c.module('pangu_events',m.SECONDARY/'tools/run_sidecar_graph_parity.py').read_events(ep/'sidecar/infer.log') if side['windows'] else []
        loads=[v for v in events if v['event']=='sidecar_load_start']
        for load in loads:
            kw=load['engine_kwargs']
            assert load['model_path']=='/data/pangu-pro-moe-model'
            assert kw['data_parallel_size']==8 and kw['tensor_parallel_size']==1
            assert kw['enable_expert_parallel'] is True and not kw['enforce_eager']
            assert kw['distributed_executor_backend']=='external_launcher'
            assert kw['max_num_seqs']==128 and kw['max_num_batched_tokens']==65536
        inference=(ep/'sidecar/infer.log').read_text(errors='replace') if side['windows'] else ''
        monitor=(ep/'sidecar/monitor.log').read_text(errors='replace')
        assert 'No available memory for the cache blocks' not in monitor
        for part in inference.split('SIDECAR_INSTANCE_MEMORY_AUDIT ')[1:]:
            budget=json.JSONDecoder().raw_decode(part)[0]
            assert budget['utilization']==.8
            assert 0<=budget['kv_budget_bytes']<=budget['instance_kv_limit_bytes']
            assert budget['kv_budget_bytes']<=budget['physical_kv_limit_bytes']
            memory_budgets.append(dict(epoch=index+1,**budget))
        assert 'ModuleNotFoundError' not in inference and not re.search(
            r'OutOfMemoryError|NPU out of memory|Memory_Allocation_Failure',inference)
        epoch_audits.append(dict(epoch=index+1,primary_responses=len(answers),
            full_rank_step_audits=len(runtime),kv_preemptions=sum(x['preemptions'] for x in runtime),
            primary_reward=epoch['reward'],primary_mean_length=epoch['mean_length'],
            primary_step_seconds=epoch['step_seconds'],sidecar_starts=len(side['windows']),
            sidecar_completed=side['completed_queries'],sidecar_completed_before_stop=side['completed_before_stop'],
            sidecar_sigterm=side['sigterm_events'],sidecar_sigkill=side['sigkill_events']))
        sides.append(side)
    run=arm/'seed_0/natural_f4/fig6'
    assert read(run/'SPLIT_STORAGE_REMOVED.json')['public_epoch0_untouched'] is True
    windows=[w for side in sides for w in side['windows']]
    completed=sum(w['completed'] for w in windows);correct=sum(w['correct'] for w in windows)
    before=sum(w['completed_before_stop'] for w in windows)
    assert before==stored['completed_before_stop'] and completed==stored['completed_including_drain']
    starts=len(windows);assert starts==stored['actual_sidecar_starts']
    reference=read(m.REFERENCE/'TRAINING_SIDECAR_RESULT.json')
    assert primary['prompt_hash']==reference['primary']['prompt_hash'],'Workload differs from the Qwen2.5 reference'
    result=dict(status='COMPLETE_FULL_ARM_AUDIT_PASS',model='PanGu-Pro-MoE',
        auditor_sha256=m.c.sha(__file__),
        formal_epochs=2,formal_primary_steps=10,primary_responses=5120,
        full_rank_step_audits=160,epochs=epoch_audits,actual_sidecar_starts=starts,
        completed_before_stop=before,completed_including_drain=completed,
        correct_including_drain=correct,strict_accuracy=correct/completed if completed else None,
        completed_query_output_tokens=sum(w['tokens'] for w in windows),
        primary_Q_tokens_per_second=primary['response_tokens_s'],primary_reward=primary['reward'],
        primary_mean_length=primary['mean_length'],primary_total_rollout_s=sum(e['rollout_s'] for e in primary['epochs']),
        primary_same_seed_OFF_Q_change_percent=stored['primary_Q_change_percent'],
        qwen25_reference_primary_Q=reference['primary']['response_tokens_s'],
        qwen25_reference_completed_before_stop=reference['completed_before_stop'],
        ready_seconds=[w['all_ready_seconds'] for w in windows],
        stop_to_physical_exit_seconds=[w['shutdown_to_exit_seconds'] for w in windows],
        windows=[dict(epoch=index+1,**w) for index,side in enumerate(sides) for w in side['windows']],
        model_and_graph_preflight_queries=proof['checked_prompts'],formal_source_hashes_match=True,
        instance_memory_budget_audits=memory_budgets,
        safety='All output/state/prompt/replay/lease-exit-ACK-before-restore checks rerun from raw data.',
        scope='All10 primary steps and every admitted window, including startup-only windows. Updated primary policies, descriptive not frozen causal comparison. Different-model tokens are not added together.',
        required_artifact_sha256={str(p.relative_to(root)):m.c.sha(p) for p in (
            root/'PROTOCOL.json',root/'COMPARISON.json',arm/'TRAINING_SIDECAR_RESULT.json',
            arm/'execution/DRIVER_RESULT.json',arm/'seed_0/natural_f4/ARM_PASS.json')})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--wait-for-pid',type=int)
    args=parser.parse_args()
    if args.wait_for_pid is not None:
        process=Path(f'/proc/{args.wait_for_pid}/stat')
        def identity():
            try:
                fields=process.read_text().rsplit(')',1)[1].split()
                return None if fields[0]=='Z' else fields[19]
            except (FileNotFoundError,ProcessLookupError):return None
        initial=identity()
        print(f'Waiting for verified controller PID {args.wait_for_pid}, start_ticks={initial}',flush=True)
        while initial is not None and identity()==initial:
            time.sleep(15)
        # Do not restart training or accept missing/failed completion artifacts.
    result=audit(args.root)
    m.c.write(args.root/'FINAL_COMPLETION_AUDIT.json',result)
    accuracy=f"{100*result['strict_accuracy']:.2f}%" if result['strict_accuracy'] is not None else 'N/A (no completed query)'
    lines=['# PanGu Ada-N4 full experiment', '',
        'Two updated-policy epochs, five steps each. Seed0, protected common epoch0, ',
        'primary FULL/TQ1/native sleep2 and unchanged Natural-F4 KV budgets. ',
        'PanGu internal DP8 x TP1 + EP8, FULL/TQ1, original strict>60s history gate.', '',
        f"Primary: {result['primary_Q_tokens_per_second']:.2f} tokens/s, total rollout {result['primary_total_rollout_s']:.2f}s.",
        f"Reward {result['primary_reward']:.6f}, mean response {result['primary_mean_length']:.2f} tokens.",
        f"Same-seed updated-policy OFF difference {result['primary_same_seed_OFF_Q_change_percent']:+.2f}% (descriptive, not frozen causal).", '',
        f"Sidecar starts {result['actual_sidecar_starts']}/10, completed before stop {result['completed_before_stop']}, including drain {result['completed_including_drain']}.",
        f"Correct {result['correct_including_drain']}, strict GSM8K accuracy including drain {accuracy}.", '',
        '| Epoch | Step | Window seconds | All ready seconds | Jobs before stop | Jobs including drain | Stop to physical exit seconds |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for w in result['windows']:
        ready=f"{w['all_ready_seconds']:.3f}" if w['all_ready_seconds'] is not None else 'not all ready'
        lines.append(f"| {w['epoch']} | {w['step']} | {w['window_seconds']:.3f} | {ready} | {w['completed_before_stop']} | {w['completed']} | {w['shutdown_to_exit_seconds']:.3f} |")
    lines+=['', 'All5120 primary responses are nonempty. All160 rank-step audits prove actual FULL decode. ',
        'Raw output/state/prompt/replay/lease exit and ACK-before-restore checks passed. ',
        'All admitted windows, including startup-only windows, remain included. ',
        'The public epoch0 is preserved. No different-model token throughputs are pooled.', '',
        'Detailed evidence: `FINAL_COMPLETION_AUDIT.json`, `pangu_moe/TRAINING_SIDECAR_RESULT.json`, ',
        '`PROTOCOL.json`, raw epoch logs and sidecar output/state files.', '']
    (args.root/'PANGU_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
