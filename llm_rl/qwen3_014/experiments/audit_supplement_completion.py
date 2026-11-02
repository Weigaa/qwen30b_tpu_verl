#!/usr/bin/env python3
"""Final offline, requirement-by-requirement audit; never launch an experiment.

An incomplete queue is not a failed experiment and not goal completion. Do
not write a final-success artifact until all five evidence sets are present
and their actual generations/scores can be independently reconstructed.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import eager_controls as eager

ROOT = eager.ROOT


def same(actual, expected, label):
    if not math.isfinite(float(actual)) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError('Reconstructed result differs: '+label)


def prerequisites(root=ROOT):
    files = ['p4_explicit_pair_report/summary.json', 'quota_pair_report/summary.json',
        'graph_control_report/summary.json', 'EAGER_AND_QUALITY_RESULT.json', 'code_task/CODE_PAIR_RESULT.json']
    return [name for name in files if not (root/name).is_file()]


def audit_raw_sidecar(pair):
    """Recompute useful queries and accuracy from original completed jobs.

    Reuse the preserved experiment auditor, including actual prompt/GT,
    completed-versus-partial identities, time windows, process exit/ACK,
    settings and frozen Sidecar model digest. No inference is launched.
    """
    on=Path(pair['on'])
    epochs=list(on.glob('*/epoch_001_mode1_planned'))
    if len(epochs)!=1:raise ValueError('Missing unique Sidecar-ON epoch')
    logs=list((epochs[0]/'logs').glob('*.txt'))
    if len(logs)!=1:raise ValueError('Missing original Sidecar primary log')
    auditor=on/'audit_sidecar_execution.py'
    spec=importlib.util.spec_from_file_location('final_original_raw_sidecar_auditor',auditor)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    primary=re.sub(r'\x1b\[[0-9;]*[mK]','',logs[0].read_text(errors='replace'))
    actual=module.audit_sidecar(epochs[0],primary)
    # JSON summaries stringify integer dictionary keys (replica identities).
    actual=json.loads(json.dumps(actual))
    if actual!=pair['sidecar_on']:
        raise ValueError('Original Sidecar query/accuracy/lifecycle events differ from paired result')
    return dict(status='RAW_SIDECAR_QUERIES_ACCURACY_AND_EXIT_RECOMPUTED',
        completed_before_stop=actual['completed_before_trainer_stop'],
        correct_before_stop=actual['correct_before_stop'],
        completed_including_drain=actual['completed_queries'],
        correct_including_drain=actual['correct_queries'],
        partial_queries=actual['partial_queries'],
        preserved_auditor_sha256=hashlib.sha256(auditor.read_bytes()).hexdigest(),
        original_primary_log_sha256=hashlib.sha256(logs[0].read_bytes()).hexdigest())


def audit_quality():
    summary = json.loads((ROOT/'EAGER_AND_QUALITY_RESULT.json').read_text())
    if summary['status'] != 'TWO_EAGER_CONTROLS_AND_MATCHED_HELDOUT_COMPLETE':
        raise ValueError('Eager/quality queue has not committed')
    result = {}
    for policy in ('vanilla','planned_f4','natural_f2_graph_trained'):
        if policy == 'natural_f2_graph_trained':
            endpoint = ROOT/'quality/ada_n2_seed0_epoch1/global_step_5'
        else:
            epoch = next((ROOT/f'eager_controls/seed_0/{policy}/fig6').glob('epoch_001_*'))
            endpoint = epoch/'checkpoints/qwen3moe_for_eagle3/global_step_5'
        scored = eager.parse_evaluation(ROOT/f'heldout_results/{policy}',endpoint)
        original = summary['quality'][policy]
        if scored != original:
            raise ValueError('Held-out raw outputs no longer match summary: '+policy)
        result[policy] = scored
    a = result['vanilla']['scores_by_prompt_sha256']
    b = result['planned_f4']['scores_by_prompt_sha256']
    c = result['natural_f2_graph_trained']['scores_by_prompt_sha256']
    if not set(a) == set(b) == set(c) or len(a) != 256:
        raise ValueError('Actual endpoint evaluations not on the same256 questions')
    same(summary['matched_quality_pair']['difference'],sum(b[k]-a[k] for k in a)/256,'paired held-out delta')
    return result


def audit_code():
    import code_task
    from code_test_wire import normalize,infrastructure_format_error
    report = json.loads((ROOT/'code_task/CODE_PAIR_RESULT.json').read_text())
    if report['status'] != 'V014_CODE_DOMAIN_FROZEN_PAIR_COMPLETE':
        raise ValueError('Code task incomplete')
    paired = {}; scores = {}
    repair=report.get('grading_repair')
    repair_protocol=None
    if repair:
        path=ROOT/'code_task/GRADING_REPAIR_PROTOCOL.json'
        if (repair['status']!='LOSSLESS_CODE_WIRE_REPAIR_COMPLETE' or repair['NPU_reruns']!=0 or
                repair['protocol_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError('Unproved grading repair lineage')
        repair_protocol=json.loads(path.read_text())
        if (repair_protocol['original_protocol_sha256']!=hashlib.sha256((ROOT/'code_task/PROTOCOL.json').read_bytes()).hexdigest()
                or repair_protocol['original_code_pair_sha256']!=hashlib.sha256(
                    (ROOT/'code_task/quality_initial_wire_format_invalid/CODE_PAIR_RESULT.json').read_bytes()).hexdigest()):
            raise ValueError('Original generation/grading provenance changed after repair')
        for key,name in [('new_adapter_sha256','code_test_wire.py'),('repair_driver_sha256','repair_code_quality_wire.py')]:
            if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=repair_protocol[key]:
                raise ValueError('Grading repair source changed')
    from rank_occupancy import parse_epoch as occupancy
    for policy in ('vanilla','planned_f4'):
        actual,outputs,selected = code_task.audit_arm(policy)
        paired[policy] = outputs
        declared = report['arms'][policy]
        for name in ('rollout_seconds','response_tokens','response_tokens_s','actual_full_audits','preemptions'):
            same(actual[name],declared[name],'code '+policy+' '+name)
        rows = list(code_task.read_rows(ROOT/f'code_task/quality/{policy}.jsonl'))
        if len(rows) != 160 or {r['prompt_hash'] for r in rows} != set(selected):
            raise ValueError('Missing/duplicate fixed sample0 sandbox scores')
        archived=None
        if repair_protocol:
            old=ROOT/f'code_task/quality_initial_wire_format_invalid/{policy}.jsonl'
            if hashlib.sha256(old.read_bytes()).hexdigest()!=repair_protocol['original_score_sha256'][policy]:
                raise ValueError('Original withdrawn grading was changed')
            archived={r['prompt_hash']:r for r in code_task.read_rows(old)}
        changed_keys=set()
        for row in rows:
            answer = selected[row['prompt_hash']]['output']
            if (row['sample_index'] != 0 or row['output_sha256'] != hashlib.sha256(answer.encode()).hexdigest()
                    or not isinstance(row['passed'],bool)):
                raise ValueError('Scored answer differs from prespecified sample0')
            if row['passed'] and (row['exit_code'] != 0 or row['timed_out']):
                raise ValueError('Failed sandbox counted as pass')
            if infrastructure_format_error(row):
                raise ValueError('Judge input-format infrastructure failure is not a measured wrong answer')
            tests=selected[row['prompt_hash']]['gts']
            tests=json.loads(tests) if isinstance(tests,str) else tests
            encoded,changed=normalize(tests)
            if row['passed'] and (row.get('tests')!=len(tests['inputs']) or
                    row.get('returned_tests')!=len(tests['inputs']) or row.get('test_passes')!=len(tests['inputs'])):
                raise ValueError('A passing score does not prove all original test cases')
            if changed:
                changed_keys.add(row['prompt_hash'])
                if not repair_protocol or not row.get('lossless_wire_normalization'):
                    raise ValueError('Native test schema has not been adapted to pinned judge wire format')
                from repair_code_quality_wire import digest
                if (row['original_test_sha256']!=digest(tests) or row['wire_test_sha256']!=digest(encoded)
                        or row['original_test_sha256']!=repair_protocol['source_test_sha256'][row['prompt_hash']]
                        or row['wire_test_sha256']!=repair_protocol['wire_test_sha256'][row['prompt_hash']]):
                    raise ValueError('Original test values or normalized wire representation changed')
            elif archived is not None and row!=archived[row['prompt_hash']]:
                raise ValueError('An unaffected answer was regraded or its original score changed')
        if repair_protocol and (len(changed_keys)!=21 or changed_keys!=set(repair_protocol['affected_prompt_hashes'])):
            raise ValueError('Not the complete predeclared schema-affected21 panel')
        same(sum(r['passed'] for r in rows)/160,declared['sandbox_sampled_pass_at_1'],'sandbox pass@1 '+policy)
        scores[policy] = {r['prompt_hash']:r['passed'] for r in rows}
        epoch=next((ROOT/f'code_task/seed_401/{policy}/fig6').glob('epoch_001_*'))
        logs=list((epoch/'logs').glob('*.txt'))
        if len(logs)!=1:raise ValueError('Ambiguous complete code-task primary log')
        text=logs[0].read_text(errors='replace')
        declared['observed_shrink_RPC_completion_records']=text.count('Elastic parallel shrink rpc done:')
        declared['Q_E_actual_occupied_intervals']=occupancy(epoch,
            {row['step']:row['response_tokens'] for row in actual['steps']},'FULL')
        if policy=='planned_f4':
            plan=json.loads((epoch/'oracle/length_sorted_rank_plan.json').read_text())
            declared['planned_floors']=[row['selected_floor'] for row in plan]
            declared['planned_response_caps']=[row['tail_guard_response_cap'] for row in plan]
        declared['shrink_scope']='Observed RPC records, not the selected minimum floor. No physical free rank-time credited from planning alone; Q/E uses actual native-sleep/restore intervals.'
    if set(paired['vanilla']) != set(paired['planned_f4']):
        raise ValueError('Code paired request identities differ')
    for key,row in paired['vanilla'].items():
        other = paired['planned_f4'][key]
        if (row['input'],row['gts']) != (other['input'],other['gts']):
            raise ValueError('Actual code input/GT differs')
    # A pair could agree while both use the wrong dataset. Verify actual
    # decoded prompts/GTs and all n16 sample identities against original data.
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from prepare_code_dataset import normalize as canonicalize_dataset
    original=pq.read_table(code_task.DATA/'train.parquet').to_pylist()
    canonical=pq.read_table(code_task.TRAIN).to_pylist()
    if len(original)!=160 or canonical!=canonicalize_dataset(original):
        raise ValueError('Canonical code data differs from original beyond identity normalization')
    tokenizer=AutoTokenizer.from_pretrained('/data/Qwen3-30B-A3B',trust_remote_code=True)
    expected={tokenizer.decode(tokenizer.apply_chat_template(row['prompt'],add_generation_prompt=True,
        tokenize=True),skip_special_tokens=True):json.loads(row['reward_model']['ground_truth']) for row in canonical}
    if len(expected)!=160:raise ValueError('Original question set is not unique160')
    for policy,actual_outputs in paired.items():
        groups={}
        for row in actual_outputs.values():
            gts=json.loads(row['gts']) if isinstance(row['gts'],str) else row['gts']
            if row['input'] not in expected or gts!=expected[row['input']]:
                raise ValueError('Actual code input/GT differs from original dataset')
            groups.setdefault(row['input'],set()).add(int(row['rollout_sample_index']))
        if set(groups)!=set(expected) or any(v!=set(range(16)) for v in groups.values()):
            raise ValueError('Not all original160 questions have exactly n16 sample identities')
    report['all_actual_original160_prompt_GT_and_n16_identities_verified']=True
    if code_task.suite().execution_digest() != report['source_sha256']:
        raise ValueError('Code runtime source changed after generation')
    provenance = json.loads((ROOT/'code_task/PROTOCOL.json').read_text())
    for name,wanted in provenance['scoring_sha256'].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() != wanted:
            raise ValueError('Scoring/plan source changed: '+name)
    report['paired_pass_at_1_difference'] = sum(scores['planned_f4'][k]-scores['vanilla'][k] for k in scores['vanilla'])/160
    a,b=(report['arms'][name] for name in ('vanilla','planned_f4'))
    report['P4_response_throughput_change_percent']=100*(b['response_tokens_s']/a['response_tokens_s']-1)
    report['P4_rollout_time_change_percent']=100*(b['rollout_seconds']/a['rollout_seconds']-1)
    report['P4_actual_response_work_change_percent']=100*(b['response_tokens']/a['response_tokens']-1)
    return report


def main():
    missing = prerequisites()
    if missing:
        print(json.dumps(dict(status='INCOMPLETE_NOT_A_RUNTIME_FAILURE',missing=missing)))
        return
    # Reconstruct raw-backed reports, not just their green status strings.
    import run_explicit_p4_off
    import sidecar_device_intervals
    import report_p4_pair
    import report_quota_pair
    import report_graph_controls
    import audit_quota_v014
    pair = run_explicit_p4_off.compare()
    if not all(r['same_input_contract'] and r['exact_response_match'] for r in pair['steps']):
        raise ValueError('Sidecar pair not exact')
    raw_sidecar=audit_raw_sidecar(pair)
    actual_intervals = sidecar_device_intervals.analyze()
    stored_intervals = json.loads((ROOT/'p4_sidecar_actual_intervals/summary.json').read_text())
    if actual_intervals != stored_intervals:
        raise ValueError('Sidecar actual device intervals changed')
    report_p4_pair.main()
    workload = ROOT/'quota_prepared/workload.json'; data = json.loads(workload.read_text())
    digest = hashlib.sha256(workload.read_bytes()).hexdigest()
    old = json.loads((ROOT/'quota_runtime/QUOTA_PAIR_RESULT.json').read_text())
    for policy,stored in zip(('min_skew','release_area'),old['arms']):
        actual = audit_quota_v014.audit_arm(ROOT/'quota_runtime'/policy,data,policy,digest)
        if actual != stored:
            raise ValueError('Quota raw event audit differs')
    report_quota_pair.main()
    report_graph_controls.main()
    quality = audit_quality(); code = audit_code()
    import report_endpoint_quality
    report_endpoint_quality.main()
    p4 = json.loads((ROOT/'p4_explicit_pair_report/summary.json').read_text())
    quota = json.loads((ROOT/'quota_pair_report/summary.json').read_text())
    graph = json.loads((ROOT/'graph_control_report/summary.json').read_text())
    report = dict(status='ALL_FIVE_MINIMAL_V014_SUPPLEMENT_REQUIREMENTS_AUDITED',
        sidecar=p4,sidecar_raw_reaudit=raw_sidecar,quota=quota,graph_controls=graph,heldout=quality,additional_code_task=code,
        constraints=['Single paired experiment/seed per supplemental contrast, not a method/seed grid.',
            'No convergence/equivalence/energy/AICore-utilization claims.',
            'N2 Graph-trained quality endpoint is from the separately requested Sidecar-ON epoch1.',
            'The held-out P4 endpoint is eager-trained; no direct P4 Graph-trained endpoint or final-convergence equivalence claim.',
            'Code task frozen; dummy timed reward is not code accuracy. Fixed sample0 sandbox pass@1.',
            'Initial code grading was withdrawn for native-list/string-wire infrastructure errors; lossless schema repair regraded21 fixed answers/arm once and retained139 untouched scores/arm. No NPU rerun.',
            'For the additional code task, observed shrink RPC records and actual occupied intervals, not planned floors, establish any rank-release evidence.',
            'Eager/TQ2 vs FULL/TQ1 is a configured-path comparison, not a pure one-bit graph causal estimate.',
            'No paper TeX changed. Protected common epoch0 checkpoints are preserved.'],
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for common in (eager.suite().COMMON,__import__('code_task').COMMON):
        if not (common/'DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT').is_file():
            raise ValueError('Protected common epoch0 missing: '+str(common))
    (ROOT/'FINAL_SUPPLEMENT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# v0.14 supplementary experiments: all five evidence sets audited','',
        '| Priority | Evidence | Detailed report |','|---|---|---|',
        '| 1 | Explicit frozen P4 Sidecar ON/OFF, real device intervals | p4_explicit_pair_report/RESULTS.md |',
        '| 2 | Same-work Quota/MinSkew, actual quorum/waiting/Q/E | quota_pair_report/RESULTS.md |',
        '| 3 | Two eager controls vs historical FULL, actual KV/input matched | graph_control_report/RESULTS.md |',
        '| 4 | Same256 held-out questions, uniform greedy serving | heldout_results/*/RESULT.json |',
        '| 5 | One frozen additional code-domain baseline/final-P4 pair | code_task/CODE_PAIR_RESULT.json |','',
        '## Endpoint held-out scores','']
    for name,row in quality.items():
        lines.append(f"- {name}: {row['mean_score']:.4%}, {row['questions']} questions, {row['empty_answers']} empty answers.")
    lines += ['','## Additional code-domain results','',
        '| Method | Rollout s | Effective response tok/s | Sandbox sampled pass@1 |','|---|---:|---:|---:|']
    for name,row in code['arms'].items():
        lines.append(f"| {name} | {row['rollout_seconds']:.2f} | {row['response_tokens_s']:.2f} | {row['sandbox_sampled_pass_at_1']:.2%} |")
    lines+=['',f"P4 throughput change {code['P4_response_throughput_change_percent']:+.3f}%, "
        f"rollout-time change {code['P4_rollout_time_change_percent']:+.3f}%, "
        f"actual response-work change {code['P4_actual_response_work_change_percent']:+.3f}%. "
        'Do not attribute a whole time change to speed when realized response work differs.','']
    for name,row in code['arms'].items():
        lines.append(f"- {name}: observed shrink RPC completion records {row['observed_shrink_RPC_completion_records']}; "
            f"KV preemptions {row['preemptions']}. "+row['shrink_scope'])
    lines += ['','## Scope and negative findings','']+['- '+s for s in report['constraints']]
    (ROOT/'RESULTS_FINAL.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'])))


if __name__ == '__main__':main()
