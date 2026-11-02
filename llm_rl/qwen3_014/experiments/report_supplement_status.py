#!/usr/bin/env python3
"""Incremental evidence inventory. Never equate preparation with NPU results."""
import json
from pathlib import Path
import re
from datetime import datetime,timezone

ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')


def main():
    p4=json.loads((ROOT/'p4_explicit_pair_report/summary.json').read_text())
    quota=json.loads((ROOT/'quota_pair_report/summary.json').read_text())
    if p4['status']!='P4_EXPLICIT_PAIR_PAPER_REPORT_AUDITED':raise RuntimeError('P4 report not complete')
    if quota['status']!='V014_FROZEN_SAME_GRAPH_QUOTA_PAIR_AUDITED':raise RuntimeError('Quota report not complete')
    final=ROOT/'FINAL_SUPPLEMENT_AUDIT.json'
    complete=(final.is_file() and json.loads(final.read_text())['status']==
        'ALL_FIVE_MINIMAL_V014_SUPPLEMENT_REQUIREMENTS_AUDITED' and
        json.loads(final.read_text())['additional_code_task'].get('grading_repair',{}).get('status')==
        'LOSSLESS_CODE_WIRE_REPAIR_COMPLETE')
    lines=['# v0.14 minimal supplementary evidence: '+('all five evidence sets audited' if complete else 'incomplete overall'),'',
        'Status generated UTC '+datetime.now(timezone.utc).isoformat(timespec='seconds')+'. '
        'No paper TeX modified; completed and pending evidence stay separate.','',
        '| Priority | Requirement | Current authoritative evidence |','|---|---|---|',
        '| 1 | P4 Sidecar actual reuse | Completed explicit frozen ON/OFF pair |',
        '| 2 | Quota mechanism | Completed same-work/full/native-sleep2 pair |']
    eager=ROOT/'EAGER_AND_QUALITY_RESULT.json';code=ROOT/'code_task/CODE_PAIR_RESULT.json'
    graph=ROOT/'graph_control_report/summary.json'
    for n,label,path in [(3,'Two matched eager controls',graph),(4,'Uniform held-out endpoint quality',eager),
                         (5,'One additional code-domain baseline/P4 pair',code)]:
        if n==3 and path.exists():
            status=json.loads(path.read_text())['status']
            if status!='TWO_MATCHED_V014_EAGER_GRAPH_CONTROLS_REAUDITED':raise RuntimeError('Invalid Graph control report')
            evidence='Completed; both5-step eager arms re-audited against historical FULL results'
        elif n==4 and path.exists() and (ROOT/'heldout_report/summary.json').exists():
            status=json.loads((ROOT/'heldout_report/summary.json').read_text())['status']
            evidence=('Completed; all3 endpoints scored on the same256 questions, raw scores/load paths re-audited'
                if status=='HELDOUT_THREE_ENDPOINTS_AUDITED' else 'Primary pair complete; Graph-trained endpoint pending')
        elif n==5 and complete:
            evidence='Completed; frozen160-question/2560-request pair and all320 fixed-sample0 OS-sandbox scores audited; grading format repair disclosed'
        else:evidence='Result exists; requires full final audit' if path.exists() else 'Not completed'
        lines.append(f'| {n} | {label} | {evidence} |')
    lines+=['','## Completed P4 Sidecar evidence','',
        f"Primary throughput change {p4['response_throughput_change_pct']:+.3f}%, all2560 answers exact. "
        f"{p4['sidecar']['completed_before_trainer_stop']} queries before stop, strict accuracy "
        f"{100*p4['sidecar']['strict_accuracy_before_stop']:.3f}%. "
        f"Assigned lease including startup/drain {p4['actual_device_intervals']['assigned_lease_including_drain_rank_seconds']:.3f} rank-s. "
        'Not AICore utilization. Single seed101, frozen/no TailGuard/fixed planned survivor identities, '
        'not silently the updated-policy/natural-member-selection main-P4 trajectory.','',
        'Full report: `p4_explicit_pair_report/RESULTS.md`.','',
        '## Completed Quota mechanism evidence','',
        f"Same1710400 response tokens. Q gain {quota['metrics']['Q_gain_percent']:.3f}%, "
        f"E gain {quota['metrics']['E_gain_percent']:.3f}%, "
        f"waiting-interval change {quota['waiting_interval_change_percent']:.3f}%. "
        'One prespecified forced-length case, not a population/quality/global-optimal-real-QE claim. '
        'Actual native-sleep and restore intervals determine H; EP detach alone receives no physical-free credit.','',
        'Full report: `quota_pair_report/RESULTS.md`.','',
        '## Eager controls and endpoint quality','']
    for policy in ('vanilla','planned_f4'):
        logs=list((ROOT/f'eager_controls/seed_0/{policy}/fig6').glob('epoch_001*/logs/*.txt'))
        if not logs:lines.append(f'- {policy}: not started.');continue
        text=logs[0].read_text(errors='replace')
        metrics=[line for line in text.splitlines() if re.search(r'step:\d+ - ',line)]
        result=ROOT/f'eager_controls/seed_0/{policy}/EAGER_RESULT.json'
        if result.exists():
            r=json.loads(result.read_text())
            lines.append(f"- {policy}: completed5-step eager epoch, rollout {r['rollout_seconds']:.2f}s, "
                f"Q {r['response_tokens_s']:.2f} response tok/s, reward {r['reward']:.6f}.")
        else:lines.append(f'- {policy}: {len(metrics)}/5 committed step metrics; not yet a validated epoch.')
        scored=ROOT/f'heldout_results/{policy}/RESULT.json'
        if scored.exists():
            r=json.loads(scored.read_text())
            lines.append(f"  Held-out actual {r['questions']} questions, mean {r['mean_score']:.6f}, empty answers {r['empty_answers']}.")
        else:lines.append('  Held-out endpoint quality not yet complete.')
    if graph.exists():lines+=['','Full Graph control report: `graph_control_report/RESULTS.md`.']
    n2=ROOT/'heldout_results/natural_f2_graph_trained/RESULT.json'
    if n2.exists():
        row=json.loads(n2.read_text())
        lines+=['',f"Graph-trained N2+Sidecar-ON epoch1 endpoint: {row['mean_score']:.6f} on the same256 questions, "
            f"empty answers {row['empty_answers']}. This is not a P4 Graph-trained endpoint or a graph-only precision effect."]
    lines+=['','## Additional code-domain actual execution','']
    for policy in ('vanilla','planned_f4'):
        arm=ROOT/f'code_task/seed_401/{policy}'
        audited=arm/'GENERATION_RESULT.json'
        if audited.exists():
            row=json.loads(audited.read_text())
            if json.loads((arm/'DRIVER_RESULT.json').read_text())['exit_code']!=0:
                raise RuntimeError('Code generation report without actual parent exit0')
            lines.append(f"- {policy}: completed5-step frozen epoch, {row['actual_full_audits']} actual FULL rank-step audits, "
                f"rollout {row['rollout_seconds']:.2f}s, effective response throughput {row['response_tokens_s']:.2f} tok/s, "
                f"{row['preemptions']} recovered KV preemptions. Primary reward is dummy0, not code accuracy.")
        elif (arm/'LAUNCH.json').exists():
            lines.append(f'- {policy}: launched; complete generation/performance/quality not yet validated.')
        else:lines.append(f'- {policy}: not launched.')
    if complete:
        report=json.loads(code.read_text())
        for name,row in report['arms'].items():
            lines.append(f"- Final {name} code sampled pass@1: {row['sandbox_sampled_pass_at_1']:.4%}, "
                f"{row['scored_answers']} fixed sample0 answers. Initial native-test wire-format scores were withdrawn; "
                '21 schema-affected answers/arm regraded once,139 untouched scores/arm retained, no NPU rerun.')
        lines+=['','All five minimal evidence sets passed the final raw-backed audit. Full results and scope: `RESULTS_FINAL.md`.']
    else:lines+=['','Code-domain final comparison and sampled pass@1 remain pending until both complete arms and fixed-sample0 sandbox grading pass.']
    lines+=['',
        'The earlier pre-NPU multi-turn history-mapping failure is archived; it is not a timing sample. '
        'Original baseline native recovery audit passed unaided; no derived log or primary-source modification was used.']
    lines+=['','The formal final eager protocol uses the exact historical Graph source fingerprint, '
        'same public actor, seed0/epoch1/input/KV/native-sleep2/lifecycle1, CPU metadata0 in both paths. '
        'Eager/TQ2 versus FULL/TQ1 is a configured-path comparison, not a one-bit replay causal estimate. '
        'The device-metadata1 pilot was intentionally stopped for protocol alignment and is archived, not a runtime failure or final timing sample.','',
        '## Additional preparation and scope','',
        '- Held-out: actual pinned CPU dataset loader verified256 disjoint fixed questions and GTs. '
        'A separate32-row train-only bookkeeping loader is unused by val_only. Completed endpoint scores above come only from actual NPU inference, never CPU loader checks.',
        '- Code domain: same ordered160 original questions/labels, identity metadata normalized only, '
        'completed31 repair swaps replayed and all groups freshly verified feasible. '
        'CPU admission proof is separate from the actual NPU arm status above. '
        'Quality uses fixed sample_index0, OS-isolated full-suite grading, no host execution.',
        '- Training quality statements remain short-endpoint/local-held-out scope. No convergence or equivalence claim.','',
        'Live detail/provenance/negative findings: `WORKLOG.md`.']
    (ROOT/'RESULTS_STATUS.md').write_text('\n'.join(lines)+'\n')
    print('Wrote RESULTS_STATUS.md; all-five-audited='+str(complete))


if __name__=='__main__':main()
