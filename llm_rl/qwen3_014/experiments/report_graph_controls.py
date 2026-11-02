#!/usr/bin/env python3
"""Re-audit the two completed eager controls against historical FULL results."""
import hashlib
import json
from pathlib import Path
import re
import eager_controls as protocol


def main():
    suite = protocol.suite(); reports = []
    for policy in ('vanilla', 'planned_f4'):
        arm = protocol.ROOT/f'eager_controls/seed_0/{policy}'
        result = protocol.parse_eager(arm, policy)
        epoch = next((arm/'fig6').glob('epoch_001_*'))
        reference = protocol.GRAPH/f'seed_0/{policy}/fig6'/epoch.name
        common_actor = suite.COMMON/'epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5/actor'
        load_counts = {}
        for name, path in (('eager',epoch),('graph',reference)):
            text = next((path/'logs').glob('*.txt')).read_text(errors='replace')
            loads = re.findall(r'\[Rank (\d+)\] Loaded sharded model checkpoint from '+re.escape(str(common_actor)),text)
            if len(loads) != 16 or set(map(int,loads)) != set(range(16)):
                raise RuntimeError('Same common actor load not proved on all16 ranks')
            load_counts[name] = len(loads)
        def audits(path):
            return {(v['step'],v['rank']):v for v in suite.json_records(path,'MODE1_STEP_RUNTIME_AUDIT ')}
        eager, graph = audits(epoch), audits(reference)
        expected = {(s,r) for s in range(1,6) for r in range(16)}
        if set(eager) != expected or set(graph) != expected:
            raise RuntimeError('Missing actual rank-step coverage')
        if any(any(eager[k][field] != graph[k][field]
                for field in ('kv_blocks','kv_tokens','worker_kv_blocks','floor')) for k in expected):
            raise RuntimeError('Actual eager/graph per-rank KV or configured floor differs')
        hashes = {}; first_step = None
        for step in range(1,6):
            a = list(protocol.records(epoch/f'rollout_data/{step}.jsonl'))
            b = list(protocol.records(reference/f'rollout_data/{step}.jsonl'))
            if len(a) != 512 or len(b) != 512 or any((x['input'],x['gts']) != (y['input'],y['gts']) for x,y in zip(a,b)):
                raise RuntimeError('Actual ordered input/GT differs')
            hashes[step] = hashlib.sha256(json.dumps([(x['input'],x['gts']) for x in a],ensure_ascii=False).encode()).hexdigest()
            if step == 1:
                lengths_a = list(map(int,(epoch/'rollout_length/length_1.txt').read_text().split()))
                lengths_b = list(map(int,(reference/'rollout_length/length_1.txt').read_text().split()))
                first_step = dict(questions=512, identical_answer_texts=sum(x['output']==y['output'] for x,y in zip(a,b)),
                    identical_entire_response_length_vector=lengths_a==lengths_b,
                    reward_eager=sum(float(x['score']) for x in a)/512,
                    reward_graph=sum(float(x['score']) for x in b)/512,
                    scope='Before the first new actor update; no claim of token-level parity from text equality alone.')
        original = json.loads((arm/f'EAGER_TRAINING_{policy}_RESULT.json').read_text())
        if original['exit_code'] != 0:
            recovery = json.loads((arm/'POST_VALIDATION_RECOVERY.json').read_text())
            if recovery['status'] != 'EAGER_CHILD0_POST_VALIDATION_ONLY_RECOVERED':
                raise RuntimeError('Nonzero parent without independent native release audit')
            from native_eager_handoff import validate
            validate(next((epoch/'logs').glob('*.txt')).read_text(errors='replace'))
        g = result['graph_reference']
        reports.append(dict(policy=policy,eager_rollout_seconds=result['rollout_seconds'],
            graph_rollout_seconds=g['rollout_s'],eager_response_tokens=result['response_tokens'],
            graph_response_tokens=g['response_tokens'],eager_Q=result['response_tokens_s'],
            graph_Q=g['response_tokens_s'],graph_Q_gain_percent=100*(g['response_tokens_s']/result['response_tokens_s']-1),
            eager_reward=result['reward'],graph_reward=g['reward'],
            eager_Q_E=result['eager_Q_E'],graph_Q_E=result['graph_Q_E'],
            rank_step_actual_KV_floor_pairs_verified=80,ordered_input_GT_step_sha256=hashes,
            original_training_parent_result=original,first_step=first_step,
            common_actor_checkpoint=str(common_actor),actual_common_actor_load_rank_counts=load_counts,
            source_sha256=suite.execution_digest(),preemptions_eager=result['preemptions'],preemptions_graph=g['preemptions']))
    report = dict(status='TWO_MATCHED_V014_EAGER_GRAPH_CONTROLS_REAUDITED',seed=0,epoch=1,steps=5,arms=reports,
        report_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Identical common initial actor, ordered input/GT, actual per-rank/per-step KV/floor and historical execution source. '
        'Native sleep2/lifecycle1 in both. Configured fastest paths eager/TQ2 versus FULL/TQ1; not a one-bit graph-only causal effect. '
        'One training seed, five actual updates per arm. Later responses/weights may diverge; Q normalizes each actual response work total. '
        'No final convergence, statistical equivalence or population confidence interval claimed.')
    out = protocol.ROOT/'graph_control_report';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# v0.14 Graph controls: two matched eager epoch1 arms','',
        '| Method | Eager/TQ2 rollout s | FULL/TQ1 rollout s | Eager response tok/s | FULL response tok/s | Q gain | Eager reward | FULL reward |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        lines.append(f"| {r['policy']} | {r['eager_rollout_seconds']:.2f} | {r['graph_rollout_seconds']:.2f} | "
            f"{r['eager_Q']:.2f} | {r['graph_Q']:.2f} | {r['graph_Q_gain_percent']:+.2f}% | {r['eager_reward']:.6f} | {r['graph_reward']:.6f} |")
    lines += ['',report['scope'],'','Actual80 rank-step KV/floor matches and all2560 ordered input/GT pairs checked per method. '
        'Reward is on-policy training reward, not held-out quality. Q is effective response tokens divided by rollout return time '
        '(not console prompt-plus-response throughput). E=W/H charges native-sleep release and restore reoccupation; '
        'recorded-clock bounds and per-step evidence are in summary.json.','',
        'Planned-F4 original child completed all5steps with exit0; its parent exit3 was an offline eager validator requiring a graph-only '
        'invalidation event. Original result/log retained. Independent80-worker-step native sleep completion and positive allocator '
        'release counters audited before post-validation recovery; no training rerun or fabricated graph event.','',
        '## First step before any new actor update','']
    for r in reports:
        f=r['first_step']
        lines.append(f"- {r['policy']}: {f['identical_answer_texts']}/512 identical answer texts; entire response-length vector equality "
            f"{f['identical_entire_response_length_vector']}. Eager/Graph reward {f['reward_eager']:.6f}/{f['reward_graph']:.6f}.")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='arms'}))
    for r in reports:
        print(r['policy'],r['graph_Q_gain_percent'],r['first_step'])


if __name__ == '__main__':main()
