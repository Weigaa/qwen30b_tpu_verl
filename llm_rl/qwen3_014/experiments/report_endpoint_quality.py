#!/usr/bin/env python3
"""Report actual fixed-panel endpoint inference, never training reward."""
import json
import eager_controls as protocol


def main():
    scores = {}; pending = []
    for policy in ('vanilla','planned_f4','natural_f2_graph_trained'):
        output = protocol.ROOT/f'heldout_results/{policy}'
        if not (output/'RESULT.json').is_file():
            pending.append(policy);continue
        if policy == 'natural_f2_graph_trained':
            endpoint = protocol.ROOT/'quality/ada_n2_seed0_epoch1/global_step_5'
        else:
            epoch = next((protocol.ROOT/f'eager_controls/seed_0/{policy}/fig6').glob('epoch_001_*'))
            endpoint = epoch/'checkpoints/qwen3moe_for_eagle3/global_step_5'
        scores[policy] = protocol.parse_evaluation(output,endpoint)
    if not {'vanilla','planned_f4'} <= scores.keys():
        raise RuntimeError('Both primary endpoint evaluations must actually finish')
    a = scores['vanilla']['scores_by_prompt_sha256']; b = scores['planned_f4']['scores_by_prompt_sha256']
    if set(a) != set(b):raise RuntimeError('Different held-out questions')
    training_inputs = {}
    for policy in ('vanilla','planned_f4'):
        epoch = next((protocol.ROOT/f'eager_controls/seed_0/{policy}/fig6').glob('epoch_001_*'))
        training_inputs[policy] = {(r['input'],str(r['gts'])) for step in range(1,6)
            for r in protocol.records(epoch/f'rollout_data/{step}.jsonl')}
    if training_inputs['vanilla'] != training_inputs['planned_f4'] or len(training_inputs['vanilla']) != 160:
        raise RuntimeError('Training prompt/GT sets differ')
    result = dict(status='HELDOUT_ENDPOINTS_PARTIAL' if pending else 'HELDOUT_THREE_ENDPOINTS_AUDITED',
        endpoints=scores,pending=pending,questions=256,
        training_input_GT_sets_equal=True,training_unique_prompts=160,
        planned_minus_vanilla_percentage_points=100*sum(b[k]-a[k] for k in a)/256,
        vanilla_only_correct=sum(a[k]==1 and b[k]!=1 for k in a),
        planned_only_correct=sum(b[k]==1 and a[k]!=1 for k in a),
        scope='Five real updates from shared math epoch0, one training seed. Vanilla/P4 training prompt sets match, '
            'but method-specific batching/order and generation trajectories may differ. Same prespecified256 disjoint held-out questions; '
            'greedy n1, nonelastic eager/TQ2/native-sleep2 serving for every endpoint. '
            'The separate N2 endpoint was trained with FULL graph and Sidecar ON. '
            'Point differences are observed evidence, not proof of no quality effect, equivalence or final convergence.')
    out = protocol.ROOT/'heldout_report';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Short-training endpoint held-out quality','',
        '| Trained endpoint | Correct / questions | Accuracy | Empty answers |',
        '|---|---:|---:|---:|']
    for name,row in scores.items():
        lines.append(f"| {name} | {sum(row['scores_by_prompt_sha256'].values()):.0f}/256 | {100*row['mean_score']:.2f}% | {row['empty_answers']} |")
    lines += ['',f"P4 minus Vanilla {result['planned_minus_vanilla_percentage_points']:+.6f} percentage points. "
        f"Vanilla-only correct {result['vanilla_only_correct']}; P4-only correct {result['planned_only_correct']}.",
        '',result['scope'],'','Pending: '+(', '.join(pending) if pending else 'none')+'.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='endpoints'}))


if __name__ == '__main__':main()
