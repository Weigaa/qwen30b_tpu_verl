#!/usr/bin/env python3
"""One prespecified Natural-F4 graph Sidecar arm, same seed0 and >60s gate as N2.

The existing runtime is read-only. Two updated epochs resume public epoch0 and
this run's own epoch1. Actual floor4 is sufficient; floor2 is never requested.
"""
import argparse
import json
from pathlib import Path
import shutil
import statistics

import run_n2_f8_campaign as c

NEW_N2 = Path('/data/adafloor_graph_results/mlsys_n2_f8_349952_20261025T143000Z')
PARITY = Path('/data/adafloor_graph_results/mlsys_v014_fullgraph_sidecar_parity_20261024T161726Z_native_nofx/PARITY_RESULT.json')
CAPS = '4:280576,8:349952,16:380800'


def helper():
    return c.module('n4_sidecar_original', c.SIDECAR/'tools/run_v014_sidecar_training.py')


def build_launch(root):
    h = helper()
    env, _ = h.build_launch(root, 'graph', 60.0, 'gt')
    n2 = h.suite.arm_env(root, 0, 'natural_f2')
    n4 = h.suite.arm_env(root, 0, 'natural_f4')
    # Keep every Sidecar override, replace only the primary method-specific base.
    env.update({key: value for key, value in n4.items() if n2.get(key) != value})
    env['FLOOR_KV_CAPS'] = CAPS
    command = h.suite.arm_command(0, 'natural_f4') + [
        'actor_rollout_ref.rollout.shrink_aware.target_policy=natural']
    assert env['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8'] == '349952'
    assert env['VERL_SIDECAR_ENABLE'] == '1' and env['VERL_SIDECAR_TASK_QUEUE_ENABLE'] == '1'
    assert env['FAIR_OUTPUT_ROOT'].endswith('/seed_0/natural_f4')
    assert env['ADAFLOOR_GRAPH_BASE_RUNNER'].endswith('adaptive_floor4_natural_full3.sh')
    assert env['VERL_SIDECAR_WINDOW_GATE_STRICT_MINIMUM'] == '1'
    return h, env, command


def validate_parity(h):
    proof = json.loads(PARITY.read_text())
    shadow = proof.get('same_input_shadow', {})
    exact_shadow = (proof.get('same_padded_input_graph_parity_exact') is True
        and shadow.get('status') == 'ALL_SAME_PADDED_INPUT_SHADOW_CHECKS_PASSED'
        and shadow.get('max_hidden_abs') == shadow.get('max_logits_abs') == 0.0
        and shadow.get('post_checks', 0) > 0
        and shadow.get('post_checks') == shadow.get('pre_checks')
        and shadow.get('post_mismatches') == shadow.get('pre_argmax_mismatches') == 0
        and shadow.get('diagnostic_generation_unchanged') is True)
    assert proof['status'] == 'PASS' and proof['checked_prompts'] == 16
    assert proof.get('greedy_token_ids_exact') is True or exact_shadow
    for key, path in [('sidecar_script_sha256', 'internal/run_elastic_sidecar_infer.sh'),
                      ('aclgraph_sha256', 'vllm_ascend/compilation/acl_graph.py'),
                      ('model_runner_sha256', 'vllm_ascend/worker/model_runner_v1.py')]:
        assert proof[key] == h.suite.digest(c.SIDECAR/path)
    return proof


def summarize(root):
    h = helper()
    primary = h.suite.parse_arm(root, 0, 'natural_f4')
    baseline = h.suite.parse_arm(c.OLD, 0, 'natural_f4')
    side = [h.sidecar_summary(Path(e['epoch_dir']), 'graph', 60., True)
            for e in primary['epochs']]
    n2 = json.loads((NEW_N2/'graph_sidecar/TRAINING_SIDECAR_RESULT.json').read_text())
    windows = [w for epoch in side for w in epoch['windows']]
    completed = sum(w['completed'] for w in windows)
    result = dict(status='PASS', policy='natural_f4', seed=0, epochs=[1, 2],
        actor_frozen=False, sidecar_execution='graph', floor_kv_caps=CAPS,
        window_gate_minimum_seconds=60, window_gate_comparison='>',
        baseline_comparison_scope='Same-seed updated-policy descriptive reference, not frozen causal ON/OFF',
        primary=primary, baseline=baseline, sidecar=side,
        actual_sidecar_starts=len(windows),
        completed_before_stop=sum(w['completed_before_stop'] for w in windows),
        completed_including_drain=completed,
        correct_including_drain=sum(w['correct'] for w in windows),
        strict_accuracy_including_drain=(sum(w['correct'] for w in windows)/completed if completed else None),
        primary_Q_change_percent=100*(primary['response_tokens_s']/baseline['response_tokens_s']-1),
        n2_reference=dict(path=str(NEW_N2/'graph_sidecar/TRAINING_SIDECAR_RESULT.json'),
            completed_before_stop=n2['completed_before_stop'],
            actual_sidecar_starts=n2['actual_sidecar_starts'],
            primary_Q=n2['primary']['response_tokens_s']),
        public_epoch0_preserved=h.suite.COMMON.is_dir())
    for epoch in primary['epochs']:
        audits = h.suite.json_records(Path(epoch['epoch_dir']), 'MODE1_STEP_RUNTIME_AUDIT ')
        assert len(audits) == 80 and all(a['complete'] and a['full_iterations'] > 0 for a in audits)
    c.write(root/'TRAINING_SIDECAR_RESULT.json', result)
    (root/'RESULTS.md').write_text('# Ada-N4 graph Sidecar\n\n'
        f"Primary Q {primary['response_tokens_s']:.2f} tokens/s. "
        f"Same-seed OFF difference {result['primary_Q_change_percent']:+.2f}%.\n\n"
        f"Starts {len(windows)}/10. Jobs before stop {result['completed_before_stop']}. "
        f"Including drain {completed}, correct {result['correct_including_drain']}.\n\n"
        'All starts use the same strict >60s gate and eight FULL/TQ1 TP1 replicas. '
        'Updated primary trajectories are not an exact frozen causal ON/OFF pair. '
        'Per-window startup, shutdown, decisions and replay evidence are in TRAINING_SIDECAR_RESULT.json.\n')
    c.state(root, 'COMPLETE', starts=len(windows), completed_before_stop=result['completed_before_stop'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('dry-run', 'run', 'summarize'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.mode == 'summarize':
        summarize(root)
        return
    h, env, command = build_launch(root)
    proof = validate_parity(h)
    preview = dict(policy='natural_f4', seed=0, epochs=[1, 2],
        common_epoch0=str(h.suite.COMMON), floor_kv_caps=CAPS,
        sidecar_execution='graph', gate='>60s', command=command,
        execution_sha256=h.suite.execution_digest(),
        sidecar_parity_proof_path=str(PARITY), sidecar_parity_proof_sha256=c.sha(PARITY),
        env={k: v for k, v in env.items() if k.startswith(('FAIR_', 'VLLM_', 'VERL_', 'DYNAMIC_', 'ADAFLOOR_'))
             or k in ('FLOOR_KV_CAPS', 'TASK_QUEUE_ENABLE', 'COMMON_EPOCH0_ROOT', 'PYTORCH_NPU_ALLOC_CONF')})
    if args.mode == 'dry-run':
        print(json.dumps(preview, indent=2))
        return
    root.mkdir(parents=True, exist_ok=False)
    c.write(root/'PROTOCOL.json', dict(
        status='PRESPECIFIED_SINGLE_N4_GRAPH_SIDECAR_ARM', method='natural_f4', seed=0,
        epochs=[1, 2], steps_each=5, same_seed_and_public_epoch0_as_new_N2=True,
        floor_kv_caps=CAPS, gate='>60s', primary='FULL/TQ1/native_sleep2/lifecycle1/KV-first',
        secondary='Qwen2.5-1.5B-Instruct/GSM8K train, eight TP1 FULL/TQ1 replicas',
        controller_sha256=c.sha(c.__file__), n4_controller_sha256=c.sha(__file__),
        execution_sha256={'qwen3_graph_sidecar_v014': h.suite.execution_digest()},
        baseline=str(c.OLD/'seed_0/natural_f4'), new_n2_reference=str(NEW_N2/'graph_sidecar'),
        stop_policy='Stop on failure, retain evidence, no extra seeds, no capacity reduction or eager fallback',
        checkpoint_policy='Own epoch1 actor/history to epoch2, original validation before private cleanup'))
    c.write(root/'v014_launch.json', preview)
    c.write(root/'SIDECAR_GREEDY_PARITY_PROOF.json', proof)
    shutil.copyfile(__file__, root/'controller_snapshot.py')
    storage = c.split_prepare(root, 'sidecar_n4')
    run = root/'seed_0/natural_f4/fig6'
    try:
        c.launch(root, 'GRAPH_SIDECAR_N4', root/'execution', c.SIDECAR, env, command,
                 run_dir=run, storage=storage)
        result = summarize(root)
        c.write(root/'seed_0/natural_f4/ARM_PASS.json', result['primary'])
        c.split_cleanup(root, 'sidecar_n4', storage, run)
    except Exception as error:
        c.state(root, 'STOPPED_WITH_ERROR', error=repr(error))
        raise


if __name__ == '__main__':
    main()
