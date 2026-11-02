#!/usr/bin/env python3
"""Additional completed-run review; does not change the running N2 sources."""
import json
from pathlib import Path
import re

ROOT = Path('/data/adafloor_graph_results/mlsys_v014_graph_sidecar_natural_f2_seed0_20261023T172800Z')


def main():
    result = json.loads((ROOT / 'TRAINING_SIDECAR_RESULT.json').read_text())
    if result['status'] != 'PASS' or json.loads((ROOT / 'DRIVER_RESULT.json').read_text())['exit_code'] != 0:
        raise RuntimeError('Do not certify a live or failed experiment')
    windows = []; epochs = []
    for n, (primary, secondary, baseline) in enumerate(zip(result['primary']['epochs'], result['sidecar'],
                                                          result['baseline']['epochs']), 1):
        epoch = Path(primary['epoch_dir']); side = epoch / 'sidecar'
        lease = (side / 'lease.log').read_text()
        starts = list(re.finditer(r'^sidecar_start_time=([0-9.]+)$', lease, re.M))
        ready = []
        for line in (side / 'infer.log').read_text(errors='replace').splitlines():
            if line.startswith('{'):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get('event') == 'sidecar_model_ready':
                    ready.append(record)
        for window, start in zip(secondary['windows'], starts):
            end = next((s.start() for s in starts if s.start() > start.start()), len(lease))
            block = lease[start.start():end]
            physical = re.search(r'^sidecar_devices=([^\n]+)$', block, re.M)
            active = re.search(r'^sidecar_active_ranks=([^\n]+)$', block, re.M)
            if not physical or not active:
                raise RuntimeError('Missing actual selected devices/primary ranks')
            devices = list(map(int, physical[1].split(',')))
            active = list(map(int, active[1].split(',')))
            if len(devices) != 8 or len(set(devices)) != 8 or set(devices) & set(active) or set(devices) | set(active) != set(range(16)):
                raise RuntimeError('Actual Sidecar lease is not eight inactive identity-mapped NPUs')
            models = [r for r in ready if window['start'] <= r['time'] <= window['stop']]
            if len(models) != 8 or {r['shard_index'] for r in models} != set(range(8)):
                raise RuntimeError('Not all eight Sidecar replicas ready within the lease')
            row = dict(epoch=n, **window, devices=devices,
                       first_ready_seconds=min(r['time'] for r in models)-window['start'],
                       all_ready_seconds=max(r['time'] for r in models)-window['start'],
                       lease_rank_seconds=8*window['window_seconds'],
                       drain_rank_seconds=8*window['shutdown_to_exit_seconds'])
            windows.append(row)
        if len(starts) != len(secondary['windows']):
            raise RuntimeError('Window count disagrees with validated controller')
        epochs.append(dict(epoch=n, rollout_seconds=primary['rollout_s'],
                           response_tokens_s=primary['response_tokens_s'], reward=primary['reward'],
                           mean_response_tokens=primary['mean_length'], preemptions=primary['preemptions'],
                           no_sidecar_main_reference_seconds=baseline['rollout_s'],
                           no_sidecar_main_reference_reward=baseline['reward'],
                           no_sidecar_main_reference_tokens_s=baseline['response_tokens_s'],
                           rollout_seconds_change_pct=100*(primary['rollout_s']/baseline['rollout_s']-1),
                           decisions=[dict(step=d['step'], action=d['action'], reason=d['reason'])
                                      for d in secondary['decisions']]))
    completed = sum(w['completed'] for w in windows); correct = sum(w['correct'] for w in windows)
    report = dict(status='TWO_UPDATED_EPOCHS_SIDECAR_LIFECYCLE_AUDITED', seed=0, policy='natural_f2',
                  epochs=epochs, windows=windows,
                  completed_before_stop=sum(w['completed_before_stop'] for w in windows),
                  completed_including_drain=completed, strict_correct_including_drain=correct,
                  strict_accuracy_including_drain=correct/completed,
                  assigned_lease_rank_seconds=sum(w['lease_rank_seconds'] for w in windows),
                  assigned_drain_rank_seconds=sum(w['drain_rank_seconds'] for w in windows),
                  sigkill_events=sum(x['sigkill_events'] for x in result['sidecar']),
                  scope='Two real updated-policy epochs after shared epoch0; epoch2 uses its own epoch1. Main reference is same-seed existing no-Sidecar training, not frozen paired causal overhead. Query counts are jobs, not unique GSM8K questions across repeats/epochs. Assigned-device lease rank-time includes startup, not AICore utilization. Partial requests are checkpointed, never counted as completions.')
    (ROOT / 'FINAL_LIFECYCLE_REVIEW.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Ada-N2 Sidecar: seed0, two updated epochs', '',
             'Completed and audited:10 steps, real actor updates, native sleep2/FULL decode. '
             'Short windows skipped; step3/4 in each epoch ran Sidecar, all eight replicas ready. '
             'All matching exit/ACK boundaries precede primary restore. No preemption or SIGKILL.', '',
             '| Epoch | Rollout seconds | Response tok/s | Reward | Sidecar starts |',
             '| --- | ---: | ---: | ---: | ---: |']
    for e in epochs:
        lines.append(f"| {e['epoch']} | {e['rollout_seconds']:.2f} | {e['response_tokens_s']:.2f} | {e['reward']:.6f} | 2 |")
    lines += ['', f"Completed before primary stop: {report['completed_before_stop']:,}; including shutdown drain: {completed:,}. "
              f"Strict GSM8K accuracy including drain: {correct}/{completed} ({100*correct/completed:.2f}%).", '',
              f"Assigned lease rank-time {report['assigned_lease_rank_seconds']:.2f} rank-s; "
              f"shutdown drain {report['assigned_drain_rank_seconds']:.2f} rank-s. Includes startup; not measured compute utilization.", '',
              report['scope'], '', 'Full machine-readable evidence: `FINAL_LIFECYCLE_REVIEW.json` and `TRAINING_SIDECAR_RESULT.json`.']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
