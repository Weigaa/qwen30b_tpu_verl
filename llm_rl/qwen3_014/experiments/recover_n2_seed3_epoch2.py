#!/usr/bin/env python3
"""Recover only a proven pre-execution Ray startup failure; preserve epoch1.

This host-only adapter leaves every pinned benchmark source unchanged. The
original fair runner's completion, checkpoint and quality validators are kept.
It extends its Planned-only completed-epoch resume guard to Natural-F2 for
this single, explicitly audited incident. No generation is retried or selected.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

import run_n2_f8_campaign as campaign


def require_preexecution_failure(text):
    if ('The actor never ran - it was cancelled before it started running.' not in text
            or "Owner's node has crashed" not in text):
        raise RuntimeError('Not the audited Ray pre-execution failure')
    for marker in ('rollout_output_time_s:', 'MODE1_STEP_RUNTIME_AUDIT ',
                   'rollout_forward_begin', 'Initializing a V1 LLM engine'):
        if marker in text:
            raise RuntimeError('Epoch2 executed benchmark work; refuse automatic retry')


def resume_script(original, source, storage, run):
    anchor='SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)'
    guard='    if ! is_planned_variant \\\n'
    checkpoint='    resume_epoch1="$VALIDATED_EPOCH_DIR"\n'
    if any(original.count(x)!=1 for x in (anchor, guard, checkpoint)):
        raise RuntimeError('Pinned fair resume layout changed')
    text=original.replace(anchor, 'SCRIPT_DIR='+shlex.quote(str(source)))
    text=text.replace(guard, '    if [[ "$variant" != adafloor_n_f2 ]] \\\n')
    # The resume guard must see only epoch1. Install the owned split-disk link
    # AFTER that guard and epoch1 validation, before epoch2 planning/execution.
    target=run/'epoch_002_mode1_natural'
    setup=(checkpoint+'    mkdir -p -- '+shlex.quote(str(target))+'\n'
           +'    ln -s -- '+shlex.quote(str(storage))+' '
           +shlex.quote(str(target/'checkpoints'))+'\n')
    return text.replace(checkpoint, setup)


def natural_resume_script(original, source):
    anchor='SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)'
    guard='if [[ ! -d "$BASELINE_EPOCH0/rollout_data" ]]; then'
    if original.count(anchor)!=1 or original.count(guard)!=1:
        raise RuntimeError('Pinned Natural wrapper layout changed')
    # The existing dynamic epoch driver accepts colon-separated EMA history.
    # Its outer Natural wrapper validated that list as if it were one path.
    # Validate the public first component here, keep the COMPLETE list passed
    # to the unchanged planner, and retain every runtime variable verbatim.
    return original.replace(anchor,'SCRIPT_DIR='+shlex.quote(str(source))).replace(
        guard,'if [[ ! -d "${BASELINE_EPOCH0%%:*}/rollout_data" ]]; then')


def recover(root, seed):
    if seed!=3:raise RuntimeError('Recovery restricted to the diagnosed seed3 incident')
    campaign.check_sources(root);campaign.idle(root)
    suite=campaign.main_suite();arm=root/'primary/seed_3/natural_f2';run=arm/'fig6'
    stage=arm/'ray_startup_recovery_v2'
    if stage.exists():raise RuntimeError('Recovery already attempted; inspect it, no blind retry')
    first=run/'epoch_001_mode1_natural';second=run/'epoch_002_mode1_natural'
    first_result=suite.parse_epoch(first,require_tailguard=True)
    if not (first/'offline_planning_history.json').is_file():
        raise RuntimeError('Committed epoch1 history missing')
    shards=list((first/'checkpoints/qwen3moe_for_eagle3/global_step_5/actor/dist_ckpt').glob('*.distcp'))
    if len(shards)!=32:raise RuntimeError('Expected original32 epoch1 actor shards')
    failed=arm/'ray_startup_recovery/failed_epoch2_before_actor_execution'
    prior=arm/'ray_startup_recovery'
    if (json.loads((prior/'DRIVER_RESULT.json').read_text())['exit_code']!=2
            or 'missing reusable epoch0 rollout data:' not in (prior/'driver.log').read_text()
            or any(p.is_file() for p in second.rglob('*') if not p.is_symlink())):
        raise RuntimeError('Unexpected first host adapter result; refuse retry')
    logs=list((failed/'logs').glob('*.txt'))
    if len(logs)!=1:raise RuntimeError('Ambiguous failed epoch2 logs')
    text=logs[0].read_text();require_preexecution_failure(text)
    old_launch=json.loads((arm/'LAUNCH.json').read_text())
    ray=Path(old_launch['env']['RAY_TMPDIR'])/'ray/session_latest/logs'
    ray_error=(ray/'raylet.err').read_text()
    if 'runtime env agent timed out' not in ray_error:
        raise RuntimeError('Missing diagnosed Ray runtime-env-agent startup timeout')
    storage=Path('/workspace/adafloor_graph_results')/(root.name+'_epoch2_math_seed3')
    ownership=dict(campaign=str(root),label='math_seed3',storage=str(storage))
    if json.loads((storage/'OWNED_BY_N2_F8_CAMPAIGN.json').read_text())!=ownership:
        raise RuntimeError('Wrong checkpoint storage ownership')
    if any(p.is_symlink() or (p.is_file() and p!=storage/'OWNED_BY_N2_F8_CAMPAIGN.json')
           for p in storage.rglob('*')):
        raise RuntimeError('Failed epoch2 already wrote checkpoint data')
    if not (second/'checkpoints').is_symlink() or (second/'checkpoints').resolve()!=storage:
        raise RuntimeError('Wrong epoch2 checkpoint storage link')
    stage.mkdir()
    campaign.snapshot_raw(stage/'failed_raw_taskrunner_logs',old_launch['env']['RAY_TMPDIR'])
    archive=stage/'failed_host_preflight_no_trainer'
    second.rename(archive)
    original=campaign.MAIN/'run_paper_fair_epoch1_2_from_common_epoch0.sh'
    script=stage/'resume_fair_natural_epoch2.sh'
    script.write_text(resume_script(original.read_text(),campaign.MAIN,storage,run))
    natural=Path(old_launch['env']['ADAFLOOR_GRAPH_BASE_RUNNER'])
    natural_snapshot=stage/'natural_wrapper_colon_history.sh'
    natural_snapshot.write_text(natural_resume_script(natural.read_text(),campaign.MAIN))
    natural_snapshot.chmod(0o755)
    env=campaign.new_capacity(suite.arm_env(root/'primary',3,'natural_f2'))
    env.update(FAIR_RESUME_EXISTING='1',RAY_agent_register_timeout_ms='120000')
    env['ADAFLOOR_GRAPH_BASE_RUNNER']=str(natural_snapshot)
    command=suite.arm_command(3,'natural_f2');command[1]=str(script)
    campaign.write(stage/'RECOVERY_PROTOCOL.json',dict(
        status='PROVEN_PREEXECUTION_STARTUP_FAILURE_ONLY',
        recovery_controller_sha256=campaign.sha(__file__),
        original_fair_runner=str(original),original_fair_runner_sha256=campaign.sha(original),
        host_resume_snapshot_sha256=campaign.sha(script),
        original_natural_wrapper_sha256=campaign.sha(natural),
        host_natural_snapshot_sha256=campaign.sha(natural_snapshot),
        original_epoch1=first_result,original_epoch1_shards=[str(p) for p in shards],
        failed_log_sha256=campaign.sha(logs[0]),original_failed_epoch=str(failed),
        original_launch_sha256=campaign.sha(arm/'LAUNCH.json'),
        failed_driver_receipt=json.loads((arm/'DRIVER_RESULT.json').read_text()),
        raylet_failure=ray_error,ray_startup_timeout_ms=120000,
        budgets_and_benchmark_execution_unchanged=True,
        generation_repeated=False,scope='Seed3 epoch2 actor never started. Preserve epoch1 and original failed attempt. '
        'Resume only epoch2 from original epoch1 checkpoint/history. Original validator audits both epochs before cleanup. '
        '120s Ray agent registration grace is startup-only, outside rollout timing.'))
    campaign.launch(root,'MATH_N2_SEED3_RECOVER_EPOCH2',stage,campaign.MAIN,env,command)
    # Fresh evidence, not a synthetic successful receipt for the failed child.
    result=suite.parse_arm(root/'primary',3,'natural_f2')
    if result['epochs'][0]!=first_result:raise RuntimeError('Previously completed epoch1 changed')
    campaign.occupancy_for(suite,result,arm/'RANK_OCCUPANCY.json')
    campaign.write(arm/'ARM_PASS.json',result)
    campaign.write(arm/'REVIEW.json',dict(status='NEW_BUDGET_SEED_AUDITED',new=result,
        old_budget=suite.parse_arm(campaign.OLD,3,'natural_f2'),
        unchanged_full16_reference=suite.parse_arm(campaign.OLD,3,'vanilla'),
        source_unchanged=True,preexecution_recovery=str(stage/'RECOVERY_PROTOCOL.json')))
    campaign.split_cleanup(root,'math_seed3',storage,run)
    campaign.write(stage/'RECOVERY_PASS.json',dict(
        status='ORIGINAL_EPOCH1_AND_RECOVERED_EPOCH2_VALIDATED',
        original_failure_preserved=True,original_epoch1_unchanged=True,
        both_epochs_actual_full_graph_audits=160,preemptions=result['preemptions']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve()
    campaign.math_arm=recover
    # Applied only to Ray initialization, never to model or planner settings.
    os.environ['RAY_agent_register_timeout_ms']='120000'
    sys.argv=['run_n2_f8_campaign.py','resume','--root',str(root)]
    campaign.main()


if __name__=='__main__':main()
