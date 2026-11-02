"""One bounded full-capacity decode/stop test, excluded from formal timings."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import run_n4_sidecar_models as m


def group_alive(pgid):
    output=subprocess.run(['ps','-eo','pgid=,stat='],capture_output=True,text=True,check=True).stdout
    return any(int(parts[0])==pgid and not parts[1].startswith('Z')
               for row in output.splitlines() if len(parts:=row.split())==2)


def events(path):
    reader=m.c.module('stop_event_reader',m.SECONDARY/'tools/run_sidecar_graph_parity.py')
    return reader.read_events(path) if path.exists() else []


def verify(root,name):
    destination=root/'stop_preflight'/name
    result_path=destination/'RESULT.json'
    if result_path.exists():
        value=json.loads(result_path.read_text())
        if value['status']!='PASS':raise RuntimeError('Failed stop check retained: '+name)
        return value
    m.c.idle(root);destination.mkdir(parents=True,exist_ok=False)
    from sidecar_residual_preflight import reserve
    saved=json.loads((root/'preflight'/name/'graph/launch.json').read_text())
    env=os.environ.copy();env.update(saved)
    residual_gib=json.loads((root/'PROTOCOL.json').read_text()).get('residual_preflight_gib_per_device',{}).get(name,0)
    with reserve(destination/'co_resident_residual',residual_gib,env):
        return _verify_reserved(root,name,destination)


def _verify_reserved(root,name,destination):
    result_path=destination/'RESULT.json'
    saved=json.loads((root/'preflight'/name/'graph/launch.json').read_text())
    env=os.environ.copy();env.update(saved)
    env.update(VERL_SIDECAR_MAX_PROMPTS='1024',VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE='128',
        VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA='128',VERL_SIDECAR_REPEAT_UNTIL_KILLED='1',
        VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS='0',
        VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM='1',
        VERL_SIDECAR_OUTPUT_FILE=str(destination/'outputs.jsonl'),
        VERL_SIDECAR_LOG_FILE=str(destination/'infer.log'),
        VERL_SIDECAR_STATE_DIR=str(destination/'state'),
        VERL_SIDECAR_STOP_FILE=str(destination/'stop'),
        XDG_CACHE_HOME=str(destination/'cache'),ASCEND_CACHE_PATH=str(destination/'cache/ascend'))
    env['PYTHONPATH']=str(m.c.SIDECAR)+':'+env.get('PYTHONPATH','')
    m.c.write(destination/'LAUNCH.json',dict(diagnostic_only=True,
        cwd=str(m.c.SIDECAR),wrapper=str(m.WRAPPER),
        env={k:v for k,v in env.items() if k.startswith(('VERL_','VLLM_','PYTHONPATH'))},
        protocol='Full serving capacities, file stop after all8 worker FULL replay receipts; original3s soft/2s TERM grace; require durable checkpoint and no SIGKILL'))
    m.c.state(root,'STOP_PREFLIGHT_'+name)
    stop_time=None;term_time=None;kill_time=None
    try:
        with (destination/'driver.log').open('x') as log:
            # Match the formal watcher's cwd and wrapper, not the standalone
            # sidecar checkout. Python -m registry imports must stay isolated
            # even with a primary checkout first on the inherited path.
            process=subprocess.Popen(['bash',str(m.WRAPPER)],
                cwd=m.c.SIDECAR,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+1200
            replay_at=None
            while group_alive(process.pid):
                if time.monotonic()>deadline:raise RuntimeError('Bounded stop check timed out')
                now=time.monotonic()
                if stop_time is None:
                    records=events(destination/'infer.log')
                    replay={x['pid'] for x in records if x['event']=='sidecar_aclgraph_replay'
                            and x['runtime_mode']=='FULL' and not x['capture_phase']}
                    if len(replay)==8 and replay_at is None:replay_at=now
                    if replay_at is not None and now-replay_at>=5:
                        (destination/'stop').write_text('diagnostic file stop\n')
                        stop_time=now;stop_wall=time.time()
                elif now-stop_time>=3 and term_time is None:
                    os.killpg(process.pid,signal.SIGTERM);term_time=now
                elif now-stop_time>=5 and kill_time is None:
                    os.killpg(process.pid,signal.SIGKILL);kill_time=now
                time.sleep(.2)
            code=process.wait(timeout=10)
    except BaseException:
        if 'process' in locals() and group_alive(process.pid):
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=20)
        raise
    elapsed=time.monotonic()-stop_time if stop_time is not None else None
    records=events(destination/'infer.log')
    checkpoints=[x for x in records if x['event']=='sidecar_soft_stop_checkpointed']
    # Count all8 physical workers, not just the DP shard leaders.
    checkpoint_count=len(checkpoints)
    outputs=[]
    aggregate=destination/'outputs.jsonl'
    if aggregate.exists():outputs=[json.loads(x) for x in aggregate.read_text().splitlines() if x]
    states=[json.loads(x) for p in (destination/'state').glob('completed.shard*.jsonl')
            for x in p.read_text().splitlines() if x]
    key=lambda row:(row.get('sidecar_epoch',0),row['prompt_id'])
    inflight=[p.name for p in (destination/'state').glob('inflight.shard*.json')
              if json.loads(p.read_text()).get('prompt_ids')]
    partial_bytes=sum(p.stat().st_size for p in (destination/'state').rglob('partials.shard*.jsonl'))
    passed=(stop_time is not None and kill_time is None and checkpoint_count==8
            and not inflight and {key(x) for x in outputs}=={key(x) for x in states}
            and partial_bytes<32*1024**2 and code in (0,-signal.SIGTERM,128+signal.SIGTERM))
    value=dict(status='PASS' if passed else 'FAIL',diagnostic_only=True,exit_code=code,
        stop_wall=stop_wall if stop_time is not None else None,
        stop_to_physical_group_exit_s=elapsed,sigterm=term_time is not None,sigkill=kill_time is not None,
        checkpointed_workers=checkpoint_count,completed=len(outputs),completed_states=len(states),
        uncleared_inflight=inflight,partial_snapshot_bytes=partial_bytes,
        all8_workers_actual_FULL_replay=True if stop_time is not None else False)
    m.c.write(result_path,value)
    if not passed:raise RuntimeError(f'{name} full-capacity stop check failed: {value}')
    return value
