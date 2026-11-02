#!/usr/bin/env python3
"""One explicit OFF arm after the existing N2 job; never relaunch its handle."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time

ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')
SOURCE=Path('/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main')
ORIGINAL_SOURCE='/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main'
ON=Path('/data/adafloor_graph_results/mlsys_v014_graph_sidecar_full_epoch_seed101_calibrated_20261023T151600Z')
N2=Path('/data/adafloor_graph_results/mlsys_v014_graph_sidecar_natural_f2_seed0_20261023T172800Z')
OFF=ROOT/'p4_explicit_off'
PID=2920069


def state(value,**extra):
    ROOT.mkdir(parents=True,exist_ok=True)
    p=ROOT/'P4_OFF_QUEUE_STATE.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=value,observed_time=time.time(),**extra),indent=2)+'\n');tmp.replace(p)


def signature(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()[19]
    except FileNotFoundError:return None


def wait_for_n2():
    original=signature(PID)
    while original is not None and signature(PID)==original:
        state('WAITING_FOR_LIVE_N2_CONTROLLER',controller_pid=PID,start_ticks=original)
        time.sleep(15)
    result=N2/'DRIVER_RESULT.json';audit=N2/'TRAINING_SIDECAR_RESULT.json'
    if (not result.exists() or json.loads(result.read_text())['exit_code']!=0
            or not audit.exists() or json.loads(audit.read_text())['status']!='PASS'):
        raise RuntimeError('N2 ended without successful two-epoch audit; no automatic restart or next GPU launch')
    while True:
        smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
        if smi.returncode or 'Process id' not in smi.stdout:raise RuntimeError('Cannot verify idle NPUs')
        footer=smi.stdout.rsplit('Process id',1)[1]
        processes=re.findall(r'^\|\s+\d+\s+\d+\s+(\d+)\s+',footer,re.M)
        if not processes:return smi.stdout
        state('WAITING_FOR_IDLE_NPUS',observed_process_ids=processes);time.sleep(15)


def primary_code_matches():
    count=0
    with tarfile.open(ON/'execution_sources_complete.tar.gz') as archive:
        for member in archive.getmembers():
            name=member.name.removeprefix('./')
            if not member.isfile() or name.split('/')[0] not in ('verl','vllm','vllm_ascend','megatron'):continue
            path=SOURCE/name;count+=1
            if not path.exists() or hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(archive.extractfile(member).read()).digest():
                raise RuntimeError('Primary source differs from original ON: '+name)
    return count


def normalize(value):
    return str(value).replace(str(ON),str(OFF)).replace(ORIGINAL_SOURCE,str(SOURCE))


def input_digest(path):
    h=hashlib.sha256();count=0
    with path.open() as stream:
        for line in stream:
            row=json.loads(line);count+=1
            values=[row[k] for k in ('input','prompts','rollout_prompt_hash','prompt_occurrence_ordinal',
                                     'rollout_sample_index','rollout_request_seed','rollout_rank','gts')]
            h.update(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()+b'\n')
    if count!=512:raise RuntimeError('Incomplete paired input')
    return h.hexdigest()


def compare():
    spec=importlib.util.spec_from_file_location('off_paired_rows',ON/'audit_full_epoch.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    current=next(OFF.glob('*/epoch_001_mode1_planned'));reference=next(ON.glob('*/epoch_001_mode1_planned'))
    logs=list((current/'logs').glob('*.txt'))
    if len(logs)!=1:raise RuntimeError('Missing unique OFF log')
    text=logs[0].read_text(errors='replace')
    if not re.search(r'\[run\] end_time=\S+ exit_code=0',text):raise RuntimeError('OFF child did not finish')
    if list((current/'sidecar').glob('infer.log')):raise RuntimeError('OFF unexpectedly created Sidecar inference')
    audits=[json.loads(x) for x in re.findall(r'MODE1_STEP_RUNTIME_AUDIT (\{[^\n]+\})',text)]
    if len(audits)!=80 or any(not a['complete'] or a.get('full_iterations',0)<=0 for a in audits):
        raise RuntimeError('Missing actual OFF FULL graph execution')
    metrics=[dict(re.findall(r'([\w/]+):([^ ]+)',line)) for line in text.splitlines() if re.search(r'step:\d+ - ',line)]
    if len(metrics)!=5:raise RuntimeError('Missing five OFF metrics')
    on=json.loads((ON/'FINAL_EPOCH_AUDIT.json').read_text());rows=[]
    for step,m in enumerate(metrics,1):
        a=current/f'rollout_data/{step}.jsonl';b=reference/f'rollout_data/{step}.jsonl'
        if input_digest(a)!=input_digest(b):raise RuntimeError('Paired checkpoint/input/seed/rank contract mismatch')
        x=helper.rows_digest(a);y=helper.rows_digest(b);duration=float(m['rollout/rollout_output_time_s'])
        if float(m['actor/lr'])!=0 or float(m['response/aborted_ratio'])!=0:raise RuntimeError('Not frozen or aborted')
        rows.append(dict(step=step,off_seconds=duration,on_seconds=on['steps'][step-1]['rollout_seconds'],
            off_response_tokens=x['valid_response_tokens'],on_response_tokens=y['valid_response_tokens'],
            off_reward=x['reward_mean'],on_reward=y['reward_mean'],same_input_contract=True,
            exact_response_match=x['response_sha256']==y['response_sha256']))
    result=dict(status='EXPLICIT_OFF_ON_PAIR_COMPLETE',on=str(ON),off=str(OFF),steps=rows,
        off_flag='VERL_SIDECAR_ENABLE=0',same_primary_code_files=primary_code_matches(),
        sidecar_on=on['sidecar'],scope='One frozen seed101 pair; no confidence interval or universal overhead bound.')
    (ROOT/'P4_EXPLICIT_PAIR_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    primary_code_matches();smi=wait_for_n2()
    primary_code_matches()
    if OFF.exists():raise RuntimeError('Refusing existing OFF directory')
    OFF.mkdir();(OFF/'NPU_BEFORE.txt').write_text(smi)
    spec=importlib.util.spec_from_file_location('off_launch_base',SOURCE/'tools/run_mlsys_fig6.py')
    suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
    original=json.loads((ON/'v014_launch.json').read_text());env=suite.arm_env(OFF,101,'planned_f4')
    env.update({k:normalize(v) for k,v in original['env'].items()})
    env['VERL_SIDECAR_ENABLE']='0';env['PYTHONPATH']=str(SOURCE)
    command=[normalize(x) for x in original['command']]
    for key in ('XDG_CACHE_HOME','TORCHAIR_CACHE_HOME','TRITON_CACHE_DIR','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','RAY_TMPDIR'):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    launch=dict(command=command,execution_sha256=suite.execution_digest(),
        env={k:v for k,v in env.items() if k.startswith(('VLLM_','VERL_','FAIR_','ADAFLOOR_','DYNAMIC_'))
             or k in ('TASK_QUEUE_ENABLE','HCCL_BUFFSIZE','PYTORCH_NPU_ALLOC_CONF','HCCL_IF_BASE_PORT')})
    (OFF/'v014_launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    with tarfile.open(OFF/'execution_sources_complete.tar.gz','w:gz') as archive:
        files=list(SOURCE.glob('*.sh'))
        for directory in ('tools','internal','verl','vllm','vllm_ascend','megatron'):
            files.extend(p for p in (SOURCE/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.yaml'))
        for p in sorted(set(files)):archive.add(p,arcname=str(p.relative_to(SOURCE)))
    state('OFF_RUNNING',command=command)
    with (OFF/'driver.log').open('x') as log:rc=subprocess.run(command,env=env,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT).returncode
    (OFF/'DRIVER_RESULT.json').write_text(json.dumps(dict(exit_code=rc))+'\n')
    if rc:raise RuntimeError(f'OFF child failed with exit{rc}; retained, not restarted')
    if suite.execution_digest()!=launch['execution_sha256']:raise RuntimeError('OFF runtime sources changed')
    compare();state('PAIR_COMPLETE')


if __name__=='__main__':
    try:main()
    except Exception as error:
        state('STOPPED_WITH_ERROR',error=repr(error));raise
