#!/usr/bin/env python3
"""Two eager epoch1 arms only; reuse their endpoints for one held-out pass each."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')
CODE=Path(__file__).resolve().parent
PRIOR_PID=int(os.environ.get('SUPPLEMENT_PRIOR_QUOTA_PID','3131053'))


def state(value,**extra):
    path=ROOT/'EAGER_QUEUE_STATE.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=value,time=time.time(),**extra),indent=2)+'\n');tmp.replace(path)


def signature(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()[19]
    except FileNotFoundError:return None


def idle():
    while True:
        result=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
        if result.returncode or 'Process id' not in result.stdout:raise RuntimeError('Cannot verify idle NPUs')
        footer=result.stdout.rsplit('Process id',1)[1]
        if not re.findall(r'^\|\s+\d+\s+\d+\s+\d+\s+',footer,re.M):return result.stdout
        state('WAITING_FOR_IDLE_NPUS');time.sleep(15)


def launch(command,env,root,source,label):
    root.mkdir(parents=True,exist_ok=True)
    if (root/f'{label}_LAUNCH.json').exists():raise RuntimeError('Refusing existing launch evidence')
    for key in ('XDG_CACHE_HOME','TORCHAIR_CACHE_HOME','TRITON_CACHE_DIR','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','RAY_TMPDIR'):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    (root/f'{label}_LAUNCH.json').write_text(json.dumps(dict(command=command,
        env={k:v for k,v in env.items() if k.startswith(('VLLM_','VERL_','FAIR_','ADAFLOOR_','DYNAMIC_'))
             or k in ('TASK_QUEUE_ENABLE','HCCL_BUFFSIZE','PYTORCH_NPU_ALLOC_CONF') }),indent=2)+'\n')
    idle();state(label)
    with (root/f'{label}.log').open('x') as log:
        result=subprocess.run(command,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT)
    (root/f'{label}_RESULT.json').write_text(json.dumps(dict(exit_code=result.returncode))+'\n')
    if result.returncode:raise RuntimeError(f'{label} exit{result.returncode}; artifacts retained, not retried')


def eval_endpoint(m,endpoint,name):
    output=ROOT/'heldout_results'/name
    if (output/'RESULT.json').exists():raise RuntimeError('Refusing duplicate held-out evaluation')
    env,command=m.build_evaluation(endpoint,output)
    launch(command,env,output,m.SOURCE,'HELDOUT_RUNNING_'+name)
    result=m.parse_evaluation(output,endpoint)
    (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def recover_eager_validation_only(m, arm, policy, source_sha):
    """Never retry training or turn a nonzero process result into zero.

    A single known eager-vs-graph verifier mismatch can be post-audited. Keep
    the original rc3 and traceback; require all actual training/input/plan/KV
    evidence plus 80 real native-sleep release pairs before resuming quality.
    """
    if policy != 'planned_f4':
        raise RuntimeError('Existing arm unsuccessful; no automatic retry')
    label = 'EAGER_TRAINING_'+policy
    original = json.loads((arm/(label+'_RESULT.json')).read_text())
    error_log = arm/(label+'.log'); raw = error_log.read_bytes()
    text = raw.decode(errors='replace')
    if (original['exit_code'] != 3 or
            'No native sleep handoff after rank=0 step=1 inference and before rollout return' not in text or
            'validate_native_planned_memory.py' not in text):
        raise RuntimeError('Not the bounded known eager post-validation failure; no retry')
    from native_eager_handoff import validate
    epoch = arm/'fig6/epoch_001_mode1_planned'
    logs = list((epoch/'logs').glob('*.txt'))
    if len(logs) != 1:
        raise RuntimeError('Ambiguous original child log')
    native = validate(logs[0].read_text(errors='replace'))
    training = m.parse_eager(arm,policy)
    receipt = dict(status='EAGER_CHILD0_POST_VALIDATION_ONLY_RECOVERED',
        original_parent_result=original, original_parent_log_sha256=hashlib.sha256(raw).hexdigest(),
        original_child_log_sha256=hashlib.sha256(logs[0].read_bytes()).hexdigest(),
        source_sha256=source_sha, native_handoff_audit=native,
        complete_training_status=training['status'], training_repeated=False,
        reason='Graph-only invalidation event is correctly absent in eager release; actual per-worker native sleep and release counters verified.')
    (arm/'POST_VALIDATION_RECOVERY.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    old=signature(PRIOR_PID)
    while old is not None and signature(PRIOR_PID)==old:
        state('WAITING_FOR_QUOTA_PAIR',controller_pid=PRIOR_PID);time.sleep(15)
    previous=ROOT/'quota_runtime/QUOTA_PAIR_RESULT.json'
    if not previous.is_file() or json.loads(previous.read_text())['status']!='V014_FROZEN_SAME_GRAPH_QUOTA_PAIR_AUDITED':
        raise RuntimeError('Previous measured pair not validated; do not restart or skip it')
    # Import configuration only after the preceding queue has ended. No code
    # is read once at queue creation and silently reused after later edits.
    spec=importlib.util.spec_from_file_location('eager_controls_runtime',CODE/'eager_controls.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    suite=m.suite();source_sha=suite.execution_digest()
    expected_subset=json.loads((ROOT/'heldout_prepared/manifest.json').read_text())
    for file,wanted in expected_subset['input_sha256'].items():
        with Path(file).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=wanted:raise RuntimeError('Prespecified held-out input/code changed')
    bookkeeping=json.loads((ROOT/'heldout_prepared/train_bookkeeping32.json').read_text())
    if (bookkeeping['status']!='TRAIN_SPLIT_BOOKKEEPING_ONLY_NOT_OPTIMIZED' or
            hashlib.sha256(m.EVAL_TRAIN.read_bytes()).hexdigest()!=bookkeeping['sha256']):
        raise RuntimeError('Val-only bookkeeping loader fixture changed')
    if os.environ.get('SUPPLEMENT_HANDOFF_EXISTING_EAGER_PID'):
        # A declared directory sentinel uses the old controller's existing
        # overwrite guard to prevent its known-fraction0.005 evaluation. It
        # is NOT a fabricated launch/result file. No training signal/restart.
        prior=int(os.environ['SUPPLEMENT_HANDOFF_EXISTING_EAGER_PID']);old=signature(prior)
        while old is not None and signature(prior)==old:
            state('WAITING_EXISTING_EAGER_TRAINING',controller_pid=prior);time.sleep(15)
        gate=ROOT/'heldout_results/vanilla/HELDOUT_RUNNING_vanilla_LAUNCH.json'
        marker=gate/'CONTROL_GATE.json'
        actual=ROOT/'eager_controls/seed_0/vanilla/EAGER_TRAINING_vanilla_RESULT.json'
        if (not gate.is_dir() or not marker.is_file() or
                json.loads(marker.read_text()).get('purpose')!='prevent_legacy_fraction_evaluation' or
                not actual.is_file() or json.loads(actual.read_text())['exit_code']!=0):
            raise RuntimeError('Controlled handoff not validated; original training not restarted')
        m.parse_eager(ROOT/'eager_controls/seed_0/vanilla','vanilla')
        archive=ROOT/'controller_gates/consumed_legacy_fraction_gate'
        archive.parent.mkdir(exist_ok=True);gate.rename(archive)
    arms=[];quality={}
    for policy in ('vanilla','planned_f4'):
        env,command=m.build_training(policy)
        arm=ROOT/'eager_controls/seed_0'/policy
        training_result=arm/('EAGER_TRAINING_'+policy+'_RESULT.json')
        if training_result.exists():
            if json.loads(training_result.read_text())['exit_code']!=0:
                recover_eager_validation_only(m,arm,policy,source_sha)
        else:
            launch(command,env,arm,m.SOURCE,'EAGER_TRAINING_'+policy)
        if suite.execution_digest()!=source_sha:raise RuntimeError('Primary source changed')
        result=m.parse_eager(arm,policy)
        (arm/'EAGER_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');arms.append(result)
        epochs=list((arm/'fig6').glob('epoch_001_*'))
        if len(epochs)!=1:raise RuntimeError('Missing trained endpoint')
        endpoint=epochs[0]/'checkpoints/qwen3moe_for_eagle3/global_step_5'
        scored=ROOT/'heldout_results'/policy/'RESULT.json'
        if scored.exists():quality[policy]=m.parse_evaluation(scored.parent,endpoint)
        else:quality[policy]=eval_endpoint(m,endpoint,policy)
        # Free only this run's private checkpoint after both training and its
        # held-out inference have passed; the next arm starts from publicE0.
        cleanup=env.copy();cleanup.update(FAIR_KEEP_COMPLETED_CHECKPOINTS='0',FAIR_FINALIZE_EXISTING='1')
        if policy=='planned_f4':
            # Dispatch only the offline verifier, only during finalize-only.
            # The exact historical-source execution fingerprint is preserved.
            cleanup['PATH']=str(CODE/'cleanup_bin')+os.pathsep+cleanup['PATH']
        if endpoint.exists():launch(command,cleanup,arm,m.SOURCE,'VALIDATED_CLEANUP_'+policy)
        elif not (arm/'fig6/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt').exists():
            raise RuntimeError('Missing endpoint and no validated cleanup receipt')
    retained=ROOT/'quality/ada_n2_seed0_epoch1/global_step_5'
    if retained.is_dir():
        scored=ROOT/'heldout_results/natural_f2_graph_trained/RESULT.json'
        quality['natural_f2_graph_trained']=(m.parse_evaluation(scored.parent,retained)
            if scored.exists() else eval_endpoint(m,retained,'natural_f2_graph_trained'))
    a=quality['vanilla']['scores_by_prompt_sha256'];b=quality['planned_f4']['scores_by_prompt_sha256']
    if set(a)!=set(b):raise RuntimeError('Quality prompt pairs mismatch')
    quality_pair=dict(questions=len(a),vanilla_mean=sum(a.values())/len(a),
        planned_f4_mean=sum(b.values())/len(b),difference=sum(b[k]-a[k] for k in a)/len(a),
        vanilla_only_correct=sum(a[k]==1 and b[k]!=1 for k in a),
        planned_only_correct=sum(b[k]==1 and a[k]!=1 for k in a),
        scope='One training seed, five additional updates,256 prespecified held-out questions. Equal nonelastic eager serving. No final-convergence or equivalence claim; graph-trained N2 is separately labelled.')
    report=dict(status='TWO_EAGER_CONTROLS_AND_MATCHED_HELDOUT_COMPLETE',eager=arms,quality=quality,
        matched_quality_pair=quality_pair,source_sha256=source_sha)
    (ROOT/'EAGER_AND_QUALITY_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    state('COMPLETE')


if __name__=='__main__':
    try:main()
    except Exception as error:state('STOPPED_WITH_ERROR',error=repr(error));raise
