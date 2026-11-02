#!/usr/bin/env python3
"""Queue one additional-domain FULL pair after the eager/quality queue."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

CODE = Path(__file__).resolve().parent
ROOT = Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')
PRIOR_PID = int(os.environ.get('SUPPLEMENT_PRIOR_EAGER_PID', '3193789'))


def state(value, **extra):
    path = ROOT / 'CODE_QUEUE_STATE.json'; temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(state=value, time=time.time(), **extra), indent=2)+'\n')
    temporary.replace(path)


def signature(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def idle():
    while True:
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
        if result.returncode or 'Process id' not in result.stdout:
            raise RuntimeError('NPU process-idle check failed')
        footer = result.stdout.rsplit('Process id', 1)[1]
        if not re.findall(r'^\|\s+\d+\s+\d+\s+\d+\s+', footer, re.M):
            return
        state('WAITING_FOR_IDLE_NPUS'); time.sleep(15)


def main():
    old = signature(PRIOR_PID)
    while old is not None and signature(PRIOR_PID) == old:
        state('WAITING_FOR_EAGER_AND_QUALITY', controller_pid=PRIOR_PID); time.sleep(15)
    previous = ROOT / 'EAGER_AND_QUALITY_RESULT.json'
    if not previous.exists() or json.loads(previous.read_text())['status'] != 'TWO_EAGER_CONTROLS_AND_MATCHED_HELDOUT_COMPLETE':
        raise RuntimeError('Previous queue not validated; not automatically skipped/restarted')
    spec = importlib.util.spec_from_file_location('code_domain_runtime', CODE / 'code_task.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    suite = module.suite(); source_sha = suite.execution_digest()
    manifest = json.loads((module.DATA / 'manifest.json').read_text())
    for key in ('train_parquet', 'test_parquet'):
        with Path(manifest[key]).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != manifest[key+'_sha256']:
            raise RuntimeError('Prespecified code dataset changed')
    if not (module.COMMON / 'DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT').exists():
        raise RuntimeError('Missing protected code-task common checkpoint')
    view=json.loads((ROOT/'code_prepared/manifest.json').read_text())
    if (view['status']!='ALL160_CODE_SEMANTICS_UNCHANGED_IDENTITIES_CANONICAL' or
            hashlib.sha256(module.TRAIN.read_bytes()).hexdigest()!=view['sha256']):
        raise RuntimeError('Canonical code view changed or lost semantic equivalence proof')
    preview=ROOT/'code_plan_preview_after_multiturn_fix/RESULT.json'
    if not preview.is_file() or json.loads(preview.read_text()).get('status')!='ALL160_CODE_PROMPTS_PLANNED_FEASIBLY_CPU_ONLY':
        raise RuntimeError('All160 code prompts have not passed CPU KV-feasibility preflight; no NPU arm launched')
    preview_launch=json.loads((preview.parent/'LAUNCH.json').read_text())
    for path,wanted in preview_launch['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=wanted:
            raise RuntimeError('Current code planner CPU preflight input changed: '+path)
    vanilla_preview=ROOT/'code_vanilla_plan_preview/aclgraph_bucket_plan.audit.json'
    if (not vanilla_preview.is_file() or json.loads(vanilla_preview.read_text()).get('status')!=
            'ALL160_VANILLA_CHAT_HISTORY_BUCKETS_AUDITED_CPU_ONLY' or
            json.loads(vanilla_preview.read_text())['train_sha256']!=view['sha256']):
        raise RuntimeError('Vanilla multi-turn history/bucket CPU preflight missing or stale')
    vanilla_proof=json.loads(vanilla_preview.read_text())
    if (vanilla_proof['plan_sha256']!=hashlib.sha256(vanilla_preview.with_name('aclgraph_bucket_plan.json').read_bytes()).hexdigest()
            or vanilla_proof['wrapper_sha256']!=hashlib.sha256((CODE/'code_vanilla_bucket_plan.py').read_bytes()).hexdigest()):
        raise RuntimeError('Vanilla CPU bucket plan or tokenizer helper changed')
    # Isolate every actual generated answer. Fail closed if isolation breaks.
    subprocess.run(['python3', '-m', 'unittest', '-v', 'test_sandbox_code.py'], cwd=CODE, check=True)
    protocol = dict(status='PRESPECIFIED_NOT_COMPLETED', seed=401, actor_frozen=True,
                    policies=['vanilla', 'planned_f4'], epochs=[1], steps=5,
                    offline_repair_budget=32,offline_exact_peak_memoization=True,
                    prompts=160, rollout_n=16, common=str(module.COMMON),
                    dataset_manifest=manifest, source_sha256=source_sha,
                    canonical_dataset_manifest=view,
                    cpu_vanilla_history_preflight=vanilla_proof,
                    cpu_P4_planner_preflight=json.loads(preview.read_text()),
                    repair_trace_sha256=hashlib.sha256(module.REPAIR_TRACE.read_bytes()).hexdigest(),
                    reward='Dummy0 in frozen timed primary; no online host code execution.',
                    quality='Offline OS-sandboxed full-suite pass@1, sample_index0 on every paired prompt; not16-sample best-of.')
    protocol['scoring_sha256'] = {name:hashlib.sha256((CODE / name).read_bytes()).hexdigest()
        for name in ('code_task.py', 'cached_planner.py', 'code_vanilla_bucket_plan.py', 'prepare_code_dataset.py', 'offline_bin/python3', 'sandbox_code.py', 'sandbox_grader.py',
                     '.deps/pyext-0.7/pyext.py', '.deps/bwrap-root/usr/bin/bwrap')}
    destination = ROOT / 'code_task'; destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'PROTOCOL.json').exists():
        raise RuntimeError('Refusing existing code-task execution')
    (destination / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for policy in ('vanilla', 'planned_f4'):
        env, command = module.build_training(policy)
        arm = destination / f'seed_401/{policy}'; arm.mkdir(parents=True)
        for key in ('XDG_CACHE_HOME', 'TORCHAIR_CACHE_HOME', 'TRITON_CACHE_DIR', 'ASCEND_CACHE_PATH', 'ASCEND_WORK_PATH', 'RAY_TMPDIR'):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        idle(); state('RUNNING_'+policy)
        (arm / 'LAUNCH.json').write_text(json.dumps(dict(command=command, env={
            k:v for k,v in env.items() if k.startswith(('VLLM_', 'VERL_', 'ADAFLOOR_', 'FAIR_', 'DYNAMIC_'))
            or k in ('COMMON_EPOCH0_ROOT', 'TRAIN_FILE', 'TRAIN_FILE_ORIG', 'TEST_FILE', 'TASK_QUEUE_ENABLE',
                     'MAX_CROSS_STEP_REPAIR_SWAPS','PATH')}), indent=2)+'\n')
        with (arm / 'driver.log').open('x') as log:
            result = subprocess.run(command, cwd=module.SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        (arm / 'DRIVER_RESULT.json').write_text(json.dumps(dict(exit_code=result.returncode))+'\n')
        if result.returncode or suite.execution_digest() != source_sha:
            raise RuntimeError('Code-task child failed or primary source changed; evidence retained')
        report, _, _ = module.audit_arm(policy)
        (arm / 'GENERATION_RESULT.json').write_text(json.dumps(report, indent=2)+'\n')
    state('SCORING_IN_OS_SANDBOXES')
    report = module.score_pair(); report['source_sha256'] = source_sha
    (destination / 'CODE_PAIR_RESULT.json').write_text(json.dumps(report, indent=2)+'\n')
    state('COMPLETE')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state('STOPPED_WITH_ERROR', error=repr(error)); raise
