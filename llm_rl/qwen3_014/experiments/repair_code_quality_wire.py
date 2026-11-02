#!/usr/bin/env python3
"""Regrade schema-affected fixed answers only; never rerun NPU generation."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import code_task
from code_test_wire import normalize,infrastructure_format_error
from sandbox_code import run_case

ROOT=code_task.ROOT/'code_task'
ARCHIVE=ROOT/'quality_initial_wire_format_invalid'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def state(value,**extra):
    path=ROOT/'GRADING_REPAIR_STATE.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=value,time=time.time(),**extra),indent=2)+'\n');tmp.replace(path)


def main():
    if (ROOT/'CODE_PAIR_RESULT.json').exists() or (ROOT/'quality').exists():
        raise RuntimeError('Refusing to overwrite/repeat repaired quality')
    initial=json.loads((ARCHIVE/'CODE_PAIR_RESULT.json').read_text())
    declared=json.loads((ROOT/'PROTOCOL.json').read_text())
    if code_task.suite().execution_digest()!=declared['source_sha256']:
        raise RuntimeError('Runtime source changed; no generation repeated')
    for name,wanted in declared['scoring_sha256'].items():
        if hashlib.sha256((code_task.CODE/name).read_bytes()).hexdigest()!=wanted:
            raise RuntimeError('Original pinned judge/sandbox/protocol changed: '+name)
    arms={};selected={};outputs={};old={};affected={};wire={}
    for policy in ('vanilla','planned_f4'):
        arms[policy],outputs[policy],selected[policy]=code_task.audit_arm(policy)
        if arms[policy]!=dict((k,v) for k,v in initial['arms'][policy].items()
                if k not in ('sandbox_sampled_pass_at_1','scored_answers','sandbox_failures')):
            raise RuntimeError('Original generation evidence changed')
        scores=list(code_task.read_rows(ARCHIVE/f'{policy}.jsonl'))
        old[policy]={r['prompt_hash']:r for r in scores}
        if len(scores)!=160 or set(old[policy])!=set(selected[policy]):
            raise RuntimeError('Original scores lack complete fixed160 panel')
        affected[policy]=set();wire[policy]={}
        for key,row in selected[policy].items():
            prior=old[policy][key]
            if prior['sample_index']!=0 or prior['output_sha256']!=hashlib.sha256(row['output'].encode()).hexdigest():
                raise RuntimeError('Original fixed sample0 answer changed')
            tests=json.loads(row['gts']) if isinstance(row['gts'],str) else row['gts']
            encoded,changed=normalize(tests);wire[policy][key]=(tests,encoded)
            if changed:affected[policy].add(key)
            if infrastructure_format_error(prior) and not changed:
                raise RuntimeError('Known format error outside affected schema set')
        if len(affected[policy])!=21:raise RuntimeError('Not the fixed21 schema-affected questions')
    if set(outputs['vanilla'])!=set(outputs['planned_f4']) or affected['vanilla']!=affected['planned_f4']:
        raise RuntimeError('Paired requests or schema-affected panel differs')
    for key,row in outputs['vanilla'].items():
        other=outputs['planned_f4'][key]
        if (row['input'],row['gts'])!=(other['input'],other['gts']):
            raise RuntimeError('Paired actual prompt/test-suite differs')
    protocol=dict(status='PRESPECIFIED_LOSSLESS_WIRE_REPAIR_NOT_COMPLETED',
        original_protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),
        original_code_pair_sha256=hashlib.sha256((ARCHIVE/'CODE_PAIR_RESULT.json').read_bytes()).hexdigest(),
        original_score_sha256={p:hashlib.sha256((ARCHIVE/f'{p}.jsonl').read_bytes()).hexdigest() for p in old},
        new_adapter_sha256=hashlib.sha256((code_task.CODE/'code_test_wire.py').read_bytes()).hexdigest(),
        repair_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_runtime_and_judge_hashes_unchanged=True,NPU_reruns=0,
        original_test_values_and_expected_outputs_unchanged=True,all_questions=160,
        rescored_per_arm=21,reused_per_arm=139,sample_index=0,
        affected_prompt_hashes=sorted(affected['vanilla']),
        source_test_sha256={k:digest(wire['vanilla'][k][0]) for k in sorted(affected['vanilla'])},
        wire_test_sha256={k:digest(wire['vanilla'][k][1]) for k in sorted(affected['vanilla'])},
        selection='Entire schema-affected panel, not errors-only or score-based. One new grade per fixed answer.',
        sandbox_limits='Unchanged: original OS isolation,1GiB AS,64MiB tmpfs,8s CPU,20s wall, pinned PRIME full-suite judge.')
    protocol_path=ROOT/'GRADING_REPAIR_PROTOCOL.json'
    if protocol_path.exists():raise RuntimeError('Repair already attempted; inspect original evidence')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    destination=ROOT/'quality';destination.mkdir()
    for policy in ('vanilla','planned_f4'):
        state('REGRADING_NATIVE_SCHEMA_'+policy)
        def grade(key):
            if key not in affected[policy]:return old[policy][key]
            answer=selected[policy][key]['output'];tests,encoded=wire[policy][key]
            result=run_case(answer,encoded)
            if infrastructure_format_error(result):
                raise RuntimeError('Wire adapter failed; not a model wrong answer')
            return dict(prompt_hash=key,sample_index=0,
                output_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                original_test_sha256=digest(tests),wire_test_sha256=digest(encoded),
                lossless_wire_normalization=True,**result)
        scored=[]
        with (destination/f'{policy}.jsonl').open('x') as stream,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for row in pool.map(grade,sorted(selected[policy])):
                stream.write(json.dumps(row)+'\n');stream.flush();scored.append(row)
        if len(scored)!=160:raise RuntimeError('Repaired grading lost answers')
        arms[policy].update(sandbox_sampled_pass_at_1=sum(r['passed'] for r in scored)/160,
            scored_answers=160,sandbox_failures=sum(r['exit_code']!=0 or r['timed_out'] for r in scored))
        print(policy,'correct',sum(r['passed'] for r in scored),'of160',flush=True)
    receipt=dict(status='LOSSLESS_CODE_WIRE_REPAIR_COMPLETE',protocol=str(protocol_path),
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),NPU_reruns=0,
        rescored_per_arm=21,reused_per_arm=139,affected_prompt_hashes=sorted(affected['vanilla']))
    initial.update(arms=arms,grading_repair=receipt)
    initial['scope']+=' Original native list-valued test representations were losslessly adapted to the pinned judge wire format;21 fixed schema-affected answers/arm regraded once,139 untouched scores reused, original invalid grading archived. No generation retry.'
    tmp=ROOT/'CODE_PAIR_RESULT.tmp'
    tmp.write_text(json.dumps(initial,indent=2)+'\n');tmp.replace(ROOT/'CODE_PAIR_RESULT.json')
    state('COMPLETE',receipt=receipt)


if __name__=='__main__':
    try:main()
    except Exception as error:state('STOPPED_WITH_ERROR',error=repr(error));raise
