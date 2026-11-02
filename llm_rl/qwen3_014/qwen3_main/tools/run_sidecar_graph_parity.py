#!/usr/bin/env python3
"""One minimal, sequential eager/ACLGraph greedy check of the Sidecar itself.

This is a correctness preflight, not a timed ON/OFF primary experiment.
Uses the same model, data, engine capacities and sampling as the full run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def read_events(path):
    events = []
    for line in path.read_text(errors='replace').splitlines():
        start = line.find('{')
        if start < 0:
            continue
        try:
            row = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if 'event' in row:
            events.append(row)
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--shadow-proof', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.summarize_only)
    runs = {}; summaries = {}
    for mode in (() if args.summarize_only else ('eager', 'graph')):
        arm = root / mode
        arm.mkdir()
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('VLLM_', 'VERL_', 'ADAFLOOR_'))}
        env.update(PYTHONPATH=str(REPO)+':'+os.environ.get('PYTHONPATH',''),
            VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM='1',
            VERL_SIDECAR_MODEL_PATH='/data/Qwen2.5-1.5B-Instruct',
            VERL_SIDECAR_PROMPTS_FILE='/data/gsm8k',
            VERL_SIDECAR_DATA_SPLIT='train',
            VERL_SIDECAR_NPU_DEVICES=str(args.device),
            VERL_SIDECAR_PARALLEL_MODE='dp', VERL_SIDECAR_TENSOR_PARALLEL_SIZE='1',
            VERL_SIDECAR_DATA_PARALLEL_SIZE='1', VERL_SIDECAR_REPLICA_COUNT='1',
            VERL_SIDECAR_ENABLE_EXPERT_PARALLEL='0',
            VERL_SIDECAR_TASK_QUEUE_ENABLE='1' if mode=='graph' else '2',
            VERL_SIDECAR_ENFORCE_EAGER='0' if mode=='graph' else '1',
            VERL_SIDECAR_GRAPH_AUDIT='1' if mode=='graph' else '0',
            VERL_SIDECAR_CUDAGRAPH_MODE='FULL_DECODE_ONLY',
            VERL_SIDECAR_CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,16,32,64,128]',
            VERL_SIDECAR_N='1', VERL_SIDECAR_TEMPERATURE='0.0', VERL_SIDECAR_TOP_P='1.0',
            VERL_SIDECAR_GPU_MEMORY_UTILIZATION='0.80', VERL_SIDECAR_MAX_MODEL_LEN='6144',
            VERL_SIDECAR_MAX_TOKENS='4096', VERL_SIDECAR_MAX_NUM_SEQS='128',
            VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS='65536',
            VERL_SIDECAR_MAX_PROMPTS='16', VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE='16',
            VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA='16', VERL_SIDECAR_GENERATE_CHUNK_SIZE='128',
            VERL_SIDECAR_REPEAT_UNTIL_KILLED='0', VERL_SIDECAR_STREAM_CHECKPOINT='1',
            VERL_SIDECAR_ENGINE_MULTIPROCESSING='1', VERL_SIDECAR_SINGLETON_GROUP_REUSE='0',
            VERL_SIDECAR_SAVE_TOKEN_IDS='1',
            VERL_SIDECAR_OUTPUT_FILE=str(arm/'outputs.jsonl'),
            VERL_SIDECAR_LOG_FILE=str(arm/'infer.log'), VERL_SIDECAR_STATE_DIR=str(arm/'state'),
            VERL_SIDECAR_STOP_FILE=str(arm/'stop'),
            XDG_CACHE_HOME=str(arm/'cache'), ASCEND_CACHE_PATH=str(arm/'cache/ascend'))
        (arm/'launch.json').write_text(json.dumps({k:v for k,v in env.items()
            if k.startswith(('VERL_', 'PYTHONPATH'))}, indent=2)+'\n')
        print(f'[Sidecar parity] starting {mode}, device={args.device}', flush=True)
        with (arm/'driver.log').open('w') as log:
            rc = subprocess.run(['bash', str(REPO/'internal/run_elastic_sidecar_infer.sh')],
                env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            raise RuntimeError(f'{mode} Sidecar failed rc={rc}; see {arm}/driver.log')
        rows = [json.loads(line) for line in (arm/'outputs.jsonl').open()]
        runs[mode] = {row['prompt_id']:row for row in rows}
        events = read_events(arm/'infer.log')
        summaries[mode] = dict(load=[e for e in events if e['event']=='sidecar_load_start'],
            ready=[e for e in events if e['event']=='sidecar_model_ready'],
            capture=[e for e in events if e['event']=='sidecar_aclgraph_capture'],
            replay=[e for e in events if e['event']=='sidecar_aclgraph_replay'])
    if args.summarize_only:
        for mode in ('eager','graph'):
            arm=root/mode
            if 'sidecar_exit_code=0' not in (arm/'infer.log').read_text():
                raise RuntimeError('Cannot summarize a failed/incomplete Sidecar')
            rows=[json.loads(line) for line in (arm/'outputs.jsonl').open()]
            if len(rows)!=16:raise RuntimeError('Missing/duplicate parity outputs')
            runs[mode]={row['prompt_id']:row for row in rows}
            events=read_events(arm/'infer.log')
            summaries[mode]={key:[e for e in events if e['event']==event] for key,event in
                [('load','sidecar_load_start'),('ready','sidecar_model_ready'),
                 ('capture','sidecar_aclgraph_capture'),('replay','sidecar_aclgraph_replay')]}
    if set(runs['eager']) != set(range(16)) or set(runs['graph']) != set(range(16)):
        raise RuntimeError('Missing/duplicate parity inputs')
    mismatches=[]
    for prompt_id in range(16):
        left, right = (runs[m][prompt_id] for m in ('eager','graph'))
        if left['prompt'] != right['prompt'] or left['prompt_source'] != right['prompt_source']:
            raise RuntimeError('Parity input mismatch')
        fields=[field for field in ('text','token_ids','finish_reason')
                if left['outputs'][0][field]!=right['outputs'][0][field]]
        if fields:mismatches.append(dict(prompt_id=prompt_id,fields=fields))
    if (not summaries['graph']['capture'] or not summaries['graph']['replay']
            or any(e['runtime_mode']!='FULL' or e['capture_phase'] for e in summaries['graph']['replay'])):
        raise RuntimeError('Missing native FULL graph capture/replay proof')
    shadow=None
    if mismatches:
        if args.shadow_proof:
            shadow_path=args.shadow_proof.resolve()
        else:
            shadow_dir=root/'same_input_shadow'
            rc=subprocess.run([sys.executable,str(REPO/'tools/diagnose_sidecar_graph_parity.py'),
                '--input-run',str(root),'--output',str(shadow_dir)],cwd=REPO).returncode
            if rc:raise RuntimeError('Same-input graph diagnostic failed; do not launch primary')
            shadow_path=shadow_dir/'SHADOW_RESULT.json'
        shadow=json.loads(shadow_path.read_text())
        shadow_rows={row['prompt_id']:row for row in map(json.loads,
                     (shadow_path.parent/'outputs.jsonl').open())}
        expected_steps=max(r['outputs'][0]['token_ids_len'] for r in runs['graph'].values())-1
        if (shadow['status']!='ALL_SAME_PADDED_INPUT_SHADOW_CHECKS_PASSED'
                or shadow['primary_exit_code']!=0
                or shadow['post_checks']!=expected_steps or shadow['pre_checks']!=expected_steps
                or shadow['post_mismatches'] or shadow['pre_argmax_mismatches']
                or shadow['max_hidden_abs']!=0.0 or shadow['max_logits_abs']!=0.0
                or set(shadow_rows)!=set(runs['graph'])
                or any(shadow_rows[k]['outputs'][0]['token_ids']!=r['outputs'][0]['token_ids']
                       for k,r in runs['graph'].items())):
            raise RuntimeError('Same-input graph proof failed or diagnostic changed generation')
        shadow=dict(**shadow,proof_path=str(shadow_path),
                    proof_sha256=hashlib.sha256(shadow_path.read_bytes()).hexdigest(),
                    diagnostic_generation_unchanged=True)
    report = dict(status='PASS', checked_prompts=16, greedy_token_ids_exact=not mismatches,
        free_batch_greedy_exact_count=16-len(mismatches),free_batch_mismatches=mismatches,
        same_padded_input_graph_parity_exact=shadow is not None,same_input_shadow=shadow,
        output_tokens=sum(r['outputs'][0]['token_ids_len'] for r in runs['graph'].values()),
        model='/data/Qwen2.5-1.5B-Instruct', seed=0,
        data='/data/gsm8k/train.parquet', diagnostics=summaries,
        sidecar_script_sha256=hashlib.sha256((REPO/'internal/run_elastic_sidecar_infer.sh').read_bytes()).hexdigest(),
        aclgraph_sha256=hashlib.sha256((REPO/'vllm_ascend/compilation/acl_graph.py').read_bytes()).hexdigest(),
        model_runner_sha256=hashlib.sha256((REPO/'vllm_ascend/worker/model_runner_v1.py').read_bytes()).hexdigest(),
        end_utc=datetime.now(timezone.utc).isoformat(),
        scope='16 shared greedy queries; if dynamic eager/graph batching changes free-running trajectories, require bit-exact hidden/logit and argmax comparisons on every same-padded-input decode step before/after replay. No numerical threshold relaxation, no eager result fed into generation. Not universal parity or primary ON/OFF/throughput proof.')
    (root/'PARITY_RESULT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='diagnostics'}, indent=2), flush=True)


if __name__=='__main__':
    main()
