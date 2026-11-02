#!/usr/bin/env python3
"""Reuse the completed16-query eager arm, finish graph, then resume queue."""
import json
import os
from pathlib import Path
import subprocess
import sys

import run_n4_sidecar_models as m


def main():
    root=Path('/data/adafloor_graph_results/mlsys_n4_sidecar_internal_dp_deterministic_20261027T081200Z')
    m.check_contract(root);m.c.idle(root)
    preflight=root/'preflight/qwen3_8b'
    saved=json.loads((preflight/'eager/launch.json').read_text())
    env=os.environ.copy();env['PYTHONPATH']=saved['PYTHONPATH']
    command=[sys.executable,str(m.SECONDARY/'tools/run_sidecar_graph_parity.py'),
        '--output',str(preflight),'--model','/data/Qwen3-8B',
        '--tensor-parallel-size','4','--data-parallel-size','2','--resume-after-eager']
    m.c.state(root,'RESUMING_GRAPH_PREFLIGHT_ONLY',reused_eager_answers=16)
    with (root/'GRAPH_PREFLIGHT_RECOVERY.log').open('x') as log:
        code=subprocess.run(command,cwd=m.SECONDARY,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    m.c.write(root/'GRAPH_PREFLIGHT_RECOVERY_RESULT.json',dict(exit_code=code,reused_eager_answers=16))
    if code:
        m.c.state(root,'STOPPED_WITH_ERROR',error='Resumed graph preflight failed, evidence retained')
        raise SystemExit(code)
    m.proof(root,'qwen3_8b');m.check_contract(root)
    raise SystemExit(subprocess.run([sys.executable,str(m.c.CODE/'run_n4_sidecar_models.py'),
        'run','--output',str(root)],cwd=m.c.CODE).returncode)


if __name__=='__main__':main()
