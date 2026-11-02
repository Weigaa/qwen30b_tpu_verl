#!/usr/bin/env python3
"""Finish successful Sidecar training after a missing manifest hash stopped summary.

No repeated training, generation, sidecar windows or scoring. The missing
summary metadata is filled ONLY from the original pre-run pinned protocol,
after checking it against current execution sources. Preserve original bytes.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import run_n2_f8_campaign as campaign


def finalize_existing_sidecar(root):
    campaign.check_sources(root)
    side=root/'graph_sidecar'
    result=json.loads((side/'DRIVER_RESULT.json').read_text())
    if result.get('exit_code')!=0 or result.get('retained_endpoint') is not True:
        raise RuntimeError('Sidecar training was not successful with retained endpoint')
    helper=campaign.module('n2_postrun_sidecar',campaign.SIDECAR/'tools/run_v014_sidecar_training.py')
    helper.BASELINE=root/'primary'
    # Check all ten steps, actor updates, true FULL replay, saved outputs,
    # cleanup validation and own epoch1->epoch2 chain BEFORE metadata repair.
    primary=helper.suite.parse_arm(side,0,'natural_f2')
    contract=json.loads((root/'PROTOCOL.json').read_text())
    pinned=contract['execution_sha256']['qwen3_graph_sidecar_v014']
    if helper.suite.execution_digest()!=pinned:
        raise RuntimeError('Current sidecar runtime differs from pre-run pinned hash')
    path=side/'v014_launch.json';launch=json.loads(path.read_text())
    stage=root/'postrun_metadata_recovery';stage.mkdir(exist_ok=True)
    original=stage/'v014_launch.original.json'
    if 'execution_sha256' not in launch:
        if original.exists():raise RuntimeError('Ambiguous incomplete metadata repair')
        shutil.copyfile(path,original)
        launch['execution_sha256']=pinned
        campaign.write(path,launch)
        campaign.write(stage/'MANIFEST_REPAIR.json',dict(
            status='MISSING_SUMMARY_FIELD_FROM_ORIGINAL_PINNED_PROTOCOL',
            original_manifest=str(original),original_manifest_sha256=campaign.sha(original),
            repaired_manifest_sha256=campaign.sha(path),execution_sha256=pinned,
            source_protocol=str(root/'PROTOCOL.json'),source_protocol_sha256=campaign.sha(root/'PROTOCOL.json'),
            child_result=result,changed_fields=['execution_sha256'],training_repeated=False,
            runtime_and_budget_unchanged=True,reason='Successful ten-step child exited0. Summary raised KeyError because host campaign omitted execution_sha256 from v014_launch.json.'))
    else:
        if launch['execution_sha256']!=pinned or not original.is_file():
            raise RuntimeError('Unexpected prior metadata instead of diagnosed repair')
    campaign.state(root,'SUMMARIZING_COMPLETED_GRAPH_SIDECAR_NO_REGENERATION')
    summary=helper.summarize(side)
    if summary['primary']!=primary:raise RuntimeError('Primary evidence changed during summary')
    storage=Path('/workspace/adafloor_graph_results')/(root.name+'_epoch2_graph_sidecar')
    campaign.split_cleanup(root,'graph_sidecar',storage,side/'seed_0/natural_f2/fig6')
    campaign.write(stage/'POSTRUN_RECOVERY_PASS.json',dict(
        status='SUCCESSFUL_EXISTING_TRAINING_SUMMARIZED_WITHOUT_REGENERATION',
        execution_sha256=pinned,metadata_only=True,sidecar_result_sha256=campaign.sha(side/'TRAINING_SIDECAR_RESULT.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve()
    campaign.sidecar_arm=finalize_existing_sidecar
    os.environ['RAY_agent_register_timeout_ms']='120000'
    sys.argv=['run_n2_f8_campaign.py','resume','--root',str(root)]
    campaign.main()


if __name__=='__main__':main()
