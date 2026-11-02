#!/usr/bin/env python3
"""Use the proven v0.14 FULL/sleep2 envelope for the original sidecar pair."""
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v014_sidecar_envelope',REPO/'tools/run_mlsys_fig6.py')
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)


def main():
    assert len(sys.argv)>1 and sys.argv[1]=='adafloor_p_f4'
    args=sys.argv[2:]
    seed=int(next(x.split('=',1)[1] for x in args if x.startswith('actor_rollout_ref.rollout.seed=')))
    assert seed in (101,202,303)
    arm_root=Path(os.environ['FAIR_OUTPUT_ROOT']).resolve()
    save_freq=int(os.environ.get('FAIR_TRAIN_STEPS','1'))
    env=suite.arm_env(arm_root,seed,'planned_f4')
    # Preserve the exact old experiment's one-window/frozen/request-seeded
    # contract while replacing only the primary execution envelope.
    for key,value in os.environ.items():
        if key.startswith('VERL_SIDECAR_') or key in (
            'COMMON_EPOCH0_ROOT','FAIR_OUTPUT_ROOT','DYNAMIC_RUN_NAME',
            'FAIR_START_EPOCH','FAIR_TOTAL_EPOCHS','FAIR_PROMPTS_PER_EPOCH',
            'FAIR_TRAIN_BATCH_SIZE','FAIR_TRAIN_STEPS','FAIR_ROLLOUT_N',
            'FAIR_MAX_RESPONSE_LENGTH','FAIR_FREEZE_ACTOR','FAIR_DRY_RUN',
            'FAIR_KEEP_COMPLETED_CHECKPOINTS','VERL_PAIRED_REQUEST_SAMPLING_SEEDS',
            'DYNAMIC_DISABLE_TAIL_GUARD','DYNAMIC_EXPECT_NO_RESPONSE_CAPS',
            'DYNAMIC_SHORT_STEP_CAP_ENABLE'):
            env[key]=value
    env.update(ADAFLOOR_GRAPH_SIDECAR_ENABLE='1',
        VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY='planned',
        VERL_MEGATRON_EAGER_WEIGHT_SYNC_GROUP_INIT='0',
        VERL_HCCL_IF_BASE_PORT_START='12000',HCCL_IF_BASE_PORT='12000',
        VERL_HCCL_IF_BASE_PORT_BLOCK='16384',VERL_SIDECAR_TASK_QUEUE_ENABLE='2',
        RAY_TMPDIR=f'/tmp/sc14_{seed}_{os.getpid()}',TRAINER_SAVE_FREQ=str(save_freq))
    env['FAIR_RUN_NAME']=env['DYNAMIC_RUN_NAME']
    for key in ('XDG_CACHE_HOME','TORCHAIR_CACHE_HOME','TRITON_CACHE_DIR','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','RAY_TMPDIR'):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    command=suite.arm_command(seed,'planned_f4')
    command+=[f'trainer.save_freq={save_freq}', 'actor_rollout_ref.rollout.shrink_aware.target_policy=planned']+args
    arm_root.mkdir(parents=True,exist_ok=True)
    path=arm_root/'v014_launch.json'
    if path.exists():raise RuntimeError('Retain existing launch evidence; do not overwrite')
    suite.write_json(path,dict(command=command,execution_sha256=suite.execution_digest(),env={k:v for k,v in env.items()
        if k.startswith(('VLLM_','VERL_','FAIR_','ADAFLOOR_','DYNAMIC_')) or k in ('TASK_QUEUE_ENABLE','HCCL_BUFFSIZE','PYTORCH_NPU_ALLOC_CONF','HCCL_IF_BASE_PORT')}))
    print(f'[sidecar v0.14] primary=FULL_DECODE_ONLY/TQ1/native_sleep2 sidecar=eager/TQ2 seed={seed}',flush=True)
    os.chdir(REPO)
    os.execvpe(command[0],command,env)


if __name__=='__main__':main()
