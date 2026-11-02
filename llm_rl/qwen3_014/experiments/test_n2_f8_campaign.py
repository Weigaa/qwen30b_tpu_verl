from pathlib import Path
from run_n2_f8_campaign import new_capacity,calibration_launch,CAPS


def test_only_n2_f8_changes_other_budgets_and_runtime_stay_pinned():
    from run_n2_f8_campaign import main_suite
    suite=main_suite()
    before=suite.arm_env(Path('/tmp/n2_f8_test'),0,'natural_f2')
    after=new_capacity(before)
    changed={k for k in after if before.get(k)!=after[k]}
    assert changed=={'VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8','FLOOR_KV_CAPS'}
    assert before['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8']=='288256'
    assert after['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8']=='349952'
    assert after['FLOOR_KV_CAPS']==CAPS
    for key,wanted in [('FLOOR2','131072'),('FLOOR4','280576'),('FLOOR16','380800')]:
        assert after['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_'+key]==wanted
    assert after['VLLM_ROLLOUT_SLEEP_LEVEL']=='2'
    assert after['VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE']=='1'
    assert after['TASK_QUEUE_ENABLE']=='1'


def test_warm_capacity_gate_really_targets_f8_after_low_floor_and_twice():
    env,command=calibration_launch(Path('/tmp/n2_f8_test'))
    assert env['DYNAMIC_FORCE_SELECTED_FLOORS']=='2,8,8,16,16'
    assert env['DYNAMIC_TRAIN_STEPS']=='3'
    assert env['DYNAMIC_PLAN_STEPS']=='5'
    assert env['DYNAMIC_ENABLE_CKPT_CHAIN']=='0'
    assert env['SAVE_CKPT_ENABLE']=='0'
    assert env['FAIR_FREEZE_ACTOR']=='0'
    assert 'trainer.total_training_steps=3' in command
    assert 'actor_rollout_ref.actor.optim.lr=1e-6' in command
    assert 'trainer.save_freq=-1' in command
    assert 'actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY' in command
    assert env['ADAFLOOR_GRAPH_BASE_RUNNER'].endswith('floor2_natural_tailguard_reuse_epoch0_2epoch.sh')
    assert env['VLLM_ASCEND_ACLGRAPH_BUCKET_COUNT']=='8'


def test_calibration_does_not_relax_generation_limits_or_add_synthetic_tail():
    env,command=calibration_launch(Path('/tmp/n2_f8_test'))
    assert env['FAIR_MAX_RESPONSE_LENGTH']=='16384'
    assert env['VERL_PAIRED_REQUEST_SAMPLING_SEEDS']=='0'
    assert env['VERL_SIDECAR_ENABLE']=='0'
    assert 'VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS' not in env
    assert 'actor_rollout_ref.rollout.seed=0' in command


def test_only_exact_tbe_cleanup_race_is_classified_as_shutdown_noise():
    from run_n2_f8_campaign import calibration_tracebacks
    trace='''Traceback (most recent call last):
  File "/usr/local/python3.11.14/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/local/Ascend/cann-8.5.0/python/site-packages/tbe/common/repository_manager/utils/common.py", line 110, in daemon_process
    os.kill(mgr_pid, signal.SIGKILL)
ProcessLookupError: [Errno 3] No such process'''
    benign,fatal=calibration_tracebacks(trace)
    assert len(benign)==1 and not fatal
    assert calibration_tracebacks(trace.replace('daemon_process','model_forward'))[1]
    assert calibration_tracebacks(trace.replace('ProcessLookupError','OutOfMemoryError'))[1]
    assert calibration_tracebacks(trace+'\nTraceback (most recent call last):\nRuntimeError: real failure')[1]
    assert calibration_tracebacks('Traceback (most recent call last):\nValueError: wrong shape')[1]
