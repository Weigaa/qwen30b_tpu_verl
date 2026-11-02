import json
from monitor_n2_f8_campaign import progress_record,ray_environment


def test_progress_is_actual_forward_event_not_config_or_submission():
    line='INFO 10-26 02:01:22 [model_runner_v1.py:1877] rollout_forward_begin dp_rank=4 iteration=256 local_tokens=32 padded_tokens=32 requests=31 max_query_len=1 global_prefill=False graph_mode=FULL'
    assert progress_record(line)==dict(time='02:01:22',rank=4,iteration=256,requests=31,graph_mode='FULL')
    assert progress_record(line.replace('rollout_forward_begin','rollout_forward_submitted')) is None
    assert progress_record('graph_mode=FULL configured') is None


def test_ray_environment_exposes_only_selected_directory(tmp_path):
    p=tmp_path/'environ';p.write_bytes(b'IGNORED=private\0RAY_TMPDIR=/tmp/scope\0ALSO_IGNORED=private\0')
    assert ray_environment(p)=='/tmp/scope'
    assert ray_environment(tmp_path/'missing') is None


def test_code_launch_without_ray_key_uses_only_actual_code_trainer(tmp_path,monkeypatch):
    import monitor_n2_f8_campaign as m
    root=tmp_path/'campaign';root.mkdir();(root/'STATE.json').write_text(json.dumps(dict(state='RUNNING_NEW_BUDGET_CODE_N2')))
    code=tmp_path/'code';launch=code/'seed_401/natural_f2/LAUNCH.json';launch.parent.mkdir(parents=True)
    launch.write_text(json.dumps(dict(env={})))
    ray=tmp_path/'actual_ray';(ray/'ray/session_latest/logs').mkdir(parents=True)
    monkeypatch.setattr(m,'code_root',lambda root:code)
    monkeypatch.setattr(m,'processes',lambda root:[dict(trainer=True,scope='code',ray_tmpdir=str(ray))])
    result=m.observe(root)
    assert result['last_launch']==str(launch)
    assert result['ray_session_present']
    assert result['ray_tmpdir_source']=='actual_scoped_trainer_environment'
    monkeypatch.setattr(m,'processes',lambda root:[])
    result=m.observe(root)
    assert not result['ray_session_present']
    assert 'No scoped live trainer' in result['runtime_observation']
