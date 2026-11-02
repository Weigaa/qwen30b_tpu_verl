import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('endpoint_retainer_test',Path(__file__).with_name('retain_training_endpoint.py'))
retainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(retainer)


def committed(tmp_path,shards=32):
    epoch=tmp_path/'epoch';(epoch/'logs').mkdir(parents=True)
    (epoch/'logs/run.txt').write_text('\n'.join(f'step:{i} - actor/lr:1e-6' for i in range(1,6))+
                                      '\n[run] end_time=2026-10-24T03:00:00+0800 exit_code=0\n')
    (epoch/'offline_planning_history.json').write_text('{}')
    folder=epoch/'checkpoints/qwen3moe_for_eagle3/global_step_5/actor/dist_ckpt';folder.mkdir(parents=True)
    (folder/'.metadata').write_bytes(b'metadata')
    for i in range(shards):(folder/f'{i}.distcp').write_bytes(b'committed')
    return epoch


def test_incomplete_checkpoint_not_accepted(tmp_path):
    assert retainer.ready(committed(tmp_path,31)) is None


def test_hardlinks_preserve_model_after_source_cleanup(tmp_path):
    source=retainer.ready(committed(tmp_path));destination=tmp_path/'retained/global_step_5'
    receipt=retainer.retain(source,destination)
    original=source/'actor/dist_ckpt/0.distcp';retained=destination/'actor/dist_ckpt/0.distcp'
    assert original.stat().st_ino==retained.stat().st_ino
    original.unlink()
    assert retained.read_bytes()==b'committed'
    assert receipt['status']=='COMMITTED_ENDPOINT_HARDLINKED'
    with pytest.raises(RuntimeError,match='existing endpoint'):
        retainer.retain(source,destination)
