import json
import os
import pytest

from observe_n2_endpoint_receipt import capture


def test_missing_endpoint_is_only_a_wait(tmp_path):
    assert capture(tmp_path) is False
    assert not (tmp_path/'ENDPOINT_RETENTION_AUDIT.json').exists()


def test_observer_preserves_only_small_provenance_not_weights(tmp_path):
    source=tmp_path/'graph_sidecar/seed_0/natural_f2/fig6/epoch_001_mode1_natural/checkpoints/qwen3moe_for_eagle3/global_step_5'
    endpoint=tmp_path/'heldout_checkpoint/global_step_5'
    source.mkdir(parents=True);endpoint.mkdir(parents=True)
    (source/'actor.bin').write_bytes(b'test actor')
    os.link(source/'actor.bin',endpoint/'actor.bin')
    stat=(endpoint/'actor.bin').stat()
    value=dict(status='COMMITTED_ENDPOINT_HARDLINKED',source=str(source),endpoint=str(endpoint),
        additional_actor_updates_from_public_epoch0=5,actor_frozen=False,seed=0,
        files=[dict(path='actor.bin',device=stat.st_dev,inode=stat.st_ino,bytes=stat.st_size)])
    (endpoint/'RETENTION_RECEIPT.json').write_text(json.dumps(value))
    assert capture(tmp_path) is True
    report=json.loads((tmp_path/'ENDPOINT_RETENTION_AUDIT.json').read_text())
    assert report['receipt']==value
    assert report['weight_data_copied'] is False
    assert report['extra_weight_hardlinks_created'] is False
    assert (endpoint/'actor.bin').stat().st_nlink==stat.st_nlink
    (endpoint/'actor.bin').unlink()
    (endpoint/'actor.bin').write_bytes(b'different')
    with pytest.raises(RuntimeError,match='identity changed'):
        capture(tmp_path)
