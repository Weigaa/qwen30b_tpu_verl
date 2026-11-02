import pytest
import torch

from verl.utils.revision_expert_audit import fingerprint


def test_sampled_fingerprint_detects_changed_current_expert():
    value = torch.arange(64,dtype=torch.float32).reshape(2,4,8)
    before = fingerprint(value,0)
    value[0,0,0] += 1
    assert fingerprint(value,0) != before


def test_logical_transpose_is_normalized_but_unknown_shape_rejected():
    value = torch.arange(64,dtype=torch.float32).reshape(2,4,8)
    assert fingerprint(value,0) == fingerprint(value.transpose(1,2),0,[4,8])
    with pytest.raises(RuntimeError,match='logical shape'):
        fingerprint(value,0,[2,16])
