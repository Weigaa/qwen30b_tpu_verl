from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('implementation',['base','ascend_patch'])
@pytest.mark.parametrize('enabled,world_size,has_world,borrow',[
    ('1',1,True,True),('0',1,True,False),('1',1,False,False),
    ('1',16,True,False),
])
def test_borrowed_singleton_group_ownership(monkeypatch,enabled,world_size,has_world,borrow,implementation):
    from vllm.distributed import parallel_state as state
    from vllm import platforms
    from vllm_ascend.patch.worker import patch_distributed
    cls=(patch_distributed.GroupCoordinatorPatch if implementation=='ascend_patch'
         else patch_distributed.GroupCoordinatorPatch.__mro__[1])
    monkeypatch.setattr(patch_distributed,'create_hccl_pg_options',lambda _:None)
    monkeypatch.setattr(state.torch.npu,'current_device',lambda:0)
    monkeypatch.setattr(platforms,'current_platform',SimpleNamespace(
        is_cuda_alike=lambda:False,is_xpu=lambda:False,is_out_of_tree=lambda:False,
        is_tpu=lambda:False,is_cpu=lambda:True))
    monkeypatch.setenv('VERL_SIDECAR_SINGLETON_GROUP_REUSE',enabled)
    monkeypatch.setattr(state.torch.distributed,'get_rank',lambda:0)
    monkeypatch.setattr(state.torch.distributed,'get_world_size',lambda:world_size)
    world=SimpleNamespace(device_group=object(),cpu_group=object())
    monkeypatch.setattr(state,'_WORLD',world if has_world else None)
    created=[];destroyed=[]
    def new_group(ranks,backend,**kwargs):
        group=object();created.append((ranks,backend,group));return group
    monkeypatch.setattr(state.torch.distributed,'new_group',new_group)
    monkeypatch.setattr(state.torch.distributed,'destroy_process_group',destroyed.append)
    group=cls([[0]],0,'hccl',False,group_name='tp')
    assert len(created)==(0 if borrow else 2)
    assert group._owns_process_groups is not borrow
    if borrow:
        assert group.device_group is world.device_group
        assert group.cpu_group is world.cpu_group
    group.destroy()
    assert len(destroyed)==(0 if borrow else 2)
    assert not hasattr(group,'device_group')
    assert not hasattr(group,'cpu_group')
