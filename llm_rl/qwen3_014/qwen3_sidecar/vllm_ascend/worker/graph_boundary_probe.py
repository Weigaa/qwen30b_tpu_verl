"""Opt-in first-decode boundary snapshots, not enabled in performance runs."""
import json
import os
from pathlib import Path

import torch
from vllm.forward_context import get_forward_context
from vllm.config import CUDAGraphMode


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in tensors(v)]
    return []


class GraphBoundaryProbe:
    def __init__(self, model):
        self.phase = None
        self.reference = {}
        self.captured = {}
        self.done = False
        self.in_first_attention = False
        self.active_reduce = None
        for name, module in model.named_modules():
            if (name == 'model.embed_tokens' or name == 'model.norm' or
                    name.startswith('model.layers.0.')):
                module.register_forward_hook(self.hook(name))
                if name.endswith('o_proj') or name.endswith('down_proj'):
                    self.trace_mm(name,module)
                if name.endswith('rotary_emb'):
                    module.register_forward_pre_hook(self.pre_hook(name))
        attention = model.get_submodule('model.layers.0.self_attn')
        attention.register_forward_pre_hook(lambda *_: self.set_attention(True))
        attention.register_forward_hook(lambda *_: self.set_attention(False))
        from vllm.distributed import get_tp_group
        communicator = get_tp_group().device_communicator
        original_reduce = communicator.all_reduce
        def traced_reduce(value):
            name=self.active_reduce
            if name:self.snapshot(name+'.reduce_input',value)
            result=original_reduce(value)
            if name:self.snapshot(name+'.reduce_output',result)
            return result
        communicator.all_reduce=traced_reduce

    def set_attention(self, value):
        self.in_first_attention=value

    def trace_mm(self,name,module):
        original=module.quant_method.apply
        def traced(*args,**kwargs):
            result=original(*args,**kwargs)
            self.snapshot(name+'.local_mm',result)
            self.active_reduce=name
            return result
        module.quant_method.apply=traced

    def snapshot(self,name,output):
        if self.done:return
        context=get_forward_context()
        if self.phase=='reference':
            self.reference[name]=[t.detach().clone() for t in tensors(output)]
        elif context.cudagraph_runtime_mode==CUDAGraphMode.FULL and context.capturing:
            key=context.batch_descriptor.num_tokens
            self.captured.setdefault(key,{})[name]=[t.detach().clone() for t in tensors(output)]

    def pre_hook(self,name):
        def snapshot(module,args):
            if name.endswith('rotary_emb') and not self.in_first_attention:return
            self.snapshot(name+'.inputs',args)
        return snapshot

    def hook(self, name):
        def snapshot(module, args, output):
            if name.endswith('rotary_emb') and not self.in_first_attention:return
            self.snapshot(name,output)
            if name.endswith('o_proj') or name.endswith('down_proj'):self.active_reduce=None
        return snapshot

    def compare(self, key, valid_rows):
        if self.done or not self.reference:
            return
        torch.npu.synchronize()
        graph = self.captured[key]
        rows = []
        for name, refs in self.reference.items():
            assert name in graph, name
            for index, (ref, got) in enumerate(zip(refs, graph[name])):
                left, right = ref[:valid_rows].float(), got[:valid_rows].float()
                rows.append(dict(name=name,index=index,shape=list(ref.shape),
                    max_abs=float((left-right).abs().max().item()) if left.numel() else 0.0,
                    equal=torch.equal(left,right)))
        rank = torch.distributed.get_rank()
        root = Path(os.environ['VLLM_ASCEND_GRAPH_BOUNDARY_PROBE'])
        root.mkdir(parents=True,exist_ok=True)
        (root/f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,key=key,
            valid_rows=valid_rows,boundaries=rows),indent=2)+'\n')
        print('GRAPH_BOUNDARY_PROBE '+json.dumps(dict(rank=rank,
            first_mismatch=next((r for r in rows if not r['equal']),None))),flush=True)
        self.done = True
