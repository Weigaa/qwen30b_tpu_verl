# vllm/utils/moe_stats.py
# SPDX-License-Identifier: Apache-2.0
import os, json, threading
from collections import defaultdict
from typing import Optional
import torch

class MoEStats:
    # current_batch_seq_ids = None  # type: Optional[torch.Tensor]
    def get_current_batch_seq_ids(self, num_scheduled_tokens):
        # print("[sched] scheduler_output[num_scheduled_tokens]:", scheduler_output.num_scheduled_tokens)
        out = []
        for rid, c in num_scheduled_tokens.items():
            v = int(rid.replace('-', '')[:16], 16)  # 取前16个hex
            if v >= (1 << 63):
                v -= (1 << 64)  # 映射到有符号int64
            out.extend([v] * int(c))
        self.current_batch_seq_ids = torch.tensor(out, dtype=torch.long)
        self.schedule_count += 1

    def __init__(self):
        self.enabled = os.getenv("VLLM_MOE_STATS", "1") == "1"
        print(f"MoEStats enabled: {self.enabled}")
        # 若只统计 response（解码阶段），设为1；若想 prefill+decode 全记，设为0
        self.decode_only = os.getenv("VLLM_MOE_STATS_DECODE_ONLY", "1") == "1"
        self._lock = threading.Lock()
        self.current_batch_seq_ids=None
        self.schedule_count = 0
        self.top_k_count = 0
        self.step_layer_topk: Optional[List[Optional[torch.Tensor]]] = None
        self.reset_epoch()
    
    @torch.no_grad()
    def record_layer_topk(self, layer_id: int, topk_ids: torch.Tensor):
        
        layer_id = int(layer_id)

        # 建议统一到 CPU，避免占用设备显存/graph 相关问题
        topk_ids_cpu = topk_ids.detach().to("cpu")

        if self.step_layer_topk is None:
            self.step_layer_topk = []

        if len(self.step_layer_topk) <= layer_id:
            self.step_layer_topk.extend([None] * (layer_id + 1 - len(self.step_layer_topk)))

        # 覆盖式记录：本 step 最新一次该层的 topk_ids
        self.step_layer_topk[layer_id] = topk_ids_cpu


    def reset_epoch(self):
        with self._lock:
            # per_prompt[prompt_id][layer_idx] = 1D tensor[num_experts]
            self.per_prompt = {}
            print("successfully reset MoEStats for new epoch.")

    def _ensure_vec(self, prompt_id: int, layer_idx: int, num_experts: int):
        d = self.per_prompt.setdefault(int(prompt_id), {})
        if layer_idx not in d:
            d[layer_idx] = torch.zeros(num_experts, dtype=torch.float32)

    @torch.no_grad()
    def record(self,
               layer_idx: int,
               topk_ids: torch.Tensor,
               num_experts: int,  # 移除 topk_weights 参数
               token_types: Optional[torch.Tensor] = None):
        if not self.enabled:
            return
        # 统一到 CPU，避免 graph/compile 干扰
        seq_ids = self.current_batch_seq_ids
        
        #确认schedule和topk_ids匹配
        self.top_k_count += 1

        if seq_ids is None:
            #print("MoEStats: current_batch_seq_ids is None, cannot record stats.") 
            #igonre dummy stats
            return
        # 确保是 1D Tensor
        assert isinstance(seq_ids, torch.Tensor) and seq_ids.dim() == 1, \
            f"seq_ids must be a 1D tensor, got {type(seq_ids)} with dim={getattr(seq_ids, 'dim', lambda:None)()}"
        seq_ids = seq_ids.detach().to("cpu")
        topk_ids = topk_ids.detach().to("cpu")

        # 仅统计 decode（response）部分：约定 token_types != 0 为 decode。
        # if token_types is not None and self.decode_only:
        #     mask = (token_types != 0)
        # else:
        #     mask = torch.ones(seq_ids.shape[0], dtype=torch.bool)

        # idxs = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        idxs = torch.arange(seq_ids.shape[0], device=seq_ids.device).tolist()
        #在graph模式下会padding token，所以会出现不匹配的现象，仍然也需要统计
        # if seq_ids.shape[0] != topk_ids.shape[0]:
        #     print(f"seq_ids.shape[0] ({seq_ids.shape[0]}) != topk_ids.shape[0] ({topk_ids.shape[0]}), cannot record stats.")
        #     print(f"seq_ids: {seq_ids}")
        #     print(f"topk_ids: {topk_ids}")
        #     return
        for i in idxs:
            pid = int(seq_ids[i].item())
            self._ensure_vec(pid, layer_idx, num_experts)
            ids = topk_ids[i].tolist()
            # 移除权重处理，直接每个选中专家+1
            for e in ids:
                self.per_prompt[pid][layer_idx][int(e)] += 1.0
        # print(f"MoEStats recorded layer {layer_idx} for {len(idxs)} tokens.")
        # print("current result:", self.per_prompt)

    def snapshot(self):
        # 转成纯 Python dict（便于 json.dump）
        out = {}
        for pid, layers in self.per_prompt.items():
            out[str(pid)] = {str(l): v.tolist() for l, v in layers.items()}
        print("current result:", out)
        print("current result size:", len(out))
        return out

# 全局实例
moe_stats = MoEStats()
