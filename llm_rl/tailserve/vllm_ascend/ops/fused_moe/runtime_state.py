# SPDX-License-Identifier: Apache-2.0
"""Restore derived MoE constants without changing graph-captured addresses."""
import sys

import torch


def restore_expert_ids_(expert_ids: torch.Tensor, num_local_experts: int) -> int:
    if num_local_experts <= 0 or expert_ids.ndim != 1:
        raise ValueError("Expected a 1D expert-id table and a positive local expert count")
    expected = torch.arange(expert_ids.numel(), dtype=expert_ids.dtype,
                            device=expert_ids.device).remainder_(num_local_experts)
    changed = int(torch.count_nonzero(expert_ids != expected).item())
    expert_ids.copy_(expected)
    return changed


def restore_loaded_moe_runtime_state_after_sleep() -> list[dict]:
    """Repair constants in both installed MoE registries after weight wake.

    AdaFloor's legacy stack and the v0.14 stack use separate registries and
    dispatcher classes. These dispatchers are not nn.Modules: their expert
    IDs are allocated during model load in CaMem's weights pool, but are not
    included in NPUWorker's named_buffers snapshot. Level-2 sleep discards
    them. Only inspect already loaded registries, and preserve tensor identity
    and address so any graph/dispatcher aliases remain valid.
    """
    reports = []
    seen = set()
    for module_name in (
        "vllm_ascend.ops.moe.moe_comm_method",
        "vllm_ascend.ops.fused_moe.moe_comm_method",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for method in getattr(module, "_MoECommMethods", {}).values():
            dispatcher = getattr(method, "token_dispatcher", None)
            if dispatcher is None or id(dispatcher) in seen:
                continue
            seen.add(id(dispatcher))
            ids = getattr(dispatcher, "expert_ids_per_ep_rank", None)
            if not isinstance(ids, torch.Tensor):
                continue
            local_experts = int(dispatcher.num_local_experts)
            before = ids[:16].cpu().tolist()
            changed = restore_expert_ids_(ids, local_experts)
            reports.append({
                "registry": module_name,
                "local_experts": local_experts,
                "entries": ids.numel(),
                "changed": changed,
                "address": ids.data_ptr(),
                "before": before,
                "after": ids[:16].cpu().tolist(),
            })
    return reports
