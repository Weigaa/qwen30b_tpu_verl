"""
Elastic Expert Parallelism - enables ranks to exit EP group dynamically.

When a DP rank finishes processing its prompts during rollout generation,
it can exit the EP group. The remaining ranks rebuild their EP/MC2/DP groups
and update expert mappings to ensure MOE dispatch/combine operates correctly.

Key design decisions:
- A saved "coordination group" (the original full DP group) is used for
  done-flag exchange. This group is NEVER rebuilt.
- EP/MC2/model-DP groups are rebuilt to exclude exited ranks.
- log2phy and expert_map are recomputed: new_moe_expert_num = new_ep_size * experts_per_rank.
  Orphaned experts (from exited ranks) are remapped to remaining ranks.
- TokenDispatcherWithMC2 is refreshed to use the new HCCL communicator.
"""

import torch
import torch.distributed as dist
from typing import List, Optional, Set

from vllm.logger import init_logger

logger = init_logger(__name__)


def create_elastic_group(active_global_ranks: List[int],
                         local_rank: int,
                         backend: str,
                         group_name: str):
    """Create a new process group for elastic EP.

    ALL world ranks must call this (collective operation).
    Returns GroupCoordinator for member ranks, None for others.
    """
    from vllm.distributed.parallel_state import init_model_parallel_group

    current_rank = dist.get_rank()

    if current_rank in active_global_ranks:
        return init_model_parallel_group(
            [active_global_ranks], local_rank, backend,
            group_name=group_name)
    else:
        # Non-member ranks must still participate in the collective
        dist.new_group(active_global_ranks, backend=backend)
        dist.new_group(active_global_ranks, backend="gloo")
        return None


class ElasticEPManager:
    """Manages dynamic EP group membership for elastic expert parallelism.

    When a rank finishes its prompts, it marks itself as done. At each
    synchronization point, all ranks exchange done flags using the
    coordination group. When new exits are detected, all ranks participate
    in rebuilding EP/MC2/DP groups.
    """

    def __init__(self, dp_size: int, dp_rank: int,
                 original_ep_size: int, experts_per_rank: int,
                 num_moe_experts: int,
                 global_redundant_expert_num: int = 0):
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.original_ep_size = original_ep_size
        self.current_ep_size = original_ep_size
        self.experts_per_rank = experts_per_rank
        self.num_moe_experts = num_moe_experts
        self.global_redundant_expert_num = global_redundant_expert_num
        self.global_num_experts = num_moe_experts + global_redundant_expert_num

        self.done_ranks: Set[int] = set()
        self.is_done = False

        # Save the original (full) DP group for coordination - NEVER rebuilt
        from vllm.distributed.parallel_state import get_dp_group, get_ep_group
        self._coord_group = get_dp_group().device_group
        self._original_ep_ranks = list(get_ep_group().ranks)

        # Save original group objects for reset between waves
        self._original_ep_group = get_ep_group()
        from vllm_ascend.distributed.parallel_state import get_mc2_group
        self._original_mc2_group = get_mc2_group()
        self._original_dp_group = get_dp_group()

    def sync_and_rebuild_if_needed(self, mark_self_done: bool = False) -> bool:
        """Sync done flags and rebuild groups if needed.

        ALL DP ranks (including already-parked ones) must call this
        at the same time (it contains collectives).

        Args:
            mark_self_done: If True, mark this rank as permanently done.

        Returns:
            True if groups were rebuilt (new exits detected).
        """
        if mark_self_done:
            self.is_done = True

        # All-gather done flags using the coordination group
        local_flag = torch.tensor(
            [1 if self.is_done else 0],
            dtype=torch.int32, device="npu")
        all_flags = [torch.zeros(1, dtype=torch.int32, device="npu")
                     for _ in range(self.dp_size)]
        dist.all_gather(all_flags, local_flag, group=self._coord_group)

        new_done_ranks = set()
        for r in range(self.dp_size):
            if all_flags[r].item() > 0 and r not in self.done_ranks:
                new_done_ranks.add(r)
                self.done_ranks.add(r)

        if not new_done_ranks:
            return False

        active_ranks = self.get_active_dp_ranks()
        logger.info(
            "ElasticEP: Ranks %s exited. Remaining active: %s (EP %d -> %d)",
            sorted(new_done_ranks), active_ranks,
            self.current_ep_size, len(active_ranks))

        # Rebuild all groups (ALL ranks must participate)
        self._rebuild_groups(active_ranks)
        return True

    def reset(self):
        """Reset state for the next rollout wave.

        Restores original EP/MC2/DP groups and clears done flags.
        Does NOT require collective operations - just restores saved refs.
        """
        if not self.done_ranks:
            return  # Nothing to reset

        from vllm.distributed.parallel_state import set_ep_group, set_dp_group
        from vllm_ascend.distributed.parallel_state import set_mc2_group

        set_ep_group(self._original_ep_group)
        set_mc2_group(self._original_mc2_group)
        set_dp_group(self._original_dp_group)

        self.done_ranks.clear()
        self.is_done = False
        self.current_ep_size = self.original_ep_size

        # Refresh MC2 dispatcher back to original group
        self._refresh_mc2_dispatcher()

        logger.info("ElasticEP: Reset to original groups (ep_size=%d)",
                    self.original_ep_size)

    def get_active_dp_ranks(self) -> List[int]:
        return sorted(r for r in range(self.dp_size) if r not in self.done_ranks)

    def _rebuild_groups(self, active_dp_ranks: List[int]):
        """Rebuild EP, MC2, and model-level DP groups.

        ALL world ranks must call this (contains collectives).
        """
        new_ep_size = len(active_dp_ranks)
        if new_ep_size == 0:
            return

        self.current_ep_size = new_ep_size

        from vllm.distributed.parallel_state import (
            get_world_group, set_ep_group, set_dp_group)
        from vllm_ascend.distributed.parallel_state import set_mc2_group

        backend = dist.get_backend(get_world_group().device_group)
        local_rank = get_world_group().local_rank

        # Map active DP ranks to global ranks in EP group
        # For TP=1: DP rank == EP rank == global rank (common MoE case)
        tp_size = len(self._original_ep_ranks) // self.dp_size
        active_ep_global_ranks = []
        for dp_rank in active_dp_ranks:
            start = dp_rank * tp_size
            end = start + tp_size
            active_ep_global_ranks.extend(
                self._original_ep_ranks[start:end])

        # Rebuild EP group
        new_ep = create_elastic_group(
            active_ep_global_ranks, local_rank, backend, "ep")
        if new_ep is not None:
            set_ep_group(new_ep)

        # Rebuild MC2 group (same ranks as EP)
        new_mc2 = create_elastic_group(
            active_ep_global_ranks, local_rank, backend, "mc2")
        if new_mc2 is not None:
            set_mc2_group(new_mc2)

        # Rebuild model-level DP group (active DP ranks only)
        # For TP>1, each TP worker has its own DP group
        current_rank = dist.get_rank()
        # Find which TP-local index this rank has
        tp_offset = 0
        for dp_rank in range(self.dp_size):
            base = dp_rank * tp_size
            for t in range(tp_size):
                if self._original_ep_ranks[base + t] == current_rank:
                    tp_offset = t
                    break

        # DP group for this TP offset
        active_dp_global_ranks = [
            self._original_ep_ranks[dp_rank * tp_size + tp_offset]
            for dp_rank in active_dp_ranks
        ]
        new_dp = create_elastic_group(
            active_dp_global_ranks, local_rank, backend, "dp")
        if new_dp is not None:
            set_dp_group(new_dp)

        # Update MC2 dispatcher and expert mappings (active ranks only)
        if not self.is_done:
            self._refresh_mc2_dispatcher()

    def _refresh_mc2_dispatcher(self):
        """Refresh TokenDispatcherWithMC2 with new MC2 group info."""
        from vllm_ascend.distributed.parallel_state import get_mc2_group
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.ops.moe.moe_comm_method import get_moe_comm_method

        mc2_method = get_moe_comm_method(MoECommType.MC2)
        if mc2_method is None:
            return

        new_mc2 = get_mc2_group()
        dispatcher = mc2_method.token_dispatcher

        device_group = new_mc2.device_group
        local_rank = dist.get_rank(group=device_group)
        backend_obj = device_group._get_backend(torch.device("npu"))
        dispatcher.moe_all_to_all_group_name = (
            backend_obj.get_hccl_comm_name(local_rank))
        dispatcher.ep_rank_id = new_mc2.rank_in_group
        dispatcher.ep_world_size = new_mc2.world_size

        logger.info(
            "ElasticEP: Refreshed MC2 dispatcher: ep_world_size=%d, "
            "ep_rank_id=%d", dispatcher.ep_world_size, dispatcher.ep_rank_id)

    def compute_new_expert_mapping(self):
        """Compute new log2phy and expert_map for the reduced EP group.

        Returns:
            (log2phy, expert_map, new_moe_expert_num) or
            (None, None, None) if this rank is exited.
        """
        active_ranks = self.get_active_dp_ranks()
        new_ep_size = len(active_ranks)
        if new_ep_size == 0:
            return None, None, None

        epr = self.experts_per_rank
        new_moe_expert_num = new_ep_size * epr

        # Mapping from original DP/EP rank to new EP rank
        orig_to_new = {orig: new_r for new_r, orig in enumerate(active_ranks)}

        if self.dp_rank not in orig_to_new:
            return None, None, None  # This rank exited

        # Compute log2phy: logical expert ID -> new physical ID
        total_experts = self.global_num_experts
        log2phy = torch.zeros(total_experts, dtype=torch.int32)

        for e in range(total_experts):
            orig_rank = e // epr
            local_idx = e % epr

            if orig_rank in orig_to_new:
                # Expert is on an active rank
                new_rank = orig_to_new[orig_rank]
                log2phy[e] = new_rank * epr + local_idx
            else:
                # Orphaned expert: distribute round-robin to active ranks
                target_new_rank = e % new_ep_size
                log2phy[e] = target_new_rank * epr + local_idx

        # Compute expert_map for this rank
        my_new_rank = orig_to_new[self.dp_rank]
        expert_map = torch.full((new_moe_expert_num,), -1, dtype=torch.int32)
        for i in range(epr):
            physical_id = my_new_rank * epr + i
            if physical_id < new_moe_expert_num:
                expert_map[physical_id] = i

        return log2phy, expert_map, new_moe_expert_num
