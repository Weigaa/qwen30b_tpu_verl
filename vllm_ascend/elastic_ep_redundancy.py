"""
Expert Redundancy Placement for Elastic EP.

Manages pre-loading of redundant expert copies so that when EP ranks exit,
their experts are already available on remaining ranks (zero-copy switch).

Key concepts:
- Primary experts: base allocation, experts_per_rank per rank
- Redundant experts: extra copies pre-loaded for fault tolerance
- Redundancy budget: max extra experts each rank can store (memory-limited)
- Placement strategy: decides WHICH experts to redundantly store WHERE

Design:
- Weight tensors are allocated with (experts_per_rank + redundancy_budget) slots
- During model init, both primary and redundant experts are loaded
- expert_map includes redundant entries so weight_loader loads them
- After loading, runtime expert_map only exposes primary experts
- On rank exit, expert_map/log2phy are updated to activate redundant copies
- For missing experts (not cached), CPU loading fills remaining slots

Theory (from the paper):
- Minimum redundancy for n-node fault tolerance: n * k (k = experts_per_rank)
- If n-node full redundancy is infeasible, uniform placement maximizes
  expected recovery hit rate under random exits
- Predictive placement uses sequence length distributions to estimate
  per-rank exit probability and prioritize redundancy accordingly
"""

import torch
from typing import Dict, List, Optional, Set, Tuple

from vllm.logger import init_logger

logger = init_logger(__name__)


class RedundancyPlacement:
    """Base class for expert redundancy placement strategies.

    Subclasses implement _compute_placement() to decide which experts
    each rank redundantly stores.
    """

    def __init__(self, num_experts: int, ep_size: int,
                 experts_per_rank: int, redundancy_budget: int):
        """
        Args:
            num_experts: Total number of logical experts (e.g. 128)
            ep_size: Number of EP ranks (e.g. 16)
            experts_per_rank: Primary experts per rank (e.g. 8)
            redundancy_budget: Max extra experts per rank (memory-limited)
        """
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.experts_per_rank = experts_per_rank
        self.redundancy_budget = min(redundancy_budget,
                                     num_experts - experts_per_rank)

        # placement[rank] = sorted list of global expert IDs redundantly stored
        self.placement: Dict[int, List[int]] = {}
        self._compute_placement()

    def _compute_placement(self):
        raise NotImplementedError

    def get_redundant_experts(self, rank: int) -> List[int]:
        """Return list of global expert IDs redundantly stored on this rank."""
        return self.placement.get(rank, [])

    def get_total_local_experts(self, rank: int) -> int:
        """Total experts on this rank (primary + redundant)."""
        return self.experts_per_rank + len(self.get_redundant_experts(rank))

    def get_primary_range(self, rank: int) -> Tuple[int, int]:
        """Return (start, end) global expert ID range for primary experts."""
        start = rank * self.experts_per_rank
        end = min(start + self.experts_per_rank, self.num_experts)
        return start, end

    def get_expert_map_for_loading(self, rank: int) -> Tuple[int, torch.Tensor]:
        """Get expert_map including redundant experts (for weight loading).

        Returns:
            (total_local_experts, expert_map) where expert_map has
            entries for both primary and redundant experts.
        """
        epr = self.experts_per_rank
        redundant = self.get_redundant_experts(rank)
        total_local = epr + len(redundant)

        expert_map = torch.full((self.num_experts,), -1, dtype=torch.int32)

        # Primary experts
        start, end = self.get_primary_range(rank)
        for i, gid in enumerate(range(start, end)):
            expert_map[gid] = i

        # Redundant experts (stored after primary in weight tensor)
        for idx, gid in enumerate(redundant):
            expert_map[gid] = epr + idx

        return total_local, expert_map

    def get_expert_map_primary_only(self, rank: int) -> torch.Tensor:
        """Get expert_map with only primary experts (for runtime use).

        Redundant experts are masked out (set to -1) so MC2 dispatch
        does not route tokens to them during normal operation.
        """
        expert_map = torch.full((self.num_experts,), -1, dtype=torch.int32)
        start, end = self.get_primary_range(rank)
        for i, gid in enumerate(range(start, end)):
            expert_map[gid] = i
        return expert_map

    def get_experts_on_rank_for_cpu_cache(self, rank: int) -> List[int]:
        """Get primary experts on this rank that should be pre-cached to CPU.

        These are experts that are NOT redundantly stored on any other rank,
        meaning if this rank exits, they'd need CPU loading.
        """
        start, end = self.get_primary_range(rank)
        my_primary = set(range(start, end))

        # Check which of my primary experts are cached on other ranks
        cached_elsewhere = set()
        for other_rank in range(self.ep_size):
            if other_rank == rank:
                continue
            cached_elsewhere.update(
                e for e in self.get_redundant_experts(other_rank)
                if e in my_primary)

        # Return uncovered experts (not cached elsewhere)
        return sorted(my_primary - cached_elsewhere)

    def get_log2phy_for_loading(self, rank: int) -> torch.Tensor:
        """Get log2phy map that includes redundant expert mappings.

        During normal operation, each rank preferentially routes to its
        own local copies (primary + redundant).
        """
        from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map

        # Build expert_map_all including redundant experts
        expert_map_all = torch.full(
            (self.ep_size, self.num_experts), -1, dtype=torch.int32)

        for r in range(self.ep_size):
            _, emap = self.get_expert_map_for_loading(r)
            expert_map_all[r] = emap

        log2phy_all = generate_log2phy_map(expert_map_all)
        return log2phy_all[rank]


class UniformRedundancyPlacement(RedundancyPlacement):
    """Uniform placement: distribute redundant experts as evenly as possible.

    Each rank stores experts from other ranks in a round-robin pattern,
    ensuring maximal coverage diversity. This maximizes expected hit rate
    when exits are uniformly random.
    """

    def _compute_placement(self):
        epr = self.experts_per_rank
        budget = self.redundancy_budget

        for rank in range(self.ep_size):
            primary_start, primary_end = self.get_primary_range(rank)
            primary_set = set(range(primary_start, primary_end))

            # Collect all experts not primary to this rank
            other_experts = [e for e in range(self.num_experts)
                             if e not in primary_set]

            # Select up to budget experts, spread evenly across other ranks
            # Strategy: interleave from all other ranks for maximal diversity
            selected = []
            if budget > 0 and other_experts:
                # Group by source rank
                other_ranks = [r for r in range(self.ep_size) if r != rank]
                # Round-robin: take one expert from each other rank in turn
                expert_idx_per_rank = {r: 0 for r in other_ranks}
                while len(selected) < budget:
                    added_this_round = False
                    for src_rank in other_ranks:
                        if len(selected) >= budget:
                            break
                        src_start, src_end = self.get_primary_range(src_rank)
                        idx = expert_idx_per_rank[src_rank]
                        if src_start + idx < src_end:
                            selected.append(src_start + idx)
                            expert_idx_per_rank[src_rank] = idx + 1
                            added_this_round = True
                    if not added_this_round:
                        break

            self.placement[rank] = sorted(selected)


class PredictiveRedundancyPlacement(RedundancyPlacement):
    """Placement based on predicted exit probabilities.

    Ranks with higher exit probability (shorter sequences, finish earlier)
    have their experts preferentially stored on ranks with lower exit
    probability (longer sequences, finish later).

    This is a greedy strategy: iterate over high-exit-probability ranks
    and assign their experts to low-exit-probability ranks.
    """

    def __init__(self, num_experts: int, ep_size: int,
                 experts_per_rank: int, redundancy_budget: int,
                 exit_probabilities: Optional[List[float]] = None):
        """
        Args:
            exit_probabilities: Per-rank probability of early exit.
                Higher value = more likely to finish early.
                If None, falls back to uniform placement.
        """
        if exit_probabilities is None:
            exit_probabilities = [1.0 / ep_size] * ep_size
        assert len(exit_probabilities) == ep_size
        self.exit_probabilities = exit_probabilities
        super().__init__(num_experts, ep_size, experts_per_rank,
                         redundancy_budget)

    def _compute_placement(self):
        epr = self.experts_per_rank
        budget = self.redundancy_budget

        # Initialize
        for rank in range(self.ep_size):
            self.placement[rank] = []

        # Sort ranks by exit probability
        # risky_order: most likely to exit first
        risky_order = sorted(range(self.ep_size),
                             key=lambda r: self.exit_probabilities[r],
                             reverse=True)
        # safe_order: least likely to exit first (prefer storing on these)
        safe_order = sorted(range(self.ep_size),
                            key=lambda r: self.exit_probabilities[r])

        remaining_budget = {r: budget for r in range(self.ep_size)}

        # Greedily assign experts from risky ranks to safe ranks
        for risky_rank in risky_order:
            risky_start, risky_end = self.get_primary_range(risky_rank)
            risky_experts = list(range(risky_start, risky_end))

            for expert_id in risky_experts:
                # Find safest rank with available budget
                for safe_rank in safe_order:
                    if safe_rank == risky_rank:
                        continue
                    if remaining_budget[safe_rank] <= 0:
                        continue
                    # Skip if already primary on safe_rank
                    s_start, s_end = self.get_primary_range(safe_rank)
                    if s_start <= expert_id < s_end:
                        continue
                    # Skip if already placed
                    if expert_id in self.placement[safe_rank]:
                        continue

                    self.placement[safe_rank].append(expert_id)
                    remaining_budget[safe_rank] -= 1
                    break

        # Sort placements for deterministic behavior
        for rank in range(self.ep_size):
            self.placement[rank] = sorted(self.placement[rank])


# ---- Module-level placement registry ----
_REDUNDANCY_PLACEMENT: Optional[RedundancyPlacement] = None


def get_or_create_placement(
    num_experts: int, ep_size: int,
    experts_per_rank: int, redundancy_budget: int,
    strategy: str = "uniform",
    exit_probabilities: Optional[List[float]] = None
) -> RedundancyPlacement:
    """Get or create the global RedundancyPlacement singleton."""
    global _REDUNDANCY_PLACEMENT
    if _REDUNDANCY_PLACEMENT is not None:
        return _REDUNDANCY_PLACEMENT

    if strategy == "predictive":
        _REDUNDANCY_PLACEMENT = PredictiveRedundancyPlacement(
            num_experts, ep_size, experts_per_rank, redundancy_budget,
            exit_probabilities)
    else:
        _REDUNDANCY_PLACEMENT = UniformRedundancyPlacement(
            num_experts, ep_size, experts_per_rank, redundancy_budget)

    logger.info(
        "ElasticEP redundancy: strategy=%s, budget=%d, ep_size=%d, "
        "num_experts=%d, epr=%d",
        strategy, redundancy_budget, ep_size, num_experts, experts_per_rank)
    for r in range(ep_size):
        redundant = _REDUNDANCY_PLACEMENT.get_redundant_experts(r)
        if redundant:
            logger.info("  Rank %d redundant experts: %s", r, redundant)

    return _REDUNDANCY_PLACEMENT


def get_placement() -> Optional[RedundancyPlacement]:
    """Get the current placement, or None if not initialized."""
    return _REDUNDANCY_PLACEMENT


def reset_placement():
    """Reset the placement singleton (for testing)."""
    global _REDUNDANCY_PLACEMENT
    _REDUNDANCY_PLACEMENT = None


class ExpertRedundancyManager:
    """Manages redundant expert copies for elastic EP.

    Provides:
    1. Expert map computation with redundancy for model init
    2. Hit/miss analysis when ranks exit
    3. New expert mapping computation using cached redundant experts
    4. CPU weight cache management for uncached experts
    """

    def __init__(self, placement: RedundancyPlacement,
                 hidden_size: int, intermediate_size_per_partition: int,
                 params_dtype: torch.dtype):
        self.placement = placement
        self.num_experts = placement.num_experts
        self.ep_size = placement.ep_size
        self.epr = placement.experts_per_rank
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition
        self.params_dtype = params_dtype

        # CPU weight cache: layer_idx -> expert_id -> {'w13': tensor, 'w2': tensor}
        self._cpu_expert_cache: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

    def analyze_rank_exit(
        self, exited_ranks: Set[int], active_ranks: List[int]
    ) -> Dict[int, Dict[str, List[int]]]:
        """Analyze which orphaned experts are cached vs need CPU loading.

        Args:
            exited_ranks: Ranks that have exited
            active_ranks: Remaining active ranks (sorted)

        Returns:
            Dict[rank] -> {
                'hits': [global_expert_ids cached on this rank],
                'misses': [global_expert_ids assigned to this rank for CPU load]
            }
        """
        # Collect all orphaned experts
        orphaned = set()
        for rank in exited_ranks:
            start, end = self.placement.get_primary_range(rank)
            orphaned.update(range(start, end))

        # Find hits: which active rank has which orphaned expert cached
        global_hits: Dict[int, int] = {}  # expert_id -> caching_rank
        for rank in active_ranks:
            redundant = set(self.placement.get_redundant_experts(rank))
            for e in orphaned:
                if e in redundant and e not in global_hits:
                    global_hits[e] = rank

        # Remaining misses: orphaned experts not cached anywhere
        misses = sorted(orphaned - set(global_hits.keys()))

        # Compute per-rank hit counts and remaining capacity
        hits_per_rank: Dict[int, List[int]] = {r: [] for r in active_ranks}
        for e, r in global_hits.items():
            hits_per_rank[r].append(e)

        budget = self.placement.redundancy_budget
        remaining_capacity = {}
        for rank in active_ranks:
            # Capacity = budget - hits_already_using_redundant_slots
            remaining_capacity[rank] = max(
                0, budget - len(hits_per_rank[rank]))

        # Distribute misses to active ranks, respecting capacity
        miss_assignment: Dict[int, List[int]] = {r: [] for r in active_ranks}
        unassigned = []
        for i, expert_id in enumerate(misses):
            # Round-robin, but skip ranks without capacity
            assigned = False
            for j in range(len(active_ranks)):
                target = active_ranks[(i + j) % len(active_ranks)]
                if remaining_capacity[target] > 0:
                    miss_assignment[target].append(expert_id)
                    remaining_capacity[target] -= 1
                    assigned = True
                    break
            if not assigned:
                unassigned.append(expert_id)

        if unassigned:
            logger.warning(
                "ElasticEP: %d orphaned experts cannot fit in weight "
                "tensors (budget=%d). These experts will use approximate "
                "routing.", len(unassigned), budget)

        # Build per-rank result
        result = {}
        for rank in active_ranks:
            result[rank] = {
                'hits': sorted(hits_per_rank[rank]),
                'misses': miss_assignment[rank],
            }

        total_hits = sum(len(v['hits']) for v in result.values())
        total_misses = sum(len(v['misses']) for v in result.values())
        logger.info(
            "ElasticEP redundancy: %d orphaned experts, %d hits (%.1f%%), "
            "%d misses need CPU loading",
            len(orphaned), total_hits,
            100.0 * total_hits / max(len(orphaned), 1), total_misses)

        return result

    def compute_elastic_expert_mapping(
        self, active_ranks: List[int], my_rank: int,
        hit_miss_info: Dict[int, Dict[str, List[int]]]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        """Compute new log2phy and expert_map after ranks exit, using redundancy.

        Uses redundant copies for hits and assigns physical slots for misses.
        The physical ID space is expanded to accommodate extra experts:
        slots_per_rank = epr + max_extra_per_rank.

        Returns:
            (log2phy, expert_map, new_moe_expert_num) or None if rank exited
        """
        if my_rank not in active_ranks:
            return None

        epr = self.epr
        new_ep_size = len(active_ranks)
        orig_to_new = {orig: new_r for new_r, orig in enumerate(active_ranks)}

        # Determine slots_per_rank: epr + max extra experts any rank handles
        max_extra = 0
        for rank in active_ranks:
            info = hit_miss_info.get(rank, {'hits': [], 'misses': []})
            extra = len(info['hits']) + len(info['misses'])
            max_extra = max(max_extra, extra)

        slots_per_rank = epr + max_extra
        new_moe_expert_num = new_ep_size * slots_per_rank

        # Build expert_map for my_rank
        my_new_rank = orig_to_new[my_rank]
        my_info = hit_miss_info.get(my_rank, {'hits': [], 'misses': []})

        expert_map = torch.full(
            (new_moe_expert_num,), -1, dtype=torch.int32)

        # Primary experts: weight indices 0..epr-1
        for i in range(epr):
            phys_id = my_new_rank * slots_per_rank + i
            expert_map[phys_id] = i

        # Redundant hits: weight indices epr..epr+len(hits)-1
        # These are already loaded in the weight tensor
        my_redundant = self.placement.get_redundant_experts(my_rank)
        for slot_offset, expert_id in enumerate(my_info['hits']):
            phys_id = my_new_rank * slots_per_rank + epr + slot_offset
            # Find weight index: position in redundant experts list
            redundant_idx = my_redundant.index(expert_id)
            weight_idx = epr + redundant_idx
            expert_map[phys_id] = weight_idx

        # CPU-loaded misses: weight indices for dynamically loaded experts
        # These will be loaded into available weight slots
        miss_offset = len(my_info['hits'])
        for slot_offset, expert_id in enumerate(my_info['misses']):
            phys_id = (my_new_rank * slots_per_rank
                       + epr + miss_offset + slot_offset)
            # Weight index: use slots after existing redundant experts
            # The actual weight data must be loaded from CPU
            weight_idx = epr + len(my_redundant) + slot_offset
            expert_map[phys_id] = weight_idx

        # Build log2phy: logical expert ID -> physical ID
        log2phy = torch.zeros(self.num_experts, dtype=torch.int32)

        # Build a global routing table: for each expert, which new rank
        # and which physical slot?
        expert_to_phys: Dict[int, int] = {}

        for rank in active_ranks:
            new_r = orig_to_new[rank]
            r_info = hit_miss_info.get(rank, {'hits': [], 'misses': []})

            # Primary experts
            p_start, p_end = self.placement.get_primary_range(rank)
            for i, gid in enumerate(range(p_start, p_end)):
                expert_to_phys[gid] = new_r * slots_per_rank + i

            # Redundant hits
            for slot_offset, expert_id in enumerate(r_info['hits']):
                expert_to_phys[expert_id] = (
                    new_r * slots_per_rank + epr + slot_offset)

            # CPU-loaded misses
            miss_off = len(r_info['hits'])
            for slot_offset, expert_id in enumerate(r_info['misses']):
                expert_to_phys[expert_id] = (
                    new_r * slots_per_rank + epr + miss_off + slot_offset)

        for e in range(self.num_experts):
            if e in expert_to_phys:
                log2phy[e] = expert_to_phys[e]
            else:
                # Should not happen if all experts are covered
                # Fallback: map to first active rank's first slot
                log2phy[e] = 0
                logger.warning(
                    "ElasticEP: Expert %d has no physical mapping!", e)

        return log2phy, expert_map, new_moe_expert_num

    def save_expert_weights_to_cpu(
        self, layer_idx: int, expert_id: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor
    ):
        """Save expert weights to CPU cache.

        Args:
            layer_idx: MOE layer index
            expert_id: Global expert ID
            w13_weight: Weight data for this expert (single expert slice)
            w2_weight: Weight data for this expert (single expert slice)
        """
        if layer_idx not in self._cpu_expert_cache:
            self._cpu_expert_cache[layer_idx] = {}
        self._cpu_expert_cache[layer_idx][expert_id] = {
            'w13': w13_weight.cpu().clone(),
            'w2': w2_weight.cpu().clone(),
        }

    def load_expert_from_cpu(
        self, layer_idx: int, expert_id: int,
        w13_target: torch.Tensor,
        w2_target: torch.Tensor,
        weight_idx: int
    ) -> bool:
        """Load expert weights from CPU cache into weight tensors.

        Args:
            layer_idx: MOE layer index
            expert_id: Global expert ID to load
            w13_target: The w13_weight parameter tensor
            w2_target: The w2_weight parameter tensor
            weight_idx: Local weight index to load into

        Returns:
            True if loaded successfully, False if not in cache
        """
        layer_cache = self._cpu_expert_cache.get(layer_idx, {})
        if expert_id not in layer_cache:
            logger.warning(
                "ElasticEP: Expert %d layer %d not in CPU cache",
                expert_id, layer_idx)
            return False

        cached = layer_cache[expert_id]
        w13_target.data[weight_idx].copy_(cached['w13'].to(w13_target.device))
        w2_target.data[weight_idx].copy_(cached['w2'].to(w2_target.device))
        return True
