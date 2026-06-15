from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class RankRolePlan:
    donor_ranks: list[int]
    wave2_ranks: list[int]
    intermediate_survivor_ranks: list[int]
    final_survivor_ranks: list[int]
    package_topology: list[list[int]]
    intermediate_survivor_packages: list[list[int]]
    final_survivor_packages: list[list[int]]
    package_locality_score: float
    fallback_reason: Optional[str] = None


def default_package_topology(world_size: int) -> list[list[int]]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if world_size % 2 == 0:
        return [[rank, rank + 1] for rank in range(0, world_size, 2)]
    return [[rank] for rank in range(world_size)]


def parse_rank_topology(value: object,
                        world_size: Optional[int] = None) -> Optional[list[list[int]]]:
    if value is None or value == "":
        return None
    parsed = _parse_jsonish(value)
    if parsed is None:
        return None
    if not isinstance(parsed, list):
        raise ValueError("package_topology must be a list of rank lists")
    topology: list[list[int]] = []
    for package in parsed:
        if not isinstance(package, (list, tuple)):
            raise ValueError("package_topology must be a list of rank lists")
        ranks = [int(rank) for rank in package]
        topology.append(ranks)
    _validate_topology(topology, world_size)
    return topology


def parse_rank_list(value: object) -> Optional[list[int]]:
    if value is None or value == "":
        return None
    parsed = _parse_jsonish(value)
    if parsed is None:
        return None
    if isinstance(parsed, str):
        parsed = [item for item in parsed.split(",") if item.strip()]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"rank list must be a list or comma string, got {value!r}")
    return [int(rank) for rank in parsed]


def plan_survivor_ranks(
    world_size: int,
    shrink_stages: Sequence[int] = (8, 4),
    package_topology: Optional[Sequence[Sequence[int]]] = None,
    policy: str = "topology_aware",
    intermediate_survivor_ranks: Optional[Sequence[int]] = None,
    final_survivor_ranks: Optional[Sequence[int]] = None,
) -> RankRolePlan:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    stages = [int(stage) for stage in shrink_stages]
    if len(stages) != 2:
        raise ValueError(f"shrink_stages must contain exactly two stages, got {stages}")
    intermediate_size, final_size = stages
    if not (world_size >= intermediate_size >= final_size > 0):
        raise ValueError(
            "shrink_stages must satisfy world_size >= intermediate >= final > 0, "
            f"got world_size={world_size}, stages={stages}")

    normalized_topology = [list(map(int, pkg)) for pkg in (
        package_topology if package_topology is not None
        else default_package_topology(world_size))]
    _validate_topology(normalized_topology, world_size)

    policy = (policy or "topology_aware").lower().strip()
    fallback_reason: Optional[str] = None
    if policy == "manual":
        if intermediate_survivor_ranks is None or final_survivor_ranks is None:
            raise ValueError(
                "manual survivor selection requires both intermediate and final survivor ranks")
        intermediate = _validate_rank_set(
            intermediate_survivor_ranks, world_size, intermediate_size,
            "intermediate_survivor_ranks")
        final = _validate_rank_set(
            final_survivor_ranks, world_size, final_size,
            "final_survivor_ranks")
        _validate_subset(final, intermediate)
    elif policy == "contiguous":
        final = list(range(final_size))
        intermediate = list(range(intermediate_size))
    elif policy == "topology_aware":
        if intermediate_survivor_ranks is not None or final_survivor_ranks is not None:
            intermediate = _validate_rank_set(
                intermediate_survivor_ranks or [], world_size, intermediate_size,
                "intermediate_survivor_ranks")
            final = _validate_rank_set(
                final_survivor_ranks or [], world_size, final_size,
                "final_survivor_ranks")
            _validate_subset(final, intermediate)
            fallback_reason = "manual_ranks_override_topology_policy"
        else:
            intermediate, final, fallback_reason = _topology_aware_survivors(
                world_size, normalized_topology, intermediate_size, final_size)
    else:
        raise ValueError(
            "survivor_selection_policy must be topology_aware, contiguous, or manual, "
            f"got {policy!r}")

    intermediate = sorted(intermediate)
    final = sorted(final)
    _validate_subset(final, intermediate)

    all_ranks = set(range(world_size))
    final_set = set(final)
    intermediate_set = set(intermediate)
    donor = sorted(all_ranks - intermediate_set)
    wave2 = sorted(intermediate_set - final_set)
    intermediate_packages = _complete_packages_for_ranks(
        intermediate, normalized_topology)
    final_packages = _complete_packages_for_ranks(final, normalized_topology)
    locality = _package_locality_score(
        intermediate, final, normalized_topology)

    return RankRolePlan(
        donor_ranks=donor,
        wave2_ranks=wave2,
        intermediate_survivor_ranks=intermediate,
        final_survivor_ranks=final,
        package_topology=normalized_topology,
        intermediate_survivor_packages=intermediate_packages,
        final_survivor_packages=final_packages,
        package_locality_score=locality,
        fallback_reason=fallback_reason,
    )


def _parse_jsonish(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                if "," in stripped:
                    return [item.strip() for item in stripped.split(",") if item.strip()]
                raise
    return value


def _validate_rank_set(ranks: Sequence[int], world_size: int, size: int,
                       name: str) -> list[int]:
    values = [int(rank) for rank in ranks]
    if len(values) != size:
        raise ValueError(f"{name} must contain {size} ranks, got {len(values)}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate ranks: {values}")
    bad = [rank for rank in values if rank < 0 or rank >= world_size]
    if bad:
        raise ValueError(f"{name} contains out-of-range ranks for world_size={world_size}: {bad}")
    return sorted(values)


def _validate_subset(final: Sequence[int], intermediate: Sequence[int]) -> None:
    missing = sorted(set(final) - set(intermediate))
    if missing:
        raise ValueError(
            "final_survivor_ranks must be a subset of intermediate_survivor_ranks, "
            f"missing={missing}")


def _validate_topology(topology: Sequence[Sequence[int]],
                       world_size: Optional[int]) -> None:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for package in topology:
        if not package:
            raise ValueError("package_topology cannot contain empty packages")
        for rank in package:
            rank = int(rank)
            if world_size is not None and (rank < 0 or rank >= world_size):
                raise ValueError(
                    f"package_topology rank {rank} is out of range for world_size={world_size}")
            if rank in seen:
                duplicates.add(rank)
            seen.add(rank)
    if duplicates:
        raise ValueError(f"package_topology contains duplicate ranks: {sorted(duplicates)}")


def _topology_aware_survivors(
    world_size: int,
    topology: Sequence[Sequence[int]],
    intermediate_size: int,
    final_size: int,
) -> tuple[list[int], list[int], Optional[str]]:
    complete_final = _select_complete_packages(topology, final_size)
    complete_intermediate = _select_complete_packages(
        topology, intermediate_size, must_include=complete_final)
    if complete_final is not None and complete_intermediate is not None:
        final = _flatten(complete_final)
        intermediate = _flatten(complete_intermediate)
        return sorted(intermediate), sorted(final), None

    intermediate = list(range(intermediate_size))
    final = list(range(final_size))
    reason = (
        "topology_package_fit_failed;"
        f"fallback=contiguous;world_size={world_size};"
        f"intermediate_size={intermediate_size};final_size={final_size}")
    return intermediate, final, reason


def _select_complete_packages(
    topology: Sequence[Sequence[int]],
    target_size: int,
    must_include: Optional[Sequence[Sequence[int]]] = None,
) -> Optional[list[list[int]]]:
    selected = [list(pkg) for pkg in (must_include or [])]
    selected_keys = {_package_key(pkg) for pkg in selected}
    current = sum(len(pkg) for pkg in selected)
    if current > target_size:
        return None
    if current == target_size:
        return selected
    for package in topology:
        key = _package_key(package)
        if key in selected_keys:
            continue
        package_size = len(package)
        if current + package_size > target_size:
            continue
        selected.append(list(package))
        selected_keys.add(key)
        current += package_size
        if current == target_size:
            return selected
    return None


def _package_key(package: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(int(rank) for rank in package))


def _flatten(packages: Iterable[Sequence[int]]) -> list[int]:
    return [int(rank) for package in packages for rank in package]


def _complete_packages_for_ranks(
    ranks: Sequence[int],
    topology: Sequence[Sequence[int]],
) -> list[list[int]]:
    rank_set = set(int(rank) for rank in ranks)
    return [list(package) for package in topology if set(package).issubset(rank_set)]


def _package_locality_score(
    intermediate: Sequence[int],
    final: Sequence[int],
    topology: Sequence[Sequence[int]],
) -> float:
    def score_for(ranks: Sequence[int]) -> float:
        rank_set = set(ranks)
        if not rank_set:
            return 1.0
        complete = sum(len(pkg) for pkg in topology if set(pkg).issubset(rank_set))
        return complete / float(len(rank_set))

    return (score_for(intermediate) + score_for(final)) / 2.0
