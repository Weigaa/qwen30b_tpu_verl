#!/usr/bin/env python3
"""Generate floor4 MC2 live-pool rank-set strings for experiments."""
from __future__ import annotations

import argparse
import itertools


def fmt_group(group: tuple[int, ...]) -> str:
    sorted_group = tuple(sorted(group))
    if len(sorted_group) == 4 and sorted_group == tuple(range(sorted_group[0], sorted_group[0] + 4)):
        return f"{sorted_group[0]}-{sorted_group[-1]}"
    return ",".join(str(rank) for rank in sorted_group)


def sliding_groups(world_size: int, width: int, step: int) -> list[tuple[int, ...]]:
    groups = []
    for start in range(0, max(world_size - width + 1, 0), step):
        groups.append(tuple(range(start, start + width)))
    return groups


def block_groups(world_size: int, width: int) -> list[tuple[int, ...]]:
    groups = []
    for start in range(0, world_size, width):
        group = tuple(range(start, min(start + width, world_size)))
        if len(group) == width:
            groups.append(group)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--mode", choices=["all", "sliding", "blocks", "custom"], default="sliding")
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--custom", default="", help="Semicolon-separated groups, e.g. '12-15;10-13'.")
    parser.add_argument("--max-groups", type=int, default=0, help="Optional cap after generation.")
    args = parser.parse_args()

    if args.width != 4:
        raise SystemExit("This helper is intentionally limited to width=4 floor4 groups.")
    if args.mode == "all":
        groups = [tuple(group) for group in itertools.combinations(range(args.world_size), 4)]
    elif args.mode == "sliding":
        groups = sliding_groups(args.world_size, args.width, args.step)
    elif args.mode == "blocks":
        groups = block_groups(args.world_size, args.width)
    else:
        groups = []
        for item in args.custom.replace("|", ";").split(";"):
            item = item.strip()
            if not item:
                continue
            ranks = []
            for part in item.replace("+", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    start_s, end_s = part.split("-", 1)
                    start = int(start_s.strip())
                    end = int(end_s.strip())
                    step = 1 if end >= start else -1
                    ranks.extend(range(start, end + step, step))
                else:
                    ranks.append(int(part))
            group = tuple(sorted(dict.fromkeys(ranks)))
            if len(group) != 4:
                raise SystemExit(f"invalid custom group {item!r}: expected 4 unique ranks, got {group}")
            groups.append(group)

    deduped = []
    seen = set()
    for group in groups:
        group = tuple(sorted(group))
        if group in seen:
            continue
        if len(group) != 4:
            continue
        if any(rank < 0 or rank >= args.world_size for rank in group):
            raise SystemExit(f"rank out of range in group {group}")
        seen.add(group)
        deduped.append(group)

    if args.max_groups > 0:
        deduped = deduped[:args.max_groups]

    value = ";".join(fmt_group(group) for group in deduped)
    print(value)
    print(f"# groups={len(deduped)} mode={args.mode} world_size={args.world_size}")
    print(f"export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS='{value}'")
    print(f"export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_MAX_GROUPS={max(len(deduped) * 2, 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
