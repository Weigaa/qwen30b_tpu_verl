#!/usr/bin/env python3
"""Extract a conservative per-rank KV capacity from a vLLM startup log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CAPACITY_RE = re.compile(r"GPU KV cache size:\s*([0-9,]+) tokens")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--block-size", required=True, type=int)
    return parser.parse_args()


def extract(log_path: Path, world_size: int, block_size: int) -> int:
    if world_size <= 0 or block_size <= 0:
        raise ValueError("world size and block size must be positive")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    capacities = [
        int(raw.replace(",", "")) for raw in CAPACITY_RE.findall(text)
    ]
    if len(capacities) != world_size:
        raise ValueError(
            f"{log_path}: expected {world_size} per-rank KV capacities, "
            f"found {len(capacities)}"
        )
    invalid = [
        value for value in capacities if value <= 0 or value % block_size != 0
    ]
    if invalid:
        raise ValueError(f"{log_path}: invalid KV capacities {invalid}")
    return min(capacities)


def main() -> int:
    args = parse_args()
    print(extract(args.log, args.world_size, args.block_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
