#!/usr/bin/env python3
"""Summarize sampled per-bucket NPU op profiler traces.

This script expects traces produced with:

  VLLM_ASCEND_BUCKET_OP_PROFILE=1
  VLLM_ASCEND_BUCKET_OP_PROFILE_DIR=/home/sharedata/wj_records/op_profiles/...

Each sampled step is stored under:

  bucket_<step>/rank_<rank>_<live|dummy>_step_<step>/

The output is intentionally coarse: op names differ across CANN / torch_npu
versions, so we aggregate by regex buckets that are useful for comparing
baseline EP=16 with shrink EP=8.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PATH_RE = re.compile(r"bucket_(?P<bucket>\d+).*rank_(?P<rank>\d+)_(?P<kind>live|dummy)_step_(?P<step>\d+)")

CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("moe_dispatch", re.compile(r"dispatch|token_dispatch|moe.*send|send.*moe", re.I)),
    ("moe_combine", re.compile(r"combine|token_combine|moe.*recv|recv.*moe", re.I)),
    ("comm", re.compile(r"hccl|alltoall|all_to_all|allgather|all_gather|reducescatter|reduce_scatter|broadcast", re.I)),
    ("grouped_matmul", re.compile(r"grouped.*matmul|groupedmatmul|aclnngroupedmatmul|batchmatmul|matmul|mm\\b|bmm\\b", re.I)),
    ("attention", re.compile(r"attention|flash|pagedattention|pa_\\w|reshapeandcache|rope|rotary", re.I)),
    ("routing_index", re.compile(r"topk|nonzero|where|sort|argsort|cumsum|gather|scatter|index|onehot|histogram", re.I)),
    ("sampling", re.compile(r"sample|sampling|softmax|multinomial|argmax|logits", re.I)),
    ("copy_mem", re.compile(r"copy|memcpy|transdata|transpose|contiguous|cast|permute", re.I)),
]


def iter_json_files(root: Path) -> Iterable[Path]:
    for pattern in ("*.json", "*.json.gz"):
        yield from root.rglob(pattern)


def load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def iter_events(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        events = obj.get("traceEvents")
        if isinstance(events, list):
            for ev in events:
                if isinstance(ev, dict):
                    yield ev
        elif "name" in obj:
            yield obj
    elif isinstance(obj, list):
        for ev in obj:
            if isinstance(ev, dict):
                yield ev


def event_duration_us(ev: dict[str, Any]) -> float:
    dur = ev.get("dur", ev.get("duration", 0.0))
    try:
        return float(dur or 0.0)
    except (TypeError, ValueError):
        return 0.0


def categorize(name: str) -> str:
    for category, pattern in CATEGORY_PATTERNS:
        if pattern.search(name):
            return category
    return "other"


def profile_key(path: Path) -> tuple[int, int, str, int] | None:
    text = str(path)
    match = PATH_RE.search(text)
    if not match:
        return None
    return (
        int(match.group("bucket")),
        int(match.group("rank")),
        match.group("kind"),
        int(match.group("step")),
    )


def _latest_profile_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    if not candidates:
        raise SystemExit(f"No profile directories found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profile_dir",
        nargs="?",
        type=Path,
        help=(
            "Profile directory. Defaults to latest under "
            "$WJ_RECORDS_DIR/op_profiles."),
    )
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    if args.profile_dir is None:
        records_dir = Path(os.getenv("WJ_RECORDS_DIR", "/home/sharedata/wj_records"))
        args.profile_dir = _latest_profile_dir(records_dir / "op_profiles")

    totals = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    top_ops = defaultdict(Counter)
    files = list(iter_json_files(args.profile_dir))
    parsed_files = 0
    events_used = 0

    for path in files:
        key = profile_key(path)
        if key is None:
            continue
        try:
            obj = load_json(path)
        except Exception:
            continue
        parsed_files += 1
        for ev in iter_events(obj):
            name = str(ev.get("name", ""))
            dur_us = event_duration_us(ev)
            if dur_us <= 0 or not name:
                continue
            category = categorize(name)
            totals[key][category] += dur_us / 1000.0  # ms
            totals[key]["total_ms"] += dur_us / 1000.0
            counts[key][category] += 1
            top_ops[key][name] += dur_us / 1000.0
            events_used += 1

    categories = [
        "total_ms",
        "moe_dispatch",
        "moe_combine",
        "comm",
        "grouped_matmul",
        "attention",
        "routing_index",
        "sampling",
        "copy_mem",
        "other",
    ]
    print(f"profile_dir={args.profile_dir}")
    print(f"json_files={len(files)} parsed_json_files={parsed_files} events_used={events_used}")
    if not totals:
        print("No profiler op events found. Check VLLM_ASCEND_BUCKET_OP_PROFILE_DIR and profiler export files.")
        return

    print("\nper_rank_bucket_ms")
    print("bucket,rank,kind,step," + ",".join(categories))
    for key in sorted(totals):
        bucket, rank, kind, step = key
        values = [totals[key].get(cat, 0.0) for cat in categories]
        print(f"{bucket},{rank},{kind},{step}," + ",".join(f"{v:.3f}" for v in values))

    bucket_totals = defaultdict(lambda: defaultdict(float))
    bucket_counts = Counter()
    for (bucket, _rank, kind, _step), values in totals.items():
        bkey = (bucket, kind)
        bucket_counts[bkey] += 1
        for cat, value in values.items():
            bucket_totals[bkey][cat] += value

    print("\nper_bucket_rank_avg_ms")
    print("bucket,kind,num_ranks," + ",".join(categories))
    for bkey in sorted(bucket_totals):
        bucket, kind = bkey
        n = bucket_counts[bkey]
        values = [bucket_totals[bkey].get(cat, 0.0) / n for cat in categories]
        print(f"{bucket},{kind},{n}," + ",".join(f"{v:.3f}" for v in values))

    print("\ntop_ops")
    for key in sorted(top_ops):
        bucket, rank, kind, step = key
        ops = "; ".join(
            f"{name}:{dur:.3f}ms"
            for name, dur in top_ops[key].most_common(args.top_n))
        print(f"bucket={bucket} rank={rank} kind={kind} step={step} {ops}")


if __name__ == "__main__":
    main()
