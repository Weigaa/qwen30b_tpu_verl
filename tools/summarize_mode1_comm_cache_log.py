#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


LINE_RE = re.compile(r"Mode1 comm cache state: (?P<body>.*)")
KV_CAP_RE = re.compile(r"Capping mode1 parity KV blocks to mode1 budget: (?P<body>.*)")
KV_SIZE_RE = re.compile(r"GPU KV cache size: (?P<tokens>[0-9,]+) tokens")
RESIZE_START_RE = re.compile(r"Mode1 adaptive KV resize phase=start (?P<body>.*)")
RELEASE_RE = re.compile(r"Mode1 MoE comm runtime transient release: (?P<body>.*)")
FIELD_KEY_RE = re.compile(r"(?<!\S)(\w+)=")


@dataclass
class Row:
    tag: str
    occurrence: int
    rank: int
    cached_groups: str
    registry_groups: str
    topology_count: int
    topology_methods: int
    topology_groups: str
    free_gib: float
    non_torch_gib: float


@dataclass
class KvCapRow:
    floor: str
    cap_source: str
    max_tokens: int
    capped_tokens: int
    headroom_tokens: int


@dataclass
class ResizeRow:
    target_floor: str
    target_tokens: int
    effective_target_tokens: int
    headroom_tokens: int


@dataclass
class KvSizeRow:
    tokens: int


@dataclass
class ReleaseRow:
    reason: str
    methods: int
    topologies: int
    method_attrs: int
    dispatcher_attrs: int
    prepare_attrs: int
    tensors: int
    tensor_bytes: int


def _parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    matches = list(FIELD_KEY_RE.finditer(body))
    for index, match in enumerate(matches):
        key = match.group(1)
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        fields[key] = body[value_start:value_end].strip()
    return fields


def _bytes_to_gib(raw: str) -> float:
    try:
        return float(raw) / (1024.0**3)
    except (TypeError, ValueError):
        return 0.0


def _int_field(fields: dict[str, str], name: str, default: int = 0) -> int:
    try:
        return int(float(fields.get(name, str(default)).replace(",", "")))
    except (AttributeError, TypeError, ValueError):
        return default


def parse_log(path: Path) -> tuple[list[Row], list[KvCapRow], list[ResizeRow],
                                  list[KvSizeRow], list[ReleaseRow]]:
    rows: list[Row] = []
    kv_caps: list[KvCapRow] = []
    resize_rows: list[ResizeRow] = []
    kv_sizes: list[KvSizeRow] = []
    release_rows: list[ReleaseRow] = []
    for line in path.read_text(errors="replace").splitlines():
        match = LINE_RE.search(line)
        if match:
            fields = _parse_fields(match.group("body"))
            try:
                rank = int(fields.get("rank", "-1"))
            except ValueError:
                rank = -1
            if rank >= 0:
                rows.append(
                    Row(
                        tag=fields.get("tag", ""),
                        occurrence=0,
                        rank=rank,
                        cached_groups=fields.get("cached_groups", ""),
                        registry_groups=fields.get("registry_groups", ""),
                        topology_count=_int_field(fields, "topology_count"),
                        topology_methods=_int_field(fields, "topology_methods"),
                        topology_groups=fields.get("topology_groups", ""),
                        free_gib=_bytes_to_gib(fields.get("free_bytes", "0")),
                        non_torch_gib=_bytes_to_gib(
                            fields.get("non_torch", "0")),
                    ))
            continue

        match = KV_CAP_RE.search(line)
        if match:
            fields = _parse_fields(match.group("body"))
            kv_caps.append(
                KvCapRow(
                    floor=fields.get("effective_floor",
                                     fields.get("configured_floor", "")),
                    cap_source=fields.get("cap_source", ""),
                    max_tokens=_int_field(fields, "max_tokens"),
                    capped_tokens=_int_field(fields, "capped_tokens"),
                    headroom_tokens=_int_field(
                        fields, "planned_floor_headroom_tokens"),
                ))
            continue

        match = RESIZE_START_RE.search(line)
        if match:
            fields = _parse_fields(match.group("body"))
            resize_rows.append(
                ResizeRow(
                    target_floor=fields.get("target_floor", ""),
                    target_tokens=_int_field(fields, "target_tokens"),
                    effective_target_tokens=_int_field(
                        fields, "effective_target_tokens"),
                    headroom_tokens=_int_field(
                        fields, "planned_floor_headroom_tokens"),
                ))
            continue

        match = KV_SIZE_RE.search(line)
        if match:
            kv_sizes.append(
                KvSizeRow(tokens=int(match.group("tokens").replace(",", ""))))
            continue

        match = RELEASE_RE.search(line)
        if match:
            fields = _parse_fields(match.group("body"))
            release_rows.append(
                ReleaseRow(
                    reason=fields.get("reason", ""),
                    methods=_int_field(fields, "methods"),
                    topologies=_int_field(fields, "topologies"),
                    method_attrs=_int_field(fields, "method_attrs"),
                    dispatcher_attrs=_int_field(fields, "dispatcher_attrs"),
                    prepare_attrs=_int_field(fields, "prepare_attrs"),
                    tensors=_int_field(fields, "tensors"),
                    tensor_bytes=_int_field(fields, "tensor_bytes"),
                ))
    return rows, kv_caps, resize_rows, kv_sizes, release_rows


def assign_occurrences(rows: list[Row]) -> list[Row]:
    occurrence_by_tag: dict[str, int] = defaultdict(int)
    ranks_seen_by_tag: dict[str, set[int]] = defaultdict(set)
    assigned: list[Row] = []
    for row in rows:
        seen = ranks_seen_by_tag[row.tag]
        if row.rank in seen:
            occurrence_by_tag[row.tag] += 1
            seen.clear()
        occurrence = occurrence_by_tag[row.tag]
        seen.add(row.rank)
        assigned.append(
            Row(
                tag=row.tag,
                occurrence=occurrence,
                rank=row.rank,
                cached_groups=row.cached_groups,
                registry_groups=row.registry_groups,
                topology_count=row.topology_count,
                topology_methods=row.topology_methods,
                topology_groups=row.topology_groups,
                free_gib=row.free_gib,
                non_torch_gib=row.non_torch_gib,
            ))
    return assigned


def _rank_band(rank: int) -> str:
    if rank < 8:
        return "r00-07"
    if rank < 12:
        return "r08-11"
    return "r12-15"


def _short_groups(groups: str) -> str:
    if not groups:
        return "-"
    groups = groups.replace(" ", "")
    groups = groups.replace("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", "0..15")
    groups = groups.replace("8,9,10,11,12,13,14,15", "8..15")
    groups = groups.replace("12,13,14,15", "12..15")
    groups = groups.replace("[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]", "[0..15]")
    groups = groups.replace("[8,9,10,11,12,13,14,15]", "[8..15]")
    groups = groups.replace("[12,13,14,15]", "[12..15]")
    return groups


def print_summary(rows: list[Row], kv_caps: list[KvCapRow],
                  resize_rows: list[ResizeRow],
                  kv_sizes: list[KvSizeRow],
                  release_rows: list[ReleaseRow]) -> None:
    rows = assign_occurrences(rows)
    if not rows:
        print("No 'Mode1 comm cache state' rows found.")
    else:
        _print_comm_cache_summary(rows)
    _print_kv_summary(kv_caps, resize_rows, kv_sizes)
    _print_release_summary(release_rows)


def _print_comm_cache_summary(rows: list[Row]) -> None:
    by_tag_band: dict[tuple[str, int, str], list[Row]] = defaultdict(list)
    by_tag: dict[tuple[str, int], list[Row]] = defaultdict(list)
    tag_order: list[tuple[str, int]] = []
    for row in rows:
        tag_key = (row.tag, row.occurrence)
        if tag_key not in tag_order:
            tag_order.append(tag_key)
        by_tag[tag_key].append(row)
        by_tag_band[(row.tag, row.occurrence, _rank_band(row.rank))].append(row)

    baseline_total_by_band: dict[str, float] = {}
    for band in ("all", "r00-07", "r08-11", "r12-15"):
        if band == "all":
            items = by_tag.get(("rollout_step_start", 0), [])
        else:
            items = by_tag_band.get(("rollout_step_start", 0, band), [])
        if items:
            baseline_total_by_band[band] = sum(
                item.non_torch_gib for item in items)

    print("# comm_cache_state")
    print(
        "tag\tocc\tband\tranks\ttopology_count\ttopology_methods\tfree_gib\t"
        "non_torch_gib\tfree_gib_total\tnon_torch_gib_total\t"
        "non_torch_delta_from_step1_start\tcached_groups\ttopology_groups")
    for tag, occurrence in tag_order:
        all_items = by_tag[(tag, occurrence)]
        all_ranks = ",".join(str(item.rank) for item in sorted(all_items, key=lambda item: item.rank))
        all_topology_counts = sorted({item.topology_count for item in all_items})
        all_topology_methods = sorted({item.topology_methods for item in all_items})
        all_free_vals = [item.free_gib for item in all_items]
        all_non_torch_vals = [item.non_torch_gib for item in all_items]
        all_non_torch_total = sum(all_non_torch_vals)
        all_delta = all_non_torch_total - baseline_total_by_band.get(
            "all", all_non_torch_total)
        print(
            f"{tag}\t{occurrence + 1}\tall\t{all_ranks}\t"
            f"{all_topology_counts}\t{all_topology_methods}\t"
            f"{min(all_free_vals):.2f}-{max(all_free_vals):.2f}\t"
            f"{min(all_non_torch_vals):.2f}-{max(all_non_torch_vals):.2f}\t"
            f"{sum(all_free_vals):.2f}\t{all_non_torch_total:.2f}\t"
            f"{all_delta:.2f}\t-\t-")
        for band in ("r00-07", "r08-11", "r12-15"):
            items = by_tag_band.get((tag, occurrence, band), [])
            if not items:
                continue
            ranks = ",".join(str(item.rank) for item in sorted(items, key=lambda item: item.rank))
            topology_counts = sorted({item.topology_count for item in items})
            topology_methods = sorted({item.topology_methods for item in items})
            free_vals = [item.free_gib for item in items]
            non_torch_vals = [item.non_torch_gib for item in items]
            non_torch_total = sum(non_torch_vals)
            delta = non_torch_total - baseline_total_by_band.get(
                band, non_torch_total)
            cached = _short_groups(sorted(items, key=lambda item: item.rank)[0].cached_groups)
            topology = _short_groups(sorted(items, key=lambda item: item.rank)[0].topology_groups)
            print(
                f"{tag}\t{occurrence + 1}\t{band}\t{ranks}\t"
                f"{topology_counts}\t{topology_methods}\t"
                f"{min(free_vals):.2f}-{max(free_vals):.2f}\t"
                f"{min(non_torch_vals):.2f}-{max(non_torch_vals):.2f}\t"
                f"{sum(free_vals):.2f}\t{non_torch_total:.2f}\t"
                f"{delta:.2f}\t"
                f"{cached}\t{topology}")


def _print_kv_summary(kv_caps: list[KvCapRow], resize_rows: list[ResizeRow],
                      kv_sizes: list[KvSizeRow]) -> None:
    print("# kv_capacity")
    print("kind\tfloor\teffective_tokens\theadroom_tokens\tcap_source\tcount")
    cap_counts: dict[tuple[str, int, int, str], int] = defaultdict(int)
    for row in kv_caps:
        cap_counts[(row.floor, row.capped_tokens, row.headroom_tokens,
                    row.cap_source)] += 1
    for (floor, tokens, headroom, cap_source), count in sorted(
            cap_counts.items(), key=lambda item:
            (item[0][0], item[0][1], item[0][2], item[0][3])):
        print(
            f"cap\t{floor}\t{tokens}\t{headroom}\t{cap_source}\t{count}")

    resize_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    for row in resize_rows:
        resize_counts[(row.target_floor, row.effective_target_tokens,
                       row.headroom_tokens)] += 1
    for (floor, tokens, headroom), count in sorted(
            resize_counts.items(), key=lambda item:
            (item[0][0], item[0][1], item[0][2])):
        print(f"resize\t{floor}\t{tokens}\t{headroom}\t-\t{count}")

    size_counts: dict[int, int] = defaultdict(int)
    for row in kv_sizes:
        size_counts[row.tokens] += 1
    for tokens, count in sorted(size_counts.items()):
        print(f"gpu_size\t-\t{tokens}\t-\t-\t{count}")


def _print_release_summary(release_rows: list[ReleaseRow]) -> None:
    print("# runtime_transient_release")
    print(
        "reason\tcount\tmethods_range\ttopologies_range\tmethod_attrs\t"
        "dispatcher_attrs\tprepare_attrs\ttensors\ttensor_gib")
    if not release_rows:
        return
    by_reason: dict[str, list[ReleaseRow]] = defaultdict(list)
    for row in release_rows:
        by_reason[row.reason].append(row)
    for reason, items in by_reason.items():
        methods = [item.methods for item in items]
        topologies = [item.topologies for item in items]
        tensor_gib = sum(item.tensor_bytes for item in items) / (1024.0**3)
        print(
            f"{reason}\t{len(items)}\t{min(methods)}-{max(methods)}\t"
            f"{min(topologies)}-{max(topologies)}\t"
            f"{sum(item.method_attrs for item in items)}\t"
            f"{sum(item.dispatcher_attrs for item in items)}\t"
            f"{sum(item.prepare_attrs for item in items)}\t"
            f"{sum(item.tensors for item in items)}\t{tensor_gib:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Mode1 comm-cache state rows from a rollout log.")
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print_summary(*parse_log(args.log))


if __name__ == "__main__":
    main()
