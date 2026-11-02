#!/usr/bin/env python3
"""Summarize stage decode MSTX ranges from torch_npu profiler traces."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


MARKER_RE = re.compile(r"\bvllm_stage_decode_(?P<kind>\w+)\b")
FIELD_RE = re.compile(r"\b(?P<key>rank|mode|stage|layer)=(-?\d+)\b")
PATH_RE = re.compile(r"\bpath=([^\s]+)")
BUCKET_RE = re.compile(
    r"bucket_stage(?P<stage>\d+)_sample(?P<sample>\d+)_step(?P<step>\d+)")


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


def event_duration_ms(ev: dict[str, Any]) -> float:
    dur = ev.get("dur")
    if dur is None:
        dur = ev.get("duration")
    try:
        return float(dur or 0.0) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def event_ts_us(ev: dict[str, Any]) -> float:
    ts = ev.get("ts")
    if ts is None:
        ts = ev.get("timestamp")
    try:
        return float(ts or 0.0)
    except (TypeError, ValueError):
        return 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def parse_marker(name: str) -> tuple[str, dict[str, int], str] | None:
    match = MARKER_RE.search(name)
    if not match:
        return None
    fields = {m.group("key"): int(m.group(2)) for m in FIELD_RE.finditer(name)}
    path_match = PATH_RE.search(name)
    path = path_match.group(1) if path_match else ""
    return match.group("kind"), fields, path


def _latest_profile_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    if not candidates:
        raise SystemExit(f"No profile directories found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def fmt_pair(values: list[float]) -> str:
    return f"{percentile(values, 0.50):.3f} / {percentile(values, 0.90):.3f}"


def paired_wait_gaps(
        grouped: dict[tuple[int, int, int, str], list[float]],
        *,
        stage: int | None = None,
        layer: int | None = None) -> list[float]:
    gaps: list[float] = []
    keys = {
        (s, r, l)
        for s, r, l, kind in grouped
        if kind in ("compute_window", "comm_window")
    }
    for s, rank, l in sorted(keys):
        if stage is not None and s != stage:
            continue
        if layer is not None and l != layer:
            continue
        compute = sorted(grouped.get((s, rank, l, "compute_window"), []))
        comm = sorted(grouped.get((s, rank, l, "comm_window"), []))
        for cpt, cmt in zip(compute, comm):
            gaps.append(max(0.0, cmt - cpt))
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profile_dir",
        nargs="?",
        type=Path,
        help=(
            "Profiler directory. Defaults to latest under "
            "$WJ_RECORDS_DIR/op_profiles."),
    )
    args = parser.parse_args()

    if args.profile_dir is None:
        records_dir = Path(os.getenv("WJ_RECORDS_DIR", "/home/sharedata/wj_records"))
        args.profile_dir = _latest_profile_dir(records_dir / "op_profiles")

    rows: list[dict[str, Any]] = []
    files = list(iter_json_files(args.profile_dir))
    parsed_files = 0
    for path in files:
        bucket_match = BUCKET_RE.search(str(path))
        bucket_sample = int(bucket_match.group("sample")) if bucket_match else -1
        bucket_step = int(bucket_match.group("step")) if bucket_match else -1
        try:
            obj = load_json(path)
        except Exception:
            continue
        parsed_files += 1
        for ev in iter_events(obj):
            name = str(ev.get("name", ""))
            parsed = parse_marker(name)
            if parsed is None:
                continue
            dur_ms = event_duration_ms(ev)
            if dur_ms <= 0:
                continue
            kind, fields, marker_path = parsed
            ts_us = event_ts_us(ev)
            rows.append({
                "kind": kind,
                "rank": fields.get("rank", -1),
                "mode": fields.get("mode", -1),
                "stage": fields.get("stage", -1),
                "layer": fields.get("layer", -1),
                "sample": bucket_sample,
                "step": bucket_step,
                "path": marker_path,
                "dur_ms": dur_ms,
                "ts_us": ts_us,
                "end_us": ts_us + dur_ms * 1000.0,
            })

    print(f"profile_dir={args.profile_dir}")
    print(f"json_files={len(files)} parsed_json_files={parsed_files} markers={len(rows)}")
    if not rows:
        print("No vllm_stage_decode_* MSTX markers found.")
        return

    by_stage_kind: dict[tuple[int, str], list[float]] = defaultdict(list)
    by_stage_layer_kind: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    by_stage_rank_kind: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    by_stage_rank_layer_kind: dict[tuple[int, int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        stage = int(row["stage"])
        layer = int(row["layer"])
        rank = int(row["rank"])
        kind = str(row["kind"])
        dur_ms = float(row["dur_ms"])
        by_stage_kind[(stage, kind)].append(dur_ms)
        by_stage_layer_kind[(stage, layer, kind)].append(dur_ms)
        by_stage_rank_kind[(stage, rank, kind)].append(dur_ms)
        by_stage_rank_layer_kind[(stage, rank, layer, kind)].append(dur_ms)

    stages = sorted({stage for stage, _kind in by_stage_kind if stage >= 0},
                    reverse=True)
    print("\nstage_summary_ms")
    print("stage,compute_window p50/p90,comm_window p50/p90,wait_gap p50/p90,attention_compute p50/p90,ffn_compute p50/p90,bind_wait p50/p90,prefetch_submit p50/p90")
    for stage in stages:
        compute = by_stage_kind.get((stage, "compute_window"), [])
        comm = by_stage_kind.get((stage, "comm_window"), [])
        wait_gap = paired_wait_gaps(by_stage_rank_layer_kind, stage=stage)
        attention = by_stage_kind.get((stage, "attention_compute"), [])
        ffn = by_stage_kind.get((stage, "ffn_compute"), [])
        bind = by_stage_kind.get((stage, "bind_wait"), [])
        submit = by_stage_kind.get((stage, "prefetch_submit"), [])
        print(
            f"{stage},{fmt_pair(compute)},{fmt_pair(comm)},"
            f"{fmt_pair(wait_gap)},{fmt_pair(attention)},"
            f"{fmt_pair(ffn)},"
            f"{fmt_pair(bind)},{fmt_pair(submit)}")

    print("\nstage_rank_counts")
    print("stage,rank,compute_window_n,comm_window_n,attention_compute_n,ffn_compute_n,bind_wait_n,prefetch_submit_n")
    for stage in stages:
        ranks = sorted({
            rank
            for s, rank, _kind in by_stage_rank_kind
            if s == stage
        })
        for rank in ranks:
            print(
                f"{stage},{rank},"
                f"{len(by_stage_rank_kind.get((stage, rank, 'compute_window'), []))},"
                f"{len(by_stage_rank_kind.get((stage, rank, 'comm_window'), []))},"
                f"{len(by_stage_rank_kind.get((stage, rank, 'attention_compute'), []))},"
                f"{len(by_stage_rank_kind.get((stage, rank, 'ffn_compute'), []))},"
                f"{len(by_stage_rank_kind.get((stage, rank, 'bind_wait'), []))},"
                f"{len(by_stage_rank_kind.get((stage, rank, 'prefetch_submit'), []))}")

    print("\nwindow_decomposition_ms")
    print("filter=sample>1,layer>1")
    print("stage,overlap_window p50/p90,attention_compute p50/p90,prev_ffn_compute p50/p90,effective_compute p50/p90,prefetch_submit p50/p90,pre_ffn_gap p50/p90,post_prev_ffn_to_bind p50/p90,through_prev_ffn p50/p90,n")
    by_instance_kind: dict[tuple[int, int, int, int, int, str],
                           list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if float(row.get("ts_us", 0.0)) <= 0:
            continue
        key = (
            int(row["sample"]),
            int(row["step"]),
            int(row["rank"]),
            int(row["stage"]),
            int(row["layer"]),
            str(row["kind"]),
        )
        by_instance_kind[key].append(row)
    for key in by_instance_kind:
        by_instance_kind[key].sort(key=lambda row: float(row["ts_us"]))

    decomp: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["kind"] != "compute_window":
            continue
        sample = int(row["sample"])
        layer = int(row["layer"])
        if sample <= 1 or layer <= 1:
            continue
        step = int(row["step"])
        rank = int(row["rank"])
        stage = int(row["stage"])
        bind_rows = by_instance_kind.get(
            (sample, step, rank, stage, layer, "bind_wait"), [])
        prev_ffn_rows = by_instance_kind.get(
            (sample, step, rank, stage, layer - 1, "ffn_compute"), [])
        attention_rows = by_instance_kind.get(
            (sample, step, rank, stage, layer, "attention_compute"), [])
        submit_rows = by_instance_kind.get(
            (sample, step, rank, stage, layer, "prefetch_submit"), [])
        if not bind_rows or not prev_ffn_rows:
            continue
        bind = bind_rows[0]
        prev_ffn = prev_ffn_rows[-1]
        submit = submit_rows[0] if submit_rows else None
        window_start = float(row["ts_us"])
        prev_ffn_start = float(prev_ffn["ts_us"])
        prev_ffn_end = float(prev_ffn["end_us"])
        bind_start = float(bind["ts_us"])
        # Prefer the explicit attention marker.  Older traces do not have it, so
        # keep the previous gap-based approximation as a fallback.
        if attention_rows:
            attention_ms = float(attention_rows[0]["dur_ms"])
        else:
            attention_ms = max(0.0, (bind_start - prev_ffn_end) / 1000.0)
        effective_compute_ms = float(prev_ffn["dur_ms"]) + attention_ms
        decomp[(stage, "overlap_window")].append(float(row["dur_ms"]))
        decomp[(stage, "attention_compute")].append(attention_ms)
        decomp[(stage, "prev_ffn_compute")].append(float(prev_ffn["dur_ms"]))
        decomp[(stage, "effective_compute")].append(effective_compute_ms)
        decomp[(stage, "prefetch_submit")].append(
            float(submit["dur_ms"]) if submit is not None else 0.0)
        decomp[(stage, "pre_ffn_gap")].append(
            max(0.0, (prev_ffn_start - window_start) / 1000.0))
        decomp[(stage, "post_prev_ffn_to_bind")].append(
            max(0.0, (bind_start - prev_ffn_end) / 1000.0))
        decomp[(stage, "through_prev_ffn")].append(
            max(0.0, (prev_ffn_end - window_start) / 1000.0))

    for stage in stages:
        overlap = decomp.get((stage, "overlap_window"), [])
        attention = decomp.get((stage, "attention_compute"), [])
        prev_ffn = decomp.get((stage, "prev_ffn_compute"), [])
        effective_compute = decomp.get((stage, "effective_compute"), [])
        submit = decomp.get((stage, "prefetch_submit"), [])
        pre_ffn_gap = decomp.get((stage, "pre_ffn_gap"), [])
        post_prev_ffn = decomp.get((stage, "post_prev_ffn_to_bind"), [])
        through_prev_ffn = decomp.get((stage, "through_prev_ffn"), [])
        print(
            f"{stage},{fmt_pair(overlap)},{fmt_pair(attention)},"
            f"{fmt_pair(prev_ffn)},{fmt_pair(effective_compute)},"
            f"{fmt_pair(submit)},{fmt_pair(pre_ffn_gap)},"
            f"{fmt_pair(post_prev_ffn)},{fmt_pair(through_prev_ffn)},"
            f"{len(overlap)}")

    print("\npaired_effective_overlap_ms")
    print("filter=sample>1,layer>1")
    print("stage,n,comm_window p50/p90,effective_compute p50/p90,attention_compute p50/p90,prev_ffn_compute p50/p90,comm_minus_effective_compute p50/p90,effective_compute_over_comm p50/p90,compute_window_gap p50/p90,prefetch_submit p50/p90")
    paired: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["kind"] != "comm_window":
            continue
        sample = int(row["sample"])
        layer = int(row["layer"])
        if sample <= 1 or layer <= 1:
            continue
        step = int(row["step"])
        rank = int(row["rank"])
        stage = int(row["stage"])
        attention_rows = by_instance_kind.get(
            (sample, step, rank, stage, layer, "attention_compute"), [])
        prev_ffn_rows = by_instance_kind.get(
            (sample, step, rank, stage, layer - 1, "ffn_compute"), [])
        compute_rows = by_instance_kind.get(
            (sample, step, rank, stage, layer, "compute_window"), [])
        submit_rows = by_instance_kind.get(
            (sample, step, rank, stage, layer, "prefetch_submit"), [])
        if not attention_rows or not prev_ffn_rows or not compute_rows:
            continue
        comm_ms = float(row["dur_ms"])
        attention_ms = float(attention_rows[0]["dur_ms"])
        prev_ffn_ms = float(prev_ffn_rows[-1]["dur_ms"])
        effective_compute_ms = attention_ms + prev_ffn_ms
        compute_window_ms = float(compute_rows[0]["dur_ms"])
        submit_ms = float(submit_rows[0]["dur_ms"]) if submit_rows else 0.0
        paired[(stage, "comm_window")].append(comm_ms)
        paired[(stage, "effective_compute")].append(effective_compute_ms)
        paired[(stage, "attention_compute")].append(attention_ms)
        paired[(stage, "prev_ffn_compute")].append(prev_ffn_ms)
        paired[(stage, "comm_minus_effective_compute")].append(
            comm_ms - effective_compute_ms)
        paired[(stage, "effective_compute_over_comm")].append(
            effective_compute_ms / comm_ms if comm_ms > 0 else 0.0)
        paired[(stage, "compute_window_gap")].append(
            comm_ms - compute_window_ms)
        paired[(stage, "prefetch_submit")].append(submit_ms)

    def fmt_percent_pair(values: list[float]) -> str:
        return (
            f"{percentile(values, 0.50) * 100.0:.1f}% / "
            f"{percentile(values, 0.90) * 100.0:.1f}%")

    for stage in stages:
        comm = paired.get((stage, "comm_window"), [])
        effective = paired.get((stage, "effective_compute"), [])
        attention = paired.get((stage, "attention_compute"), [])
        prev_ffn = paired.get((stage, "prev_ffn_compute"), [])
        residual = paired.get((stage, "comm_minus_effective_compute"), [])
        ratio = paired.get((stage, "effective_compute_over_comm"), [])
        compute_gap = paired.get((stage, "compute_window_gap"), [])
        submit = paired.get((stage, "prefetch_submit"), [])
        print(
            f"{stage},{len(comm)},{fmt_pair(comm)},"
            f"{fmt_pair(effective)},{fmt_pair(attention)},"
            f"{fmt_pair(prev_ffn)},{fmt_pair(residual)},"
            f"{fmt_percent_pair(ratio)},{fmt_pair(compute_gap)},"
            f"{fmt_pair(submit)}")

    print("\nstage_layer_summary_ms")
    print("stage,layer,compute_window p50/p90,comm_window p50/p90,wait_gap p50/p90,ffn_compute p50/p90,bind_wait p50/p90")
    for stage, layer in sorted({(s, l) for s, l, _k in by_stage_layer_kind},
                               reverse=True):
        compute = by_stage_layer_kind.get((stage, layer, "compute_window"), [])
        comm = by_stage_layer_kind.get((stage, layer, "comm_window"), [])
        wait_gap = paired_wait_gaps(by_stage_rank_layer_kind,
                                    stage=stage,
                                    layer=layer)
        ffn = by_stage_layer_kind.get((stage, layer, "ffn_compute"), [])
        bind = by_stage_layer_kind.get((stage, layer, "bind_wait"), [])
        print(
            f"{stage},{layer},{fmt_pair(compute)},"
            f"{fmt_pair(comm)},{fmt_pair(wait_gap)},"
            f"{fmt_pair(ffn)},{fmt_pair(bind)}")


if __name__ == "__main__":
    main()
