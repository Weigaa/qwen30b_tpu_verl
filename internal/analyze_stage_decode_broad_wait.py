#!/usr/bin/env python3
import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

MARKER_RE = re.compile(r"\bvllm_stage_decode_(?P<kind>\w+)\b")
FIELD_RE = re.compile(r"\b(?P<key>rank|mode|stage|layer)=(-?\d+)\b")
BUCKET_RE = re.compile(
    r"bucket_stage(?P<stage>\d+)_sample(?P<sample>\d+)_step(?P<step>\d+)")
RANK_RE = re.compile(r"rank_(?P<rank>\d+)_")


def load_trace(path: Path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8",
                       errors="ignore") as f:
            return json.load(f)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def iter_events(obj):
    if isinstance(obj, dict):
        evs = obj.get("traceEvents")
        if isinstance(evs, list):
            yield from (e for e in evs if isinstance(e, dict))
        elif "name" in obj:
            yield obj
    elif isinstance(obj, list):
        yield from (e for e in obj if isinstance(e, dict))


def ts_us(event):
    try:
        return float(event.get("ts", event.get("timestamp", 0)) or 0.0)
    except Exception:
        return 0.0


def dur_us(event):
    try:
        return float(event.get("dur", event.get("duration", 0)) or 0.0)
    except Exception:
        return 0.0


def pct(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def fmt_ms(values):
    return f"{pct(values, .5):.3f} / {pct(values, .9):.3f}"


def fmt_ratio(values):
    return f"{pct(values, .5) * 100:.1f}% / {pct(values, .9) * 100:.1f}%"


# Prefer runtime/device queue view. Fall back only when the trace lacks EP
# dispatch, e.g. single-rank stage can use init-routing/grouped-matmul instead
# of distributed dispatch.
START_OP_PATTERNS = [
    ("dispatch_dequeue",
     lambda n: n == "Dequeue@aclnnMoeDistributeDispatchV2"),
    ("dispatch_npu", lambda n: n == "npu::npu_moe_distribute_dispatch_v2"),
    ("dispatch_host", lambda n: n == "aclnnMoeDistributeDispatchV2"),
    ("init_routing_dequeue", lambda n: n == "Dequeue@aclnnMoeInitRoutingV3"),
    ("init_routing_npu", lambda n: n == "npu::npu_moe_init_routing_v2"),
    ("init_routing_host", lambda n: n == "aclnnMoeInitRoutingV3"),
    ("grouped_matmul_dequeue", lambda n: n == "Dequeue@aclnnGroupedMatmulV5"),
    ("grouped_matmul_npu", lambda n: n == "npu::npu_grouped_matmul"),
    ("grouped_matmul_host", lambda n: n == "aclnnGroupedMatmulV5"),
]
START_OP_PRIORITY = {name: idx for idx, (name, _pred) in enumerate(START_OP_PATTERNS)}


def main():
    parser = argparse.ArgumentParser(
        description="Summarize broad compute/wait split from stage decode traces.")
    parser.add_argument("label")
    parser.add_argument("profile_dir", type=Path)
    args = parser.parse_args()

    trace_files = sorted(args.profile_dir.rglob("trace_view.json"))
    markers = []
    ops_by_trace = []

    for path in trace_files:
        bm = BUCKET_RE.search(str(path))
        rm = RANK_RE.search(str(path))
        sample = int(bm.group("sample")) if bm else -1
        step = int(bm.group("step")) if bm else -1
        bucket_stage = int(bm.group("stage")) if bm else -1
        rank = int(rm.group("rank")) if rm else -1
        try:
            obj = load_trace(path)
        except Exception as exc:
            print(f"WARN load failed path={path} error={exc}")
            continue

        ops = []
        for event in iter_events(obj):
            name = str(event.get("name", ""))
            ts = ts_us(event)
            du = dur_us(event)
            if du <= 0:
                continue

            marker = MARKER_RE.search(name)
            if marker:
                fields = {
                    match.group("key"): int(match.group(2))
                    for match in FIELD_RE.finditer(name)
                }
                markers.append({
                    "kind": marker.group("kind"),
                    "sample": sample,
                    "step": step,
                    "rank": fields.get("rank", rank),
                    "stage": fields.get("stage", bucket_stage),
                    "layer": fields.get("layer", -1),
                    "ts": ts,
                    "end": ts + du,
                    "dur_ms": du / 1000.0,
                    "name": name,
                    "trace": str(path),
                })

            for label, pred in START_OP_PATTERNS:
                if pred(name):
                    ops.append((ts, du, label, name))
                    break

        ops.sort()
        ops_by_trace.append((sample, step, bucket_stage, rank, ops))

    by_key = defaultdict(list)
    for row in markers:
        key = (row["sample"], row["step"], row["rank"], row["stage"],
               row["layer"], row["kind"])
        by_key[key].append(row)
    for rows in by_key.values():
        rows.sort(key=lambda item: item["ts"])

    op_index = defaultdict(list)
    for sample, step, stage, rank, ops in ops_by_trace:
        op_index[(sample, step, rank, stage)].extend(ops)
    for rows in op_index.values():
        rows.sort()

    metrics = defaultdict(lambda: defaultdict(list))
    start_kind_counts = defaultdict(Counter)
    missing = Counter()
    examples = []

    for comm in [row for row in markers if row["kind"] == "comm_window"]:
        sample = comm["sample"]
        step = comm["step"]
        rank = comm["rank"]
        stage = comm["stage"]
        layer = comm["layer"]
        if sample <= 1 or layer <= 1:
            continue

        bind_rows = by_key.get(
            (sample, step, rank, stage, layer, "bind_wait"), [])
        ffn_rows = by_key.get(
            (sample, step, rank, stage, layer, "ffn_compute"), [])
        if not bind_rows or not ffn_rows:
            missing[(stage, "bind_or_ffn")] += 1
            continue

        bind = bind_rows[0]
        ffn = ffn_rows[0]
        candidates = []
        for ts, du, label, name in op_index.get((sample, step, rank, stage),
                                                []):
            if ffn["ts"] - 1 <= ts <= ffn["end"] + 1:
                candidates.append((START_OP_PRIORITY[label], ts, du, label,
                                   name))

        if not candidates:
            missing[(stage, "moe_start")] += 1
            continue

        candidates.sort(key=lambda item: (item[0], item[1]))
        _priority, start_ts, _start_dur, start_label, start_name = candidates[0]
        actual_wait_ms = (start_ts - bind["ts"]) / 1000.0
        comm_ms = comm["dur_ms"]
        broad_ms = comm_ms - actual_wait_ms

        metrics[stage]["comm_window"].append(comm_ms)
        metrics[stage]["actual_wait_to_moe_start"].append(actual_wait_ms)
        metrics[stage]["broad_compute_phase"].append(broad_ms)
        metrics[stage]["broad_over_comm"].append(
            broad_ms / comm_ms if comm_ms else 0.0)
        metrics[stage]["actual_wait_over_comm"].append(
            actual_wait_ms / comm_ms if comm_ms else 0.0)
        start_kind_counts[stage][start_label] += 1

        prev_ffn = by_key.get(
            (sample, step, rank, stage, layer - 1, "ffn_compute"), [])
        attn = by_key.get(
            (sample, step, rank, stage, layer, "attention_compute"), [])
        if prev_ffn and attn:
            op_compute = prev_ffn[-1]["dur_ms"] + attn[0]["dur_ms"]
            metrics[stage]["operator_compute_prev_ffn_plus_attention"].append(
                op_compute)

        if len(examples) < 8:
            examples.append((stage, sample, step, rank, layer, comm_ms,
                             actual_wait_ms, broad_ms, start_label,
                             start_name))

    print(f"label={args.label}")
    print(f"profile_dir={args.profile_dir}")
    print(f"trace_files={len(trace_files)} markers={len(markers)}")
    print("filter=sample>1,layer>1")
    print(
        "stage,n,comm_window p50/p90 ms,"
        "actual_wait_to_moe_start p50/p90 ms,"
        "broad_compute_phase p50/p90 ms,"
        "broad/comm p50/p90,"
        "actual_wait/comm p50/p90,"
        "operator_compute(prev_ffn+attention) p50/p90 ms")
    for stage in sorted(metrics, reverse=True):
        stage_metrics = metrics[stage]
        print(
            f"{stage},"
            f"{len(stage_metrics['comm_window'])},"
            f"{fmt_ms(stage_metrics['comm_window'])},"
            f"{fmt_ms(stage_metrics['actual_wait_to_moe_start'])},"
            f"{fmt_ms(stage_metrics['broad_compute_phase'])},"
            f"{fmt_ratio(stage_metrics['broad_over_comm'])},"
            f"{fmt_ratio(stage_metrics['actual_wait_over_comm'])},"
            f"{fmt_ms(stage_metrics['operator_compute_prev_ffn_plus_attention'])}"
        )

    print("start_op_counts")
    for stage in sorted(start_kind_counts, reverse=True):
        print(stage, dict(start_kind_counts[stage]))
    print("missing", dict(missing))
    print("examples")
    for example in examples:
        print(example)


if __name__ == "__main__":
    main()
