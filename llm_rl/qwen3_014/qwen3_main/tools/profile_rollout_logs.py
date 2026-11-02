#!/usr/bin/env python3
"""Summarize rollout performance logs.

This script intentionally parses only text logs.  It is useful for comparing
old/new runs without changing the training or rollout code paths.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

STEP_RE = re.compile(r"\bstep:(?P<step>\d+)\b(?P<body>.*)")
ROLLOUT_TIME_RE = re.compile(r"=> rollout_output_time_s:\s*(?P<value>" + FLOAT_RE + r")")
SPEED_RE = re.compile(r"=> rollouts speed tokens/s:\s*(?P<value>" + FLOAT_RE + r")")
GLOBAL_TOKENS_RE = re.compile(r"=> global_token_num:\s*(?P<value>\[.*\])")
KV_RE = re.compile(r"GPU KV cache size:\s*(?P<value>[\d,]+)\s+tokens")
CONCURRENCY_RE = re.compile(
    r"Maximum concurrency for\s*(?P<tokens>[\d,]+)\s*tokens per request:\s*(?P<value>" + FLOAT_RE + r")x"
)

PROFILE_SUMMARY_RE = re.compile(
    r"Profile execute duration summary \[(?P<phase>[^\]]+)\]\s+"
    r"steps=(?P<steps>\d+)\s+avg_reqs=(?P<avg_reqs>" + FLOAT_RE + r")\s+(?P<body>.*)"
)
PROFILE_RAW_RE = re.compile(
    r"Profile execute duration \[(?P<phase>[^\]]+)\]:(?P<body>.*)"
)
PROFILE_FIELD_RE = re.compile(
    r"\[(?P<name>[^\]]+)\]:(?:avg=)?(?P<value>" + FLOAT_RE + r")ms"
)

MOE_STAGE_RE = re.compile(
    r"MoE stage timing .*?call=(?P<call>\d+)\s+comm=(?P<comm>\S+)\s+"
    r"tokens=(?P<tokens>\d+)\s+dispatch_ms=(?P<dispatch>" + FLOAT_RE + r")\s+"
    r"mlp_ms=(?P<mlp>" + FLOAT_RE + r")\s+combine_ms=(?P<combine>" + FLOAT_RE + r")\s+"
    r"total_ms=(?P<total>" + FLOAT_RE + r")"
)
FUSED_MOE_PROFILE_RE = re.compile(
    r"FusedMoE profile .*?call=(?P<call>\d+)\s+kind=(?P<kind>\S+)\s+"
    r"comm=(?P<comm>\S+)\s+tokens=(?P<tokens>\d+)\s+(?P<body>.*?)(?:\s+summary=|$)"
)
QWEN_LAYER_RE = re.compile(
    r"Qwen[23]Moe layer profile .*?name=(?P<name>\S+)\s+call=(?P<call>\d+)\s+"
    r"prefix=(?P<prefix>\S+)\s+tokens=(?P<tokens>\d+)\s+(?P<body>.*)"
)
QWEN_LAYER_FIELD_RE = re.compile(r"(?P<name>\S+)_ms=(?P<value>" + FLOAT_RE + r")")
FUSED_MOE_PROFILE_FIELD_RE = QWEN_LAYER_FIELD_RE
FORWARD_KIND_RE = re.compile(
    r"forward_kind_debug kind=(?P<kind>\S+)\s+call=(?P<call>\d+)\s+"
    r"rank=(?P<rank>\S+)\s+dp_rank=(?P<dp_rank>\S+)\s+tokens=(?P<tokens>\d+)"
    r"(?:\s+attn_state=(?P<attn_state>\S+))?\s+elapsed_ms=(?P<elapsed>"
    + FLOAT_RE
    + r")"
)
ATTENTION_STAGE_RE = re.compile(
    r"Attention stage timing .*?call=(?P<call>\d+)\s+state=(?P<state>\S+)\s+"
    r"op=(?P<op>\S+)\s+tokens=(?P<tokens>\d+)\s+op_ms=(?P<op_ms>"
    + FLOAT_RE
    + r")"
)
DECODE_LEN_PROFILE_PART_RE = re.compile(
    r"bucket=(?P<bucket>[^,|]+),n=(?P<n>\d+),avg_reqs=(?P<avg_reqs>"
    + FLOAT_RE
    + r"),avg_max_seq=(?P<avg_max_seq>"
    + FLOAT_RE
    + r"),avg_seq=(?P<avg_seq>"
    + FLOAT_RE
    + r")(?P<body>[^|]*)"
)
DECODE_LEN_PROFILE_FIELD_RE = re.compile(
    r",(?P<name>forward|Sample|prepare_input|post_process)=(?P<value>"
    + FLOAT_RE
    + r")ms"
)


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def fmean(values: list[float]) -> float | None:
    return mean(values) if values else None


def fp50(values: list[float]) -> float | None:
    return median(values) if values else None


def fp95(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = math.ceil(0.95 * len(values)) - 1
    return values[max(0, min(idx, len(values) - 1))]


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def token_bucket(num_tokens: int) -> str:
    if num_tokens <= 0:
        return "0"
    if num_tokens <= 32:
        return "<=32"
    if num_tokens <= 64:
        return "33-64"
    if num_tokens <= 128:
        return "65-128"
    if num_tokens <= 256:
        return "129-256"
    if num_tokens <= 512:
        return "257-512"
    if num_tokens <= 1024:
        return "513-1024"
    return ">1024"


def parse_metric_body(body: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for part in body.split(" - "):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip().split()[0]
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def parse_profile_fields(body: str) -> dict[str, float]:
    return {m.group("name"): float(m.group("value")) for m in PROFILE_FIELD_RE.finditer(body)}


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "avg": fmean(values),
        "p50": fp50(values),
        "p95": fp95(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def parse_log(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "steps": [],
        "rollout_output_time_s": [],
        "rollouts_speed_tokens_s": [],
        "global_token_num": [],
        "kv_cache_tokens": [],
        "max_concurrency": [],
        "profile_summary": defaultdict(lambda: defaultdict(list)),
        "profile_raw": defaultdict(lambda: defaultdict(list)),
        "moe_stage": defaultdict(lambda: defaultdict(list)),
        "moe_stage_steady": defaultdict(lambda: defaultdict(list)),
        "fused_moe_profile": defaultdict(lambda: defaultdict(list)),
        "fused_moe_profile_steady": defaultdict(lambda: defaultdict(list)),
        "qwen_layer": defaultdict(lambda: defaultdict(list)),
        "qwen_layer_steady": defaultdict(lambda: defaultdict(list)),
        "forward_kind": defaultdict(lambda: defaultdict(list)),
        "attention_stage": defaultdict(lambda: defaultdict(list)),
        "attention_stage_steady": defaultdict(lambda: defaultdict(list)),
        "decode_len_profile": defaultdict(lambda: defaultdict(list)),
    }

    with path.open("r", errors="replace") as f:
        for raw in f:
            line = strip_ansi(raw).strip()

            if match := ROLLOUT_TIME_RE.search(line):
                result["rollout_output_time_s"].append(float(match.group("value")))

            if match := SPEED_RE.search(line):
                result["rollouts_speed_tokens_s"].append(float(match.group("value")))

            if match := KV_RE.search(line):
                result["kv_cache_tokens"].append(int(match.group("value").replace(",", "")))

            if match := CONCURRENCY_RE.search(line):
                result["max_concurrency"].append(float(match.group("value")))

            if match := GLOBAL_TOKENS_RE.search(line):
                try:
                    values = ast.literal_eval(match.group("value"))
                except (SyntaxError, ValueError):
                    values = []
                if isinstance(values, list):
                    result["global_token_num"].append([int(v) for v in values])

            if match := STEP_RE.search(line):
                metrics = parse_metric_body(match.group("body"))
                if metrics:
                    metrics["step"] = float(match.group("step"))
                    result["steps"].append(metrics)

            if match := PROFILE_SUMMARY_RE.search(line):
                phase = match.group("phase")
                fields = parse_profile_fields(match.group("body"))
                for name, value in fields.items():
                    result["profile_summary"][phase][name].append(value)
                result["profile_summary"][phase]["steps"].append(float(match.group("steps")))
                result["profile_summary"][phase]["avg_reqs"].append(float(match.group("avg_reqs")))

            if match := PROFILE_RAW_RE.search(line):
                phase = match.group("phase")
                fields = parse_profile_fields(match.group("body"))
                for name, value in fields.items():
                    result["profile_raw"][phase][name].append(value)

            if match := MOE_STAGE_RE.search(line):
                key = f"{match.group('comm')}/tokens={match.group('tokens')}"
                call = float(match.group("call"))
                result["moe_stage"][key]["dispatch_ms"].append(float(match.group("dispatch")))
                result["moe_stage"][key]["mlp_ms"].append(float(match.group("mlp")))
                result["moe_stage"][key]["combine_ms"].append(float(match.group("combine")))
                result["moe_stage"][key]["total_ms"].append(float(match.group("total")))
                result["moe_stage"][key]["call"].append(call)
                if call > 1:
                    result["moe_stage_steady"][key]["dispatch_ms"].append(float(match.group("dispatch")))
                    result["moe_stage_steady"][key]["mlp_ms"].append(float(match.group("mlp")))
                    result["moe_stage_steady"][key]["combine_ms"].append(float(match.group("combine")))
                    result["moe_stage_steady"][key]["total_ms"].append(float(match.group("total")))
                    result["moe_stage_steady"][key]["call"].append(call)

            if match := FUSED_MOE_PROFILE_RE.search(line):
                key = (
                    f"{match.group('kind')}/{match.group('comm')}/"
                    f"tokens={match.group('tokens')}"
                )
                call = float(match.group("call"))
                result["fused_moe_profile"][key]["call"].append(call)
                if call > 1:
                    result["fused_moe_profile_steady"][key]["call"].append(call)
                for field in FUSED_MOE_PROFILE_FIELD_RE.finditer(match.group("body")):
                    name = f"{field.group('name')}_ms"
                    value = float(field.group("value"))
                    result["fused_moe_profile"][key][name].append(value)
                    if call > 1:
                        result["fused_moe_profile_steady"][key][name].append(value)

            if match := QWEN_LAYER_RE.search(line):
                key = f"{match.group('name')}/tokens={match.group('tokens')}"
                call = float(match.group("call"))
                result["qwen_layer"][key]["call"].append(call)
                if call > 1:
                    result["qwen_layer_steady"][key]["call"].append(call)
                for field in QWEN_LAYER_FIELD_RE.finditer(match.group("body")):
                    name = f"{field.group('name')}_ms"
                    value = float(field.group("value"))
                    result["qwen_layer"][key][name].append(value)
                    if call > 1:
                        result["qwen_layer_steady"][key][name].append(value)

            if match := FORWARD_KIND_RE.search(line):
                attn_state = match.group("attn_state") or "-"
                key = f"{match.group('kind')}/tokens={match.group('tokens')}/attn={attn_state}"
                result["forward_kind"][key]["elapsed_ms"].append(float(match.group("elapsed")))
                result["forward_kind"][key]["call"].append(float(match.group("call")))

            if match := ATTENTION_STAGE_RE.search(line):
                call = float(match.group("call"))
                tokens = int(match.group("tokens"))
                key = (
                    f"{match.group('state')}/{match.group('op')}/"
                    f"tokens={token_bucket(tokens)}"
                )
                value = float(match.group("op_ms"))
                result["attention_stage"][key]["op_ms"].append(value)
                result["attention_stage"][key]["tokens"].append(float(tokens))
                result["attention_stage"][key]["call"].append(call)
                if call > 1:
                    result["attention_stage_steady"][key]["op_ms"].append(value)
                    result["attention_stage_steady"][key]["tokens"].append(float(tokens))
                    result["attention_stage_steady"][key]["call"].append(call)

            if "decode_len_profile" in line:
                for part in DECODE_LEN_PROFILE_PART_RE.finditer(line):
                    key = part.group("bucket")
                    result["decode_len_profile"][key]["n"].append(
                        float(part.group("n")))
                    result["decode_len_profile"][key]["avg_reqs"].append(
                        float(part.group("avg_reqs")))
                    result["decode_len_profile"][key]["avg_max_seq"].append(
                        float(part.group("avg_max_seq")))
                    result["decode_len_profile"][key]["avg_seq"].append(
                        float(part.group("avg_seq")))
                    for field in DECODE_LEN_PROFILE_FIELD_RE.finditer(
                            part.group("body")):
                        result["decode_len_profile"][key][
                            field.group("name")].append(
                                float(field.group("value")))

    # Convert defaultdicts for json friendliness.
    result["profile_summary"] = {
        phase: {name: summarize_values(values) for name, values in fields.items()}
        for phase, fields in result["profile_summary"].items()
    }
    result["profile_raw"] = {
        phase: {name: summarize_values(values) for name, values in fields.items()}
        for phase, fields in result["profile_raw"].items()
    }
    result["moe_stage"] = {
        key: {name: summarize_values(values) for name, values in fields.items()}
        for key, fields in result["moe_stage"].items()
    }
    result["moe_stage_steady"] = {
        key: {name: summarize_values(values) for name, values in fields.items()}
        for key, fields in result["moe_stage_steady"].items()
    }
    result["fused_moe_profile"] = {
        key: {name: summarize_values(values) for name, values in fields.items()}
        for key, fields in result["fused_moe_profile"].items()
    }
    result["fused_moe_profile_steady"] = {
        key: {name: summarize_values(values) for name, values in fields.items()}
        for key, fields in result["fused_moe_profile_steady"].items()
    }
    result["qwen_layer"] = {
        key: {name: summarize_values(values) for name, values in fields.items()}
        for key, fields in result["qwen_layer"].items()
    }
    result["qwen_layer_steady"] = {
        key: {name: summarize_values(values) for name, values in fields.items()}
        for key, fields in result["qwen_layer_steady"].items()
    }
    result["forward_kind"] = {
        key: {name: summarize_values(values) for name, values in fields.items()}
        for key, fields in result["forward_kind"].items()
    }
    result["attention_stage"] = {
        key: {name: summarize_values(values) for name, values in fields.items()}
        for key, fields in result["attention_stage"].items()
    }
    result["attention_stage_steady"] = {
        key: {name: summarize_values(values) for name, values in fields.items()}
        for key, fields in result["attention_stage_steady"].items()
    }
    result["decode_len_profile"] = {
        key: {name: summarize_values(values) for name, values in fields.items()}
        for key, fields in result["decode_len_profile"].items()
    }

    return result


def compact_summary(parsed: dict[str, Any]) -> dict[str, Any]:
    steps = parsed["steps"]
    token_lists = parsed["global_token_num"]
    latest_tokens = token_lists[-1] if token_lists else []

    step_metrics: dict[str, list[float]] = defaultdict(list)
    for step in steps:
        for key, value in step.items():
            if key != "step":
                step_metrics[key].append(value)

    return {
        "path": parsed["path"],
        "num_steps": len(steps),
        "rollout_output_time_s": summarize_values(parsed["rollout_output_time_s"]),
        "rollouts_speed_tokens_s": summarize_values(parsed["rollouts_speed_tokens_s"]),
        "timing_s/generate_sequences": summarize_values(step_metrics.get("timing_s/generate_sequences", [])),
        "timing_s/gen": summarize_values(step_metrics.get("timing_s/gen", [])),
        "timing_per_token_ms/gen": summarize_values(step_metrics.get("timing_per_token_ms/gen", [])),
        "response_length/mean": summarize_values(step_metrics.get("response_length/mean", [])),
        "response_length/clip_ratio": summarize_values(step_metrics.get("response_length/clip_ratio", [])),
        "perf/total_num_tokens": summarize_values(step_metrics.get("perf/total_num_tokens", [])),
        "latest_global_tokens": {
            "n": len(latest_tokens),
            "sum": sum(latest_tokens) if latest_tokens else None,
            "mean": fmean([float(v) for v in latest_tokens]),
            "max": max(latest_tokens) if latest_tokens else None,
            "min": min(latest_tokens) if latest_tokens else None,
            "ge_16000": sum(1 for v in latest_tokens if v >= 16000) if latest_tokens else None,
        },
        "kv_cache_tokens": summarize_values([float(v) for v in parsed["kv_cache_tokens"]]),
        "max_concurrency": summarize_values(parsed["max_concurrency"]),
    }


def print_markdown(summaries: list[dict[str, Any]], parsed_logs: list[dict[str, Any]]) -> None:
    print("## Step Metrics\n")
    headers = [
        "log",
        "steps",
        "rollout avg",
        "gen_seq avg",
        "gen avg",
        "ms/token avg",
        "tokens sum",
        "len mean",
        "clip avg",
        "speed avg",
        "kv avg",
        "conc avg",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for summary in summaries:
        path = Path(summary["path"])
        row = [
            path.parent.parent.name + "/" + path.name,
            summary["num_steps"],
            fmt(summary["rollout_output_time_s"]["avg"]),
            fmt(summary["timing_s/generate_sequences"]["avg"]),
            fmt(summary["timing_s/gen"]["avg"]),
            fmt(summary["timing_per_token_ms/gen"]["avg"], 4),
            fmt(summary["latest_global_tokens"]["sum"], 0),
            fmt(summary["response_length/mean"]["avg"]),
            fmt(summary["response_length/clip_ratio"]["avg"], 4),
            fmt(summary["rollouts_speed_tokens_s"]["avg"]),
            fmt(summary["kv_cache_tokens"]["avg"], 0),
            fmt(summary["max_concurrency"]["avg"], 2),
        ]
        print("| " + " | ".join(str(x) for x in row) + " |")

    print("\n## Decode Stage Metrics\n")
    headers = ["log", "source", "phase", "forward", "sample", "prepare", "post", "records"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for parsed in parsed_logs:
        path = Path(parsed["path"])
        log_name = path.parent.parent.name + "/" + path.name
        for source_name, source in (("summary", parsed["profile_summary"]), ("raw", parsed["profile_raw"])):
            for phase, fields in source.items():
                if phase != "Decode":
                    continue
                row = [
                    log_name,
                    source_name,
                    phase,
                    fmt(fields.get("forward", {}).get("avg")),
                    fmt(fields.get("Sample", {}).get("avg")),
                    fmt(fields.get("prepare input", {}).get("avg")),
                    fmt(fields.get("post process", {}).get("avg")),
                    fields.get("forward", {}).get("n", 0),
                ]
                print("| " + " | ".join(str(x) for x in row) + " |")

    print("\n## MoE Stage Metrics\n")
    headers = ["log", "view", "comm/tokens", "dispatch", "mlp", "combine", "total", "records"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for parsed in parsed_logs:
        path = Path(parsed["path"])
        log_name = path.parent.parent.name + "/" + path.name
        for view_name, stage in (("all", parsed["moe_stage"]), ("call>1", parsed["moe_stage_steady"])):
            for key, fields in sorted(stage.items()):
                row = [
                    log_name,
                    view_name,
                    key,
                    fmt(fields.get("dispatch_ms", {}).get("avg")),
                    fmt(fields.get("mlp_ms", {}).get("avg")),
                    fmt(fields.get("combine_ms", {}).get("avg")),
                    fmt(fields.get("total_ms", {}).get("avg")),
                    fields.get("total_ms", {}).get("n", 0),
                ]
                print("| " + " | ".join(str(x) for x in row) + " |")

    print("\n## Fused-MoE Internal Metrics\n")
    headers = ["log", "view", "kind/comm/tokens", "fields", "records"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for parsed in parsed_logs:
        path = Path(parsed["path"])
        log_name = path.parent.parent.name + "/" + path.name
        for view_name, stage in (
            ("all", parsed.get("fused_moe_profile", {})),
            ("call>1", parsed.get("fused_moe_profile_steady", {})),
        ):
            for key, fields in sorted(stage.items()):
                parts = []
                records = fields.get("call", {}).get("n", 0)
                for field_name, summary in sorted(fields.items()):
                    if field_name == "call":
                        continue
                    parts.append(f"{field_name.removesuffix('_ms')}={fmt(summary['avg'])}")
                row = [log_name, view_name, key, ", ".join(parts), records]
                print("| " + " | ".join(str(x) for x in row) + " |")

    print("\n## Qwen2-MoE Layer Metrics\n")
    headers = ["log", "view", "name/tokens", "fields", "records"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for parsed in parsed_logs:
        path = Path(parsed["path"])
        log_name = path.parent.parent.name + "/" + path.name
        for view_name, stage in (("all", parsed["qwen_layer"]), ("call>1", parsed["qwen_layer_steady"])):
            for key, fields in sorted(stage.items()):
                parts = []
                records = fields.get("call", {}).get("n", 0)
                for field_name, summary in sorted(fields.items()):
                    if field_name == "call":
                        continue
                    parts.append(f"{field_name.removesuffix('_ms')}={fmt(summary['avg'])}")
                row = [log_name, view_name, key, ", ".join(parts), records]
                print("| " + " | ".join(str(x) for x in row) + " |")

    print("\n## Attention Stage Metrics\n")
    headers = ["log", "view", "state/op/tokens", "op avg", "p50", "p95", "token avg", "records"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for parsed in parsed_logs:
        path = Path(parsed["path"])
        log_name = path.parent.parent.name + "/" + path.name
        for view_name, stage in (
            ("all", parsed.get("attention_stage", {})),
            ("call>1", parsed.get("attention_stage_steady", {})),
        ):
            for key, fields in sorted(stage.items()):
                op_ms = fields.get("op_ms", {})
                tokens = fields.get("tokens", {})
                row = [
                    log_name,
                    view_name,
                    key,
                    fmt(op_ms.get("avg")),
                    fmt(op_ms.get("p50")),
                    fmt(op_ms.get("p95")),
                    fmt(tokens.get("avg")),
                    op_ms.get("n", 0),
                ]
                print("| " + " | ".join(str(x) for x in row) + " |")

    print("\n## Forward Kind Debug\n")
    headers = ["log", "kind/tokens/attn", "elapsed avg", "p50", "p95", "records"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for parsed in parsed_logs:
        path = Path(parsed["path"])
        log_name = path.parent.parent.name + "/" + path.name
        for key, fields in sorted(parsed.get("forward_kind", {}).items()):
            elapsed = fields.get("elapsed_ms", {})
            row = [
                log_name,
                key,
                fmt(elapsed.get("avg")),
                fmt(elapsed.get("p50")),
                fmt(elapsed.get("p95")),
                elapsed.get("n", 0),
            ]
            print("| " + " | ".join(str(x) for x in row) + " |")

    print("\n## Decode Length Profile\n")
    headers = [
        "log",
        "bucket",
        "forward",
        "sample",
        "prepare",
        "post",
        "avg max seq",
        "avg seq",
        "records",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for parsed in parsed_logs:
        path = Path(parsed["path"])
        log_name = path.parent.parent.name + "/" + path.name
        for key, fields in sorted(parsed.get("decode_len_profile", {}).items()):
            row = [
                log_name,
                key,
                fmt(fields.get("forward", {}).get("avg")),
                fmt(fields.get("Sample", {}).get("avg")),
                fmt(fields.get("prepare_input", {}).get("avg")),
                fmt(fields.get("post_process", {}).get("avg")),
                fmt(fields.get("avg_max_seq", {}).get("avg")),
                fmt(fields.get("avg_seq", {}).get("avg")),
                fields.get("forward", {}).get("n", 0),
            ]
            print("| " + " | ".join(str(x) for x in row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    args = parser.parse_args()

    parsed_logs = [parse_log(path) for path in args.logs]
    summaries = [compact_summary(parsed) for parsed in parsed_logs]

    if args.json:
        print(json.dumps({"summaries": summaries, "logs": parsed_logs}, indent=2, sort_keys=True))
    else:
        print_markdown(summaries, parsed_logs)


if __name__ == "__main__":
    main()
