#!/usr/bin/env python3
"""Summarize Elastic group seq digest logs by rank and sequence."""
from __future__ import annotations

import argparse
import collections
import hashlib
import re
import signal
import sys
from pathlib import Path

DEFAULT_LOG_ROOT = Path("mode1_baseline_random_batch_floor4_threshold_multistep")
DIGEST_RE = re.compile(
    r"Elastic group seq digest: rank=(?P<rank>\d+) seq=(?P<seq>\d+) "
    r"event=(?P<event>\S+) kind=(?P<kind>\S*) attr=(?P<attr>\S*) "
    r"group_name=(?P<group_name>\S*) ranks=(?P<ranks>.*?) target=(?P<target>.*?) "
    r"active=(?P<active>.*?) detached=(?P<detached>\S+) seen=(?P<seen>\S+) "
    r"reason=(?P<reason>\S*) extra=(?P<extra>.*)$"
)
HCCL_DIGEST_RE = re.compile(
    r"Elastic HCCL seq digest: rank=(?P<rank>\d+) op=(?P<op>\S+) "
    r"seq=(?P<seq>\d+) event=(?P<event>\S+) "
    r"group_kind=(?P<kind>\S*) group_name=(?P<group_name>\S*) "
    r"backend=(?P<backend>\S*) group_ranks=(?P<ranks>.*?) "
    r"target=(?P<target>.*?) active=(?P<active>.*?) "
    r"reason=(?P<reason>\S*) extra=(?P<extra>.*)$"
)
HCCL_SUMMARY_RE = re.compile(
    r"Elastic HCCL seq summary: rank=(?P<rank>\d+) event=(?P<event>\S+) "
    r"create_seq=(?P<create_seq>\d+) destroy_seq=(?P<destroy_seq>\d+) "
    r"ranks=(?P<ranks>.*?) target=(?P<target>.*?) active=(?P<active>.*?) "
    r"detached=(?P<detached>\S+) seen=(?P<seen>\d+) "
    r"reason=(?P<reason>\S*) extra=(?P<extra>.*)$"
)
MOE_DISPATCH_RE = re.compile(
    r"Elastic MoE dispatch digest: rank=(?P<rank>-?\d+) seq=(?P<seq>\d+) "
    r"event=(?P<event>\S+) started=(?P<started>\S+) "
    r"mode=(?P<mode>-?\d+) layer=(?P<layer>-?\d+) "
    r"stage=(?P<stage>-?\d+) active=(?P<active>.*?) "
    r"selected=(?P<selected>\S*) effective=(?P<effective>\S*) "
    r"fused_state=(?P<fused_state>\S*) method=(?P<method>\S*) "
    r"elapsed_ms=(?P<elapsed_ms>\S*) extra=(?P<extra>.*)$"
)
DEFERRED_RE = re.compile(
    r"Elastic parallel group destroy deferred: rank=(?P<rank>\d+) "
    r"group_kind=(?P<kind>\S+) group_ranks=(?P<ranks>.*?) "
    r"reason=(?P<reason>\S+) deferred_count=(?P<deferred_count>\d+)"
)
RETIRED_RE = re.compile(
    r"Elastic parallel group retired immediately: rank=(?P<rank>\d+) "
    r"group_kind=(?P<kind>\S+) group_ranks=(?P<ranks>.*?) "
    r"reason=(?P<reason>\S+)"
)
RESTORE_RE = re.compile(
    r"Elastic parallel restore done: rank=(?P<rank>\d+) "
    r"dp_size=(?P<dp_size>\d+) ep_size=(?P<ep_size>\d+) "
    r"rebuild_ms=(?P<rebuild_ms>[0-9]+(?:\.[0-9]+)?) "
    r"refresh_ms=(?P<refresh_ms>[0-9]+(?:\.[0-9]+)?) "
    r"total_ms=(?P<total_ms>[0-9]+(?:\.[0-9]+)?)"
)
STASH_RELEASE_RE = re.compile(
    r"Elastic parallel stashed group released: rank=(?P<rank>\d+) "
    r"attr=(?P<attr>\S+) group_kind=(?P<kind>\S+) "
    r"old_ranks=(?P<ranks>.*?) reason=(?P<reason>\S+) "
    r"destroy_ms=(?P<destroy_ms>[0-9]+(?:\.[0-9]+)?) "
    r"cleanup_ms=(?P<cleanup_ms>[0-9]+(?:\.[0-9]+)?) "
    r"already_retired=(?P<already_retired>\S+)"
)
STASH_CLASSIFY_RE = re.compile(
    r"Elastic mode1 stash classify: rank=(?P<rank>\d+) "
    r"attr=(?P<attr>\S+) group_kind=(?P<kind>\S+) "
    r"old_ranks=(?P<ranks>.*?) target_ranks=(?P<target>.*?) "
    r"active_ranks=(?P<active>.*?) "
    r"full_restore_context=(?P<full_restore_context>[01]) "
    r"fixed_keep=(?P<fixed_keep>[01]) floor4_keep=(?P<floor4_keep>[01]) "
    r"should_cache=(?P<should_cache>[01]) single_live=(?P<single_live>[01]) "
    r"full_restore=(?P<full_restore>[01]) pre_rebuild=(?P<pre_rebuild>[01]) "
    r"already_retired=(?P<already_retired>[01]) "
    r"predicted_reason=(?P<predicted_reason>\S+) "
    r"defer_preview=(?P<defer_preview>[01])"
)
FEATURE_RE = re.compile(
    r"Elastic mode1 lifecycle feature: rank=(?P<rank>\d+) "
    r"feature=(?P<feature>\S+) context=(?P<context>\S+) "
    r"stash_classify=(?P<stash_classify>[01]) "
    r"group_seq_digest=(?P<group_seq_digest>[01]) "
    r"hccl_seq_digest=(?P<hccl_seq_digest>[01]) "
    r"hccl_seq_summary=(?P<hccl_seq_summary>[01]) "
    r"defer_full_restore=(?P<defer_full_restore>[01]) "
    r"defer_group_destroy=(?P<defer_group_destroy>[01])"
    r"(?: destroy_cpu_placeholder_pg=(?P<destroy_cpu_placeholder_pg>[01]))?"
)
FORCED_FULL_RESTORE_DEFER_RE = re.compile(
    r"Elastic mode1 full-restore defer forced despite env: "
    r"rank=(?P<rank>\d+) group_kind=(?P<kind>\S+) "
    r"group_ranks=(?P<ranks>.*?) keep_ranks=(?P<keep>.*?) "
    r"defer_full_restore_env=(?P<defer_full_restore_env>\S+) "
    r"defer_group_destroy_env=(?P<defer_group_destroy_env>\S+)"
)
FORCED_FLOOR_GROUP_DEFER_RE = re.compile(
    r"Elastic mode1 floor-group defer forced outside low-floor marker: "
    r"rank=(?P<rank>\d+) group_kind=(?P<kind>\S+) "
    r"group_ranks=(?P<ranks>.*?) keep_ranks=(?P<keep>.*?) "
    r"reason=(?P<reason>\S+) defer_sizes=(?P<defer_sizes>\S+)"
)
RESTORE_STALE_SWEEP_ROUTE_RE = re.compile(
    r"Elastic mode1 restore stale-cache sweep route: rank=(?P<rank>\d+) "
    r"keep_ranks=(?P<keep>.*?) reason=(?P<reason>\S+) "
    r"feature=(?P<feature>\S+) "
    r"defer_full_restore=(?P<defer_full_restore>\S+) "
    r"defer_group_destroy=(?P<defer_group_destroy>\S+) "
    r"force_non_member_placeholder=(?P<force_non_member_placeholder>\S+)"
)
SOURCE_FALLBACK_RE = re.compile(
    r"Elastic shrink source group falls back to world group: rank=(?P<rank>\d+) "
    r"previous_active_ranks=(?P<source>.*?) reason=(?P<reason>\S+)"
)
SOURCE_UNAVAILABLE_RE = re.compile(
    r"Elastic shrink source group unavailable for non-world source ranks: "
    r"rank=(?P<rank>\d+) previous_active_ranks=(?P<source>.*?) "
    r"reason=(?P<reason>\S+)"
)
PAYLOAD_SKIP_RE = re.compile(
    r"Elastic shrink payload skipped on non-source rank: rank=(?P<rank>\d+) "
    r"active_ranks=(?P<active>.*?) source_ranks=(?P<source>.*?) "
    r"detached=(?P<detached>\S+)"
)
PRELOAD_SKIP_RE = re.compile(
    r"Elastic shrink preload skipped on non-source rank: rank=(?P<rank>\d+) "
    r"active_ranks=(?P<active>.*?) source_ranks=(?P<source>.*?) "
    r"detached=(?P<detached>\S+)"
)
DETACHED_ADVANCE_RE = re.compile(
    r"Elastic parallel already detached rank advances non-member sequence: "
    r"rank=(?P<rank>\d+) active_ranks=(?P<active>.*)$"
)
DIRECT_DESTROY_FALLBACK_RE = re.compile(
    r"Elastic mode1 low-floor (?P<label>stashed group|stale cache|group) "
    r"direct destroy fallback: rank=(?P<rank>\d+) "
    r"group_kind=(?P<kind>\S+) group_ranks=(?P<ranks>.*?) "
    r"(?:(?:keep_ranks=(?P<keep>.*?) )?)"
    r"(?:(?:target_ranks=(?P<target>.*?) )?)"
    r"reason=(?P<reason>\S+) "
    r"(?:(?:allow_same_ranks=(?P<allow_same_ranks>\S+) )?)"
    r"(?:(?:defer_preview=(?P<defer_preview>\S+) )?)"
    r"(?:(?:force_destroy=(?P<force_destroy>\S+) )?)"
    r"defer_env=(?P<defer_env>\S+) "
    r"defer_sizes=(?P<defer_sizes>\S+)"
)
MODE1_SEQ_CHECKPOINT_RE = re.compile(
    r"Elastic mode1 seq checkpoint: rank=(?P<rank>\d+) tag=(?P<tag>\S+) "
    r"create_seq=(?P<create_seq>\d+) destroy_seq=(?P<destroy_seq>\d+) "
    r"worker_create_seq=(?P<worker_create_seq>\d+) "
    r"worker_destroy_seq=(?P<worker_destroy_seq>\d+) "
    r"requested=(?P<requested>\S+) active=(?P<active>\S+) "
    r"target=(?P<target>\S+) detached=(?P<detached>\S+) "
    r"cached=(?P<cached>\S+) registry=(?P<registry>\S+) "
    r"seen=(?P<seen>\S+) deferred_ids=(?P<deferred_ids>\d+) "
    r"retired_ids=(?P<retired_ids>\d+) "
    r"deferred_groups=(?P<deferred_groups>\d+) "
    r"(?:(?:deferred_device_placeholders=(?P<deferred_device_placeholders>\d+) ))?"
    r"deferred_cpu_placeholders=(?P<deferred_cpu_placeholders>\d+) "
    r"live=(?P<live>\S+) reason=(?P<reason>\S*) extra=(?P<extra>.*)$"
)
ROLLOUT_RE = re.compile(r"rollout_output_time_s:\s*(?P<seconds>[0-9]+(?:\.[0-9]+)?)")
ROLLOUT_BOUNDARY_RE = re.compile(
    r"Elastic rollout boundary: event=(?P<event>\S+) "
    r"step=(?P<step>-?\d+) epoch=(?P<epoch>-?\d+)"
    r"(?: (?P<extra>.*))?$"
)
CONFIG_LINE_MARKERS = (
    "[mode1 threshold multistep lifecycle]",
    "[mode1 lifecycle]",
    "[mode1 code]",
    "[full redundancy experiment]",
)


def resolve_log(path: Path | None) -> Path:
    if path is not None:
        return path
    candidates = sorted(
        (
            candidate for candidate in DEFAULT_LOG_ROOT.rglob("*.txt")
            if candidate.name.startswith("wjqwen30b-")
            or candidate.name.startswith("wjeagerqwen30b-")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no rollout .txt logs found under {DEFAULT_LOG_ROOT}; pass a log path explicitly")
    return candidates[0]


def parse(path: Path):
    events = []
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        match = DIGEST_RE.search(line)
        if not match:
            continue
        row = match.groupdict()
        row["lineno"] = lineno
        row["rank"] = int(row["rank"])
        row["seq"] = int(row["seq"])
        events.append(row)
    return events


def _parse_key_values(text: str) -> dict[str, str]:
    values = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", text):
        values[key] = value
    return values


def parse_config_lines(path: Path):
    rows = []
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not any(marker in line for marker in CONFIG_LINE_MARKERS):
            continue
        clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        rows.append({
            "lineno": lineno,
            "line": clean_line,
            "values": _parse_key_values(clean_line),
        })
    return rows


def _first_config_value(config_rows, *keys: str) -> str | None:
    for row in config_rows:
        for key in keys:
            value = row["values"].get(key)
            if value is not None:
                return value
    return None


def _current_repo_sha256(relative_path: str) -> str | None:
    path = Path(__file__).resolve().parent.parent / relative_path
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def parse_auxiliary(path: Path):
    hccl_events = []
    hccl_summaries = []
    moe_dispatch = []
    deferred = []
    retired = []
    restores = []
    stash_releases = []
    stash_classify = []
    feature_rows = []
    forced_full_restore_defer = []
    forced_floor_group_defer = []
    restore_stale_sweep_routes = []
    rollouts = []
    rollout_boundaries = []
    direct_destroy_fallbacks = []
    source_fallbacks = []
    source_unavailable = []
    payload_skips = []
    preload_skips = []
    detached_advances = []
    mode1_seq_checkpoints = []
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        match = HCCL_DIGEST_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            row["seq"] = int(row["seq"])
            row["domain"] = _hccl_counter_domain(row)
            hccl_events.append(row)
            continue

        match = HCCL_SUMMARY_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            row["create_seq"] = int(row["create_seq"])
            row["destroy_seq"] = int(row["destroy_seq"])
            row["seen"] = int(row["seen"])
            hccl_summaries.append(row)
            continue

        match = MOE_DISPATCH_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            row["seq"] = int(row["seq"])
            row["mode"] = int(row["mode"])
            row["layer"] = int(row["layer"])
            row["stage"] = int(row["stage"])
            try:
                row["elapsed_ms_value"] = float(row["elapsed_ms"])
            except (TypeError, ValueError):
                row["elapsed_ms_value"] = None
            moe_dispatch.append(row)
            continue

        match = DEFERRED_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            row["deferred_count"] = int(row["deferred_count"])
            row["retire_wrapper_ms"] = first_ms(line, "retire_wrapper_ms")
            row["npu_sync_ms"] = first_ms(line, "npu_sync_ms")
            row["device_pg_destroy_ms"] = first_ms(line, "device_pg_destroy_ms")
            row["cpu_pg_destroy_ms"] = first_ms(line, "cpu_pg_destroy_ms")
            row["destroy_device_pg"] = first_ms(line, "destroy_device_pg")
            row["deferred_device_pg_refs"] = first_ms(line, "deferred_device_pg_refs")
            row["deferred_cpu_pg_refs"] = first_ms(line, "deferred_cpu_pg_refs")
            deferred.append(row)
            continue

        match = RETIRED_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            row["retire_wrapper_ms"] = first_ms(line, "retire_wrapper_ms")
            row["npu_sync_ms"] = first_ms(line, "npu_sync_ms")
            row["device_pg_destroy_ms"] = first_ms(line, "device_pg_destroy_ms")
            row["cpu_pg_destroy_ms"] = first_ms(line, "cpu_pg_destroy_ms")
            retired.append(row)
            continue

        match = RESTORE_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            row["dp_size"] = int(row["dp_size"])
            row["ep_size"] = int(row["ep_size"])
            row["rebuild_ms"] = float(row["rebuild_ms"])
            row["refresh_ms"] = float(row["refresh_ms"])
            row["total_ms"] = float(row["total_ms"])
            restores.append(row)
            continue

        match = STASH_RELEASE_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            row["destroy_ms"] = float(row["destroy_ms"])
            row["cleanup_ms"] = float(row["cleanup_ms"])
            stash_releases.append(row)
            continue

        match = STASH_CLASSIFY_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            for key in (
                    "full_restore_context",
                    "fixed_keep",
                    "floor4_keep",
                    "should_cache",
                    "single_live",
                    "full_restore",
                    "pre_rebuild",
                    "already_retired",
                    "defer_preview",
            ):
                row[key] = int(row[key])
            stash_classify.append(row)
            continue

        match = FEATURE_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            for key in (
                    "stash_classify",
                    "group_seq_digest",
                    "hccl_seq_digest",
                    "hccl_seq_summary",
                    "defer_full_restore",
                    "defer_group_destroy",
                    "destroy_cpu_placeholder_pg",
            ):
                row[key] = int(row[key]) if row.get(key) is not None else None
            feature_rows.append(row)
            continue

        match = FORCED_FULL_RESTORE_DEFER_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            forced_full_restore_defer.append(row)
            continue

        match = FORCED_FLOOR_GROUP_DEFER_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            forced_floor_group_defer.append(row)
            continue

        match = RESTORE_STALE_SWEEP_ROUTE_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            restore_stale_sweep_routes.append(row)
            continue

        match = ROLLOUT_RE.search(line)
        if match:
            rollouts.append({
                "lineno": lineno,
                "seconds": float(match.group("seconds")),
            })
            continue

        match = ROLLOUT_BOUNDARY_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["step"] = int(row["step"])
            row["epoch"] = int(row["epoch"])
            row["extra"] = row.get("extra") or ""
            rollout_boundaries.append(row)
            continue

        match = SOURCE_FALLBACK_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            source_fallbacks.append(row)
            continue

        match = SOURCE_UNAVAILABLE_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            source_unavailable.append(row)
            continue

        match = PAYLOAD_SKIP_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            payload_skips.append(row)
            continue

        match = PRELOAD_SKIP_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            preload_skips.append(row)
            continue

        match = DETACHED_ADVANCE_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            detached_advances.append(row)
            continue

        match = MODE1_SEQ_CHECKPOINT_RE.search(line)
        if match:
            row = match.groupdict()
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            for key in (
                    "create_seq", "destroy_seq", "worker_create_seq",
                    "worker_destroy_seq", "deferred_ids", "retired_ids",
                    "deferred_groups", "deferred_cpu_placeholders",
                    "deferred_device_placeholders"):
                row[key] = int(row.get(key) or 0)
            mode1_seq_checkpoints.append(row)
            continue

        match = DIRECT_DESTROY_FALLBACK_RE.search(line)
        if match:
            row = match.groupdict()
            fallback_kind = "unknown"
            if row.get("label") == "stashed group":
                fallback_kind = "stashed"
            elif row.get("label") == "stale cache":
                fallback_kind = "stale_cache"
            elif row.get("label") == "group":
                fallback_kind = "group"
            row["lineno"] = lineno
            row["rank"] = int(row["rank"])
            row["fallback_kind"] = fallback_kind
            row["line"] = line
            direct_destroy_fallbacks.append(row)
            continue

        if "Elastic mode1 low-floor" in line and "direct destroy fallback" in line:
            fallback_kind = "unknown"
            if "stashed group direct destroy fallback" in line:
                fallback_kind = "stashed"
            elif "stale cache direct destroy fallback" in line:
                fallback_kind = "stale_cache"
            elif "group direct destroy fallback" in line:
                fallback_kind = "group"
            direct_destroy_fallbacks.append({
                "lineno": lineno,
                "kind": fallback_kind,
                "fallback_kind": fallback_kind,
                "rank": None,
                "reason": "",
                "ranks": "",
                "line": line,
            })
    return {
        "hccl_events": hccl_events,
        "hccl_summaries": hccl_summaries,
        "moe_dispatch": moe_dispatch,
        "deferred": deferred,
        "retired": retired,
        "restores": restores,
        "stash_releases": stash_releases,
        "stash_classify": stash_classify,
        "feature_rows": feature_rows,
        "forced_full_restore_defer": forced_full_restore_defer,
        "forced_floor_group_defer": forced_floor_group_defer,
        "restore_stale_sweep_routes": restore_stale_sweep_routes,
        "rollouts": rollouts,
        "rollout_boundaries": rollout_boundaries,
        "direct_destroy_fallbacks": direct_destroy_fallbacks,
        "source_fallbacks": source_fallbacks,
        "source_unavailable": source_unavailable,
        "payload_skips": payload_skips,
        "preload_skips": preload_skips,
        "detached_advances": detached_advances,
        "mode1_seq_checkpoints": mode1_seq_checkpoints,
    }


def compact(row):
    name = row["group_name"] or row["attr"] or row["kind"]
    return f"{row['event']}:{row['kind']}:{name}:ranks={row['ranks']} target={row['target']} reason={row['reason']} extra={row['extra']}"


def first_ms(extra: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}=([0-9]+(?:\.[0-9]+)?)", extra)
    if not match:
        return None
    return float(match.group(1))


def print_full_restore_mc2_releases(events) -> None:
    rows = [
        row for row in events
        if row["event"] == "stash_release_done"
        and row["kind"] == "mc2"
        and row["attr"] == "_MC2"
        and row["reason"] == "mode1_full_restore"
    ]
    print("\n== mode1 full-restore MC2 releases ==")
    if not rows:
        print("none")
        return

    clusters = []
    for row in rows:
        if not clusters or row["lineno"] - clusters[-1][-1]["lineno"] > 80:
            clusters.append([row])
        else:
            clusters[-1].append(row)

    for index, cluster in enumerate(clusters, 1):
        destroy_values = [
            first_ms(row["extra"], "destroy_ms")
            for row in cluster
        ]
        destroy_values = [value for value in destroy_values if value is not None]
        ranks = sorted(row["rank"] for row in cluster)
        group_ranks = sorted(set(row["ranks"] for row in cluster))
        max_destroy = max(destroy_values) if destroy_values else 0.0
        min_destroy = min(destroy_values) if destroy_values else 0.0
        print(
            f"cluster={index} lines={cluster[0]['lineno']}-{cluster[-1]['lineno']} "
            f"ranks={ranks} group_ranks={group_ranks} "
            f"destroy_ms_min={min_destroy:.2f} destroy_ms_max={max_destroy:.2f}")
        for row in cluster:
            destroy_ms = first_ms(row["extra"], "destroy_ms")
            print(
                f"  line={row['lineno']} rank={row['rank']} seq={row['seq']} "
                f"destroy_ms={destroy_ms:.2f} active={row['active']} target={row['target']}"
                if destroy_ms is not None else
                f"  line={row['lineno']} rank={row['rank']} seq={row['seq']} {compact(row)}"
            )


def print_full_restore_deferred_summary(aux) -> None:
    rows = [
        row for row in aux["deferred"]
        if row["kind"] == "mc2"
        and row["reason"] == "mode1_full_restore"
    ]
    print("\n== mode1 full-restore MC2 deferred retire ==")
    if not rows:
        print("none")
        return
    clusters = []
    for row in rows:
        if not clusters or row["lineno"] - clusters[-1][-1]["lineno"] > 120:
            clusters.append([row])
        else:
            clusters[-1].append(row)
    for index, cluster in enumerate(clusters, 1):
        ranks = sorted(row["rank"] for row in cluster)
        group_ranks = sorted(set(row["ranks"] for row in cluster))
        max_retire = max((row["retire_wrapper_ms"] or 0.0) for row in cluster)
        max_device_destroy = max((row["device_pg_destroy_ms"] or 0.0) for row in cluster)
        max_deferred_count = max(row["deferred_count"] for row in cluster)
        max_deferred_device_refs = max((row["deferred_device_pg_refs"] or 0.0) for row in cluster)
        print(
            f"cluster={index} lines={cluster[0]['lineno']}-{cluster[-1]['lineno']} "
            f"ranks={ranks} group_ranks={group_ranks} "
            f"max_retire_wrapper_ms={max_retire:.2f} "
            f"max_device_pg_destroy_ms={max_device_destroy:.2f} "
            f"max_deferred_count={max_deferred_count} "
            f"max_deferred_device_pg_refs={max_deferred_device_refs:.0f}")
        for row in cluster:
            print(
                f"  line={row['lineno']} rank={row['rank']} "
                f"deferred_count={row['deferred_count']} "
                f"retire_wrapper_ms={(row['retire_wrapper_ms'] or 0.0):.2f} "
                f"device_pg_destroy_ms={(row['device_pg_destroy_ms'] or 0.0):.2f} "
                f"deferred_device_pg_refs={(row['deferred_device_pg_refs'] or 0.0):.0f}")


def _rank_count_from_text(ranks_text: str) -> int:
    return len(re.findall(r"-?\d+", ranks_text))


def _plain_full_restore_mc2_release_rows(aux):
    return [
        row for row in aux["stash_releases"]
        if row["kind"] == "mc2"
        and row["attr"] == "_MC2"
        and row["reason"] == "mode1_full_restore"
    ]


def _plain_misclassified_floor_mc2_release_rows(aux, slow_ms: float):
    return [
        row for row in aux["stash_releases"]
        if row["kind"] == "mc2"
        and row["attr"] == "_MC2"
        and row["reason"] == "stash_group"
        and _rank_count_from_text(row["ranks"]) <= 8
        and row["destroy_ms"] >= slow_ms
    ]


def _plain_slow_stash_release_rows(aux, slow_ms: float):
    return [
        row for row in aux["stash_releases"]
        if row["destroy_ms"] >= slow_ms
    ]


def _stash_classify_full_restore_mc2_rows(aux):
    return [
        row for row in aux["stash_classify"]
        if row["kind"] == "mc2"
        and row["attr"] == "_MC2"
        and row["full_restore_context"]
    ]


def _stash_classify_misclassified_mc2_rows(aux):
    return [
        row for row in _stash_classify_full_restore_mc2_rows(aux)
        if row["predicted_reason"] == "stash_group"
    ]


def _stash_classify_fixed_keep_without_cache_rows(aux):
    return [
        row for row in _stash_classify_full_restore_mc2_rows(aux)
        if row["fixed_keep"]
        and not row["should_cache"]
    ]


def print_stash_classify_summary(aux) -> None:
    print("\n== mode1 stash classification summary ==")
    rows = aux["stash_classify"]
    if not rows:
        print("none")
        return
    full_restore_mc2 = _stash_classify_full_restore_mc2_rows(aux)
    misclassified_mc2 = _stash_classify_misclassified_mc2_rows(aux)
    fixed_keep_no_cache = _stash_classify_fixed_keep_without_cache_rows(aux)
    by_reason = collections.Counter(
        row["predicted_reason"] for row in full_restore_mc2)
    reason_text = ",".join(
        f"{reason}:{count}" for reason, count in sorted(by_reason.items()))
    print(
        f"total={len(rows)} full_restore_mc2={len(full_restore_mc2)} "
        f"misclassified_mc2={len(misclassified_mc2)} "
        f"fixed_keep_without_cache={len(fixed_keep_no_cache)} "
        f"full_restore_mc2_reasons={reason_text or 'none'}")
    for row in misclassified_mc2[:24]:
        print(
            f"  misclassified line={row['lineno']} rank={row['rank']} "
            f"ranks={row['ranks']} target={row['target']} active={row['active']} "
            f"fixed_keep={row['fixed_keep']} should_cache={row['should_cache']} "
            f"defer_preview={row['defer_preview']}")
    if not misclassified_mc2:
        for row in full_restore_mc2[:12]:
            print(
                f"  full_restore_mc2 line={row['lineno']} rank={row['rank']} "
                f"ranks={row['ranks']} predicted={row['predicted_reason']} "
                f"defer_preview={row['defer_preview']} "
                f"fixed_keep={row['fixed_keep']} should_cache={row['should_cache']}")


def _feature_summary(aux) -> dict[str, object]:
    rows = aux.get("feature_rows", [])
    restore_route_rows = aux.get("restore_stale_sweep_routes", [])
    features = sorted(set(row["feature"] for row in rows))
    ranks = sorted(set(row["rank"] for row in rows))
    return {
        "rows": rows,
        "restore_route_rows": restore_route_rows,
        "features": features,
        "ranks": ranks,
        "has_current_guard": "mode1_full_restore_defer_guard_v2" in features,
        "has_restore_stale_sweep_reason_route": any(
            row.get("feature")
            == "restore_stale_sweep_reason_mode1_full_restore"
            and row.get("reason") == "mode1_full_restore"
            for row in restore_route_rows),
        "stash_classify_enabled": any(
            row.get("stash_classify") for row in rows),
        "hccl_seq_digest_enabled": any(
            row.get("hccl_seq_digest") for row in rows),
        "defer_full_restore_enabled": any(
            row.get("defer_full_restore") for row in rows),
        "destroy_cpu_placeholder_enabled": any(
            row.get("destroy_cpu_placeholder_pg") for row in rows),
    }


def print_feature_summary(aux) -> None:
    summary = _feature_summary(aux)
    print("\n== mode1 lifecycle feature summary ==")
    rows = summary["rows"]
    if not rows:
        print("none")
        return
    print(
        f"rows={len(rows)} ranks={summary['ranks']} "
        f"features={summary['features']} "
        f"current_guard={int(bool(summary['has_current_guard']))} "
        f"restore_stale_sweep_routes={len(summary['restore_route_rows'])} "
        f"restore_reason_route={int(bool(summary['has_restore_stale_sweep_reason_route']))} "
        f"stash_classify={int(bool(summary['stash_classify_enabled']))} "
        f"hccl_seq_digest={int(bool(summary['hccl_seq_digest_enabled']))} "
        f"defer_full_restore={int(bool(summary['defer_full_restore_enabled']))} "
        f"destroy_cpu_placeholder={int(bool(summary['destroy_cpu_placeholder_enabled']))}")
    for row in rows[:16]:
        print(
            f"  line={row['lineno']} rank={row['rank']} "
            f"feature={row['feature']} context={row['context']} "
                f"stash_classify={row['stash_classify']} "
                f"hccl_seq_digest={row['hccl_seq_digest']} "
                f"defer_full_restore={row['defer_full_restore']} "
                f"destroy_cpu_placeholder={row.get('destroy_cpu_placeholder_pg')}")


def print_log_feature_status(aux) -> None:
    summary = _feature_summary(aux)
    rows = summary["rows"]
    if not rows:
        print(
            "feature_status=missing_current_guard "
            "action=log has no Elastic mode1 lifecycle feature marker; "
            "treat HCCL-order conclusions as unavailable for this run.")
        return
    status = "current" if summary["has_current_guard"] else "stale_or_unknown"
    print(
        f"feature_status={status} feature_rows={len(rows)} "
        f"feature_ranks={summary['ranks']} "
        f"current_guard={int(bool(summary['has_current_guard']))} "
        f"restore_stale_sweep_routes={len(summary['restore_route_rows'])} "
        f"restore_reason_route={int(bool(summary['has_restore_stale_sweep_reason_route']))} "
        f"stash_classify={int(bool(summary['stash_classify_enabled']))} "
        f"hccl_seq_digest={int(bool(summary['hccl_seq_digest_enabled']))} "
        f"defer_full_restore={int(bool(summary['defer_full_restore_enabled']))} "
        f"destroy_cpu_placeholder={int(bool(summary['destroy_cpu_placeholder_enabled']))}")


def print_plain_stash_release_summary(aux, slow_ms: float) -> None:
    print("\n== plain stashed group releases ==")
    rows = aux["stash_releases"]
    if not rows:
        print("none")
        return
    slow_rows = _plain_slow_stash_release_rows(aux, slow_ms)
    print(f"total={len(rows)} slow={len(slow_rows)}")
    if not slow_rows:
        return
    for row in sorted(slow_rows,
                      key=lambda item: item["destroy_ms"],
                      reverse=True)[:24]:
        print(
            f"line={row['lineno']} rank={row['rank']} attr={row['attr']} "
            f"kind={row['kind']} ranks={row['ranks']} reason={row['reason']} "
            f"destroy_ms={row['destroy_ms']:.2f} cleanup_ms={row['cleanup_ms']:.2f}")


def print_restore_rollout_summary(aux, slow_ms: float) -> None:
    print("\n== restore and rollout summary ==")
    if aux["rollouts"]:
        values = ", ".join(
            f"#{index + 1}@line{row['lineno']}={row['seconds']:.2f}s"
            for index, row in enumerate(aux["rollouts"]))
        print(f"rollouts={values}")
    else:
        print("rollouts=none")

    restores = aux["restores"]
    if not restores:
        print("restores=none")
    else:
        clusters = []
        for row in restores:
            if not clusters or row["lineno"] - clusters[-1][-1]["lineno"] > 80:
                clusters.append([row])
            else:
                clusters[-1].append(row)
        for index, cluster in enumerate(clusters, 1):
            max_total = max(row["total_ms"] for row in cluster)
            max_rebuild = max(row["rebuild_ms"] for row in cluster)
            slow_ranks = sorted(
                row["rank"] for row in cluster
                if row["total_ms"] >= slow_ms or row["rebuild_ms"] >= slow_ms)
            print(
                f"restore_cluster={index} lines={cluster[0]['lineno']}-{cluster[-1]['lineno']} "
                f"ranks={len(cluster)} max_rebuild_ms={max_rebuild:.2f} "
                f"max_total_ms={max_total:.2f} slow_ranks={slow_ranks}")

    if aux["direct_destroy_fallbacks"]:
        print("\n== direct destroy fallbacks ==")
        for row in aux["direct_destroy_fallbacks"][:40]:
            print(f"line={row['lineno']} {row['line']}")
    else:
        print("\ndirect_destroy_fallbacks=none")


def _rollout_windows(aux):
    rollouts = aux["rollouts"]
    windows = []
    start = 1
    for index, row in enumerate(rollouts, 1):
        windows.append({
            "label": f"rollout_{index}",
            "start": start,
            "end": row["lineno"],
            "rollout_seconds": row["seconds"],
        })
        start = row["lineno"] + 1
    windows.append({
        "label": f"after_rollout_{len(rollouts)}",
        "start": start,
        "end": 10**18,
        "rollout_seconds": None,
    })
    return windows


def _rows_in_window(rows, window):
    return [
        row for row in rows
        if window["start"] <= row["lineno"] <= window["end"]
    ]


def _rank_count_span(rows, ranks=range(16), key_fn=None) -> dict[str, object]:
    """Return min/max per-rank event counts for quick divergence checks."""
    counter = collections.Counter()
    for row in rows:
        if key_fn is not None and not key_fn(row):
            continue
        counter[int(row["rank"])] += 1
    counts = [counter[int(rank)] for rank in ranks]
    return {
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "missing": [int(rank) for rank in ranks if counter[int(rank)] == 0],
        "counts": counts,
    }


def _hccl_domain_count_span_summary(rows) -> str:
    if not rows:
        return "none"
    parts = []
    domains = sorted({
        row.get("domain") or _hccl_counter_domain(row)
        for row in rows
    })
    for domain in domains:
        create_span = _rank_count_span(
            rows,
            key_fn=lambda row, domain=domain: (
                (row.get("domain") or _hccl_counter_domain(row)) == domain
                and row.get("op") == "create"
            ),
        )
        destroy_span = _rank_count_span(
            rows,
            key_fn=lambda row, domain=domain: (
                (row.get("domain") or _hccl_counter_domain(row)) == domain
                and row.get("op") == "destroy"
            ),
        )
        parts.append(
            f"{domain}:create={create_span['min']}-{create_span['max']},"
            f"destroy={destroy_span['min']}-{destroy_span['max']}"
        )
    return ";".join(parts)


def collect_hccl_count_drifts(rows):
    """Find per-domain create/destroy count drift across ranks.

    A seq-signature mismatch proves that the same rank-local seq number maps to
    different groups.  Count drift is a slightly weaker but still useful signal:
    some ranks created/destroyed a different number of HCCL/Gloo PGs in the
    window, so a later collective may wait on unmatched communicator state.
    """
    if not rows:
        return []
    drifts = []
    domains = sorted({
        row.get("domain") or _hccl_counter_domain(row)
        for row in rows
    })
    for domain in domains:
        for op in ("create", "destroy"):
            span = _rank_count_span(
                rows,
                key_fn=lambda row, domain=domain, op=op: (
                    (row.get("domain") or _hccl_counter_domain(row)) == domain
                    and row.get("op") == op
                ),
            )
            if span["min"] != span["max"]:
                drifts.append({
                    "domain": domain,
                    "op": op,
                    "min": span["min"],
                    "max": span["max"],
                    "missing": span["missing"],
                    "counts": span["counts"],
                })
    return drifts


def _unified_new_group_create_rows(hccl_events):
    """Merge real GroupCoordinator creates and placeholder creates.

    GroupCoordinator and worker_v1 use independent diagnostic counters, but
    torch.distributed.new_group() ultimately consumes one process-group create
    sequence in the same process.  For the mode=1 placeholder hypothesis, the
    important check is whether the Nth create observed by each rank represents
    the same backend/ranks shape, regardless of which diagnostic domain emitted
    the row.
    """
    rows = []
    for row in sorted(hccl_events, key=lambda item: item.get("lineno", 0)):
        if row.get("op") != "create":
            continue
        event = row.get("event") or ""
        if not event.endswith("_done"):
            continue
        if event not in (
                "groupcoordinator_device_new_group_done",
                "groupcoordinator_cpu_new_group_done",
                "non_member_device_new_group_done",
                "non_member_cpu_new_group_done",
        ):
            continue
        clone = dict(row)
        clone["domain"] = row.get("domain") or _hccl_counter_domain(row)
        rows.append(clone)
    counters = collections.Counter()
    for row in rows:
        rank = int(row["rank"])
        counters[rank] += 1
        row["unified_create_seq"] = counters[rank]
    return rows


def _unified_new_group_signature(row) -> tuple[str, str, str]:
    # Active ranks log GroupCoordinator creates while detached ranks log
    # placeholder creates. Those are expected to differ by event/reason; the
    # alignment invariant is that the Nth create consumes the same backend,
    # group name and rank set.
    return (
        row.get("backend") or "",
        _canonical_group_ranks_text(row.get("ranks") or ""),
        row.get("group_name") or "",
    )


def collect_unified_new_group_order_variants(hccl_events):
    rows = _unified_new_group_create_rows(hccl_events)
    variants = []
    max_seq = max((row.get("unified_create_seq", 0) for row in rows), default=0)
    for seq in range(1, max_seq + 1):
        seq_rows = [
            row for row in rows
            if row.get("unified_create_seq") == seq
        ]
        if len(seq_rows) <= 1:
            continue
        rows_by_signature = collections.defaultdict(list)
        for row in seq_rows:
            rows_by_signature[_unified_new_group_signature(row)].append(row)
        if len(rows_by_signature) <= 1:
            continue
        variants.append({
            "seq": seq,
            "first_lineno": min(row["lineno"] for row in seq_rows),
            "last_lineno": max(row["lineno"] for row in seq_rows),
            "rows_by_signature": rows_by_signature,
        })
    return sorted(variants, key=lambda item: (
        item["first_lineno"],
        item["seq"],
    ))


def _has_unified_new_group_create_rows(hccl_events) -> bool:
    return bool(_unified_new_group_create_rows(hccl_events))


def _suppress_domain_create_count_drift(
        hccl_count_drifts, hccl_events, unified_create_variants) -> bool:
    """Avoid false positives from the expected active/detached split.

    Active ranks emit GroupCoordinator create rows; detached ranks emit worker
    placeholder create rows. Per-domain create counts therefore differ by
    design. If the merged new_group stream is present and aligned, create-only
    per-domain count drift is not a standalone problem signal.
    """
    if not hccl_count_drifts:
        return False
    if unified_create_variants:
        return False
    if not _has_unified_new_group_create_rows(hccl_events):
        return False
    return all(row.get("op") == "create" for row in hccl_count_drifts)


def _suppress_rank_band_count_drift(
        hccl_count_drifts, shared_create_seq_variants,
        unified_create_variants) -> bool:
    """Suppress normal floor=4 rank-band count drift.

    Different rank bands retire different low-floor groups by construction.
    Count drift becomes actionable only when the create-order checks show the
    same rank-local create sequence mapped to different groups.
    """
    return bool(hccl_count_drifts) and not shared_create_seq_variants and not unified_create_variants


def _unified_new_group_order_variant_summary(variants, limit: int = 3) -> str:
    if not variants:
        return "none"
    parts = []
    for item in variants[:limit]:
        parts.append(
            f"seq{item['seq']}:variants{len(item['rows_by_signature'])}"
        )
    if len(variants) > limit:
        parts.append(f"...+{len(variants) - limit}")
    return ",".join(parts)


def collect_shared_create_seq_variants(hccl_events):
    """Find shared rank-local create seq numbers that map to different groups.

    GroupCoordinator and the worker non-member placeholder path share the same
    rank-local create counter in current code.  Unlike the domain-specific
    checker, this deliberately compares across both emitters.  A variant here
    is the strongest signal for the HCCL communicator numbering/order drift
    hypothesis: the same create seq number means different backend/ranks on
    different ranks.
    """
    rows = [
        row for row in hccl_events
        if row.get("op") == "create"
        and (row.get("event") or "").endswith("_done")
    ]
    variants = []
    max_seq = max((row.get("seq", 0) for row in rows), default=0)
    for seq in range(1, max_seq + 1):
        seq_rows = [row for row in rows if row.get("seq") == seq]
        if len(seq_rows) <= 1:
            continue
        rows_by_signature = collections.defaultdict(list)
        for row in seq_rows:
            rows_by_signature[_unified_new_group_signature(row)].append(row)
        if len(rows_by_signature) <= 1:
            continue
        variants.append({
            "seq": seq,
            "first_lineno": min(row["lineno"] for row in seq_rows),
            "last_lineno": max(row["lineno"] for row in seq_rows),
            "rows_by_signature": rows_by_signature,
        })
    return sorted(variants, key=lambda item: (
        item["first_lineno"],
        item["seq"],
    ))


def _group_backend(row) -> str | None:
    values = row.get("_extra_values")
    if values is None:
        values = _parse_key_values(row.get("extra", ""))
        row["_extra_values"] = values
    return values.get("backend")


def _group_backend_phase(row) -> str:
    event = row.get("event") or ""
    if event.endswith("_start"):
        return "start"
    if event.endswith("_done"):
        return "done"
    return event


def _group_backend_op(row) -> str:
    event = row.get("event") or ""
    if "destroy" in event or "release" in event:
        return "destroy"
    return "create"


def _group_backend_signature(row) -> tuple[str, str, str, str, str]:
    return (
        _group_backend_op(row),
        _group_backend_phase(row),
        row.get("kind") or "",
        _canonical_group_ranks_text(row.get("ranks") or ""),
        row.get("reason") or "",
    )


def _group_backend_seq_rows(events, backend: str):
    """Derive rank-local backend operation order from group digest rows.

    This is not the raw HCCL counter. It is a fallback diagnostic for older logs
    that lack "Elastic HCCL seq digest" rows: if the same per-rank Nth
    backend operation maps to different group signatures, communicator creation
    order is already suspicious at the Python lifecycle layer.
    """
    wanted_events = {
        "rebuild_create_start",
        "rebuild_create_done",
        "stash_release_start",
        "stash_release_done",
        "non_member_device_new_group_start",
        "non_member_device_new_group_done",
        "non_member_cpu_new_group_start",
        "non_member_cpu_new_group_done",
        "non_member_destroy_start",
        "non_member_destroy_done",
    }
    counters = collections.Counter()
    rows = []
    for row in sorted(events, key=lambda item: item.get("lineno", 0)):
        if row.get("event") not in wanted_events:
            continue
        if _group_backend(row) != backend:
            continue
        rank = int(row["rank"])
        op = _group_backend_op(row)
        counters[(rank, op)] += 1
        clone = dict(row)
        clone["backend"] = backend
        clone["op"] = op
        clone["backend_seq"] = counters[(rank, op)]
        rows.append(clone)
    return rows


def collect_group_backend_seq_variants(events, backend: str):
    rows = _group_backend_seq_rows(events, backend)
    variants = []
    for op in ("create", "destroy"):
        max_seq = max((row["backend_seq"] for row in rows
                       if row["op"] == op),
                      default=0)
        for seq in range(1, max_seq + 1):
            rows_by_phase = collections.defaultdict(list)
            for row in rows:
                if row["op"] != op or row["backend_seq"] != seq:
                    continue
                rows_by_phase[_group_backend_phase(row)].append(row)
            for phase, phase_rows in rows_by_phase.items():
                if len(phase_rows) <= 1:
                    continue
                rows_by_signature = collections.defaultdict(list)
                for row in phase_rows:
                    rows_by_signature[_group_backend_signature(row)].append(row)
                if len(rows_by_signature) <= 1:
                    continue
                variants.append({
                    "backend": backend,
                    "op": op,
                    "seq": seq,
                    "phase": phase,
                    "first_lineno": min(row["lineno"] for row in phase_rows),
                    "last_lineno": max(row["lineno"] for row in phase_rows),
                    "rows_by_signature": rows_by_signature,
                })
    return sorted(variants, key=lambda item: (
        item["first_lineno"],
        item["backend"],
        item["op"],
        item["seq"],
        item["phase"],
    ))


def _format_group_backend_signature(signature) -> str:
    op, phase, kind, ranks, reason = signature
    return (
        f"op={op} phase={phase} kind={kind} ranks={ranks} reason={reason}"
    )


def _group_backend_variant_summary(variants, limit: int = 3) -> str:
    if not variants:
        return "none"
    parts = []
    for item in variants[:limit]:
        parts.append(
            f"{item['backend']}:{item['op']}:seq{item['seq']}:"
            f"{item['phase']}:variants{len(item['rows_by_signature'])}"
        )
    if len(variants) > limit:
        parts.append(f"...+{len(variants) - limit}")
    return ",".join(parts)


def _hccl_counter_domain(row) -> str:
    """Separate independent diagnostic seq counters.

    worker_v1 and GroupCoordinator both emit "Elastic HCCL seq digest" rows,
    but their seq counters are independent.  Comparing the same numeric seq
    across those domains would create false positives when looking for real
    rank-local communicator order drift.
    """
    kind = row.get("kind") or ""
    reason = row.get("reason") or ""
    event = row.get("event") or ""
    if kind == "groupcoordinator" or reason == "GroupCoordinator":
        return "groupcoordinator"
    if event.startswith("groupcoordinator_"):
        return "groupcoordinator"
    return "worker"


def _counter_summary(rows, key: str, limit: int = 6) -> str:
    counter = collections.Counter(row.get(key) or "unknown" for row in rows)
    if not counter:
        return "none"
    return ",".join(
        f"{name}:{count}" for name, count in counter.most_common(limit))


def _rollout_boundary_step_summary(rows, limit: int = 8) -> str:
    steps = sorted({int(row["step"]) for row in rows})
    if not steps:
        return "none"
    text = ",".join(str(step) for step in steps[:limit])
    if len(steps) > limit:
        text += f",...+{len(steps) - limit}"
    return text


def _rollout_boundary_missing_done_summary(rows) -> str:
    if not rows:
        return "none"
    events_by_step = collections.defaultdict(set)
    for row in rows:
        events_by_step[int(row["step"])].add(str(row["event"]))
    missing = [
        step for step, events in sorted(events_by_step.items())
        if ("rollout_generate_start" in events
            and "rollout_generate_done" not in events)
    ]
    if not missing:
        return "none"
    return ",".join(str(step) for step in missing[:8])


def _direct_destroy_reason_kind_summary(rows, limit: int = 8) -> str:
    counter = collections.Counter(
        (
            row.get("reason") or "unknown",
            row.get("kind") or "unknown",
            row.get("fallback_kind") or "unknown",
        )
        for row in rows
    )
    if not counter:
        return "none"
    parts = []
    for (reason, kind, fallback_kind), count in counter.most_common(limit):
        parts.append(f"{reason}:{kind}:{fallback_kind}:{count}")
    if len(counter) > limit:
        parts.append(f"...+{len(counter) - limit}")
    return ",".join(parts)


def _full_restore_mc2_direct_destroy_fallback_rows(rows):
    return [
        row for row in rows
        if (row.get("reason") == "mode1_full_restore"
            and row.get("kind") == "mc2")
    ]


def _stash_release_reason_kind_summary(rows, limit: int = 8) -> str:
    counter = collections.Counter(
        (
            row.get("reason") or "unknown",
            row.get("kind") or "unknown",
            _canonical_group_ranks_text(row.get("ranks") or ""),
        )
        for row in rows
    )
    if not counter:
        return "none"
    parts = []
    for (reason, kind, ranks), count in counter.most_common(limit):
        parts.append(f"{reason}:{kind}:{ranks}:{count}")
    if len(counter) > limit:
        parts.append(f"...+{len(counter) - limit}")
    return ",".join(parts)


def _canonical_group_ranks_text(ranks_text: str) -> str:
    numbers = re.findall(r"-?\d+", ranks_text or "")
    if not numbers:
        return ranks_text or ""
    return "(" + ",".join(numbers) + ")"


def _group_event_summary(rows, limit: int = 8) -> str:
    counter = collections.Counter(_group_seq_key(row) for row in rows)
    if not counter:
        return "none"
    parts = []
    for (kind, group_name, ranks), count in counter.most_common(limit):
        parts.append(f"{kind}:{group_name}:{ranks}:{count}")
    if len(counter) > limit:
        parts.append(f"...+{len(counter) - limit}")
    return ";".join(parts)


def _rank_summary(rows, limit: int = 16) -> str:
    ranks = sorted({int(row["rank"]) for row in rows})
    if len(ranks) <= limit:
        return "[" + ",".join(str(rank) for rank in ranks) + "]"
    head = ",".join(str(rank) for rank in ranks[:limit])
    return f"[{head},...+{len(ranks) - limit}]"


def _cache_hit_placeholder_risk(cache_hit_rows, placeholder_rows):
    if not cache_hit_rows or not placeholder_rows:
        return None
    cache_by_key = collections.defaultdict(list)
    placeholder_by_key = collections.defaultdict(list)
    for row in cache_hit_rows:
        cache_by_key[_group_seq_key(row)].append(row)
    for row in placeholder_rows:
        placeholder_by_key[_group_seq_key(row)].append(row)
    common_keys = sorted(set(cache_by_key) & set(placeholder_by_key))
    if common_keys:
        for key in common_keys:
            first_cache = min(cache_by_key[key],
                              key=lambda row: row.get("lineno", 0))
            first_placeholder = min(placeholder_by_key[key],
                                    key=lambda row: row.get("lineno", 0))
            if first_cache.get("lineno", 0) > first_placeholder.get("lineno", 0):
                return {
                    "same_group": True,
                    "cache_after_placeholder": True,
                    "common_keys": common_keys,
                    "cache_rows": cache_by_key[key],
                    "placeholder_rows": placeholder_by_key[key],
                    "first_cache": first_cache,
                    "first_placeholder": first_placeholder,
                }
        return None

    first_placeholder = min(placeholder_rows,
                            key=lambda row: row.get("lineno", 0))
    cache_after_placeholder = [
        row for row in cache_hit_rows
        if row.get("lineno", 0) > first_placeholder.get("lineno", 0)
    ]
    if not cache_after_placeholder:
        return None
    first_cache = min(cache_after_placeholder,
                      key=lambda row: row.get("lineno", 0))
    return {
        "same_group": False,
        "cache_after_placeholder": True,
        "common_keys": [],
        "cache_rows": cache_hit_rows,
        "placeholder_rows": placeholder_rows,
        "first_cache": first_cache,
        "first_placeholder": first_placeholder,
    }


def _rank_latest_value_span(rows,
                            key: str,
                            ranks=range(16)) -> dict[str, object]:
    """Compare the latest checkpoint value per rank."""
    latest: dict[int, int] = {}
    for row in sorted(rows, key=lambda item: item.get("lineno", 0)):
        rank = int(row["rank"])
        if key not in row:
            continue
        try:
            latest[rank] = int(row[key])
        except (TypeError, ValueError):
            continue
    values = [latest[int(rank)] for rank in ranks if int(rank) in latest]
    return {
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "missing": [int(rank) for rank in ranks if int(rank) not in latest],
        "values": [latest.get(int(rank), None) for rank in ranks],
    }


def _first_mode1_seq_checkpoint_divergence(rows,
                                           ranks=range(16)) -> dict[str, object] | None:
    if not rows:
        return None
    rows_by_tag: dict[tuple[str, str, str], list[dict[str, object]]] = (
        collections.defaultdict(list))
    for row in rows:
        # shrink_done / restore_enter / restore_refresh_after are transitional
        # snapshots: active ranks may still hold a floor group while detached
        # ranks have already dropped it, and wrappers can briefly differ before
        # the full restore finishes. Only restore_done is a stable boundary that
        # should drive an actionable Python-side signature divergence.
        if str(row.get("tag") or "") != "restore_done":
            continue
        key = (
            str(row.get("tag") or "unknown"),
            str(row.get("requested") or "unknown"),
            str(row.get("active") or "unknown"),
        )
        rows_by_tag[key].append(row)
    for (tag, requested, active), tag_rows in sorted(rows_by_tag.items()):
        create_span = _rank_latest_value_span(tag_rows, "create_seq", ranks)
        destroy_span = _rank_latest_value_span(tag_rows, "destroy_seq", ranks)
        # Active and detached ranks intentionally have different lifecycle
        # counts in floor=4: ranks 0-7, 8-11 and 12-15 do not traverse the same
        # shrink levels. Treat create/destroy span as auxiliary evidence only.
        # A Python-side divergence is actionable here only when ranks in the
        # same detached bucket disagree on live/cache/registry signatures.
        signature_divergence = None
        rows_by_detached: dict[str, list[dict[str, object]]] = (
            collections.defaultdict(list))
        for row in tag_rows:
            rows_by_detached[str(row.get("detached"))].append(row)
        for detached, detached_rows in sorted(rows_by_detached.items()):
            live_variants = collections.Counter(
                row.get("live") or "unknown" for row in detached_rows)
            cached_variants = collections.Counter(
                row.get("cached") or "unknown" for row in detached_rows)
            registry_variants = collections.Counter(
                row.get("registry") or "unknown" for row in detached_rows)
            if (len(live_variants) > 1 or len(cached_variants) > 1
                    or len(registry_variants) > 1):
                signature_divergence = {
                    "detached": detached,
                    "live_variants": live_variants,
                    "cached_variants": cached_variants,
                    "registry_variants": registry_variants,
                }
                break
        if signature_divergence is not None:
            first_line = min(int(row.get("lineno", 0)) for row in tag_rows)
            return {
                "tag": tag,
                "requested": requested,
                "active": active,
                "first_lineno": first_line,
                "create_span": create_span,
                "destroy_span": destroy_span,
                "signature_detached": (
                    signature_divergence["detached"]
                    if signature_divergence is not None else "none"),
                "live_variants": (
                    signature_divergence["live_variants"]
                    if signature_divergence is not None else collections.Counter()),
                "cached_variants": (
                    signature_divergence["cached_variants"]
                    if signature_divergence is not None else collections.Counter()),
                "registry_variants": (
                    signature_divergence["registry_variants"]
                    if signature_divergence is not None else collections.Counter()),
            }
    return None


def print_by_rollout_summary(events,
                             aux,
                             slow_ms: float,
                             regression_ratio: float = 1.5) -> None:
    print("\n== by-rollout communication summary ==")
    windows = _rollout_windows(aux)
    if not aux["rollouts"]:
        print("rollout_output_time_s not found; reporting whole log as after_rollout_0")
    rows = []
    for window in windows:
        win_events = _rows_in_window(events, window)
        win_aux = {
            key: _rows_in_window(value, window)
            for key, value in aux.items()
            if isinstance(value, list)
        }
        full_restore_rows = _full_restore_mc2_release_rows(win_events)
        full_restore_destroy = max(
            (first_ms(row["extra"], "destroy_ms") or 0.0
             for row in full_restore_rows),
            default=0.0,
        )
        slow_stash_rows = _slow_stash_release_rows(win_events, slow_ms)
        plain_slow_stash_rows = _plain_slow_stash_release_rows(
            win_aux, slow_ms)
        plain_misclassified_rows = _plain_misclassified_floor_mc2_release_rows(
            win_aux, slow_ms)
        classify_misclassified_rows = _stash_classify_misclassified_mc2_rows(
            win_aux)
        classify_full_restore_rows = _stash_classify_full_restore_mc2_rows(
            win_aux)
        classify_fixed_keep_no_cache_rows = (
            _stash_classify_fixed_keep_without_cache_rows(win_aux))
        plain_full_restore_rows = _plain_full_restore_mc2_release_rows(win_aux)
        plain_full_restore_destroy = max(
            (row["destroy_ms"] for row in plain_full_restore_rows),
            default=0.0,
        )
        max_restore_total = max(
            (row["total_ms"] for row in win_aux.get("restores", [])),
            default=0.0,
        )
        max_restore_rebuild = max(
            (row["rebuild_ms"] for row in win_aux.get("restores", [])),
            default=0.0,
        )
        hccl_variants = collect_hccl_seq_variants(
            win_aux.get("hccl_events", []))
        actionable_hccl_variants = _actionable_hccl_seq_variants(
            hccl_variants)
        hccl_summary_variants = collect_hccl_summary_variants(
            win_aux.get("hccl_summaries", []))
        hccl_create_span = _rank_count_span(
            win_aux.get("hccl_events", []),
            key_fn=lambda row: row.get("op") == "create")
        hccl_destroy_span = _rank_count_span(
            win_aux.get("hccl_events", []),
            key_fn=lambda row: row.get("op") == "destroy")
        hccl_domain_spans = _hccl_domain_count_span_summary(
            win_aux.get("hccl_events", []))
        unified_create_variants = collect_unified_new_group_order_variants(
            win_aux.get("hccl_events", []))
        shared_create_seq_variants = collect_shared_create_seq_variants(
            win_aux.get("hccl_events", []))
        suppress_domain_create_count_drift = _suppress_domain_create_count_drift(
            collect_hccl_count_drifts(win_aux.get("hccl_events", [])),
            win_aux.get("hccl_events", []),
            unified_create_variants,
        )
        derived_hccl_variants = collect_group_backend_seq_variants(
            win_events, "hccl")
        derived_gloo_variants = collect_group_backend_seq_variants(
            win_events, "gloo")
        moe_missing = _moe_dispatch_missing_done_rows(
            win_aux.get("moe_dispatch", []))
        moe_slow = _moe_dispatch_slow_done_rows(
            win_aux.get("moe_dispatch", []), slow_ms)
        max_moe_prepare = max(
            (row["elapsed_ms_value"] or 0.0 for row in moe_slow),
            default=0.0,
        )
        source_fallbacks = win_aux.get("source_fallbacks", [])
        source_unavailable = win_aux.get("source_unavailable", [])
        payload_skips = win_aux.get("payload_skips", [])
        preload_skips = win_aux.get("preload_skips", [])
        detached_advances = win_aux.get("detached_advances", [])
        rollout_boundaries = win_aux.get("rollout_boundaries", [])
        rollout_boundary_events = _counter_summary(
            rollout_boundaries, "event")
        rollout_boundary_steps = _rollout_boundary_step_summary(
            rollout_boundaries)
        rollout_boundary_missing_done = (
            _rollout_boundary_missing_done_summary(rollout_boundaries))
        mode1_seq_checkpoints = win_aux.get("mode1_seq_checkpoints", [])
        mode1_seq_tags = _counter_summary(mode1_seq_checkpoints, "tag")
        mode1_seq_create_span = _rank_latest_value_span(
            mode1_seq_checkpoints, "create_seq")
        mode1_seq_destroy_span = _rank_latest_value_span(
            mode1_seq_checkpoints, "destroy_seq")
        mode1_seq_divergence = _first_mode1_seq_checkpoint_divergence(
            mode1_seq_checkpoints)
        max_deferred_device_placeholders = max(
            (int(row.get("deferred_device_placeholders") or 0)
             for row in mode1_seq_checkpoints),
            default=0,
        )
        max_deferred_cpu_placeholders = max(
            (int(row.get("deferred_cpu_placeholders") or 0)
             for row in mode1_seq_checkpoints),
            default=0,
        )
        direct_destroy_fallbacks = win_aux.get("direct_destroy_fallbacks", [])
        direct_destroy_reason_summary = _counter_summary(
            direct_destroy_fallbacks, "reason")
        direct_destroy_kind_summary = _counter_summary(
            direct_destroy_fallbacks, "fallback_kind")
        direct_destroy_reason_kind_summary = (
            _direct_destroy_reason_kind_summary(direct_destroy_fallbacks))
        full_restore_mc2_direct_destroy_fallbacks = (
            _full_restore_mc2_direct_destroy_fallback_rows(
                direct_destroy_fallbacks))
        slow_stash_reason_kind_summary = _stash_release_reason_kind_summary(
            plain_slow_stash_rows)
        forced_full_restore_defer = win_aux.get(
            "forced_full_restore_defer", [])
        forced_floor_group_defer = win_aux.get(
            "forced_floor_group_defer", [])
        restore_stale_sweep_routes = win_aux.get(
            "restore_stale_sweep_routes", [])
        non_member_reuse = [
            row for row in win_events
            if row["event"] == "non_member_reuse"
        ]
        non_member_signature_done = [
            row for row in win_events
            if row["event"] == "non_member_signature_done"
        ]
        non_member_forced_placeholder = [
            row for row in non_member_signature_done
            if "force_placeholder_new_group=1" in row["extra"]
        ]
        rebuild_cache_hits = [
            row for row in win_events
            if row["event"] == "rebuild_cache_hit"
        ]
        rebuild_cache_bypasses = [
            row for row in win_events
            if row["event"] == "rebuild_cache_bypass_for_seq_alignment"
        ]
        rebuild_create_done = [
            row for row in win_events
            if row["event"] == "rebuild_create_done"
        ]
        non_member_device_new_group_done = [
            row for row in win_events
            if row["event"] == "non_member_device_new_group_done"
        ]
        non_member_cpu_new_group_done = [
            row for row in win_events
            if row["event"] == "non_member_cpu_new_group_done"
        ]
        cache_hit_placeholder_risk = _cache_hit_placeholder_risk(
            rebuild_cache_hits, non_member_device_new_group_done)
        rollout_text = (
            f"{window['rollout_seconds']:.2f}s"
            if window["rollout_seconds"] is not None else "NA")
        row_summary = {
            "label": window["label"],
            "rollout_seconds": window["rollout_seconds"],
            "full_restore_destroy": max(full_restore_destroy,
                                         plain_full_restore_destroy),
            "slow_stash": len(slow_stash_rows) + len(plain_slow_stash_rows),
            "misclassified_mc2": len(plain_misclassified_rows),
            "classify_misclassified_mc2": len(classify_misclassified_rows),
            "classify_full_restore_mc2": len(classify_full_restore_rows),
            "classify_fixed_keep_no_cache": len(classify_fixed_keep_no_cache_rows),
            "max_restore_rebuild": max_restore_rebuild,
            "max_restore_total": max_restore_total,
            "hccl_variants": len(hccl_variants),
            "actionable_hccl_variants": len(actionable_hccl_variants),
            "hccl_summary_variants": len(hccl_summary_variants),
            "hccl_create_min": hccl_create_span["min"],
            "hccl_create_max": hccl_create_span["max"],
            "hccl_destroy_min": hccl_destroy_span["min"],
            "hccl_destroy_max": hccl_destroy_span["max"],
            "unified_create_variants": len(unified_create_variants),
            "shared_create_seq_variants": len(shared_create_seq_variants),
            "suppress_domain_create_count_drift": int(
                suppress_domain_create_count_drift),
            "derived_hccl_variants": len(derived_hccl_variants),
            "derived_gloo_variants": len(derived_gloo_variants),
            "moe_missing": len(moe_missing),
            "moe_slow": len(moe_slow),
            "max_moe_prepare": max_moe_prepare,
            "source_fallbacks": len(source_fallbacks),
            "source_unavailable": len(source_unavailable),
            "payload_skips": len(payload_skips),
            "preload_skips": len(preload_skips),
            "detached_advances": len(detached_advances),
            "rollout_boundaries": len(rollout_boundaries),
            "rollout_boundary_events": rollout_boundary_events,
            "rollout_boundary_steps": rollout_boundary_steps,
            "rollout_boundary_missing_done": rollout_boundary_missing_done,
            "mode1_seq_checkpoints": len(mode1_seq_checkpoints),
            "mode1_seq_create_min": mode1_seq_create_span["min"],
            "mode1_seq_create_max": mode1_seq_create_span["max"],
            "mode1_seq_destroy_min": mode1_seq_destroy_span["min"],
            "mode1_seq_destroy_max": mode1_seq_destroy_span["max"],
            "max_deferred_device_placeholders": max_deferred_device_placeholders,
            "max_deferred_cpu_placeholders": max_deferred_cpu_placeholders,
            "direct_destroy_fallbacks": len(direct_destroy_fallbacks),
            "direct_destroy_reasons": direct_destroy_reason_summary,
            "direct_destroy_kinds": direct_destroy_kind_summary,
            "direct_destroy_reason_kinds": direct_destroy_reason_kind_summary,
            "full_restore_mc2_direct_destroy_fallbacks": (
                len(full_restore_mc2_direct_destroy_fallbacks)),
            "slow_stash_reason_kinds": slow_stash_reason_kind_summary,
            "forced_full_restore_defer": len(forced_full_restore_defer),
            "forced_floor_group_defer": len(forced_floor_group_defer),
            "restore_stale_sweep_routes": len(restore_stale_sweep_routes),
            "non_member_reuse": len(non_member_reuse),
            "non_member_signature_done": len(non_member_signature_done),
            "non_member_forced_placeholder": len(non_member_forced_placeholder),
            "rebuild_cache_hits": len(rebuild_cache_hits),
            "rebuild_cache_bypasses": len(rebuild_cache_bypasses),
            "rebuild_create_done": len(rebuild_create_done),
            "non_member_device_new_group_done": len(non_member_device_new_group_done),
            "non_member_cpu_new_group_done": len(non_member_cpu_new_group_done),
            "cache_hit_placeholder_risk": int(cache_hit_placeholder_risk is not None),
            "cache_hit_placeholder_same_group": int(
                bool(cache_hit_placeholder_risk
                     and cache_hit_placeholder_risk["same_group"])),
        }
        rows.append(row_summary)
        print(
            f"{window['label']} lines={window['start']}-{window['end']} "
            f"rollout_time={rollout_text} "
            f"events={len(win_events)} hccl={len(win_aux.get('hccl_events', []))} "
            f"moe={len(win_aux.get('moe_dispatch', []))} "
            f"max_full_restore_mc2_destroy_ms={full_restore_destroy:.2f} "
            f"plain_full_restore_mc2_destroy_ms={plain_full_restore_destroy:.2f} "
            f"slow_stash={len(slow_stash_rows) + len(plain_slow_stash_rows)} "
            f"misclassified_mc2={len(plain_misclassified_rows)} "
            f"classify_full_restore_mc2={len(classify_full_restore_rows)} "
            f"classify_misclassified_mc2={len(classify_misclassified_rows)} "
            f"classify_fixed_keep_no_cache={len(classify_fixed_keep_no_cache_rows)} "
            f"max_restore_rebuild_ms={max_restore_rebuild:.2f} "
            f"max_restore_total_ms={max_restore_total:.2f} "
            f"hccl_seq_variants={len(hccl_variants)} "
            f"actionable_hccl_seq_variants={len(actionable_hccl_variants)} "
            f"hccl_summary_variants={len(hccl_summary_variants)} "
            f"hccl_create_count_span={hccl_create_span['min']}-{hccl_create_span['max']} "
            f"hccl_destroy_count_span={hccl_destroy_span['min']}-{hccl_destroy_span['max']} "
            f"hccl_domain_count_spans={hccl_domain_spans} "
            f"unified_create_order_variants={len(unified_create_variants)} "
            f"unified_create_order_variant_summary={_unified_new_group_order_variant_summary(unified_create_variants)} "
            f"shared_create_seq_variants={len(shared_create_seq_variants)} "
            f"shared_create_seq_variant_summary={_unified_new_group_order_variant_summary(shared_create_seq_variants)} "
            f"suppress_domain_create_count_drift={int(suppress_domain_create_count_drift)} "
            f"derived_hccl_backend_order_variants={len(derived_hccl_variants)} "
            f"derived_hccl_backend_order_variant_summary={_group_backend_variant_summary(derived_hccl_variants)} "
            f"derived_gloo_backend_order_variants={len(derived_gloo_variants)} "
            f"derived_gloo_backend_order_variant_summary={_group_backend_variant_summary(derived_gloo_variants)} "
            f"direct_destroy_fallbacks={len(direct_destroy_fallbacks)} "
            f"direct_destroy_reasons={direct_destroy_reason_summary} "
            f"direct_destroy_kinds={direct_destroy_kind_summary} "
            f"direct_destroy_reason_kinds={direct_destroy_reason_kind_summary} "
            f"full_restore_mc2_direct_destroy_fallbacks={len(full_restore_mc2_direct_destroy_fallbacks)} "
            f"slow_stash_reason_kinds={slow_stash_reason_kind_summary} "
            f"source_fallbacks={len(source_fallbacks)} "
            f"source_unavailable={len(source_unavailable)} "
            f"forced_full_restore_defer={len(forced_full_restore_defer)} "
            f"forced_floor_group_defer={len(forced_floor_group_defer)} "
            f"restore_stale_sweep_routes={len(restore_stale_sweep_routes)} "
            f"rebuild_cache_hits={len(rebuild_cache_hits)} "
            f"rebuild_cache_hit_groups={_group_event_summary(rebuild_cache_hits)} "
            f"rebuild_cache_bypasses={len(rebuild_cache_bypasses)} "
            f"rebuild_cache_bypass_groups={_group_event_summary(rebuild_cache_bypasses)} "
            f"rebuild_create_done={len(rebuild_create_done)} "
            f"rebuild_create_groups={_group_event_summary(rebuild_create_done)} "
            f"non_member_device_new_group_done={len(non_member_device_new_group_done)} "
            f"non_member_device_groups={_group_event_summary(non_member_device_new_group_done)} "
            f"non_member_cpu_new_group_done={len(non_member_cpu_new_group_done)} "
            f"non_member_reuse={len(non_member_reuse)} "
            f"non_member_signature_done={len(non_member_signature_done)} "
            f"non_member_forced_placeholder={len(non_member_forced_placeholder)} "
            f"payload_skips={len(payload_skips)} "
            f"preload_skips={len(preload_skips)} "
            f"detached_advances={len(detached_advances)} "
            f"rollout_boundaries={len(rollout_boundaries)} "
            f"rollout_boundary_events={rollout_boundary_events} "
            f"rollout_boundary_steps={rollout_boundary_steps} "
            f"rollout_boundary_missing_done={rollout_boundary_missing_done} "
            f"mode1_seq_checkpoints={len(mode1_seq_checkpoints)} "
            f"mode1_seq_tags={mode1_seq_tags} "
            f"mode1_seq_create_span={mode1_seq_create_span['min']}-{mode1_seq_create_span['max']} "
            f"mode1_seq_destroy_span={mode1_seq_destroy_span['min']}-{mode1_seq_destroy_span['max']} "
            f"max_deferred_device_placeholders={max_deferred_device_placeholders} "
            f"max_deferred_cpu_placeholders={max_deferred_cpu_placeholders} "
            f"moe_missing_done={len(moe_missing)} "
            f"moe_slow_done={len(moe_slow)} "
            f"max_moe_prepare_ms={max_moe_prepare:.2f}")
        if cache_hit_placeholder_risk:
            first_cache = cache_hit_placeholder_risk["first_cache"]
            first_placeholder = cache_hit_placeholder_risk["first_placeholder"]
            print(
                "  cache_hit_placeholder_risk=1 "
                f"same_group={int(cache_hit_placeholder_risk['same_group'])} "
                f"cache_ranks={_rank_summary(cache_hit_placeholder_risk['cache_rows'])} "
                f"placeholder_ranks={_rank_summary(cache_hit_placeholder_risk['placeholder_rows'])} "
                f"first_cache_line={first_cache['lineno']} "
                f"first_cache={_short_group_seq_token(first_cache)} "
                f"first_placeholder_line={first_placeholder['lineno']} "
                f"first_placeholder={_short_group_seq_token(first_placeholder)}")
        if mode1_seq_divergence:
            print(
                "  first_mode1_seq_checkpoint_divergence "
                f"line={mode1_seq_divergence['first_lineno']} "
                f"tag={mode1_seq_divergence['tag']} "
                f"requested={mode1_seq_divergence.get('requested', 'unknown')} "
                f"active={mode1_seq_divergence.get('active', 'unknown')} "
                f"create_values={mode1_seq_divergence['create_span']['values']} "
                f"destroy_values={mode1_seq_divergence['destroy_span']['values']} "
                f"create_missing={mode1_seq_divergence['create_span']['missing']} "
                f"destroy_missing={mode1_seq_divergence['destroy_span']['missing']} "
                f"live_variants={len(mode1_seq_divergence['live_variants'])} "
                f"cached_variants={len(mode1_seq_divergence['cached_variants'])} "
                f"registry_variants={len(mode1_seq_divergence['registry_variants'])}")
        if hccl_variants:
            first = hccl_variants[0]
            print(
                f"  first_hccl_variant line={first['first_lineno']} "
                f"domain={first.get('domain', 'unknown')} "
                f"op={first['op']} seq={first['seq']} phase={first['phase']} "
                f"classification=diagnostic_rank_band_history")
        if (hccl_create_span["min"] != hccl_create_span["max"]
                or hccl_destroy_span["min"] != hccl_destroy_span["max"]):
            print(
                "  hccl_rank_count_divergence "
                f"create_counts={hccl_create_span['counts']} "
                f"destroy_counts={hccl_destroy_span['counts']} "
                f"create_missing={hccl_create_span['missing']} "
                f"destroy_missing={hccl_destroy_span['missing']}")
        if hccl_summary_variants:
            first = hccl_summary_variants[0]
            print(
                f"  first_hccl_summary_variant line={first['first_lineno']} "
                f"event={first['event']} ranks={first['ranks']}")
        if unified_create_variants:
            first = unified_create_variants[0]
            print(
                f"  first_unified_create_order_variant "
                f"line={first['first_lineno']} seq={first['seq']} "
                f"variant_count={len(first['rows_by_signature'])}")
            for signature, signature_rows in list(
                    first["rows_by_signature"].items())[:4]:
                ranks = sorted({int(row["rank"]) for row in signature_rows})
                print(
                    f"    ranks={ranks} backend={signature[0]} "
                    f"group_ranks={signature[1]} group_name={signature[2]}")
        if shared_create_seq_variants:
            first = shared_create_seq_variants[0]
            print(
                f"  first_shared_create_seq_variant "
                f"line={first['first_lineno']} seq={first['seq']} "
                f"variant_count={len(first['rows_by_signature'])}")
            for signature, signature_rows in list(
                    first["rows_by_signature"].items())[:4]:
                ranks = sorted({int(row["rank"]) for row in signature_rows})
                print(
                    f"    ranks={ranks} backend={signature[0]} "
                    f"group_ranks={signature[1]} group_name={signature[2]}")
        if derived_hccl_variants:
            first = derived_hccl_variants[0]
            print(
                f"  first_derived_hccl_backend_order_variant "
                f"line={first['first_lineno']} op={first['op']} "
                f"seq={first['seq']} phase={first['phase']} "
                f"variant_count={len(first['rows_by_signature'])}")
            for signature, signature_rows in list(
                    first["rows_by_signature"].items())[:4]:
                ranks = sorted({int(row["rank"]) for row in signature_rows})
                print(
                    f"    ranks={ranks} "
                    f"{_format_group_backend_signature(signature)}")
        if direct_destroy_fallbacks:
            first = direct_destroy_fallbacks[0]
            print(
                f"  first_direct_destroy_fallback line={first['lineno']} "
                f"rank={first.get('rank')} kind={first.get('kind')} "
                f"fallback_kind={first.get('fallback_kind')} "
                f"ranks={first.get('ranks')} reason={first.get('reason')} "
                f"target={first.get('target')} "
                f"allow_same_ranks={first.get('allow_same_ranks')} "
                f"defer_preview={first.get('defer_preview')} "
                f"defer_env={first.get('defer_env')} "
                f"defer_sizes={first.get('defer_sizes')}")
        if full_restore_mc2_direct_destroy_fallbacks:
            first = full_restore_mc2_direct_destroy_fallbacks[0]
            print(
                f"  first_full_restore_mc2_direct_destroy_fallback "
                f"line={first['lineno']} rank={first.get('rank')} "
                f"ranks={first.get('ranks')} target={first.get('target')} "
                f"allow_same_ranks={first.get('allow_same_ranks')} "
                f"defer_preview={first.get('defer_preview')} "
                f"defer_env={first.get('defer_env')} "
                f"defer_sizes={first.get('defer_sizes')}")
        if source_fallbacks:
            first = source_fallbacks[0]
            print(
                f"  first_source_fallback line={first['lineno']} "
                f"rank={first['rank']} source={first['source']} "
                f"reason={first['reason']}")
        if source_unavailable:
            first = source_unavailable[0]
            print(
                f"  first_source_unavailable line={first['lineno']} "
                f"rank={first['rank']} source={first['source']} "
                f"reason={first['reason']}")
        if forced_full_restore_defer:
            first = forced_full_restore_defer[0]
            print(
                f"  first_forced_full_restore_defer line={first['lineno']} "
                f"rank={first['rank']} kind={first['kind']} "
                f"ranks={first['ranks']} keep={first['keep']} "
                f"env_full_restore={first['defer_full_restore_env']} "
                f"env_group_destroy={first['defer_group_destroy_env']}")
        if forced_floor_group_defer:
            first = forced_floor_group_defer[0]
            print(
                f"  first_forced_floor_group_defer line={first['lineno']} "
                f"rank={first['rank']} kind={first['kind']} "
                f"ranks={first['ranks']} keep={first['keep']} "
                f"reason={first['reason']} defer_sizes={first['defer_sizes']}")
        if restore_stale_sweep_routes:
            first = restore_stale_sweep_routes[0]
            print(
                f"  first_restore_stale_sweep_route line={first['lineno']} "
                f"rank={first['rank']} reason={first['reason']} "
                f"feature={first['feature']} "
                f"defer_full_restore={first['defer_full_restore']} "
                f"defer_group_destroy={first['defer_group_destroy']} "
                f"force_non_member_placeholder={first['force_non_member_placeholder']}")
        if non_member_reuse:
            first = non_member_reuse[0]
            print(
                f"  first_non_member_reuse line={first['lineno']} "
                f"rank={first['rank']} kind={first['kind']} "
                f"name={first['group_name']} ranks={first['ranks']} "
                f"extra={first['extra']}")
        if non_member_forced_placeholder:
            first = non_member_forced_placeholder[0]
            print(
                f"  first_non_member_forced_placeholder line={first['lineno']} "
                f"rank={first['rank']} kind={first['kind']} "
                f"name={first['group_name']} ranks={first['ranks']} "
                f"extra={first['extra']}")
        if payload_skips:
            first = payload_skips[0]
            print(
                f"  first_payload_skip line={first['lineno']} "
                f"rank={first['rank']} active={first['active']} "
                f"source={first['source']} detached={first['detached']}")
        if preload_skips:
            first = preload_skips[0]
            print(
                f"  first_preload_skip line={first['lineno']} "
                f"rank={first['rank']} active={first['active']} "
                f"source={first['source']} detached={first['detached']}")
        if detached_advances:
            first = detached_advances[0]
            print(
                f"  first_detached_advance line={first['lineno']} "
                f"rank={first['rank']} active={first['active']}")
        if plain_misclassified_rows:
            first = plain_misclassified_rows[0]
            print(
                f"  first_misclassified_mc2 line={first['lineno']} "
                f"rank={first['rank']} ranks={first['ranks']} "
                f"destroy_ms={first['destroy_ms']:.2f}")
        if classify_misclassified_rows:
            first = classify_misclassified_rows[0]
            print(
                f"  first_classify_misclassified_mc2 line={first['lineno']} "
                f"rank={first['rank']} ranks={first['ranks']} "
                f"target={first['target']} fixed_keep={first['fixed_keep']} "
                f"should_cache={first['should_cache']} "
                f"defer_preview={first['defer_preview']}")
        if moe_missing:
            first = moe_missing[0]
            print(
                f"  first_moe_missing line={first['lineno']} "
                f"rank={first['rank']} seq={first['seq']} "
                f"stage={first['stage']} method={first['method']}")
        if moe_slow:
            first = max(moe_slow,
                        key=lambda row: row["elapsed_ms_value"] or 0.0)
            print(
                f"  slowest_moe_prepare line={first['lineno']} "
                f"rank={first['rank']} seq={first['seq']} "
                f"prepare_ms={(first['elapsed_ms_value'] or 0.0):.2f} "
                f"stage={first['stage']} method={first['method']}")
    rollout_rows = [
        row for row in rows
        if row["rollout_seconds"] is not None
    ]
    if rollout_rows:
        baseline = max(rollout_rows[0]["rollout_seconds"] or 0.0, 1e-9)
        first_bad = None
        for row in rollout_rows[1:]:
            ratio = (row["rollout_seconds"] or 0.0) / baseline
            if (ratio >= regression_ratio
                    or row["full_restore_destroy"] >= slow_ms
                    or row["misclassified_mc2"] > 0
                    or row["classify_misclassified_mc2"] > 0
                    or row["actionable_hccl_variants"] > 0
                    or row["unified_create_variants"] > 0
                    or row["shared_create_seq_variants"] > 0
                    or (row["hccl_create_min"] != row["hccl_create_max"]
                        and not row["suppress_domain_create_count_drift"]
                        and (row["hccl_variants"] > 0
                             or row["unified_create_variants"] > 0
                             or row["shared_create_seq_variants"] > 0))
                    or (row["hccl_destroy_min"] != row["hccl_destroy_max"]
                        and (row["actionable_hccl_variants"] > 0
                             or row["shared_create_seq_variants"] > 0
                             or row["moe_missing"] > 0
                             or row["moe_slow"] > 0))
                    or row["derived_hccl_variants"] > 0
                    or row["full_restore_mc2_direct_destroy_fallbacks"] > 0
                    or row["source_unavailable"] > 0
                    or row["forced_full_restore_defer"] > 0
                    or row["forced_floor_group_defer"] > 0
                    or row["restore_stale_sweep_routes"] == 0
                    or row["cache_hit_placeholder_risk"] > 0
                    or row["non_member_reuse"] > 0
                    or (row["payload_skips"] > 0
                        and row["detached_advances"] == 0)
                    or row["moe_missing"] > 0
                    or row["moe_slow"] > 0):
                first_bad = (row, ratio)
                break
        if first_bad is None:
            print(
                f"rollout_regression status=none baseline={baseline:.2f}s "
                f"ratio_threshold={regression_ratio:.2f}")
        else:
            row, ratio = first_bad
            print(
                f"rollout_regression status=detected first_bad={row['label']} "
                f"baseline={baseline:.2f}s current={row['rollout_seconds']:.2f}s "
                f"ratio={ratio:.2f} ratio_threshold={regression_ratio:.2f} "
                f"full_restore_destroy_ms={row['full_restore_destroy']:.2f} "
                f"restore_rebuild_ms={row['max_restore_rebuild']:.2f} "
                f"restore_total_ms={row['max_restore_total']:.2f} "
                f"misclassified_mc2={row['misclassified_mc2']} "
                f"classify_misclassified_mc2={row['classify_misclassified_mc2']} "
                f"hccl_seq_variants={row['hccl_variants']} "
                f"hccl_summary_variants={row['hccl_summary_variants']} "
                f"unified_create_order_variants={row['unified_create_variants']} "
                f"shared_create_seq_variants={row['shared_create_seq_variants']} "
                f"suppress_domain_create_count_drift={row['suppress_domain_create_count_drift']} "
                f"hccl_create_count_span={row['hccl_create_min']}-{row['hccl_create_max']} "
                f"hccl_destroy_count_span={row['hccl_destroy_min']}-{row['hccl_destroy_max']} "
                f"derived_hccl_backend_order_variants={row['derived_hccl_variants']} "
                f"derived_gloo_backend_order_variants={row['derived_gloo_variants']} "
                f"direct_destroy_fallbacks={row['direct_destroy_fallbacks']} "
                f"direct_destroy_reasons={row['direct_destroy_reasons']} "
                f"direct_destroy_kinds={row['direct_destroy_kinds']} "
                f"direct_destroy_reason_kinds={row['direct_destroy_reason_kinds']} "
                f"full_restore_mc2_direct_destroy_fallbacks={row['full_restore_mc2_direct_destroy_fallbacks']} "
                f"slow_stash_reason_kinds={row['slow_stash_reason_kinds']} "
                f"source_fallbacks={row['source_fallbacks']} "
                f"source_unavailable={row['source_unavailable']} "
                f"forced_full_restore_defer={row['forced_full_restore_defer']} "
                f"forced_floor_group_defer={row['forced_floor_group_defer']} "
                f"restore_stale_sweep_routes={row['restore_stale_sweep_routes']} "
                f"rebuild_cache_hits={row['rebuild_cache_hits']} "
                f"rebuild_cache_bypasses={row['rebuild_cache_bypasses']} "
                f"rebuild_create_done={row['rebuild_create_done']} "
                f"non_member_device_new_group_done={row['non_member_device_new_group_done']} "
                f"cache_hit_placeholder_risk={row['cache_hit_placeholder_risk']} "
                f"cache_hit_placeholder_same_group={row['cache_hit_placeholder_same_group']} "
                f"non_member_reuse={row['non_member_reuse']} "
                f"non_member_forced_placeholder={row['non_member_forced_placeholder']} "
                f"payload_skips={row['payload_skips']} "
                f"preload_skips={row['preload_skips']} "
                f"detached_advances={row['detached_advances']} "
                f"rollout_boundaries={row['rollout_boundaries']} "
                f"rollout_boundary_events={row['rollout_boundary_events']} "
                f"rollout_boundary_steps={row['rollout_boundary_steps']} "
                f"rollout_boundary_missing_done={row['rollout_boundary_missing_done']} "
                f"moe_missing_done={row['moe_missing']} "
                f"moe_slow_done={row['moe_slow']}")


def _moe_dispatch_missing_done_rows(moe_rows):
    starts = {}
    done_keys = set()
    for row in moe_rows:
        key = (row["rank"], row["seq"], row["layer"])
        if row["event"] == "prepare_start":
            starts[key] = row
        elif row["event"] == "prepare_done":
            done_keys.add(key)
    return [
        row for key, row in sorted(starts.items(), key=lambda item: item[1]["lineno"])
        if key not in done_keys
    ]


def _moe_dispatch_slow_done_rows(moe_rows, slow_ms: float):
    return [
        row for row in moe_rows
        if row["event"] == "prepare_done"
        and row["elapsed_ms_value"] is not None
        and row["elapsed_ms_value"] >= slow_ms
    ]


def print_moe_dispatch_summary(aux, slow_ms: float) -> None:
    rows = aux["moe_dispatch"]
    print("\n== MoE dispatch prepare summary ==")
    if not rows:
        print("none; run with VLLM_ASCEND_MOE_DISPATCH_SEQ_DIGEST_LOG=1")
        return
    by_rank = collections.defaultdict(list)
    for row in rows:
        by_rank[row["rank"]].append(row)
    for rank in sorted(by_rank):
        rank_rows = by_rank[rank]
        starts = [row for row in rank_rows if row["event"] == "prepare_start"]
        dones = [row for row in rank_rows if row["event"] == "prepare_done"]
        max_elapsed = max(
            (row["elapsed_ms_value"] or 0.0 for row in dones),
            default=0.0,
        )
        slow_count = sum(
            1 for row in dones
            if (row["elapsed_ms_value"] or 0.0) >= slow_ms)
        print(
            f"rank={rank:02d} prepare_start={len(starts)} "
            f"prepare_done={len(dones)} max_prepare_ms={max_elapsed:.2f} "
            f"slow={slow_count}")
    missing = _moe_dispatch_missing_done_rows(rows)
    if missing:
        print("\nmoe_dispatch_missing_done:")
        for row in missing[:40]:
            print(
                f"line={row['lineno']} rank={row['rank']} seq={row['seq']} "
                f"layer={row['layer']} stage={row['stage']} "
                f"selected={row['selected']} effective={row['effective']} "
                f"method={row['method']} active={row['active']}")
    slow_rows = _moe_dispatch_slow_done_rows(rows, slow_ms)
    if slow_rows:
        print("\nmoe_dispatch_slow_prepare_done:")
        for row in sorted(slow_rows,
                          key=lambda item: item["elapsed_ms_value"] or 0.0,
                          reverse=True)[:40]:
            print(
                f"line={row['lineno']} rank={row['rank']} seq={row['seq']} "
                f"prepare_ms={(row['elapsed_ms_value'] or 0.0):.2f} "
                f"layer={row['layer']} stage={row['stage']} "
                f"selected={row['selected']} effective={row['effective']} "
                f"method={row['method']} active={row['active']}")


def _full_restore_mc2_release_rows(events):
    return [
        row for row in events
        if row["event"] == "stash_release_done"
        and row["kind"] == "mc2"
        and row["attr"] == "_MC2"
        and row["reason"] == "mode1_full_restore"
    ]


def _slow_stash_release_rows(events, slow_ms: float):
    rows = []
    for row in events:
        if row["event"] != "stash_release_done":
            continue
        destroy_ms = first_ms(row["extra"], "destroy_ms")
        if destroy_ms is not None and destroy_ms >= slow_ms:
            rows.append(row)
    return rows


def _hccl_seq_variant_count(hccl_events) -> int:
    return len(collect_hccl_seq_variants(hccl_events))


def _actionable_hccl_seq_variants(hccl_variants):
    """Return raw HCCL seq variants that should fail the run.

    Floor=4 staged shrink naturally gives rank bands different destroy/history
    paths: ranks 0-7, 8-11 and 12-15 do not all retire the same low-floor
    groups. Those raw destroy/summary variants are useful diagnostics, but not
    failures when the unified create stream, registry and MoE dispatch are
    clean. Hard failures are checked separately by shared/unified create-order
    and MoE/destroy signals.
    """
    return []


def collect_hccl_summary_variants(hccl_summaries):
    """Find rank-local create/destroy counter divergence at lifecycle edges."""
    grouped = collections.defaultdict(list)
    for row in hccl_summaries:
        grouped[(row["event"], row["ranks"], row["reason"])].append(row)

    variants = []
    for (event, ranks, reason), rows in grouped.items():
        if len(rows) <= 1:
            continue
        rows_by_signature = collections.defaultdict(list)
        for row in rows:
            signature = (
                row["create_seq"],
                row["destroy_seq"],
            )
            rows_by_signature[signature].append(row)
        if len(rows_by_signature) <= 1:
            continue
        variants.append({
            "event": event,
            "ranks": ranks,
            "reason": reason,
            "first_lineno": min(row["lineno"] for row in rows),
            "last_lineno": max(row["lineno"] for row in rows),
            "rows_by_signature": rows_by_signature,
        })
    return sorted(variants, key=lambda item: item["first_lineno"])


def _summary_extra_digest(row) -> tuple[str | None, str | None, str | None, str | None]:
    values = row.get("_extra_values")
    if values is None:
        values = _parse_key_values(row.get("extra", ""))
        row["_extra_values"] = values
    return (
        values.get("group_name"),
        values.get("seen_before"),
        values.get("seen_after"),
        values.get("unseen_groups"),
    )


def _format_hccl_variant_signature(signature) -> str:
    op, phase, action, backend, ranks, reason = signature
    return (
        f"op={op} phase={phase} action={action} "
        f"backend={backend} ranks={ranks} reason={reason}"
    )


def collect_hccl_seq_variants(hccl_events):
    """Find rank-local HCCL/Gloo seq numbers that map to different groups.

    The suspected failure mode is not merely "rank A is slow"; it is that after
    one shrink/restore cycle, rank-local new_group/destroy call order diverges.
    That shows up as the same (op, seq, phase) carrying different backend/ranks
    signatures across ranks.
    """
    variants = []
    rows_by_domain = collections.defaultdict(list)
    for row in hccl_events:
        rows_by_domain[row.get("domain") or _hccl_counter_domain(row)].append(row)
    for domain, domain_rows in rows_by_domain.items():
        for op in ("create", "destroy"):
            max_seq = max((row["seq"] for row in domain_rows
                           if row["op"] == op),
                          default=0)
            for seq in range(1, max_seq + 1):
                rows_by_phase = collections.defaultdict(list)
                for row in domain_rows:
                    if row["op"] != op or row["seq"] != seq:
                        continue
                    phase = _normalize_hccl_event_for_seq(row)[1]
                    rows_by_phase[phase].append(row)
                for phase, rows in rows_by_phase.items():
                    if len(rows) <= 1:
                        continue
                    rows_by_signature = collections.defaultdict(list)
                    for row in rows:
                        rows_by_signature[_normalize_hccl_event_for_seq(row)].append(row)
                    if len(rows_by_signature) <= 1:
                        continue
                    variants.append({
                        "domain": domain,
                        "op": op,
                        "seq": seq,
                        "phase": phase,
                        "first_lineno": min(row["lineno"] for row in rows),
                        "last_lineno": max(row["lineno"] for row in rows),
                        "rows_by_signature": rows_by_signature,
                    })
    return sorted(
        variants,
        key=lambda item: (
            item["first_lineno"],
            item["domain"],
            item["op"],
            item["seq"],
            item["phase"],
        ),
    )


def _filter_aux_before_line(aux, max_lineno: int):
    return {
        "hccl_events": [
            row for row in aux["hccl_events"]
            if row["lineno"] <= max_lineno
        ],
        "hccl_summaries": [
            row for row in aux["hccl_summaries"]
            if row["lineno"] <= max_lineno
        ],
        "deferred": [
            row for row in aux["deferred"]
            if row["lineno"] <= max_lineno
        ],
        "retired": [
            row for row in aux["retired"]
            if row["lineno"] <= max_lineno
        ],
        "stash_releases": [
            row for row in aux["stash_releases"]
            if row["lineno"] <= max_lineno
        ],
        "stash_classify": [
            row for row in aux["stash_classify"]
            if row["lineno"] <= max_lineno
        ],
        "feature_rows": [
            row for row in aux["feature_rows"]
            if row["lineno"] <= max_lineno
        ],
        "forced_full_restore_defer": [
            row for row in aux.get("forced_full_restore_defer", [])
            if row["lineno"] <= max_lineno
        ],
        "forced_floor_group_defer": [
            row for row in aux.get("forced_floor_group_defer", [])
            if row["lineno"] <= max_lineno
        ],
        "restore_stale_sweep_routes": [
            row for row in aux.get("restore_stale_sweep_routes", [])
            if row["lineno"] <= max_lineno
        ],
        "moe_dispatch": [
            row for row in aux["moe_dispatch"]
            if row["lineno"] <= max_lineno
        ],
        "restores": [
            row for row in aux["restores"]
            if row["lineno"] <= max_lineno
        ],
        "rollouts": [
            row for row in aux["rollouts"]
            if row["lineno"] <= max_lineno
        ],
        "rollout_boundaries": [
            row for row in aux.get("rollout_boundaries", [])
            if row["lineno"] <= max_lineno
        ],
        "direct_destroy_fallbacks": [
            row for row in aux["direct_destroy_fallbacks"]
            if row["lineno"] <= max_lineno
        ],
        "source_fallbacks": [
            row for row in aux.get("source_fallbacks", [])
            if row["lineno"] <= max_lineno
        ],
        "source_unavailable": [
            row for row in aux.get("source_unavailable", [])
            if row["lineno"] <= max_lineno
        ],
        "payload_skips": [
            row for row in aux.get("payload_skips", [])
            if row["lineno"] <= max_lineno
        ],
        "preload_skips": [
            row for row in aux.get("preload_skips", [])
            if row["lineno"] <= max_lineno
        ],
        "detached_advances": [
            row for row in aux.get("detached_advances", [])
            if row["lineno"] <= max_lineno
        ],
    }


def print_early_fail_summary(events, aux, slow_ms: float) -> int:
    """Report whether the bad communication path is visible before rollout 1 ends."""
    print("\n== early failure summary ==")
    if not aux["rollouts"]:
        print("rollout_output_time_s not found; using entire log as current partial run")
        cutoff = 10**18
    else:
        cutoff = aux["rollouts"][0]["lineno"]
        print(
            f"cutoff=first_rollout_output line={cutoff} "
            f"seconds={aux['rollouts'][0]['seconds']:.2f}")
    early_events = [row for row in events if row["lineno"] <= cutoff]
    early_aux = _filter_aux_before_line(aux, cutoff)
    verdict, actions = diagnosis_verdict(early_events, early_aux, slow_ms)
    print(verdict)
    for action in actions:
        print(action)

    fallback_count = len(early_aux["direct_destroy_fallbacks"])
    if fallback_count:
        kinds = collections.Counter(
            row.get("kind", "unknown")
            for row in early_aux["direct_destroy_fallbacks"])
        print(
            "early_direct_destroy_fallbacks="
            + ",".join(f"{kind}:{count}" for kind, count in sorted(kinds.items())))
        for row in early_aux["direct_destroy_fallbacks"][:12]:
            print(f"  line={row['lineno']} kind={row.get('kind', 'unknown')} {row['line'][:260]}")
    else:
        print("early_direct_destroy_fallbacks=none")
    forced_count = len(early_aux.get("forced_full_restore_defer", []))
    route_count = len(early_aux.get("restore_stale_sweep_routes", []))
    if forced_count:
        print(f"early_forced_full_restore_defer={forced_count}")
        for row in early_aux["forced_full_restore_defer"][:12]:
            print(
                f"  line={row['lineno']} rank={row['rank']} "
                f"kind={row['kind']} ranks={row['ranks']} keep={row['keep']} "
                f"env_full_restore={row['defer_full_restore_env']} "
                f"env_group_destroy={row['defer_group_destroy_env']}")
    else:
        print("early_forced_full_restore_defer=none")
    if route_count:
        print(f"early_restore_stale_sweep_routes={route_count}")
        for row in early_aux["restore_stale_sweep_routes"][:12]:
            print(
                f"  line={row['lineno']} rank={row['rank']} "
                f"reason={row['reason']} feature={row['feature']} "
                f"defer_full_restore={row['defer_full_restore']} "
                f"defer_group_destroy={row['defer_group_destroy']} "
                f"force_non_member_placeholder={row['force_non_member_placeholder']}")
    else:
        print("early_restore_stale_sweep_routes=none")

    slow_restores = [
        row for row in early_aux["restores"]
        if row["total_ms"] >= slow_ms or row["rebuild_ms"] >= slow_ms
    ]
    if slow_restores:
        print(f"early_slow_restores={len(slow_restores)}")
        for row in slow_restores[:16]:
            print(
                f"  line={row['lineno']} rank={row['rank']} "
                f"rebuild_ms={row['rebuild_ms']:.2f} "
                f"refresh_ms={row['refresh_ms']:.2f} total_ms={row['total_ms']:.2f}")
    else:
        print("early_slow_restores=none")

    early_moe_missing = _moe_dispatch_missing_done_rows(
        early_aux["moe_dispatch"])
    early_moe_slow = _moe_dispatch_slow_done_rows(early_aux["moe_dispatch"],
                                                  slow_ms)
    early_classify_misclassified = _stash_classify_misclassified_mc2_rows(
        early_aux)
    if early_classify_misclassified:
        first = early_classify_misclassified[0]
        print(
            f"early_classify_misclassified_mc2={len(early_classify_misclassified)} "
            f"first_line={first['lineno']} rank={first['rank']} "
            f"ranks={first['ranks']} target={first['target']} "
            f"fixed_keep={first['fixed_keep']} should_cache={first['should_cache']}")
    else:
        print("early_classify_misclassified_mc2=none")
    print(
        f"early_moe_dispatch_events={len(early_aux['moe_dispatch'])} "
        f"missing_done={len(early_moe_missing)} slow_done={len(early_moe_slow)}")
    for row in early_moe_missing[:8]:
        print(
            f"  missing line={row['lineno']} rank={row['rank']} "
            f"seq={row['seq']} stage={row['stage']} method={row['method']}")

    variants = collect_hccl_seq_variants(early_aux["hccl_events"])
    if not early_aux["hccl_events"]:
        print("early_hccl_seq_variants=unknown_no_hccl_digest")
    elif not variants:
        print("early_hccl_seq_variants=none")
    else:
        first = variants[0]
        print(
            f"early_hccl_seq_variants={len(variants)} "
            f"first_line={first['first_lineno']} op={first['op']} "
            f"seq={first['seq']} phase={first['phase']}")
    return 0


def diagnosis_verdict(events, aux, slow_ms: float) -> tuple[str, list[str]]:
    restore_slow_ms = max(slow_ms * 30.0, 30000.0)
    full_restore_rows = _full_restore_mc2_release_rows(events)
    full_restore_destroy_ms = [
        first_ms(row["extra"], "destroy_ms") or 0.0
        for row in full_restore_rows
    ]
    max_full_restore_destroy = (
        max(full_restore_destroy_ms) if full_restore_destroy_ms else 0.0)

    full_restore_deferred_rows = [
        row for row in aux["deferred"]
        if row["kind"] == "mc2"
        and row["reason"] == "mode1_full_restore"
    ]
    plain_full_restore_rows = _plain_full_restore_mc2_release_rows(aux)
    plain_full_restore_destroy = max(
        (row["destroy_ms"] for row in plain_full_restore_rows),
        default=0.0,
    )
    plain_misclassified_rows = _plain_misclassified_floor_mc2_release_rows(
        aux, slow_ms)
    classify_misclassified_rows = _stash_classify_misclassified_mc2_rows(aux)
    classify_fixed_keep_no_cache_rows = (
        _stash_classify_fixed_keep_without_cache_rows(aux))
    plain_slow_stash_rows = _plain_slow_stash_release_rows(aux, slow_ms)
    max_plain_misclassified_destroy = max(
        (row["destroy_ms"] for row in plain_misclassified_rows),
        default=0.0,
    )
    max_plain_slow_stash_destroy = max(
        (row["destroy_ms"] for row in plain_slow_stash_rows),
        default=0.0,
    )
    max_deferred_device_destroy = max(
        ((row["device_pg_destroy_ms"] or 0.0)
         for row in full_restore_deferred_rows),
        default=0.0,
    )
    max_restore_total = max(
        (row["total_ms"] for row in aux["restores"]),
        default=0.0,
    )
    max_restore_rebuild = max(
        (row["rebuild_ms"] for row in aux["restores"]),
        default=0.0,
    )
    hccl_variant_count = _hccl_seq_variant_count(aux["hccl_events"])
    actionable_hccl_variant_count = len(
        _actionable_hccl_seq_variants(
            collect_hccl_seq_variants(aux["hccl_events"])))
    hccl_count_drifts = collect_hccl_count_drifts(aux["hccl_events"])
    unified_create_variants = collect_unified_new_group_order_variants(
        aux["hccl_events"])
    shared_create_seq_variants = collect_shared_create_seq_variants(
        aux["hccl_events"])
    suppress_domain_create_count_drift = _suppress_domain_create_count_drift(
        hccl_count_drifts, aux["hccl_events"], unified_create_variants)
    suppress_rank_band_count_drift = _suppress_rank_band_count_drift(
        hccl_count_drifts, shared_create_seq_variants,
        unified_create_variants)
    derived_hccl_variant_count = len(
        collect_group_backend_seq_variants(events, "hccl"))
    rebuild_cache_hits = [
        row for row in events
        if row["event"] == "rebuild_cache_hit"
    ]
    non_member_device_new_group_done = [
        row for row in events
        if row["event"] == "non_member_device_new_group_done"
    ]
    cache_hit_placeholder_risk = _cache_hit_placeholder_risk(
        rebuild_cache_hits, non_member_device_new_group_done)
    direct_fallback_count = len(aux["direct_destroy_fallbacks"])
    forced_defer_count = len(aux.get("forced_full_restore_defer", []))
    source_fallback_count = len(aux.get("source_fallbacks", []))
    source_unavailable_count = len(aux.get("source_unavailable", []))
    payload_skip_count = len(aux.get("payload_skips", []))
    preload_skip_count = len(aux.get("preload_skips", []))
    detached_advance_count = len(aux.get("detached_advances", []))
    restore_route_count = len(aux.get("restore_stale_sweep_routes", []))
    moe_missing_done_rows = _moe_dispatch_missing_done_rows(aux["moe_dispatch"])
    moe_slow_done_rows = _moe_dispatch_slow_done_rows(aux["moe_dispatch"],
                                                      slow_ms)
    slow_stash_release_rows = _slow_stash_release_rows(events, slow_ms)
    max_slow_stash_destroy = max(
        (first_ms(row["extra"], "destroy_ms") or 0.0
         for row in slow_stash_release_rows),
        default=0.0,
    )
    feature_summary = _feature_summary(aux)
    has_current_guard = bool(feature_summary["has_current_guard"])
    has_restore_reason_route = bool(
        feature_summary["has_restore_stale_sweep_reason_route"])

    if max_full_restore_destroy >= slow_ms:
        if not has_current_guard or not has_restore_reason_route:
            return (
                "verdict=STALE_LOG_FULL_RESTORE_MC2_DIRECT_DESTROY_SLOW "
                f"max_destroy_ms={max_full_restore_destroy:.2f} "
                f"restore_stale_sweep_routes={restore_route_count} "
                f"derived_hccl_order_variants={derived_hccl_variant_count}",
                [
                    "action=this log shows the full-restore MC2 direct-destroy "
                    "slow path, but it lacks the current restore stale-cache "
                    "reason-route marker. Treat it as a pre-fix or "
                    "feature-incomplete run; rerun before changing the lifecycle "
                    "policy again."
                ],
            )
        return (
            "verdict=CURRENT_CODE_FULL_RESTORE_MC2_DIRECT_DESTROY_STILL_SLOW "
            f"max_destroy_ms={max_full_restore_destroy:.2f} "
            f"restore_stale_sweep_routes={restore_route_count} "
            f"derived_hccl_order_variants={derived_hccl_variant_count}",
            [
                "action=the restore stale-cache reason-route marker is present, "
                "but hot-path MC2 destroy still happened. Inspect force_destroy, "
                "stashed-release classification, or another destroy caller that "
                "bypasses _drop_stale_cached_elastic_parallel_groups(reason="
                "mode1_full_restore)."
            ],
        )
    if plain_full_restore_destroy >= slow_ms:
        if not has_current_guard or not has_restore_reason_route:
            return (
                "verdict=STALE_LOG_FULL_RESTORE_MC2_DIRECT_DESTROY_SLOW "
                f"max_destroy_ms={plain_full_restore_destroy:.2f} "
                f"restore_stale_sweep_routes={restore_route_count} "
                f"derived_hccl_order_variants={derived_hccl_variant_count}",
                [
                    "action=plain release rows show full-restore MC2 direct "
                    "destroy, but the current restore stale-cache reason-route "
                    "marker is missing. This cannot prove the latest defer guard "
                    "failed."
                ],
            )
        return (
            "verdict=CURRENT_CODE_FULL_RESTORE_MC2_DIRECT_DESTROY_STILL_SLOW "
            f"max_destroy_ms={plain_full_restore_destroy:.2f} "
            f"restore_stale_sweep_routes={restore_route_count} "
            f"derived_hccl_order_variants={derived_hccl_variant_count}",
            [
                "action=plain stashed-release logs show stale floor MC2 direct "
                "destroy despite the restore stale-cache reason route. Find the "
                "specific caller/reason that still reaches direct destroy before "
                "lower-level HCCL sequence speculation."
            ],
        )
    if classify_misclassified_rows:
        first_row = min(classify_misclassified_rows,
                        key=lambda row: row["lineno"])
        return (
            "verdict=FULL_RESTORE_MC2_CLASSIFY_STILL_STASH_GROUP "
            f"count={len(classify_misclassified_rows)} "
            f"first_line={first_row['lineno']} "
            f"fixed_keep_no_cache={len(classify_fixed_keep_no_cache_rows)}",
            [
                "action=stash classification still predicts reason=stash_group "
                "for a full-restore MC2 group. The full-restore classification "
                "fix did not trigger for this run; inspect fixed_keep/should_cache "
                "and target_ranks in the printed classify rows before changing "
                "lower-level HCCL behavior."
            ],
        )
    if plain_misclassified_rows:
        first_row = min(plain_misclassified_rows,
                        key=lambda row: row["lineno"])
        return (
            "verdict=FULL_RESTORE_MC2_MISCLASSIFIED_AS_STASH_SLOW "
            f"count={len(plain_misclassified_rows)} "
            f"first_line={first_row['lineno']} "
            f"max_destroy_ms={max_plain_misclassified_destroy:.2f}",
            [
                "action=floor-sized MC2 group was released as reason=stash_group "
                "and hit the 100s destroy path. Check stash classification "
                "logs: target/full-restore context or fixed-topology keep policy "
                "is likely bypassing mode1_full_restore defer."
            ],
        )
    if slow_stash_release_rows:
        first_row = min(slow_stash_release_rows, key=lambda row: row["lineno"])
        return (
            "verdict=LOW_FLOOR_STASH_RELEASE_SLOW "
            f"count={len(slow_stash_release_rows)} "
            f"first_line={first_row['lineno']} "
            f"max_destroy_ms={max_slow_stash_destroy:.2f}",
            [
                "action=low-floor stashed group release itself is slow; fix the "
                "remaining hot-path destroy/release before chasing lower-level HCCL "
                "sequence issues."
            ],
        )
    if plain_slow_stash_rows:
        first_row = min(plain_slow_stash_rows, key=lambda row: row["lineno"])
        return (
            "verdict=LOW_FLOOR_STASH_RELEASE_SLOW "
            f"count={len(plain_slow_stash_rows)} "
            f"first_line={first_row['lineno']} "
            f"max_destroy_ms={max_plain_slow_stash_destroy:.2f}",
            [
                "action=plain stashed-release logs show slow hot-path destroy. "
                "Fix the remaining stash classification/defer path before "
                "chasing DispatchV2 or KV-cache explanations."
            ],
        )
    if payload_skip_count > 0 and detached_advance_count == 0:
        first_row = min(aux["payload_skips"], key=lambda row: row["lineno"])
        return (
            "verdict=NON_SOURCE_PAYLOAD_SKIPPED_WITHOUT_PLACEHOLDER_ADVANCE "
            f"payload_skips={payload_skip_count} preload_skips={preload_skip_count} "
            f"first_line={first_row['lineno']} rank={first_row['rank']}",
            [
                "action=non-source ranks skipped payload/preload but did not "
                "advance placeholder communicator sequence; fix rebuild_elastic_ep_group "
                "non-member path before interpreting HCCL sequence variants."
            ],
        )
    if source_unavailable_count > 0:
        first_row = min(aux["source_unavailable"],
                        key=lambda row: row["lineno"])
        return (
            "verdict=SOURCE_GROUP_UNAVAILABLE_NON_WORLD "
            f"count={source_unavailable_count} "
            f"first_line={first_row['lineno']} rank={first_row['rank']} "
            f"reason={first_row['reason']}",
            [
                "action=a non-world shrink source group was unavailable. This "
                "usually means a rank detached/lost its live DP group before "
                "source collectives. Check that detached ranks still advance "
                "placeholder communicator sequence and source ranks never fall "
                "back to default world collectives."
            ],
        )
    if source_fallback_count > 0 and payload_skip_count == 0:
        first_row = min(aux["source_fallbacks"], key=lambda row: row["lineno"])
        return (
            "verdict=SOURCE_GROUP_FALLBACK_WITHOUT_NON_SOURCE_SKIP "
            f"source_fallbacks={source_fallback_count} "
            f"first_line={first_row['lineno']} rank={first_row['rank']}",
            [
                "action=a rank lost its live DP group and fell back to world "
                "group, but no non-source payload skip was logged. Verify that "
                "detached ranks do not enter source collectives."
            ],
        )
    if cache_hit_placeholder_risk:
        first_cache = cache_hit_placeholder_risk["first_cache"]
        first_placeholder = cache_hit_placeholder_risk["first_placeholder"]
        return (
            "verdict=CACHE_HIT_PLACEHOLDER_SEQUENCE_RISK "
            f"same_group={int(cache_hit_placeholder_risk['same_group'])} "
            f"first_cache_line={first_cache['lineno']} "
            f"first_placeholder_line={first_placeholder['lineno']} "
            f"cache_ranks={_rank_summary(cache_hit_placeholder_risk['cache_rows'])} "
            f"placeholder_ranks={_rank_summary(cache_hit_placeholder_risk['placeholder_rows'])}",
            [
                "action=active ranks reused cached DP/EP/MC2 wrappers while "
                "detached ranks consumed placeholder new_group calls in the "
                "same mode=1 shrink/restore run. Disable rebuild cache hits "
                "whenever placeholder alignment is enabled, including the "
                "full-world restore phase."
            ],
        )
    if shared_create_seq_variants:
        first = shared_create_seq_variants[0]
        return (
            "verdict=SHARED_CREATE_SEQ_MISMATCH_CANDIDATE "
            f"variant_count={len(shared_create_seq_variants)} "
            f"first_line={first['first_lineno']} first_seq={first['seq']}",
            [
                "action=the same shared rank-local create seq number maps to "
                "different backend/ranks across ranks. This is the direct "
                "signature of HCCL new_group order/numbering drift; compare "
                "the first_shared_create_seq_variant rows before changing "
                "MoE/DispatchV2 code."
            ],
        )
    if actionable_hccl_variant_count > 0:
        return (
            "verdict=HCCL_SEQUENCE_MISMATCH_CANDIDATE "
            f"variant_count={actionable_hccl_variant_count}",
            [
                "action=compare HCCL/Gloo seq-number variants; prioritize aligning "
                "new_group/destroy_process_group call order across same-role ranks."
            ],
        )
    if unified_create_variants:
        first = unified_create_variants[0]
        return (
            "verdict=UNIFIED_NEW_GROUP_ORDER_MISMATCH_CANDIDATE "
            f"variant_count={len(unified_create_variants)} "
            f"first_line={first['first_lineno']} first_seq={first['seq']}",
            [
                "action=when GroupCoordinator real creates and non-member "
                "placeholder creates are merged into one rank-local "
                "new_group stream, the same Nth create maps to different "
                "backend/ranks. This directly supports the HCCL internal "
                "communicator order/numbering drift hypothesis."
            ],
        )
    if (hccl_count_drifts and not suppress_domain_create_count_drift
            and not suppress_rank_band_count_drift):
        first = hccl_count_drifts[0]
        return (
            "verdict=HCCL_SEQUENCE_COUNT_DRIFT_CANDIDATE "
            f"drift_count={len(hccl_count_drifts)} "
            f"first_domain={first['domain']} first_op={first['op']} "
            f"count_span={first['min']}-{first['max']}",
            [
                "action=raw HCCL/Gloo digest shows different create/destroy "
                "event counts across ranks. This supports a communicator "
                "lifecycle/order drift hypothesis even if the same seq-number "
                "signature mismatch has not appeared yet. Compare the by-rollout "
                "hccl_rank_count_divergence and mode1_seq_checkpoint lines."
            ],
        )
    if suppress_domain_create_count_drift or suppress_rank_band_count_drift:
        return (
            "verdict=RANK_BAND_COUNT_DRIFT_SUPPRESSED_BY_CLEAN_CREATE_ORDER "
            f"drift_count={len(hccl_count_drifts)} "
            f"domain_create_suppressed={int(suppress_domain_create_count_drift)} "
            f"rank_band_suppressed={int(suppress_rank_band_count_drift)}",
            [
                "action=rank-band create/destroy counts differ because active "
                "and detached ranks traverse different shrink levels, but no "
                "shared/unified create-order mismatch was found. Treat this as "
                "a diagnostic detail, not a blocking failure."
            ],
        )
    if derived_hccl_variant_count > 0:
        return (
            "verdict=DERIVED_HCCL_ORDER_MISMATCH_CANDIDATE "
            f"variant_count={derived_hccl_variant_count}",
            [
                "action=raw HCCL seq digest is unavailable or clean, but group "
                "digest implies different rank-local backend=hccl operation "
                "signatures. Align new_group/destroy order before chasing "
                "operator-level issues."
            ],
        )
    if forced_defer_count > 0:
        first_row = min(aux["forced_full_restore_defer"],
                        key=lambda row: row["lineno"])
        return (
            "verdict=FULL_RESTORE_DEFER_FORCED_ENV_DRIFT_HANDLED "
            f"count={forced_defer_count} first_line={first_row['lineno']} "
            f"rank={first_row['rank']} kind={first_row['kind']}",
            [
                "action=an env/script still attempted to disable full-restore "
                "defer, but the guard prevented hot-path destroy. If rollout is "
                "still slow and HCCL variants are absent, continue with DispatchV2 "
                "or non-group runtime analysis rather than reverting this guard."
            ],
        )
    if full_restore_deferred_rows and max_deferred_device_destroy >= slow_ms:
        return (
            "verdict=RAW_PG_RETIRE_DESTROY_SLOW "
            f"max_device_pg_destroy_ms={max_deferred_device_destroy:.2f}",
            [
                "action=the wait moved from GroupCoordinator.destroy() to raw PG "
                "destroy; evaluate CPU/device PG deferred destroy policy carefully."
            ],
        )
    if moe_missing_done_rows:
        first_row = moe_missing_done_rows[0]
        return (
            "verdict=MOE_DISPATCH_PREPARE_STUCK "
            f"count={len(moe_missing_done_rows)} "
            f"first_line={first_row['lineno']} "
            f"rank={first_row['rank']} seq={first_row['seq']}",
            [
                "action=a layer0 MoE prepare_start has no matching prepare_done; "
                "focus on the real DispatchV2/MC2 communication path rather than "
                "group lifecycle teardown."
            ],
        )
    if moe_slow_done_rows:
        max_prepare = max(row["elapsed_ms_value"] or 0.0
                          for row in moe_slow_done_rows)
        first_row = min(moe_slow_done_rows, key=lambda row: row["lineno"])
        return (
            "verdict=MOE_DISPATCH_PREPARE_SLOW "
            f"count={len(moe_slow_done_rows)} "
            f"first_line={first_row['lineno']} "
            f"max_prepare_ms={max_prepare:.2f}",
            [
                "action=MoE prepare/DispatchV2 returns but is slow; inspect "
                "MC2/all2all runtime communication and token distribution."
            ],
        )
    if max_restore_total >= restore_slow_ms or max_restore_rebuild >= restore_slow_ms:
        return (
            "verdict=RESTORE_REBUILD_OR_REFRESH_SLOW "
            f"max_rebuild_ms={max_restore_rebuild:.2f} "
            f"max_total_ms={max_restore_total:.2f} "
            f"threshold_ms={restore_slow_ms:.2f}",
            [
                "action=inspect restore cluster details; slow point is no longer "
                "the old stash_release_done destroy_ms, and no layer0 MoE "
                "prepare stuck/slow signal was observed."
            ],
        )
    if has_restore_reason_route:
        first_row = min(aux["restore_stale_sweep_routes"],
                        key=lambda row: row["lineno"])
        return (
            "verdict=RESTORE_STALE_SWEEP_REASON_ROUTE_ACTIVE "
            f"routes={restore_route_count} "
            f"deferred_rows={len(full_restore_deferred_rows)} "
            f"first_line={first_row['lineno']}",
            [
                "action=full-restore stale sweep is routed through "
                "mode1_full_restore and no higher-priority slow group lifecycle "
                "signal was found. If rollout2+ is still slow, prioritize HCCL "
                "seq variants and MoE dispatch timing over the old direct-destroy "
                "path."
            ],
        )
    if full_restore_deferred_rows:
        return (
            "verdict=FULL_RESTORE_MC2_DEFERRED_NOT_SLOW "
            f"deferred_rows={len(full_restore_deferred_rows)} "
            f"max_device_pg_destroy_ms={max_deferred_device_destroy:.2f}",
            [
                "action=the previous full-restore MC2 direct destroy slow path "
                "is deferred in this log; if rollout is still slow, inspect HCCL "
                "seq variants or model/runtime paths next."
            ],
        )
    if not aux["hccl_events"]:
        if not has_current_guard:
            return (
                "verdict=NEEDS_FRESH_FEATURE_LOG",
                [
                    "action=this log lacks both HCCL seq digest rows and the "
                    "current mode1 lifecycle feature marker; rerun with the "
                    "current script/code before inferring communicator sequence "
                    "mismatch."
                ],
            )
        return (
            "verdict=NEEDS_FRESH_HCCL_SEQ_LOG",
            [
                "action=this log predates low-level HCCL seq instrumentation or the "
                "flag was off; rerun with VLLM_ASCEND_ELASTIC_HCCL_SEQ_DIGEST_LOG=1."
            ],
        )
    if direct_fallback_count > 0:
        kinds = collections.Counter(
            row.get("kind", "unknown")
            for row in aux["direct_destroy_fallbacks"])
        kind_text = ",".join(
            f"{kind}:{count}" for kind, count in sorted(kinds.items()))
        first_lineno = min(row["lineno"] for row in aux["direct_destroy_fallbacks"])
        return (
            "verdict=LOW_FLOOR_DIRECT_DESTROY_FALLBACK_PRESENT_BUT_NOT_SLOW "
            f"count={direct_fallback_count} first_line={first_lineno} "
            f"kinds={kind_text}",
            [
                "action=direct destroy fallback warnings remain, but no release "
                "exceeded the slow threshold in this window; treat as lifecycle "
                "hygiene unless later slow rows appear."
            ],
        )
    return (
        "verdict=NO_SLOW_GROUP_LIFECYCLE_SIGNAL_FOUND",
        [
            "action=group lifecycle logs do not show >threshold slow events; "
            "look at model forward/dispatch or non-group runtime paths."
        ],
    )


def print_diagnosis_verdict(events, aux, slow_ms: float) -> None:
    print("\n== diagnosis verdict ==")
    verdict, actions = diagnosis_verdict(events, aux, slow_ms)
    print(verdict)
    for action in actions:
        print(action)


def _group_seq_key(row):
    group_name = row["group_name"] or row["attr"] or row["kind"]
    ranks = _canonical_group_ranks_text(row["ranks"])
    return (row["kind"], group_name, ranks)


def _short_group_seq_token(row):
    event = row["event"]
    kind, group_name, ranks = _group_seq_key(row)
    reason = row["reason"]
    extra = row["extra"]
    ms = None
    for key in ("destroy_ms", "create_ms", "device_ms", "cpu_ms", "total_ms"):
        value = first_ms(extra, key)
        if value is not None:
            ms = f"{key}={value:.0f}"
            break
    reason_part = f":{reason}" if reason else ""
    ms_part = f":{ms}" if ms else ""
    return f"{event}:{kind}:{group_name}:{ranks}{reason_part}{ms_part}"


def print_rank_group_sequence_summary(events) -> None:
    """Show per-rank communicator lifecycle shape.

    Rank-local seq numbers are expected to differ across active and non-member
    ranks. This summary instead checks whether ranks in the same role issue the
    same *kind* of HCCL/Gloo group operation counts, which is closer to the
    suspected HCCL internal communicator-index mismatch.
    """
    print("\n== group lifecycle sequence summary ==")
    by_rank = collections.defaultdict(list)
    for row in events:
        by_rank[row["rank"]].append(row)

    interesting_events = {
        "rebuild_create_start",
        "rebuild_create_done",
        "rebuild_cache_hit",
        "rebuild_cache_bypass_for_seq_alignment",
        "stash_release_start",
        "stash_release_done",
        "non_member_reuse",
        "non_member_device_new_group_start",
        "non_member_device_new_group_done",
        "non_member_cpu_new_group_start",
        "non_member_cpu_new_group_done",
        "non_member_destroy_start",
        "non_member_destroy_done",
        "non_member_signature_done",
    }
    for rank in sorted(by_rank):
        rows = [
            row for row in by_rank[rank]
            if row["event"] in interesting_events
        ]
        counts = collections.Counter(row["event"] for row in rows)
        slow_rows = []
        for row in rows:
            values = []
            for key in ("destroy_ms", "create_ms", "device_ms", "cpu_ms",
                        "total_ms"):
                value = first_ms(row["extra"], key)
                if value is not None:
                    values.append(value)
            if values and max(values) >= 1000.0:
                slow_rows.append((max(values), row))
        print(
            f"rank={rank:02d} lifecycle_events={len(rows)} "
            f"create_done={counts['rebuild_create_done']} "
            f"cache_hit={counts['rebuild_cache_hit']} "
            f"cache_bypass={counts['rebuild_cache_bypass_for_seq_alignment']} "
            f"release_done={counts['stash_release_done']} "
            f"non_member_reuse={counts['non_member_reuse']} "
            f"non_member_device_done={counts['non_member_device_new_group_done']} "
            f"non_member_cpu_done={counts['non_member_cpu_new_group_done']} "
            f"non_member_destroy_done={counts['non_member_destroy_done']} "
            f"slow={len(slow_rows)}")
        if slow_rows:
            for value, row in sorted(slow_rows, reverse=True)[:6]:
                print(
                    f"  slow line={row['lineno']} seq={row['seq']} "
                    f"max_ms={value:.2f} {_short_group_seq_token(row)}")

    print("\n== lifecycle count variants ==")
    signatures = collections.defaultdict(list)
    for rank in sorted(by_rank):
        rows = [
            row for row in by_rank[rank]
            if row["event"] in interesting_events
        ]
        counter = collections.Counter(
            (row["event"], row["kind"], row["group_name"], row["ranks"],
             row["reason"])
            for row in rows)
        signature = tuple(sorted(counter.items()))
        signatures[signature].append(rank)
    for index, (signature, ranks) in enumerate(signatures.items(), 1):
        print(f"variant={index} ranks={ranks} entries={len(signature)}")
        for (event, kind, group_name, ranks_text, reason), count in signature[:24]:
            print(
                f"  count={count} event={event} kind={kind} "
                f"group={group_name} ranks={ranks_text} reason={reason}")
        if len(signature) > 24:
            print(f"  ... {len(signature) - 24} more entries")


def _hccl_short_token(row):
    create_ms = first_ms(row["extra"], "create_ms")
    suffix = f":create_ms={create_ms:.0f}" if create_ms is not None else ""
    pg_id_match = re.search(r"pg_id=([0-9]+)", row["extra"])
    pg_suffix = f":pg={pg_id_match.group(1)}" if pg_id_match else ""
    return (
        f"{row['op']}:{row['event']}:{row['kind']}:{row['group_name']}:"
        f"{row['backend']}:ranks={row['ranks']}:reason={row['reason']}"
        f"{suffix}{pg_suffix}")


def _normalize_hccl_event_for_seq(row):
    event = row["event"]
    if event.endswith("_start"):
        phase = "start"
    elif event.endswith("_done"):
        phase = "done"
    else:
        phase = event
    if "device_new_group" in event:
        action = "device_new_group"
    elif "cpu_new_group" in event:
        action = "cpu_new_group"
    elif "destroy" in event:
        action = "destroy"
    else:
        action = event
    return (
        row["op"],
        phase,
        action,
        row["backend"],
        row["ranks"],
        row["reason"],
    )


def print_hccl_sequence_summary(hccl_events, slow_ms: float) -> None:
    print("\n== HCCL/Gloo low-level sequence summary ==")
    if not hccl_events:
        print("none; run with VLLM_ASCEND_ELASTIC_HCCL_SEQ_DIGEST_LOG=1")
        return

    by_rank = collections.defaultdict(list)
    for row in hccl_events:
        by_rank[row["rank"]].append(row)

    for rank in sorted(by_rank):
        rows = by_rank[rank]
        create_rows = [row for row in rows if row["op"] == "create"]
        destroy_rows = [row for row in rows if row["op"] == "destroy"]
        slow_rows = []
        for row in rows:
            values = []
            for key in ("create_ms", "destroy_ms", "device_pg_destroy_ms",
                        "cpu_pg_destroy_ms"):
                value = first_ms(row["extra"], key)
                if value is not None:
                    values.append(value)
            if values and max(values) >= slow_ms:
                slow_rows.append((max(values), row))
        print(
            f"rank={rank:02d} hccl_events={len(rows)} "
            f"create_events={len(create_rows)} max_create_seq={max((row['seq'] for row in create_rows), default=0)} "
            f"destroy_events={len(destroy_rows)} max_destroy_seq={max((row['seq'] for row in destroy_rows), default=0)} "
            f"slow={len(slow_rows)}")
        for value, row in sorted(slow_rows, reverse=True)[:6]:
            print(
                f"  slow line={row['lineno']} seq={row['seq']} "
                f"max_ms={value:.2f} {_hccl_short_token(row)}")

    print("\n== HCCL/Gloo sequence count variants ==")
    signatures = collections.defaultdict(list)
    for rank in sorted(by_rank):
        counter = collections.Counter(
            (row["op"], row["event"], row["kind"], row["group_name"],
             row["backend"], row["ranks"], row["reason"])
            for row in by_rank[rank])
        signature = tuple(sorted(counter.items()))
        signatures[signature].append(rank)
    for index, (signature, ranks) in enumerate(signatures.items(), 1):
        print(f"variant={index} ranks={ranks} entries={len(signature)}")
        for (op, event, kind, group_name, backend, ranks_text,
             reason), count in signature[:32]:
            print(
                f"  count={count} op={op} event={event} kind={kind} "
                f"group={group_name} backend={backend} ranks={ranks_text} "
                f"reason={reason}")
        if len(signature) > 32:
            print(f"  ... {len(signature) - 32} more entries")

    print_hccl_seq_mismatch_summary(hccl_events)


def print_hccl_summary_report(hccl_summaries) -> None:
    print("\n== HCCL/Gloo lifecycle counter summary ==")
    if not hccl_summaries:
        print("none; run with VLLM_ASCEND_ELASTIC_HCCL_SEQ_SUMMARY_LOG=1")
        return
    by_rank = collections.defaultdict(list)
    for row in hccl_summaries:
        by_rank[row["rank"]].append(row)
    for rank in sorted(by_rank):
        last = by_rank[rank][-1]
        print(
            f"rank={rank:02d} summaries={len(by_rank[rank])} "
            f"last_event={last['event']} create_seq={last['create_seq']} "
            f"destroy_seq={last['destroy_seq']} ranks={last['ranks']} "
            f"seen={last['seen']} detached={last['detached']}")

    variants = collect_hccl_summary_variants(hccl_summaries)
    print("\n== HCCL/Gloo lifecycle counter variants ==")
    if not variants:
        print("none")
        return
    for variant in variants[:20]:
        print(
            f"line={variant['first_lineno']}-{variant['last_lineno']} "
            f"event={variant['event']} ranks={variant['ranks']} "
            f"reason={variant['reason']} signatures={len(variant['rows_by_signature'])}")
        for signature, rows in variant["rows_by_signature"].items():
            ranks = [row["rank"] for row in rows]
            create_seq, destroy_seq = signature
            details = sorted({
                (row["seen"], row["detached"], row["target"], row["active"],
                 _summary_extra_digest(row))
                for row in rows
            })
            print(
                f"  ranks={ranks} create_seq={create_seq} "
                f"destroy_seq={destroy_seq} state_variants={details}")
    if len(variants) > 20:
        print(f"... {len(variants) - 20} more lifecycle counter variants")


def print_hccl_seq_mismatch_summary(hccl_events, limit: int = 20) -> None:
    print("\n== HCCL/Gloo seq-number variants ==")
    if not hccl_events:
        print("no HCCL/Gloo digest events; run a fresh test with VLLM_ASCEND_ELASTIC_HCCL_SEQ_DIGEST_LOG=1")
        return
    by_rank = collections.defaultdict(list)
    for row in hccl_events:
        by_rank[row["rank"]].append(row)
    print("rank_seq_counts:")
    for rank in sorted(by_rank):
        create_max = max((row["seq"] for row in by_rank[rank]
                          if row["op"] == "create"), default=0)
        destroy_max = max((row["seq"] for row in by_rank[rank]
                           if row["op"] == "destroy"), default=0)
        create_events = sum(1 for row in by_rank[rank]
                            if row["op"] == "create")
        destroy_events = sum(1 for row in by_rank[rank]
                             if row["op"] == "destroy")
        print(
            f"  rank={rank:02d} create_events={create_events} "
            f"max_create_seq={create_max} destroy_events={destroy_events} "
            f"max_destroy_seq={destroy_max}")
    variants = collect_hccl_seq_variants(hccl_events)
    if not variants:
        print("none")
        return
    print(f"total_variants={len(variants)}")
    first = variants[0]
    print(
        f"first_variant_line={first['first_lineno']} "
        f"domain={first.get('domain', 'unknown')} op={first['op']} "
        f"seq={first['seq']} phase={first['phase']} "
        f"line_span={first['first_lineno']}-{first['last_lineno']}")
    for item in variants[:limit]:
        print(
            f"domain={item.get('domain', 'unknown')} "
            f"op={item['op']} seq={item['seq']} phase={item['phase']} "
            f"lines={item['first_lineno']}-{item['last_lineno']} "
            f"variants={len(item['rows_by_signature'])}")
        for signature, rows in item["rows_by_signature"].items():
            ranks = sorted(row["rank"] for row in rows)
            line_span = f"{min(row['lineno'] for row in rows)}-{max(row['lineno'] for row in rows)}"
            print(
                f"  ranks={ranks} lines={line_span} "
                f"{_format_hccl_variant_signature(signature)}")
    if len(variants) > limit:
        print(f"... {len(variants) - limit} more seq-number variants")


def print_source_participation_summary(aux) -> None:
    print("\n== shrink source participation summary ==")
    rows_by_rank = collections.defaultdict(lambda: {
        "source_fallbacks": 0,
        "source_unavailable": 0,
        "payload_skips": 0,
        "preload_skips": 0,
        "detached_advances": 0,
        "first_line": None,
    })
    for key in (
            "source_fallbacks",
            "source_unavailable",
            "payload_skips",
            "preload_skips",
            "detached_advances",
    ):
        for row in aux.get(key, []):
            rank = row["rank"]
            rows_by_rank[rank][key] += 1
            first_line = rows_by_rank[rank]["first_line"]
            if first_line is None or row["lineno"] < first_line:
                rows_by_rank[rank]["first_line"] = row["lineno"]
    if not rows_by_rank:
        print("none")
        return
    for rank in sorted(rows_by_rank):
        row = rows_by_rank[rank]
        print(
            f"rank={rank:02d} source_fallbacks={row['source_fallbacks']} "
            f"source_unavailable={row['source_unavailable']} "
            f"payload_skips={row['payload_skips']} "
            f"preload_skips={row['preload_skips']} "
            f"detached_advances={row['detached_advances']} "
            f"first_line={row['first_line']}")


def print_config_summary(config_rows) -> None:
    print("\n== run config summary ==")
    if not config_rows:
        print("config_status=missing_header")
        print("action=log lacks lifecycle header; verify this was run after the latest wrapper/launcher changes.")
        return

    for row in config_rows[:6]:
        print(f"line={row['lineno']} {row['line'][:360]}")

    defer_full_restore = _first_config_value(
        config_rows, "defer_full_restore", "mode1_defer_full_restore")
    destroy_cpu_pg = _first_config_value(
        config_rows, "destroy_cpu_pg", "mode1_destroy_cpu_pg_on_retire")
    destroy_cpu_placeholder_pg = _first_config_value(
        config_rows, "destroy_cpu_placeholder_pg")
    group_digest = _first_config_value(
        config_rows, "group_digest", "group_seq_digest",
        "mode1_group_seq_digest")
    hccl_digest = _first_config_value(
        config_rows, "hccl_seq_digest", "mode1_hccl_seq_digest")
    hccl_summary = _first_config_value(
        config_rows, "hccl_seq_summary", "mode1_hccl_seq_summary")
    moe_dispatch_digest = _first_config_value(
        config_rows, "moe_dispatch_digest", "mode1_moe_dispatch_digest")
    moe_dispatch_sample_every = _first_config_value(
        config_rows, "moe_dispatch_sample_every",
        "mode1_moe_dispatch_sample_every")
    mode1_seq_checkpoint = _first_config_value(
        config_rows, "mode1_seq_checkpoint")
    force_non_member_placeholder = _first_config_value(
        config_rows, "force_non_member_placeholder")
    allow_rebuild_cache_hit_with_placeholder = _first_config_value(
        config_rows, "allow_rebuild_cache_hit_with_placeholder")
    post_restore_mc2_warmup = _first_config_value(
        config_rows, "post_restore_mc2_warmup",
        "mode1_post_restore_mc2_warmup")
    post_restore_alltoall_warmup = _first_config_value(
        config_rows, "post_restore_alltoall_warmup",
        "mode1_post_restore_alltoall_warmup")
    post_shrink_warmup = _first_config_value(
        config_rows, "post_shrink_warmup")
    repeat_post_shrink_moe_dispatch_warmup = _first_config_value(
        config_rows, "repeat_post_shrink_moe_dispatch_warmup")
    post_shrink_release_empty_cache = _first_config_value(
        config_rows, "post_shrink_release_empty_cache")
    post_shrink_release_sync = _first_config_value(
        config_rows, "post_shrink_release_sync")
    release_warmup_cache = _first_config_value(
        config_rows, "release_warmup_cache", "mode1_release_warmup_cache")
    pre_moe_warmup_empty_cache = _first_config_value(
        config_rows, "pre_moe_warmup_empty_cache")
    force_cleanup_after_floor_group_release = _first_config_value(
        config_rows, "force_cleanup_after_floor_group_release")
    drop_old_floor_before_rebuild = _first_config_value(
        config_rows, "drop_old_floor_before_rebuild",
        "mode1_drop_old_floor_before_rebuild")
    launcher = _first_config_value(config_rows, "launcher")
    length_regroup = _first_config_value(config_rows, "length_regroup")
    no_resample = _first_config_value(config_rows, "no_resample")
    code_fingerprints = {
        "worker_v1_sha256": (
            _first_config_value(config_rows, "worker_v1_sha256"),
            _current_repo_sha256("vllm_ascend/worker/worker_v1.py"),
        ),
        "llm_engine_sha256": (
            _first_config_value(config_rows, "llm_engine_sha256"),
            _current_repo_sha256("vllm/v1/engine/llm_engine.py"),
        ),
        "fused_moe_sha256": (
            _first_config_value(config_rows, "fused_moe_sha256"),
            _current_repo_sha256("vllm_ascend/ops/fused_moe.py"),
        ),
        "ray_trainer_sha256": (
            _first_config_value(config_rows, "ray_trainer_sha256"),
            _current_repo_sha256("verl/trainer/ppo/ray_trainer.py"),
        ),
    }

    expected = {
        "defer_full_restore": ("1", defer_full_restore),
        "destroy_cpu_pg": ("0", destroy_cpu_pg),
        "destroy_cpu_placeholder_pg": ("0", destroy_cpu_placeholder_pg),
        "group_digest": ("1", group_digest),
        "hccl_seq_digest": ("1", hccl_digest),
        "hccl_seq_summary": ("1", hccl_summary),
        "mode1_seq_checkpoint": ("1", mode1_seq_checkpoint),
        "moe_dispatch_digest": ("1", moe_dispatch_digest),
        "force_non_member_placeholder": ("1", force_non_member_placeholder),
        "allow_rebuild_cache_hit_with_placeholder": (
            "0", allow_rebuild_cache_hit_with_placeholder),
        "post_restore_mc2_warmup": ("0", post_restore_mc2_warmup),
        "post_restore_alltoall_warmup": ("0", post_restore_alltoall_warmup),
        "post_shrink_warmup": ("0", post_shrink_warmup),
        "repeat_post_shrink_moe_dispatch_warmup": (
            "0", repeat_post_shrink_moe_dispatch_warmup),
        "post_shrink_release_empty_cache": (
            "1", post_shrink_release_empty_cache),
        "post_shrink_release_sync": ("1", post_shrink_release_sync),
        "release_warmup_cache": ("1", release_warmup_cache),
        "pre_moe_warmup_empty_cache": ("1", pre_moe_warmup_empty_cache),
        "force_cleanup_after_floor_group_release": (
            "1", force_cleanup_after_floor_group_release),
        "drop_old_floor_before_rebuild": ("1", drop_old_floor_before_rebuild),
    }
    mismatches = [
        f"{name}:expected={want}:actual={actual}"
        for name, (want, actual) in expected.items()
        if actual is None or actual != want
    ]
    status = "ok" if not mismatches else "mismatch"
    print(
        f"config_status={status} launcher={launcher} "
        f"no_resample={no_resample} length_regroup={length_regroup} "
        f"defer_full_restore={defer_full_restore} "
        f"destroy_cpu_pg={destroy_cpu_pg} "
        f"destroy_cpu_placeholder_pg={destroy_cpu_placeholder_pg} "
        f"group_digest={group_digest} "
        f"hccl_seq_digest={hccl_digest} "
        f"hccl_seq_summary={hccl_summary} "
        f"mode1_seq_checkpoint={mode1_seq_checkpoint} "
        f"force_non_member_placeholder={force_non_member_placeholder} "
        f"allow_rebuild_cache_hit_with_placeholder={allow_rebuild_cache_hit_with_placeholder} "
        f"post_restore_mc2_warmup={post_restore_mc2_warmup} "
        f"post_restore_alltoall_warmup={post_restore_alltoall_warmup} "
        f"post_shrink_warmup={post_shrink_warmup} "
        f"repeat_post_shrink_moe_dispatch_warmup={repeat_post_shrink_moe_dispatch_warmup} "
        f"post_shrink_release_empty_cache={post_shrink_release_empty_cache} "
        f"post_shrink_release_sync={post_shrink_release_sync} "
        f"release_warmup_cache={release_warmup_cache} "
        f"pre_moe_warmup_empty_cache={pre_moe_warmup_empty_cache} "
        f"force_cleanup_after_floor_group_release={force_cleanup_after_floor_group_release} "
        f"drop_old_floor_before_rebuild={drop_old_floor_before_rebuild} "
        f"moe_dispatch_digest={moe_dispatch_digest} "
        f"moe_dispatch_sample_every={moe_dispatch_sample_every}")
    if mismatches:
        print("config_mismatches=" + ",".join(mismatches))

    missing_code = [
        name for name, (logged, _current) in code_fingerprints.items()
        if not logged
    ]
    code_mismatches = [
        f"{name}:logged={logged}:current={current}"
        for name, (logged, current) in code_fingerprints.items()
        if logged and current and logged != current
    ]
    if missing_code:
        print("code_status=missing " + ",".join(missing_code))
    elif code_mismatches:
        print("code_status=mismatch " + ",".join(code_mismatches))
    else:
        print("code_status=ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", type=Path,
                        help="Log path. If omitted, use newest *.txt under the default threshold log dir.")
    parser.add_argument("--around-slow", type=float, default=1000.0,
                        help="Print digest events with *_ms above this threshold.")
    parser.add_argument("--tail", type=int, default=40,
                        help="Print this many final digest events per rank.")
    parser.add_argument("--show-seq-variants", action="store_true",
                        help="Print raw rank-local seq variants. This is noisy because active/non-member ranks naturally log different events.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip per-rank first/last event rows and print only diagnostic summaries.")
    parser.add_argument("--verdict-only", action="store_true",
                        help="Print only the selected log path and diagnosis verdict.")
    parser.add_argument("--seq-mismatch-only", action="store_true",
                        help="Print only HCCL/Gloo rank-local seq-number mismatch summary.")
    parser.add_argument("--moe-dispatch-only", action="store_true",
                        help="Print only layer0 MoE prepare/dispatch endpoint summary.")
    parser.add_argument("--by-rollout", action="store_true",
                        help="Print communication summary split by rollout_output_time_s windows.")
    parser.add_argument("--rollout-regression-ratio", type=float, default=1.5,
                        help="Mark the first rollout whose runtime is this many times rollout 1.")
    parser.add_argument("--early-fail-only", action="store_true",
                        help="Check only events before the first rollout_output_time_s for early bad-path signals.")
    parser.add_argument("--config-only", action="store_true",
                        help="Print lifecycle/run config header summary only.")
    args = parser.parse_args()

    try:
        log_path = resolve_log(args.log)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    events = parse(log_path)
    aux = parse_auxiliary(log_path)
    config_rows = parse_config_lines(log_path)
    print(f"log={log_path}")
    print(f"digest_events={len(events)}")
    if args.config_only:
        print_config_summary(config_rows)
        print_log_feature_status(aux)
        return 0
    if not events:
        print("no Elastic group seq digest lines found; using plain restore/stash logs only")
    if args.verdict_only:
        print_config_summary(config_rows)
        print_log_feature_status(aux)
        verdict, actions = diagnosis_verdict(events, aux, args.around_slow)
        print(verdict)
        for action in actions:
            print(action)
        return 0
    if args.seq_mismatch_only:
        print_hccl_seq_mismatch_summary(aux["hccl_events"])
        print_hccl_summary_report(aux["hccl_summaries"])
        print_source_participation_summary(aux)
        return 0
    if args.moe_dispatch_only:
        print_moe_dispatch_summary(aux, args.around_slow)
        return 0
    if args.by_rollout:
        print_by_rollout_summary(events, aux, args.around_slow,
                                 args.rollout_regression_ratio)
        return 0
    if args.early_fail_only:
        return print_early_fail_summary(events, aux, args.around_slow)

    by_rank = collections.defaultdict(list)
    for event in events:
        by_rank[event["rank"]].append(event)
    if not args.summary_only:
        for rank in sorted(by_rank):
            seqs = by_rank[rank]
            print(f"rank={rank:02d} count={len(seqs)} first_seq={seqs[0]['seq']} last_seq={seqs[-1]['seq']} last_event={compact(seqs[-1])}")

    if events:
        print_full_restore_mc2_releases(events)
    print_full_restore_deferred_summary(aux)
    print_plain_stash_release_summary(aux, args.around_slow)
    print_feature_summary(aux)
    print_stash_classify_summary(aux)
    print_restore_rollout_summary(aux, args.around_slow)
    print_by_rollout_summary(events, aux, args.around_slow,
                             args.rollout_regression_ratio)
    print_config_summary(config_rows)
    print_log_feature_status(aux)
    print_diagnosis_verdict(events, aux, args.around_slow)
    if events:
        print_rank_group_sequence_summary(events)
    print_hccl_sequence_summary(aux["hccl_events"], args.around_slow)
    print_hccl_summary_report(aux["hccl_summaries"])
    print_source_participation_summary(aux)
    print_moe_dispatch_summary(aux, args.around_slow)

    if args.show_seq_variants:
        print("\n== raw rank-local seq variants ==")
        max_seq = max((event["seq"] for event in events), default=0)
        mismatch_count = 0
        for seq in range(1, max_seq + 1):
            rows = [row for row in events if row["seq"] == seq]
            if len(rows) <= 1:
                continue
            signatures = collections.defaultdict(list)
            for row in rows:
                signatures[compact(row)].append(row["rank"])
            if len(signatures) > 1:
                mismatch_count += 1
                print(f"seq={seq} variants={len(signatures)}")
                for sig, ranks in signatures.items():
                    print(f"  ranks={sorted(ranks)} {sig}")
        if mismatch_count == 0:
            print("none")

    print("\n== slow digest events ==")
    slow_re = re.compile(r"(?:destroy|create|device|cpu|total|cleanup|rebuild)_ms=([0-9]+(?:\.[0-9]+)?)")
    slow_rows = []
    for row in events:
        values = [float(value) for value in slow_re.findall(row["extra"])]
        if values and max(values) >= args.around_slow:
            slow_rows.append((max(values), row))
    if not slow_rows:
        print("none")
    for value, row in sorted(slow_rows, reverse=True)[:80]:
        print(f"line={row['lineno']} rank={row['rank']} seq={row['seq']} max_ms={value:.2f} {compact(row)}")

    if args.tail > 0:
        print("\n== tail by rank ==")
        for rank in sorted(by_rank):
            print(f"-- rank={rank:02d} --")
            for row in by_rank[rank][-args.tail:]:
                print(f"  line={row['lineno']} seq={row['seq']} {compact(row)}")
    return 0


if __name__ == "__main__":
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(main())
