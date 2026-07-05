#!/usr/bin/env python3
"""Create a compact decision summary from mode1 threshold analysis output."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _first(lines: list[str], prefix: str) -> str | None:
    for line in lines:
        if line.startswith(prefix):
            return line
    return None


def _contains(lines: list[str], text: str) -> str | None:
    for line in lines:
        if text in line:
            return line.strip()
    return None


def _value(line: str | None, key: str, default: str = "NA") -> str:
    if not line:
        return default
    match = re.search(rf"(?:^|\s){re.escape(key)}=([^\s]+)", line)
    return match.group(1) if match else default


def _rollout_label(line: str) -> str:
    return line.split(None, 1)[0] if line else "NA"


def _rollout_number(label: str) -> int | None:
    match = re.match(r"rollout_(\d+)$", label)
    return int(match.group(1)) if match else None


def _rollout_lines(lines: list[str]) -> list[str]:
    rows: list[str] = []
    for line in lines:
        if re.match(r"^(rollout_\d+|after_rollout_\d+)\s", line):
            rows.append(line)
    return rows


def _is_nonzero_span(span: str) -> bool:
    match = re.match(r"(-?\d+)-(-?\d+)$", span)
    if not match:
        return False
    return match.group(1) != match.group(2)


def _has_signature_variants(line: str | None) -> bool:
    if not line:
        return False
    for key in ("live_variants", "cached_variants", "registry_variants"):
        value = _value(line, key, "0")
        try:
            if int(value) > 1:
                return True
        except ValueError:
            continue
    return False


def _first_rollout_with_condition(rollouts: list[str], predicate) -> str | None:
    for line in rollouts:
        number = _rollout_number(_rollout_label(line))
        if number is None:
            continue
        if predicate(line):
            return line
    return None


def _first_after_rollout_one_with_condition(rollouts: list[str], predicate) -> str | None:
    for line in rollouts:
        number = _rollout_number(_rollout_label(line))
        if number is None or number <= 1:
            continue
        if predicate(line):
            return line
    return None


def _line_summary(line: str | None, keys: list[str]) -> str:
    if not line:
        return "none"
    label = _rollout_label(line)
    parts = [label]
    for key in keys:
        parts.append(f"{key}={_value(line, key)}")
    return " ".join(parts)


def _float_value(line: str | None, key: str, default: float = 0.0) -> float:
    value = _value(line, key, str(default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _status_from_line(line: str | None, key: str) -> str:
    return _value(line, key, "missing")


def _is_diagnostic_hccl_variant(line: str | None) -> bool:
    return bool(line and "classification=diagnostic_rank_band_history" in line)


def build_summary(text: str, analysis_path: Path) -> list[str]:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    config_line = _first(lines, "config_status=")
    code_line = _first(lines, "code_status=")
    feature_line = _first(lines, "feature_status=")
    verdict_line = _first(lines, "verdict=")
    regression_line = _first(lines, "rollout_regression ")
    rollouts = _rollout_lines(lines)

    first_hccl_variant = _contains(lines, "first_hccl_variant ")
    hccl_count_divergence = _contains(lines, "hccl_rank_count_divergence ")
    first_hccl_summary_variant = _contains(lines, "first_hccl_summary_variant ")
    first_unified_create_variant = _contains(
        lines, "first_unified_create_order_variant ")
    first_shared_create_seq_variant = _contains(
        lines, "first_shared_create_seq_variant ")
    first_mode1_seq_divergence = _contains(
        lines, "first_mode1_seq_checkpoint_divergence ")
    first_moe_missing = _contains(lines, "first_moe_missing ")
    slowest_moe_prepare = _contains(lines, "slowest_moe_prepare ")
    first_direct_destroy = _contains(lines, "first_direct_destroy_fallback ")
    first_misclassified_mc2 = _contains(lines, "first_misclassified_mc2 ")

    first_bad_hccl = _first_after_rollout_one_with_condition(
        rollouts,
        lambda line: (
            _value(line, "actionable_hccl_seq_variants", "0") not in ("0", "NA")
            or _value(line, "unified_create_order_variants", "0") not in ("0", "NA")
            or _value(line, "shared_create_seq_variants", "0") not in ("0", "NA")
            or _value(line, "derived_hccl_backend_order_variants", "0") not in ("0", "NA")
        ),
    )
    # In floor=4, different rank bands naturally visit different shrink levels,
    # so create/destroy count spans are expected. Only same-bucket
    # live/cache/registry signature variants are actionable.
    first_bad_mode1_seq = None
    first_bad_destroy = _first_after_rollout_one_with_condition(
        rollouts,
        lambda line: (
            _value(line, "slow_stash", "0") not in ("0", "NA")
            or _value(line, "misclassified_mc2", "0") not in ("0", "NA")
            or _value(line, "full_restore_mc2_direct_destroy_fallbacks", "0")
            not in ("0", "NA")
            or _value(line, "slow_stash_reason_kinds", "none") not in ("none", "NA")
        ),
    )
    first_bad_cache_hit_placeholder = _first_after_rollout_one_with_condition(
        rollouts,
        lambda line: (
            _value(line, "cache_hit_placeholder_risk", "0")
            not in ("0", "NA")
        ),
    )
    first_bad_moe = _first_after_rollout_one_with_condition(
        rollouts,
        lambda line: (
            _value(line, "moe_missing_done", "0") not in ("0", "NA")
            or _value(line, "moe_slow_done", "0") not in ("0", "NA")
            or _float_value(line, "max_moe_prepare_ms") > 0
        ),
    )

    verdict = verdict_line or "verdict=missing"
    config_status = _status_from_line(config_line, "config_status")
    code_status = _status_from_line(code_line, "code_status")
    feature_status = _status_from_line(feature_line, "feature_status")
    regression_status = _value(regression_line, "status", "missing")
    first_bad_rollout = _value(regression_line, "first_bad", "none")
    ratio = _value(regression_line, "ratio", "NA")

    reasons: list[str] = []
    if config_status != "ok" or code_status != "ok" or feature_status != "current":
        reasons.append("freshness_invalid")
    if (first_bad_destroy or first_misclassified_mc2
            or "DIRECT_DESTROY_STILL_SLOW" in verdict
            or "STASH_RELEASE_SLOW" in verdict
            or "MISCLASSIFIED" in verdict
            or "RAW_PG_RETIRE_DESTROY_SLOW" in verdict):
        reasons.append("destroy_or_stash_path")
    if first_bad_cache_hit_placeholder or "CACHE_HIT_PLACEHOLDER" in verdict:
        reasons.append("cache_hit_placeholder_sequence_risk")
    # Global create/destroy count span can differ normally in floor=4 because
    # 0-7, 8-11 and 12-15 naturally traverse different shrink levels. Treat
    # count divergence as an auxiliary detail unless a same-seq/order variant
    # or MoE dispatch signal is present.
    if (first_bad_hccl or first_hccl_variant
            or first_unified_create_variant
            or first_shared_create_seq_variant
            or "UNIFIED_NEW_GROUP_ORDER" in verdict
            or "SHARED_CREATE_SEQ" in verdict):
        if not (
            _is_diagnostic_hccl_variant(first_hccl_variant)
            and "RANK_BAND_COUNT_DRIFT_SUPPRESSED" in verdict
            and not first_bad_hccl
            and not first_unified_create_variant
            and not first_shared_create_seq_variant
        ):
            reasons.append("hccl_sequence_or_count_drift")
    if _has_signature_variants(first_mode1_seq_divergence):
        reasons.append("mode1_python_seq_checkpoint_drift")
    if first_bad_moe or first_moe_missing or slowest_moe_prepare:
        reasons.append("moe_dispatch_path")
    if not reasons:
        reasons.append("no_specific_signal")

    if "freshness_invalid" in reasons:
        recommendation = (
            "ignore_root_cause_for_this_log_until_fresh_header_code_feature_are_ok")
    elif "destroy_or_stash_path" in reasons:
        recommendation = (
            "fix_remaining_direct_destroy_or_stash_classification_before_hccl_hypothesis")
    elif "cache_hit_placeholder_sequence_risk" in reasons:
        recommendation = (
            "disable_rebuild_cache_hit_when_placeholder_alignment_is_enabled")
    elif first_shared_create_seq_variant or "SHARED_CREATE_SEQ" in verdict:
        recommendation = (
            "hccl_shared_create_seq_mismatch_supported_compare_first_shared_create_seq_variant")
    elif "hccl_sequence_or_count_drift" in reasons:
        recommendation = (
            "hccl_internal_sequence_hypothesis_supported_check_first_bad_rollout_rank_lines")
    elif "RANK_BAND_COUNT_DRIFT_SUPPRESSED" in verdict:
        recommendation = (
            "rank_band_history_diff_is_expected_continue_or_reduce_diagnostic_logs")
    elif "mode1_python_seq_checkpoint_drift" in reasons:
        recommendation = (
            "python_side_group_lifecycle_sequence_drift_supported")
    elif "moe_dispatch_path" in reasons:
        recommendation = (
            "group_lifecycle_looks_clean_focus_dispatchv2_mc2_runtime")
    else:
        recommendation = "need_more_targeted_logs"

    output = [
        f"analysis={analysis_path}",
        f"config_status={config_status} code_status={code_status} feature_status={feature_status}",
        verdict,
        f"rollout_regression_status={regression_status} first_bad={first_bad_rollout} ratio={ratio}",
        "suspected_reasons=" + ",".join(reasons),
        f"recommendation={recommendation}",
        "destroy_signal=" + _line_summary(
            first_bad_destroy,
            [
                "rollout_time", "direct_destroy_fallbacks", "direct_destroy_reasons",
                "slow_stash", "misclassified_mc2", "slow_stash_reason_kinds",
            ],
        ),
        "cache_hit_placeholder_signal=" + _line_summary(
            first_bad_cache_hit_placeholder,
            [
                "rollout_time", "rebuild_cache_hits", "rebuild_cache_bypasses",
                "non_member_device_new_group_done", "cache_hit_placeholder_risk",
                "cache_hit_placeholder_same_group",
            ],
        ),
        "hccl_signal=" + _line_summary(
            first_bad_hccl,
            [
                "rollout_time", "hccl_seq_variants", "hccl_summary_variants",
                "unified_create_order_variants", "shared_create_seq_variants",
                "hccl_create_count_span", "hccl_destroy_count_span",
                "derived_hccl_backend_order_variants",
            ],
        ),
        "unified_create_signal=" + _line_summary(
            first_bad_hccl,
            ["rollout_time", "unified_create_order_variants"],
        ),
        "shared_create_seq_signal=" + _line_summary(
            first_bad_hccl,
            ["rollout_time", "shared_create_seq_variants"],
        ),
        "mode1_seq_signal=" + _line_summary(
            first_mode1_seq_divergence if _has_signature_variants(first_mode1_seq_divergence) else None,
            ["rollout_time", "mode1_seq_create_span", "mode1_seq_destroy_span"],
        ),
        "moe_signal=" + _line_summary(
            first_bad_moe,
            ["rollout_time", "moe_missing_done", "moe_slow_done", "max_moe_prepare_ms"],
        ),
    ]

    detail_lines = [
        ("first_direct_destroy", first_direct_destroy),
        ("first_misclassified_mc2", first_misclassified_mc2),
        ("first_hccl_variant", first_hccl_variant),
        ("hccl_count_divergence", hccl_count_divergence),
        ("first_hccl_summary_variant", first_hccl_summary_variant),
        ("first_unified_create_order_variant", first_unified_create_variant),
        ("first_shared_create_seq_variant", first_shared_create_seq_variant),
        ("first_mode1_seq_divergence", first_mode1_seq_divergence),
        ("first_moe_missing", first_moe_missing),
        ("slowest_moe_prepare", slowest_moe_prepare),
    ]
    for name, line in detail_lines:
        if line:
            output.append(f"{name}={line}")

    rollout_preview = []
    for line in rollouts[:8]:
        rollout_preview.append(_line_summary(
            line,
            [
                "rollout_time", "direct_destroy_fallbacks", "slow_stash",
                "rebuild_cache_hits", "rebuild_cache_bypasses",
                "cache_hit_placeholder_risk",
                "unified_create_order_variants", "shared_create_seq_variants",
                "hccl_create_count_span", "hccl_destroy_count_span",
                "max_deferred_device_placeholders",
                "max_deferred_cpu_placeholders",
                "mode1_seq_create_span", "mode1_seq_destroy_span",
                "moe_missing_done", "moe_slow_done", "max_moe_prepare_ms",
            ],
        ))
    if rollout_preview:
        output.append("rollout_compact=" + " | ".join(rollout_preview))
    return output


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: extract_mode1_analysis_summary.py <analysis_path>", file=sys.stderr)
        return 2
    analysis_path = Path(sys.argv[1])
    text = analysis_path.read_text(errors="replace")
    for line in build_summary(text, analysis_path):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
