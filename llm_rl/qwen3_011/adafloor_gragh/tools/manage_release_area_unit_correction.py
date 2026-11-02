#!/usr/bin/env python3
"""Create or verify a sidecar manifest for legacy release-area units."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


MANIFEST_NAME = "RELEASE_AREA_UNIT_CORRECTION.json"
LEGACY_AGGREGATE_FIELD = "planned_release_area_rank_seconds"
LEGACY_STEP_FIELD = "planned_release_area"
CORRECT_FIELD = "predicted_release_proxy_rank_tokens"
RELEASE_AREA_UNIT = "rank_token_proxy"


class CorrectionError(RuntimeError):
    """Raised when a legacy summary cannot be corrected safely."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorrectionError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CorrectionError(f"{label} must be a JSON object: {path}")
    return payload


def _finite_nonnegative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise CorrectionError(f"{label} must be a finite nonnegative number")
    return float(value)


def build_manifest(summary_path: Path) -> dict[str, Any]:
    summary_path = summary_path.expanduser().resolve()
    summary = _load_object(summary_path, "legacy summary")
    adafloor = summary.get("adafloor")
    if not isinstance(adafloor, dict):
        raise CorrectionError("legacy summary lacks an AdaFloor object")

    aggregate_value = _finite_nonnegative(
        adafloor.get(LEGACY_AGGREGATE_FIELD),
        f"adafloor.{LEGACY_AGGREGATE_FIELD}",
    )
    coordinated_release = adafloor.get("coordinated_release")
    if not isinstance(coordinated_release, dict):
        raise CorrectionError("legacy summary lacks coordinated release data")
    measured_rank_seconds = _finite_nonnegative(
        coordinated_release.get("total_rank_seconds"),
        "adafloor.coordinated_release.total_rank_seconds",
    )
    raw_steps = coordinated_release.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise CorrectionError("legacy summary lacks per-step release data")

    step_values: list[dict[str, float | int]] = []
    for index, step in enumerate(raw_steps, start=1):
        if not isinstance(step, dict):
            raise CorrectionError(f"coordinated release step {index} is invalid")
        step_number = step.get("step")
        if isinstance(step_number, bool) or not isinstance(step_number, int):
            raise CorrectionError(f"coordinated release step {index} lacks its index")
        value = _finite_nonnegative(
            step.get(LEGACY_STEP_FIELD),
            f"coordinated release step {step_number}.{LEGACY_STEP_FIELD}",
        )
        step_values.append({"step": step_number, "value": value})
    if not math.isclose(
        sum(float(item["value"]) for item in step_values),
        aggregate_value,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise CorrectionError(
            "legacy aggregate release proxy differs from the per-step sum"
        )

    return {
        "schema_version": 1,
        "status": "PASS",
        "kind": "release_area_unit_correction",
        "source_summary": {
            "path": str(summary_path),
            "sha256": _sha256(summary_path),
        },
        "corrections": [
            {
                "json_path": f"$.adafloor.{LEGACY_AGGREGATE_FIELD}",
                "legacy_field": LEGACY_AGGREGATE_FIELD,
                "legacy_value": aggregate_value,
                "correct_field": CORRECT_FIELD,
                "unit": RELEASE_AREA_UNIT,
            },
            {
                "json_path": (
                    "$.adafloor.coordinated_release.steps[*]."
                    f"{LEGACY_STEP_FIELD}"
                ),
                "legacy_field": LEGACY_STEP_FIELD,
                "legacy_values": step_values,
                "correct_field": CORRECT_FIELD,
                "unit": RELEASE_AREA_UNIT,
            },
        ],
        "measured_release": {
            "json_path": "$.adafloor.coordinated_release.total_rank_seconds",
            "value": measured_rank_seconds,
            "unit": "rank_seconds",
        },
        "ratio_policy": {
            "may_divide_predicted_proxy_by_measured_rank_seconds": False,
            "reason": (
                "The planner value is a token-domain rank-token proxy, while "
                "coordinated release is measured in rank-seconds."
            ),
        },
    }


def _serialized(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_manifest(summary_path: Path, output_path: Path | None = None) -> Path:
    summary_path = summary_path.expanduser().resolve()
    output = (
        summary_path.parent / MANIFEST_NAME
        if output_path is None
        else output_path.expanduser().resolve()
    )
    payload = build_manifest(summary_path)
    raw = _serialized(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != raw:
            raise CorrectionError(f"existing correction manifest is stale: {output}")
        return output

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    observed = _load_object(manifest_path, "correction manifest")
    source = observed.get("source_summary")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise CorrectionError("correction manifest lacks its source summary path")
    expected = build_manifest(Path(source["path"]))
    if observed != expected:
        raise CorrectionError(f"correction manifest is stale: {manifest_path}")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--summary", required=True, type=Path)
    create.add_argument("--output", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    try:
        if args.command == "create":
            output = write_manifest(args.summary, args.output)
            print(output)
        else:
            verify_manifest(args.manifest)
            print(f"PASS: {args.manifest.resolve()}")
    except (CorrectionError, OSError) as error:
        print(f"FAIL: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
