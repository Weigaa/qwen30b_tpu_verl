from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from manage_release_area_unit_correction import (  # noqa: E402
    CORRECT_FIELD,
    MANIFEST_NAME,
    RELEASE_AREA_UNIT,
    CorrectionError,
    verify_manifest,
    write_manifest,
)


def _legacy_summary(path: Path) -> bytes:
    payload = {
        "status": "PASS",
        "adafloor": {
            "planned_release_area_rank_seconds": 30.0,
            "coordinated_release": {
                "total_rank_seconds": 12.5,
                "steps": [
                    {"step": 1, "planned_release_area": 10.0},
                    {"step": 2, "planned_release_area": 20.0},
                ],
            },
        },
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return raw


def test_correction_manifest_is_sidecar_hashed_and_idempotent(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "fixed_epoch_summary.json"
    original = _legacy_summary(summary)

    manifest_path = write_manifest(summary)
    observed = verify_manifest(manifest_path)
    repeated = write_manifest(summary)

    assert repeated == manifest_path
    assert manifest_path.name == MANIFEST_NAME
    assert summary.read_bytes() == original
    assert observed["source_summary"]["sha256"]
    assert observed["corrections"][0]["correct_field"] == CORRECT_FIELD
    assert observed["corrections"][0]["unit"] == RELEASE_AREA_UNIT
    assert observed["corrections"][1]["legacy_values"] == [
        {"step": 1, "value": 10.0},
        {"step": 2, "value": 20.0},
    ]
    assert (
        observed["ratio_policy"][
            "may_divide_predicted_proxy_by_measured_rank_seconds"
        ]
        is False
    )


def test_verify_rejects_a_changed_source_summary(tmp_path: Path) -> None:
    summary = tmp_path / "natural_epoch_summary.json"
    _legacy_summary(summary)
    manifest = write_manifest(summary)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["adafloor"]["coordinated_release"]["total_rank_seconds"] = 13.0
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorrectionError, match="stale"):
        verify_manifest(manifest)


def test_create_rejects_inconsistent_per_step_proxy_sum(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    _legacy_summary(summary)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["adafloor"]["coordinated_release"]["steps"][0][
        "planned_release_area"
    ] = 11.0
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorrectionError, match="per-step sum"):
        write_manifest(summary)
