from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.migrate_deepseek_fixed_work_cap import (
    AUTHORIZED_KEY,
    CONTINUATION_KEY,
    EXECUTION_KEY,
    MigrationError,
    migrate,
)


SOURCE_EXECUTION = "1" * 64
AUTHORIZED_RUNTIME = "2" * 64
FIXED_EXECUTION = "3" * 64


def _source(path: Path, *, overrides: dict[str, str] | None = None) -> Path:
    values = {
        EXECUTION_KEY: SOURCE_EXECUTION,
        AUTHORIZED_KEY: AUTHORIZED_RUNTIME,
        CONTINUATION_KEY: SOURCE_EXECUTION,
    }
    values.update(overrides or {})
    path.write_text(
        "# preserved physical capacities\n"
        + "\n".join(f"export {key}={value}" for key, value in values.items())
        + "\nexport DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR2=563840\n",
        encoding="ascii",
    )
    return path


def test_migration_changes_only_execution_and_continuation_hashes(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.env")
    output = tmp_path / "contracts" / "caps.env"
    manifest = tmp_path / "contracts" / "migration.json"
    payload = migrate(source, output, manifest, FIXED_EXECUTION, tmp_path / "common")

    source_text = source.read_text(encoding="ascii")
    migrated_text = output.read_text(encoding="ascii")
    assert source_text != migrated_text
    assert f"export {EXECUTION_KEY}={FIXED_EXECUTION}" in migrated_text
    assert f"export {CONTINUATION_KEY}={FIXED_EXECUTION}" in migrated_text
    assert f"export {AUTHORIZED_KEY}={AUTHORIZED_RUNTIME}" in migrated_text
    assert "DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR2=563840" in migrated_text
    assert payload["source_execution_code_sha256"] == SOURCE_EXECUTION
    assert payload["source_continuation_execution_code_sha256"] == SOURCE_EXECUTION
    assert payload["authorized_runtime_execution_code_sha256"] == AUTHORIZED_RUNTIME
    assert payload["fixed_work_execution_code_sha256"] == FIXED_EXECUTION
    assert json.loads(manifest.read_text(encoding="ascii")) == payload

    assert migrate(
        source, output, manifest, FIXED_EXECUTION, tmp_path / "common"
    ) == payload


def test_migration_rejects_non_self_consistent_source_hashes(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "source.env",
        overrides={CONTINUATION_KEY: "4" * 64},
    )
    with pytest.raises(MigrationError, match="execution and continuation"):
        migrate(
            source,
            tmp_path / "caps.env",
            tmp_path / "manifest.json",
            FIXED_EXECUTION,
            tmp_path / "common",
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "malformed"])
def test_migration_requires_exactly_one_valid_provenance_hash(
    tmp_path: Path, mutation: str
) -> None:
    source = _source(tmp_path / "source.env")
    text = source.read_text(encoding="ascii")
    line = f"export {AUTHORIZED_KEY}={AUTHORIZED_RUNTIME}\n"
    if mutation == "missing":
        text = text.replace(line, "")
    elif mutation == "duplicate":
        text += line
    else:
        text = text.replace(AUTHORIZED_RUNTIME, "not-a-sha256")
    source.write_text(text, encoding="ascii")
    with pytest.raises(MigrationError):
        migrate(
            source,
            tmp_path / "caps.env",
            tmp_path / "manifest.json",
            FIXED_EXECUTION,
            tmp_path / "common",
        )


def test_migration_refuses_stale_output_or_manifest(tmp_path: Path) -> None:
    source = _source(tmp_path / "source.env")
    output = tmp_path / "caps.env"
    manifest = tmp_path / "manifest.json"
    migrate(source, output, manifest, FIXED_EXECUTION, tmp_path / "common")

    output.write_text("stale\n", encoding="ascii")
    with pytest.raises(MigrationError, match="migrated cap contract is stale"):
        migrate(source, output, manifest, FIXED_EXECUTION, tmp_path / "common")

    output.unlink()
    manifest.write_text("{}\n", encoding="ascii")
    with pytest.raises(MigrationError, match="cap migration manifest is stale"):
        migrate(source, output, manifest, FIXED_EXECUTION, tmp_path / "common")
