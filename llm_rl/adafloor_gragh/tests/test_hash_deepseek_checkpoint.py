from pathlib import Path

import pytest

from tools.hash_deepseek_checkpoint import checkpoint_files, digest


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "global_step_5"
    distcp = checkpoint / "actor" / "dist_ckpt"
    distcp.mkdir(parents=True)
    (checkpoint / ".PRESERVE_COMMON_EPOCH0").touch()
    (distcp / ".metadata").write_bytes(b"metadata")
    (distcp / "__0_0.distcp").write_bytes(b"first")
    (distcp / "__0_1.distcp").write_bytes(b"second")
    return checkpoint


def test_checkpoint_digest_is_deterministic_and_content_sensitive(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    first = digest(checkpoint)
    assert first == digest(checkpoint)
    assert first[1:] == (4, 19)

    (checkpoint / "actor" / "dist_ckpt" / "__0_1.distcp").write_bytes(b"changed")
    assert digest(checkpoint)[0] != first[0]


def test_checkpoint_hash_requires_distcp_and_preservation_marker(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / ".PRESERVE_COMMON_EPOCH0").unlink()
    with pytest.raises(ValueError, match="preservation marker"):
        checkpoint_files(checkpoint)

    (checkpoint / ".PRESERVE_COMMON_EPOCH0").touch()
    for path in (checkpoint / "actor").rglob("*.distcp"):
        path.unlink()
    with pytest.raises(ValueError, match="no distcp shards"):
        checkpoint_files(checkpoint)
