from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from verl.utils.sidecar_restore_handshake import (
    SidecarRestoreHandshakeError,
    request_sidecar_stop_before_restore,
)


def _write_fields(path: Path, fields: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in fields.items()),
        encoding="utf-8",
    )


def test_disabled_sidecar_does_not_touch_handshake_files(tmp_path: Path) -> None:
    result = request_sidecar_stop_before_restore({
        "VERL_SIDECAR_ENABLE": "0",
        "VERL_SIDECAR_RESTORE_HANDSHAKE_DIR": str(tmp_path),
    })

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_waits_for_matching_lease_release_ack(tmp_path: Path) -> None:
    active = tmp_path / "active_lease"
    request = tmp_path / "stop_request"
    ack = tmp_path / "stop_ack"
    _write_fields(active, {"lease_id": "7", "state": "running"})
    observed: dict[str, str] = {}

    def acknowledge() -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not request.exists():
            time.sleep(0.005)
        fields = dict(
            line.split("=", 1)
            for line in request.read_text(encoding="utf-8").splitlines()
        )
        observed.update(fields)
        _write_fields(ack, {
            "request_id": fields["request_id"],
            "lease_id": fields["lease_id"],
            "status": "released",
        })

    thread = threading.Thread(target=acknowledge)
    thread.start()
    result = request_sidecar_stop_before_restore({
        "VERL_SIDECAR_ENABLE": "1",
        "VERL_SIDECAR_RESTORE_HANDSHAKE_DIR": str(tmp_path),
        "VERL_SIDECAR_STOP_ACK_TIMEOUT_SECONDS": "1",
        "VERL_SIDECAR_STOP_ACK_POLL_SECONDS": "0.005",
    })
    thread.join(timeout=2)

    assert result is not None
    assert result["status"] == "released"
    assert observed["lease_id"] == "7"
    assert observed["observed_state"] == "running"


def test_timeout_fails_before_restore_can_run(tmp_path: Path) -> None:
    _write_fields(
        tmp_path / "active_lease",
        {"lease_id": "3", "state": "running"},
    )
    restore_called = False

    with pytest.raises(SidecarRestoreHandshakeError, match="timed out"):
        request_sidecar_stop_before_restore({
            "VERL_SIDECAR_ENABLE": "1",
            "VERL_SIDECAR_RESTORE_HANDSHAKE_DIR": str(tmp_path),
            "VERL_SIDECAR_STOP_ACK_TIMEOUT_SECONDS": "0.04",
            "VERL_SIDECAR_STOP_ACK_POLL_SECONDS": "0.005",
        })
        restore_called = True

    assert not restore_called
    assert (tmp_path / "stop_request").exists()


def test_required_active_lease_rejects_armed_state(tmp_path: Path) -> None:
    _write_fields(
        tmp_path / "active_lease",
        {"lease_id": "0", "state": "armed"},
    )

    with pytest.raises(
        SidecarRestoreHandshakeError,
        match="active running sidecar lease.*last_state=armed",
    ):
        request_sidecar_stop_before_restore({
            "VERL_SIDECAR_ENABLE": "1",
            "VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE": "1",
            "VERL_SIDECAR_RESTORE_HANDSHAKE_DIR": str(tmp_path),
            "VERL_SIDECAR_STOP_ACK_TIMEOUT_SECONDS": "0.04",
            "VERL_SIDECAR_STOP_ACK_POLL_SECONDS": "0.005",
        })

    assert not (tmp_path / "stop_request").exists()
