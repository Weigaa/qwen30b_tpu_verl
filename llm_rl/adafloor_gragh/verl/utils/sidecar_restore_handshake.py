"""Fail-closed coordination between rollout restore and a sidecar lease."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Mapping, Optional


_TRUE_VALUES = {"1", "true", "yes", "on"}


class SidecarRestoreHandshakeError(RuntimeError):
    """Raised when released devices cannot be confirmed before restore."""


def _enabled(env: Mapping[str, str]) -> bool:
    return env.get("VERL_SIDECAR_ENABLE", "0").lower() in _TRUE_VALUES


def _require_active_lease(env: Mapping[str, str]) -> bool:
    return (
        env.get("VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE", "0").lower()
        in _TRUE_VALUES
    )


def _handshake_paths(env: Mapping[str, str]) -> tuple[Path, Path, Path]:
    directory_value = env.get("VERL_SIDECAR_RESTORE_HANDSHAKE_DIR")
    if not directory_value:
        log_dir = env.get("VERL_SIDECAR_LOG_DIR")
        if not log_dir:
            raise SidecarRestoreHandshakeError(
                "sidecar restore handshake requires VERL_SIDECAR_LOG_DIR or "
                "VERL_SIDECAR_RESTORE_HANDSHAKE_DIR"
            )
        directory_value = str(Path(log_dir) / "restore_handshake")
    directory = Path(directory_value)
    return (
        Path(env.get("VERL_SIDECAR_ACTIVE_LEASE_FILE", directory / "active_lease")),
        Path(env.get("VERL_SIDECAR_STOP_REQUEST_FILE", directory / "stop_request")),
        Path(env.get("VERL_SIDECAR_STOP_ACK_FILE", directory / "stop_ack")),
    )


def _read_fields(path: Path) -> dict[str, str]:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    fields: dict[str, str] = {}
    for line in content.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            fields[key] = value
    return fields


def _atomic_write_fields(path: Path, fields: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = "".join(f"{key}={value}\n" for key, value in fields.items())
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def request_sidecar_stop_before_restore(
    env: Optional[Mapping[str, str]] = None,
) -> Optional[dict[str, str]]:
    """Wait until the watcher confirms that the current lease is released.

    Returning ``None`` is reserved for the sidecar-disabled path. Any missing,
    mismatched, rejected, or late acknowledgement raises before the caller can
    issue the full-world restore RPC.
    """

    environment = os.environ if env is None else env
    if not _enabled(environment):
        return None

    try:
        timeout_s = float(
            environment.get("VERL_SIDECAR_STOP_ACK_TIMEOUT_SECONDS", "60")
        )
        poll_s = float(
            environment.get("VERL_SIDECAR_STOP_ACK_POLL_SECONDS", "0.05")
        )
    except ValueError as error:
        raise SidecarRestoreHandshakeError(
            "sidecar restore handshake timeout and poll values must be numeric"
        ) from error
    if timeout_s <= 0 or poll_s <= 0:
        raise SidecarRestoreHandshakeError(
            "sidecar restore handshake timeout and poll values must be positive"
        )

    active_path, request_path, ack_path = _handshake_paths(environment)
    deadline = time.monotonic() + timeout_s
    require_active = _require_active_lease(environment)
    active: dict[str, str] = {}
    while time.monotonic() < deadline:
        active = _read_fields(active_path)
        if active.get("lease_id") and active.get("state"):
            if not require_active:
                break
            try:
                lease_number = int(active["lease_id"])
            except ValueError:
                lease_number = 0
            if active["state"] == "running" and lease_number > 0:
                break
        time.sleep(poll_s)
    else:
        requirement = "active running sidecar lease" if require_active else "sidecar lease state"
        raise SidecarRestoreHandshakeError(
            f"timed out waiting for {requirement} at {active_path}. "
            f"last_state={active.get('state', 'missing')} "
            f"last_lease_id={active.get('lease_id', 'missing')}"
        )

    request_id = uuid.uuid4().hex
    lease_id = active["lease_id"]
    request = {
        "request_id": request_id,
        "lease_id": lease_id,
        "observed_state": active["state"],
        "request_time": f"{time.time():.9f}",
    }
    _atomic_write_fields(request_path, request)

    while time.monotonic() < deadline:
        ack = _read_fields(ack_path)
        if ack.get("request_id") != request_id:
            time.sleep(poll_s)
            continue
        if ack.get("lease_id") != lease_id:
            raise SidecarRestoreHandshakeError(
                "sidecar stop acknowledgement lease mismatch for request "
                f"{request_id}. expected {lease_id}, got {ack.get('lease_id')}"
            )
        if ack.get("status") != "released":
            raise SidecarRestoreHandshakeError(
                "sidecar watcher rejected stop request "
                f"{request_id}. status={ack.get('status', 'missing')} "
                f"reason={ack.get('reason', 'unspecified')}"
            )
        return ack

    raise SidecarRestoreHandshakeError(
        "timed out waiting for sidecar release acknowledgement for "
        f"request {request_id}, lease {lease_id}, ack {ack_path}"
    )
