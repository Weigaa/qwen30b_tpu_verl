from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHER = REPO_ROOT / "internal" / "watch_elastic_shrink_and_run_sidecar.sh"
SIDECAR_STUB = REPO_ROOT / "tests" / "fixtures" / "sidecar_stub.sh"


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _wait_for(predicate, timeout_s: float = 8.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for watcher event")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _fields(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in _text(path).splitlines()
        if "=" in line
    )


def _event_time(path: Path, key: str) -> float:
    prefix = f"{key}="
    for line in _text(path).splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix):].split()[0])
    raise AssertionError(f"missing {key} in {path}")


def _request_release(
    tmp_path: Path, *, expected_state: str = "running"
) -> dict[str, str]:
    handshake_dir = tmp_path / "restore_handshake"
    active_file = handshake_dir / "active_lease"
    request_file = handshake_dir / "stop_request"
    ack_file = handshake_dir / "stop_ack"
    _wait_for(lambda: _fields(active_file).get("state") == expected_state)
    active = _fields(active_file)
    request_id = uuid.uuid4().hex
    temporary = handshake_dir / f"stop_request.{request_id}.tmp"
    temporary.write_text(
        f"request_id={request_id}\n"
        f"lease_id={active['lease_id']}\n"
        f"request_time={time.time():.9f}\n",
        encoding="utf-8",
    )
    temporary.replace(request_file)
    _wait_for(lambda: _fields(ack_file).get("request_id") == request_id)
    return _fields(ack_file)


def _start_watcher(
    tmp_path: Path,
    *,
    start_once: bool,
    require_active: bool = False,
    require_quorum: bool = False,
) -> tuple[subprocess.Popen, Path, Path, Path]:
    train_log = tmp_path / "train.log"
    lease_log = tmp_path / "lease.log"
    stub_record = tmp_path / "stub.log"
    stop_file = tmp_path / "sidecar.stop"
    train_log.touch()

    env = os.environ.copy()
    env.update({
        "SIDECAR_STUB_RECORD": str(stub_record),
        "VERL_SIDECAR_SCRIPT": str(SIDECAR_STUB),
        "VERL_SIDECAR_LEASE_LOG": str(lease_log),
        "VERL_SIDECAR_STOP_FILE": str(stop_file),
        "VERL_SIDECAR_RESTORE_HANDSHAKE_DIR": str(
            tmp_path / "restore_handshake"
        ),
        "VERL_SIDECAR_EXPECTED_ACTIVE_RANKS": "8",
        "VERL_SIDECAR_WORLD_SIZE": "16",
        "VERL_SIDECAR_WATCH_POLL_INTERVAL": "0.02",
        "VERL_SIDECAR_GRACEFUL_KILL_SECONDS": "1",
        "VERL_SIDECAR_START_ONCE": "1" if start_once else "0",
        "VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE": (
            "1" if require_active else "0"
        ),
        "VERL_SIDECAR_REQUIRE_SHRINK_QUORUM": (
            "1" if require_quorum else "0"
        ),
    })
    env.pop("VERL_SIDECAR_NPU_DEVICES", None)
    process = subprocess.Popen(
        [str(WATCHER), str(train_log)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _wait_for(lambda: "watch_start_time=" in _text(lease_log))
    return process, train_log, lease_log, stub_record


def _stop_watcher(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


def test_start_once_zero_rearms_for_two_rollout_windows(tmp_path: Path) -> None:
    process, train_log, lease_log, stub_record = _start_watcher(
        tmp_path, start_once=False
    )
    try:
        _append_line(
            train_log,
            "Elastic parallel shrink done: rank=0 "
            "active_ranks=[0, 1, 2, 3, 4, 5, 6, 7] total_ms=1.0",
        )
        _wait_for(lambda: "devices=8,9,10,11,12,13,14,15" in _text(stub_record))
        first_ack = _request_release(tmp_path)
        _wait_for(lambda: _text(stub_record).count("stopped=1") == 1)
        _wait_for(lambda: _text(lease_log).count("sidecar_rearm_time=") == 1)
        assert first_ack["status"] == "released"
        assert first_ack["lease_id"] == "1"

        _append_line(
            train_log,
            "Elastic parallel shrink done: rank=0 "
            "active_ranks=[8, 9, 10, 11, 12, 13, 14, 15] total_ms=1.0",
        )
        _wait_for(lambda: "devices=0,1,2,3,4,5,6,7" in _text(stub_record))
        second_ack = _request_release(tmp_path)
        _wait_for(lambda: _text(stub_record).count("stopped=1") == 2)
        _wait_for(lambda: _text(lease_log).count("sidecar_rearm_time=") == 2)
        assert second_ack["status"] == "released"
        assert second_ack["lease_id"] == "2"

        assert process.poll() is None
        assert _text(lease_log).count("sidecar_start_time=") == 2
        assert _text(lease_log).count("sidecar_lease_index=") == 2
    finally:
        _stop_watcher(process)


def test_start_once_one_exits_after_first_rollout_window(tmp_path: Path) -> None:
    process, train_log, lease_log, stub_record = _start_watcher(
        tmp_path, start_once=True
    )
    try:
        _append_line(
            train_log,
            "Elastic parallel shrink done: rank=0 "
            "active_ranks=[0, 1, 2, 3, 4, 5, 6, 7] total_ms=1.0",
        )
        _wait_for(lambda: "devices=8,9,10,11,12,13,14,15" in _text(stub_record))
        ack = _request_release(tmp_path)
        _wait_for(lambda: process.poll() is not None)

        assert process.returncode == 0
        assert ack["status"] == "released"
        assert ack["lease_id"] == "1"
        request_time = _event_time(lease_log, "trainer_stop_request_time")
        exit_time = _event_time(lease_log, "sidecar_exit_confirmed_time")
        ack_time = _event_time(lease_log, "watcher_restore_ack_time")
        assert request_time <= exit_time <= ack_time
        assert ack_time == float(ack["ack_time"])
        assert _text(lease_log).index("sidecar_artifacts_durable_time=") < (
            _text(lease_log).index("watcher_restore_ack_time=")
        )
        assert _text(stub_record).count("stopped=1") == 1
        assert _text(lease_log).count("sidecar_start_time=") == 1
        assert "sidecar_rearm_time=" not in _text(lease_log)
        assert "watch_end_time=" in _text(lease_log)
    finally:
        _stop_watcher(process)


def test_require_active_mode_defers_log_stop_until_trainer_request(
    tmp_path: Path,
) -> None:
    process, train_log, lease_log, stub_record = _start_watcher(
        tmp_path, start_once=True, require_active=True
    )
    active_file = tmp_path / "restore_handshake" / "active_lease"
    try:
        _append_line(
            train_log,
            "Elastic parallel shrink done: rank=0 "
            "active_ranks=[0, 1, 2, 3, 4, 5, 6, 7] total_ms=1.0",
        )
        _wait_for(lambda: _fields(active_file).get("state") == "running")
        _append_line(
            train_log,
            "Mode1 step timeline: driver_generate_done "
            "driver_generate_done_time=100.0",
        )
        _append_line(
            train_log,
            "Elastic parallel restore requested before rollout restore rpc",
        )
        _wait_for(
            lambda: "sidecar_deadline_signal_deferred_time=" in _text(lease_log)
        )

        assert _fields(active_file)["state"] == "running"
        assert "stopped=1" not in _text(stub_record)
        ack = _request_release(tmp_path)
        _wait_for(lambda: process.poll() is not None)

        assert ack["status"] == "released"
        assert ack["lease_id"] == "1"
        assert _text(stub_record).count("stopped=1") == 1
        assert _text(lease_log).index(
            "sidecar_deadline_signal_deferred_time="
        ) < _text(lease_log).index("trainer_stop_request_time=")
    finally:
        _stop_watcher(process)


def test_legacy_mode_keeps_log_driven_stop_behavior(tmp_path: Path) -> None:
    process, train_log, lease_log, stub_record = _start_watcher(
        tmp_path, start_once=True, require_active=False
    )
    active_file = tmp_path / "restore_handshake" / "active_lease"
    try:
        _append_line(
            train_log,
            "Elastic parallel shrink done: rank=0 "
            "active_ranks=[0, 1, 2, 3, 4, 5, 6, 7] total_ms=1.0",
        )
        _wait_for(lambda: _fields(active_file).get("state") == "running")
        _append_line(
            train_log,
            "Elastic parallel restore requested before rollout restore rpc",
        )
        _wait_for(lambda: _fields(active_file).get("state") == "released")

        assert _text(stub_record).count("stopped=1") == 1
        assert "sidecar_deadline_signal_time=" in _text(lease_log)
        assert "sidecar_deadline_signal_deferred_time=" not in _text(lease_log)
        ack = _request_release(tmp_path, expected_state="released")
        _wait_for(lambda: process.poll() is not None)
        assert ack["status"] == "released"
    finally:
        _stop_watcher(process)


def test_strict_shrink_quorum_requires_all_unique_world_ranks(
    tmp_path: Path,
) -> None:
    process, train_log, lease_log, stub_record = _start_watcher(
        tmp_path,
        start_once=True,
        require_active=True,
        require_quorum=True,
    )
    target_active = "8, 9, 10, 11, 12, 13, 14, 15"
    other_active = "0, 1, 2, 3, 4, 5, 6, 7"

    def worker_done(rank: int, active: str) -> str:
        return (
            f"Elastic parallel shrink done: rank={rank} "
            f"active_ranks=[{active}] total_ms=1.0"
        )

    def rpc_done(rank: int, active: str) -> str:
        return (
            f"Elastic parallel shrink rpc done: global_rank={rank} "
            f"active_ranks=[{active}] total_ms=1.0"
        )

    try:
        _append_line(train_log, worker_done(0, target_active))
        _wait_for(lambda: "quorum_count=1 quorum_required=16" in _text(lease_log))
        assert not stub_record.exists()

        _append_line(train_log, rpc_done(0, target_active))
        for rank in range(8):
            _append_line(train_log, rpc_done(rank, other_active))
        _wait_for(
            lambda: "active_ranks=0,1,2,3,4,5,6,7 reporter=7 "
            "quorum_count=8" in _text(lease_log)
        )
        assert not stub_record.exists()

        for rank in range(1, 15):
            _append_line(train_log, rpc_done(rank, target_active))
        _append_line(train_log, worker_done(7, target_active))
        _wait_for(
            lambda: "active_ranks=8,9,10,11,12,13,14,15 reporter=14 "
            "quorum_count=15" in _text(lease_log)
        )
        assert not stub_record.exists()

        _append_line(train_log, rpc_done(15, target_active))
        _wait_for(
            lambda: "devices=0,1,2,3,4,5,6,7" in _text(stub_record)
        )

        lease_text = _text(lease_log)
        assert "quorum_ranks=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" in lease_text
        assert "quorum_count=16 quorum_required=16" in lease_text
        assert "coordinated_start_time=" in lease_text
        assert lease_text.count("sidecar_start_time=") == 1
        ack = _request_release(tmp_path)
        _wait_for(lambda: process.poll() is not None)
        assert ack["status"] == "released"
    finally:
        _stop_watcher(process)


def test_strict_shrink_quorum_never_falls_back_for_malformed_done_lines(
    tmp_path: Path,
) -> None:
    process, train_log, lease_log, stub_record = _start_watcher(
        tmp_path,
        start_once=True,
        require_active=True,
        require_quorum=True,
    )
    try:
        for rank in range(16):
            _append_line(
                train_log,
                "Elastic parallel shrink done "
                f"rank={rank} active_ranks=[8, 9, 10, 11, 12, 13, 14, 15] "
                "total_ms=1.0",
            )
        _wait_for(
            lambda: _text(lease_log).count("reason=unrecognized_done_format")
            == 16
        )

        assert not stub_record.exists()
        assert "sidecar_start_time=" not in _text(lease_log)
        assert _fields(
            tmp_path / "restore_handshake" / "active_lease"
        )["state"] == "armed"
    finally:
        _stop_watcher(process)
