from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_full_decode_preemption_recovery_smoke.sh"


def test_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_smoke_requires_preemption_eager_fallback_and_full_resume() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "GRAPH_PREEMPTION_SMOKE_KV_TOKENS:-196608" in source
    assert '"Preempting request"' in source
    assert "local preemption recovery joined the DP-synchronized eager iteration" in source
    assert "DP rank entered synchronized eager recovery" in source
    assert "DP rank resumed FULL graph after synchronized eager recovery" in source
    assert "entered_ranks != set(range(16))" in source
    assert "synchronized_preemption_rounds" in source
    assert "ranks == set(range(16))" in source
    assert "locally verified eager mode on all DP ranks" in source
    assert "no preempted rank returned from synchronized eager to FULL" in source
    assert '"responses": rollout_rows' in source
    assert '"status": "PASS"' in source


def test_smoke_rejects_mixed_dispatch_and_runtime_faults() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for marker in (
        "recovery iteration unexpectedly dispatched a graph",
        "long-tail iteration did not dispatch",
        "Refusing a mixed FULL/eager collective sequence",
        "NPU out of memory",
        "RayTaskError",
    ):
        assert marker in source
    assert r"\bERR\d+\b" in source
    assert "ERR.*HCCL" not in source
