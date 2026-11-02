from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_adafloor_full_decode_fia_v014opt_tq1.sh"
VERIFIER = ROOT / "tools/verify_qwen3_adafloor_full_decode.py"


def test_dry_run_seals_planned_and_natural_full_graph_contract(tmp_path: Path) -> None:
    common = tmp_path / "common"
    checkpoint = common / (
        "epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5"
    )
    checkpoint.mkdir(parents=True)
    (common / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT").write_text("preserve\n")
    (common / "reuse.env").write_text("test=true\n")
    history = common / "epoch_000_mode0_probe/offline_planning_history.json"
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text("[]\n")
    result_root = tmp_path / "result"
    env = os.environ.copy()
    env["COMMON_EPOCH0_ROOT"] = str(common)
    env["ADAFLOOR_FULL_DECODE_RESULT_ROOT"] = str(result_root)
    result = subprocess.run(
        [str(RUNNER), "dry-run", "all"],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "arm=planned variant=adafloor_p_f4" in result.stdout
    assert "arm=natural variant=adafloor_n_f2" in result.stdout
    protocol = dict(
        line.split("=", 1)
        for line in (result_root / "protocol.env").read_text().splitlines()
    )
    assert protocol["graph_mode"] == "FULL_DECODE_ONLY"
    assert protocol["capture_sizes"] == "[16,32]"
    assert protocol["cudagraph_copy_inputs"] == "false"
    assert protocol["attention_graph"] == "true"
    assert protocol["moe_graph"] == "true"
    assert protocol["task_queue_enable"] == "1"
    assert protocol["max_update_actor_ms_per_token"] == "0.08"
    assert protocol["natural_floor2_kv_tokens"] == "98304"
    assert protocol["planned_floor4_kv_tokens"] == "280576"
    assert protocol["planned_variant"] == "adafloor_p_f4"
    assert protocol["natural_variant"] == "adafloor_n_f2"
    assert protocol["actor_frozen"] == "true"
    assert len((result_root / "code_sha256.txt").read_text().splitlines()) == 20


def test_runner_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_verifier_reuses_strict_graph_checks_with_optimized_buckets() -> None:
    spec = importlib.util.spec_from_file_location("adafloor_full_verify", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    legacy = module.load_legacy()
    assert legacy.CAPTURE_SIZES == [16, 32]
    assert legacy.SEED == 0
    source = VERIFIER.read_text()
    assert "legacy.verify_plan" in source
    assert "legacy.verify_outputs" in source
    assert 'legacy.verify_log(log_path, "graph", policy, plan)' in source
    assert 'POLICIES = ("planned", "natural")' in source


def test_runner_pins_optimized_full_graph_runtime() -> None:
    source = RUNNER.read_text()
    required = (
        "VLLM_ROLLOUT_NATIVE_SLEEP_MODE=0",
        "VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1",
        "VLLM_ROLLOUT_ASYNC_SCHEDULING=false",
        "VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1",
        "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1",
        "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1",
        "ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY",
        "ADAFLOOR_GRAPH_CAPTURE_SIZES=\"$CAPTURE_SIZES\"",
        "TASK_QUEUE_ENABLE=1",
        "FAIR_FREEZE_ACTOR=1",
        "VERL_FULL_DECODE_MAX_UPDATE_ACTOR_SECONDS=",
        "VERL_FULL_DECODE_MAX_UPDATE_ACTOR_MS_PER_TOKEN=0.08",
    )
    for marker in required:
        assert marker in source
