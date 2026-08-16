from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools/verify_adafloor_aclgraph_smoke.py"
SPEC = importlib.util.spec_from_file_location("verify_adafloor_aclgraph_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_fixture(
    tmp_path: Path,
    *,
    full_moe_capture: bool = False,
    native_full_decode: bool = False,
    executed_steps: int = 2,
    capture_sizes: tuple[int, ...] = (1, 2),
) -> Path:
    assert executed_steps in (1, 2)
    if native_full_decode:
        full_moe_capture = True
    root = tmp_path / "smoke"
    (root / "oracle").mkdir(parents=True)
    (root / "run/logs").mkdir(parents=True)
    (root / "rollout_data").mkdir()
    (root / "rollout_length").mkdir()
    (root / "protocol.env").write_text(
        "schema_version=1\n"
        "experiment=qwen3_adafloor_full_decode_dynamic_gate\n"
        "plan_steps=2\n"
        f"executed_steps={executed_steps}\n",
        encoding="utf-8",
    )

    graph_dirs = []
    for rank in range(16):
        graph_dir = (
            root
            / "cache/vllm/torch_compile_cache"
            / f"hash_{rank}"
            / f"rank_{rank}_{rank}"
            / "backbone"
        )
        graph_dir.mkdir(parents=True)
        graph_dirs.append(graph_dir)
        classes = []
        if full_moe_capture:
            body = ["            x = torch.ops.aten.relu.default(x)\n"]
            for _ in range(48):
                body.extend(
                    [
                        "            torch.ops.vllm.unified_ascend_attention_with_output(x)\n",
                        "            torch.ops.vllm.elastic_ascend_moe_forward(x)\n",
                    ]
                )
            body.append("            return x\n")
            classes.append(
                "    class submod_0:\n"
                "        def forward(self, x):\n" + "".join(body)
            )
        else:
            for layer in range(48):
                classes.append(
                    f"    class submod_{2 * layer}:\n"
                    "        def forward(self, x):\n"
                    "            x = torch.ops.aten.relu.default(x)\n"
                    "            torch.ops.vllm.unified_ascend_attention_with_output(x)\n"
                    "            return x\n"
                )
                classes.append(
                    f"    class submod_{2 * layer + 1}:\n"
                    "        def forward(self, x):\n"
                    "            return torch.ops.vllm.elastic_ascend_moe_forward(x)\n"
                )
            classes.append(
                "    class submod_96:\n"
                "        def forward(self, x):\n"
                "            return x\n"
            )
        (graph_dir / "computation_graph.py").write_text(
            "import torch\n\nclass GraphModule:\n" + "".join(classes),
            encoding="utf-8",
        )

    plan = [
        {
            "step": 1,
            "selected_floor": 4,
            "shrink_stages": [8, 4],
            "stage_survivor_ranks": [list(range(8, 16)), list(range(12, 16))],
            "tail_guard_enabled": False,
        },
        {
            "step": 2,
            "selected_floor": 16,
            "shrink_stages": [16],
            "stage_survivor_ranks": [list(range(16))],
            "tail_guard_enabled": False,
        },
    ]
    (root / "oracle/length_sorted_rank_plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    summary_plan = [dict(row, feasible=True) for row in plan]
    (root / "oracle/length_sorted_rank_plan_summary.json").write_text(
        json.dumps(summary_plan), encoding="utf-8"
    )

    lines = []
    for rank in range(16):
        prefix = f"(WorkerDict pid={1000 + rank}) "
        graph_markers = (
            [
                prefix + "FULL_DECODE_ONLY compilation enabled on NPU",
                prefix + "Native FULL_DECODE_ONLY ACLGraph enabled: KV write, "
                "Attention read, MoE, and dense decode execute in the outer full graph",
            ]
            if native_full_decode
            else [prefix + "PIECEWISE compilation enabled on NPU"]
        )
        lines.extend(graph_markers + [
            prefix
            + 'non-default args: {"compilation_config": '
            + '{"cudagraph_capture_sizes":'
            + json.dumps(list(capture_sizes), separators=(",", ":"))
            + "}}",
            prefix
            + "Loaded ACLGraph weak-ref compatibility extension "
            + f"sha256={MODULE.EXPECTED_EXTENSION_SHA256}",
            prefix
            + f"Using cache directory: {graph_dirs[rank]} for vLLM's torch.compile",
        ])
        if not native_full_decode:
            lines.extend([
                prefix + "Elastic ACLGraph attention capture enabled",
                prefix
                + ("Elastic ACLGraph MoE capture enabled" if full_moe_capture else "Elastic ACLGraph boundary enabled"),
            ])
        if not full_moe_capture:
            lines.extend([
                prefix + "Graph capturing finished in 5 secs, took 1.66 GiB",
                prefix + "Replaying aclgraph",
            ])
    lines.append("driver_generate_start step=1")
    lines.append("Mode1 pre-resume KV cleanup target_policy=natural")
    if full_moe_capture:
        for rank in range(16):
            prefix = f"(WorkerDict pid={1000 + rank}) "
            lines.extend([
                prefix + "Elastic ACLGraph recapture starting",
                *(
                    [
                        prefix + "FULL_DECODE_ONLY ACLGraph captured Attention KV write and paged read inside the outer model graph",
                        *[
                            prefix + "FULL_DECODE_ONLY Attention maximum workspace captured: seq_len_bucket=6144 bytes=237246976 "
                            + f"shape={shape}"
                            for shape in capture_sizes
                        ],
                    ]
                    if native_full_decode
                    else []
                ),
                prefix + "Graph capturing finished in 3 secs, took 1.66 GiB",
                prefix + "Elastic ACLGraph recapture finished",
                prefix + "Elastic full-MoE ACLGraph replay: generation=0",
                *(
                    [prefix + "Attention ACLGraph metadata update active: shape=1 layers=48"]
                    if native_full_decode
                    else []
                ),
            ])
        for rank in range(8, 16):
            prefix = f"(WorkerDict pid={1000 + rank}) "
            lines.append(
                prefix + "Elastic full-MoE ACLGraph topology capture starting: "
                f"rank={rank} active_ranks={list(range(8, 16))}"
            )
        for rank in range(8, 16):
            prefix = f"(WorkerDict pid={1000 + rank}) "
            lines.extend([
                *(
                    [
                        *[
                            prefix + "FULL_DECODE_ONLY Attention maximum workspace captured: seq_len_bucket=6144 bytes=237246976 "
                            + f"shape={shape}"
                            for shape in capture_sizes
                        ],
                    ]
                    if native_full_decode
                    else []
                ),
                prefix + "Graph capturing finished in 2 secs, took 3.26 GiB",
                prefix + "Elastic full-MoE ACLGraph topology capture finished: "
                f"rank={rank} active_ranks={list(range(8, 16))}",
                prefix + "Elastic full-MoE ACLGraph replay: generation=1",
            ])
    for rank in range(16):
        lines.append(
            "Elastic parallel shrink rpc done: "
            f"global_rank={rank} active_ranks={list(range(8, 16))} "
            "total_ms=11.0"
        )
    for rank in range(8, 16):
        if full_moe_capture:
            prefix = f"(WorkerDict pid={1000 + rank}) "
            if rank >= 12:
                lines.append(
                    prefix + "Elastic full-MoE ACLGraph topology capture starting: "
                    f"rank={rank} active_ranks={list(range(12, 16))}"
                )
    if full_moe_capture:
        for rank in range(12, 16):
            prefix = f"(WorkerDict pid={1000 + rank}) "
            lines.extend([
                *(
                    [
                        *[
                            prefix + "FULL_DECODE_ONLY Attention maximum workspace captured: seq_len_bucket=6144 bytes=237246976 "
                            + f"shape={shape}"
                            for shape in capture_sizes
                        ],
                    ]
                    if native_full_decode
                    else []
                ),
                prefix + "Graph capturing finished in 2 secs, took 3.23 GiB",
                prefix + "Elastic full-MoE ACLGraph topology capture finished: "
                f"rank={rank} active_ranks={list(range(12, 16))}",
                prefix + "Elastic full-MoE ACLGraph replay: generation=2",
            ])
    for rank in range(8, 16):
        lines.append(
            "Elastic parallel shrink rpc done: "
            f"global_rank={rank} active_ranks={list(range(12, 16))} "
            "total_ms=13.0"
        )
    lines.extend(["driver_generate_done step=1", "response/aborted_ratio:0.0"])
    for rank in range(16):
        lines.append(
            "Elastic parallel restore done: "
            f"rank={rank} dp_size=16 ep_size=16 rebuild_ms=1.0 "
            "refresh_ms=1.0 total_ms=3.0"
        )
    if executed_steps == 2:
        lines.append("driver_generate_start step=2")
        for rank in range(16):
            prefix = f"(WorkerDict pid={1000 + rank}) "
            lines.append(prefix + "Elastic ACLGraph recapture starting")
            if full_moe_capture:
                if native_full_decode:
                    lines.extend(
                        prefix + "FULL_DECODE_ONLY Attention maximum workspace captured: seq_len_bucket=6144 bytes=237246976 "
                        + f"shape={shape}"
                        for shape in capture_sizes
                    )
                lines.append(prefix + "Graph capturing finished in 1 secs, took 0.04 GiB")
            lines.append(prefix + "Elastic ACLGraph recapture finished")
            if full_moe_capture:
                lines.append(prefix + "Elastic full-MoE ACLGraph replay: generation=3")
        lines.extend(
            [
                "driver_generate_done step=2",
                "response/aborted_ratio:0.0",
            ]
        )
    lines.extend(["Epoch 0 completed", "After trainer.fit"])
    (root / "run/logs/primary.txt").write_text("\n".join(lines) + "\n", "utf-8")

    for step in range(1, executed_steps + 1):
        rows = []
        for index in range(32):
            rows.append(
                {
                    "step": step,
                    "decoded_response_length": 20,
                    "response_mask": [1] * 20,
                    "responses": list(range(20)),
                    "request_id": index,
                }
            )
        (root / f"rollout_data/{step}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (root / f"rollout_length/length_{step}.txt").write_text(
            "20\n" * 32, encoding="utf-8"
        )
    return root


def test_lifecycle_smoke_fixture_passes(tmp_path: Path) -> None:
    summary = MODULE.verify(_write_fixture(tmp_path))
    assert summary["status"] == "PASS"
    assert summary["plan"]["selected_floors"] == [4, 16]
    assert summary["runtime"]["step_2_recapture_workers"] == list(range(1000, 1016))
    assert summary["outputs"]["1"]["generated_tokens"] == 640
    assert summary["runtime"]["capture_sizes"] == [1, 2]
    assert summary["runtime"]["target_policy"] == "natural"
    assert set(summary["artifacts"]["implementation_sources"]) == set(
        MODULE.IMPLEMENTATION_SOURCES
    )
    assert (
        summary["artifacts"]["ascend_extension"]["sha256"]
        == MODULE.EXPECTED_EXTENSION_SHA256
    )


def test_full_moe_lifecycle_fixture_passes_with_one_partition(tmp_path: Path) -> None:
    summary = MODULE.verify(_write_fixture(tmp_path, full_moe_capture=True))
    assert summary["status"] == "PASS"
    assert summary["runtime"]["full_moe_capture"] is True
    assert summary["runtime"]["capture_count"] == 44
    assert set(summary["runtime"]["full_moe_replay_phases"]) == {
        "step_1_full16",
        "step_1_floor8",
        "step_1_floor4",
        "step_2_full16",
    }
    assert summary["compilation_graphs"]["partitions_per_rank"] == 1
    assert summary["compilation_graphs"]["attention_calls_in_graph_per_rank"] == 48
    assert summary["compilation_graphs"]["elastic_moe_calls_in_graph_per_rank"] == 48
    assert summary["scope"]["eager_boundaries"] == []


def test_native_full_decode_lifecycle_fixture_passes(tmp_path: Path) -> None:
    summary = MODULE.verify(
        _write_fixture(tmp_path, native_full_decode=True)
    )
    assert summary["status"] == "PASS"
    assert summary["runtime"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert summary["runtime"]["native_full_decode"] is True
    assert summary["runtime"]["attention_workspace"] == {
        "capture_events": 88,
        "shapes": [
            {
                "shape": 1,
                "capture_events": 44,
                "seq_len_bucket": 6144,
                "bytes_per_rank_topology": 237246976,
            },
            {
                "shape": 2,
                "capture_events": 44,
                "seq_len_bucket": 6144,
                "bytes_per_rank_topology": 237246976,
            },
        ],
        "kv_write_capture_events": 16,
        "metadata_update_events": 16,
    }
    assert summary["compilation_graphs"]["mode"] == "native_full_decode_only"
    assert summary["compilation_graphs"]["artifact_kind"] == "runtime_aclgraph_capture"


def test_native_full_decode_single_step_fixture_passes(tmp_path: Path) -> None:
    summary = MODULE.verify(
        _write_fixture(
            tmp_path,
            native_full_decode=True,
            executed_steps=1,
        )
    )
    assert summary["status"] == "PASS"
    assert summary["runtime"]["capture_count"] == 28
    assert summary["runtime"]["step_2_recapture_workers"] == []
    assert set(summary["runtime"]["full_moe_replay_phases"]) == {
        "step_1_full16",
        "step_1_floor8",
        "step_1_floor4",
    }
    assert set(summary["outputs"]) == {"1"}
    assert summary["runtime"]["attention_workspace"]["capture_events"] == 56


def test_native_full_decode_single_shape_fixture_passes(tmp_path: Path) -> None:
    summary = MODULE.verify(
        _write_fixture(
            tmp_path,
            native_full_decode=True,
            executed_steps=1,
            capture_sizes=(2,),
        )
    )
    runtime = summary["runtime"]
    assert runtime["capture_sizes"] == [2]
    assert runtime["capture_invocations"] == 28
    assert runtime["captured_graph_instances"] == 28
    assert runtime["resident_full_decode_graphs_per_active_rank"] == 1
    assert runtime["single_full_decode_graph_per_active_rank"] is True
    assert runtime["topology_graph_cache"] is False
    assert runtime["attention_workspace"]["capture_events"] == 28
    assert runtime["attention_workspace"]["shapes"] == [
        {
            "shape": 2,
            "capture_events": 28,
            "seq_len_bucket": 6144,
            "bytes_per_rank_topology": 237246976,
        }
    ]
    assert runtime["prefill_decode_separation"] == {
        "mixed_full_capture_events": 0,
        "mixed_full_replay_events": 0,
        "prefill_execution": "eager",
        "uniform_decode_execution": "FULL_DECODE_ONLY",
    }


@pytest.mark.parametrize(
    "bad_event",
    [
        "Starting to capture ACL graphs for cases: [2], mode: FULL, "
        "uniform_decode: False",
        "Elastic full-MoE ACLGraph replay: generation=0 "
        "batch_descriptor=BatchDescriptor(num_tokens=2, uniform_decode=False)",
    ],
)
def test_native_full_decode_rejects_mixed_prefill_graph(
    tmp_path: Path,
    bad_event: str,
) -> None:
    root = _write_fixture(
        tmp_path,
        native_full_decode=True,
        executed_steps=1,
        capture_sizes=(2,),
    )
    log = root / "run/logs/primary.txt"
    log.write_text(log.read_text("utf-8") + bad_event + "\n", "utf-8")
    with pytest.raises(
        MODULE.ACLGraphSmokeVerificationError,
        match="mixed prefill/decode",
    ):
        MODULE.verify(root)


def test_lifecycle_smoke_rejects_worker_capture_size_disagreement(
    tmp_path: Path,
) -> None:
    root = _write_fixture(tmp_path, full_moe_capture=True)
    log = root / "run/logs/primary.txt"
    text = log.read_text("utf-8")
    old = (
        '(WorkerDict pid=1007) non-default args: {"compilation_config": '
        '{"cudagraph_capture_sizes":[1,2]}}'
    )
    new = old.replace("[1,2]", "[1,4]")
    assert old in text
    log.write_text(text.replace(old, new, 1), "utf-8")
    with pytest.raises(
        MODULE.ACLGraphSmokeVerificationError,
        match="disagree on capture sizes",
    ):
        MODULE.verify(root)


@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        ("Mode1 pre-resume KV cleanup", "expected one runtime target policy"),
        (
            "Mode1 pre-resume KV cleanup target_policy=planned\n"
            "Mode1 pre-resume KV cleanup target_policy=natural",
            "expected one runtime target policy",
        ),
    ],
)
def test_lifecycle_smoke_rejects_missing_or_mixed_target_policy(
    tmp_path: Path,
    replacement: str,
    error: str,
) -> None:
    root = _write_fixture(tmp_path, full_moe_capture=True)
    log = root / "run/logs/primary.txt"
    text = log.read_text("utf-8")
    marker = "Mode1 pre-resume KV cleanup target_policy=natural"
    assert marker in text
    log.write_text(text.replace(marker, replacement, 1), "utf-8")
    with pytest.raises(MODULE.ACLGraphSmokeVerificationError, match=error):
        MODULE.verify(root)


def test_full_moe_lifecycle_rejects_missing_floor_replay(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path, full_moe_capture=True)
    log = root / "run/logs/primary.txt"
    lines = log.read_text("utf-8").splitlines()
    removed = False
    kept = []
    for line in lines:
        if (
            not removed
            and "WorkerDict pid=1012" in line
            and "Elastic full-MoE ACLGraph replay: generation=2" in line
        ):
            removed = True
            continue
        kept.append(line)
    assert removed
    log.write_text("\n".join(kept) + "\n", "utf-8")
    with pytest.raises(MODULE.ACLGraphSmokeVerificationError, match="floor4"):
        MODULE.verify(root)


def test_lifecycle_smoke_rejects_missing_shrink_reporter(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    log = root / "run/logs/primary.txt"
    lines = log.read_text("utf-8").splitlines()
    marker = "global_rank=7 active_ranks=[8, 9, 10, 11, 12, 13, 14, 15]"
    log.write_text("\n".join(line for line in lines if marker not in line) + "\n", "utf-8")
    with pytest.raises(MODULE.ACLGraphSmokeVerificationError, match="reporters"):
        MODULE.verify(root)


def test_lifecycle_smoke_rejects_noncontiguous_response_mask(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    path = root / "rollout_data/1.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    rows[0]["response_mask"][10] = 0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")
    with pytest.raises(MODULE.ACLGraphSmokeVerificationError, match="not contiguous"):
        MODULE.verify(root)


def test_lifecycle_smoke_rejects_attention_left_as_boundary(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    path = next(
        (root / "cache/vllm/torch_compile_cache").glob(
            "*/rank_0_0/backbone/computation_graph.py"
        )
    )
    text = path.read_text("utf-8")
    text = text.replace(
        "            x = torch.ops.aten.relu.default(x)\n",
        "",
        1,
    )
    path.write_text(text, "utf-8")
    with pytest.raises(
        MODULE.ACLGraphSmokeVerificationError, match="standalone attention boundary"
    ):
        MODULE.verify(root)
