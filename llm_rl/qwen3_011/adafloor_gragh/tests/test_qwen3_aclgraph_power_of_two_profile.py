import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_aclgraph_power_of_two_bucket_profile.sh"
COMBINER = ROOT / "analysis_eval/combine_aclgraph_power_of_two_profile.py"


def test_power_of_two_runner_contract() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    source = RUNNER.read_text(encoding="utf-8")
    assert "COUNTS=3,4,5" in source
    assert "LAUNCH_ORDER=3,5,4" in source
    assert (
        "CAPTURE_SETS='\"" not in source
    )
    assert (
        "CAPTURE_SETS='{\"3\":[8,16,32],\"4\":[4,8,16,32],"
        "\"5\":[2,4,8,16,32]}'"
    ) in source
    assert "ACLGRAPH_BUCKET_PROFILE_EXPLICIT_CAPTURE_SETS" in source
    assert "reference count-grid profile is not PASS" in source
    assert "combine_results" in source


def test_combiner_declares_exact_power_of_two_sets() -> None:
    namespace: dict[str, object] = {}
    source = COMBINER.read_text(encoding="utf-8")
    exec(compile(source.split("def _sha", 1)[0], str(COMBINER), "exec"), namespace)
    assert namespace["TARGET_CAPTURE_SETS"] == [
        [32],
        [16, 32],
        [8, 16, 32],
        [4, 8, 16, 32],
        [2, 4, 8, 16, 32],
    ]


def test_selector_accepts_prespecified_power_of_two_sets(tmp_path: Path) -> None:
    history = tmp_path / "history.json"
    records = [{"lengths": [1 + (index % 32)] * 16} for index in range(160)]
    history.write_text(json.dumps({"records": records}), encoding="utf-8")
    train = tmp_path / "train.parquet"
    import pandas as pd

    pd.DataFrame({"row": range(160)}).to_parquet(train, index=False)
    plan = tmp_path / "plan.json"
    subset = tmp_path / "subset.parquet"
    capture_sets = '{"3":[8,16,32],"4":[4,8,16,32],"5":[2,4,8,16,32]}'
    subprocess.run(
        [
            "python3",
            str(ROOT / "analysis_eval/select_aclgraph_step_buckets.py"),
            "--history",
            str(history),
            "--train-file",
            str(train),
            "--source-step",
            "4",
            "--counts",
            "3,4,5",
            "--capture-sets-json",
            capture_sets,
            "--output-plan",
            str(plan),
            "--output-train",
            str(subset),
        ],
        check=True,
    )
    payload = json.loads(plan.read_text(encoding="utf-8"))
    assert payload["method"] == "prespecified_capture_sets"
    assert [item["capture_sizes"] for item in payload["candidates"]] == [
        [8, 16, 32],
        [4, 8, 16, 32],
        [2, 4, 8, 16, 32],
    ]
