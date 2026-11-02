import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/create_revision_forecast_scenarios.py"


class ForecastScenarioTests(unittest.TestCase):
    def history(self, root):
        source = root / "offline_planning_history.json"
        payload = {"schema_version": 1, "steps": 5, "responses_per_prompt": 16,
                   "prompt_count": 160,
                   "records": [{"input": f"prompt-{i}", "lengths": [100+i]*16}
                               for i in range(160)]}
        source.write_text(json.dumps(payload))
        (root / "rollout_data").mkdir()
        (root / "rollout_length").mkdir()
        for step in range(1, 6):
            (root / "rollout_data" / f"{step}.jsonl").write_text("")
            (root / "rollout_length" / f"length_{step}.txt").write_text("")
        return source

    def test_scenarios_are_deterministic_and_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.history(root)
            before = source.read_bytes()
            out = root / "scenarios"
            subprocess.run(["python", str(TOOL), "--history", str(source),
                            "--output-root", str(out), "--mixed-seed", "505"], check=True)
            manifest = json.loads((out / "scenario_manifest.json").read_text())
            self.assertEqual(source.read_bytes(), before)
            under = json.loads(Path(manifest["scenarios"]["under_0p7"]["history"]).read_text())
            over = json.loads(Path(manifest["scenarios"]["over_1p3"]["history"]).read_text())
            self.assertEqual(under["records"][0]["lengths"][0], 70)
            self.assertEqual(over["records"][0]["lengths"][0], 130)
            mixed = json.loads(Path(manifest["scenarios"]["mixed"]["history"]).read_text())
            vals = {row["lengths"][0] / (100 + i) for i, row in enumerate(mixed["records"])}
            self.assertTrue(all(min(abs(v - 0.7), abs(v - 1.3)) < 0.02 for v in vals))
            self.assertGreater(manifest["scenarios"]["mixed"]["under_fraction"], 0.0)
            self.assertLess(manifest["scenarios"]["mixed"]["under_fraction"], 1.0)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.history(root)
            out = root / "scenarios"
            subprocess.run(["python", str(TOOL), "--history", str(source),
                            "--output-root", str(out)], check=True)
            result = subprocess.run(["python", str(TOOL), "--history", str(source),
                                     "--output-root", str(out)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
