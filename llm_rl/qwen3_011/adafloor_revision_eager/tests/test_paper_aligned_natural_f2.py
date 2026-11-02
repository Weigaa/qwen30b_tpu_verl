import importlib.util
from pathlib import Path
import unittest

FILE = Path(__file__).resolve().parents[1] / "tools/run_paper_aligned_natural_f2_forecast.py"
spec = importlib.util.spec_from_file_location("paper_aligned", FILE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PaperAlignedProtocolTest(unittest.TestCase):
    def reference(self):
        return {
            "hydra": {"actor_rollout_ref.rollout.gpu_memory_utilization": "0.9",
                      "actor_rollout_ref.actor.optim.lr": "1e-6",
                      "actor_rollout_ref.rollout.seed": "2", "data.train_files": "/old/plan"},
            "env": dict(runner.HEADER_ENV),
        }

    def test_only_experimental_changes_are_permitted(self):
        ref = self.reference()
        actual = dict(ref["hydra"], **{"actor_rollout_ref.actor.optim.lr": "0.0",
                                      "actor_rollout_ref.rollout.seed": "505",
                                      "data.train_files": "/new/plan"})
        errors, differences = runner.compare_configuration(ref, actual, ref["env"])
        self.assertEqual(errors, [])
        self.assertEqual(len(differences), 3)

    def test_missing_profile_and_modified_kv_are_rejected(self):
        ref = self.reference()
        for key, value in [("VLLM_ASCEND_MODE1_PROFILE_MAX_TOKENS", "17408"),
                           ("VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2", "126976"),
                           ("VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE", "0")]:
            env = dict(ref["env"], **{key: value})
            errors, _ = runner.compare_configuration(ref, ref["hydra"], env)
            self.assertTrue(errors, key)

    def test_memory_utilization_tuning_is_rejected(self):
        ref = self.reference()
        actual = dict(ref["hydra"], **{"actor_rollout_ref.rollout.gpu_memory_utilization": "0.85"})
        errors, _ = runner.compare_configuration(ref, actual, ref["env"])
        self.assertTrue(errors)

    def test_scheduler_preemption_is_not_relabelled(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs/under_0p7/natural_f2/epoch_001_mode1_natural"
            for name in ("rollout_data", "rollout_length", "logs"):
                (run / name).mkdir(parents=True)
            for step in range(1, 6):
                (run / f"rollout_data/{step}.jsonl").write_text('{}\n' * 512)
                (run / f"rollout_length/length_{step}.txt").write_text('1\n' * 512)
            text = ('rollout_output_time_s: 1\n' * 5
                    + 'After trainer.fit\noriginal=17408 cap=2048 effective=2048\n'
                    + 'INFO [scheduler.py:500] Preempting request 3 for request 1\n')
            (run / "logs/test.txt").write_text(text)
            with self.assertRaisesRegex(RuntimeError, 'preemptions=1'):
                runner.audit_run(root, 'under_0p7')

    def test_inherited_revision_settings_removed(self):
        from unittest.mock import patch
        with patch.dict(runner.os.environ, {"VLLM_ASCEND_MODE1_PROFILE_MAX_TOKENS": "17408",
                                           "ADAFLOOR_REVISION_HARD_KV_GUARD": "1",
                                           "PYTHONPATH": "/bad/adafloor_gragh"}):
            env = runner.clean_environment(Path("/workspace/adafloor_revision_results/test"), "under_0p7", True)
        self.assertNotIn("ADAFLOOR_REVISION_HARD_KV_GUARD", env)
        self.assertNotIn("VLLM_ASCEND_MODE1_PROFILE_MAX_TOKENS", env)
        self.assertEqual(env["PYTHONPATH"], str(runner.PAPER))


if __name__ == "__main__":
    unittest.main()
