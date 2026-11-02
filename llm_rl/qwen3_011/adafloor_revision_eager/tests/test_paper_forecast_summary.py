import importlib.util
from pathlib import Path
import unittest

FILE = Path(__file__).resolve().parents[1] / 'analysis_eval/summarize_paper_aligned_forecast.py'
SPEC = importlib.util.spec_from_file_location('summary', FILE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class SummaryTests(unittest.TestCase):
    def test_checkpoint_inventory_requires_all_nonempty_shards(self):
        import tempfile
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            folder = root / 'actor/dist_ckpt'
            folder.mkdir(parents=True)
            for name in ('.metadata', 'metadata.json', 'common.pt'):
                (folder / name).write_text('test')
            (root / 'data.pt').write_text('test')
            for rank in range(16):
                for part in range(2):
                    (folder / f'__{rank}_{part}.distcp').write_text('test')
            self.assertTrue(m.checkpoint_inventory(root)['complete'])
            (folder / '__3_1.distcp').write_text('')
            self.assertFalse(m.checkpoint_inventory(root)['complete'])
            (folder / '__3_1.distcp').unlink()
            self.assertFalse(m.checkpoint_inventory(root)['complete'])

    def test_explicit_tailguard_cap_precedes_short_step_fallback(self):
        self.assertEqual(m.effective_cap({'selected_floor': 2, 'predicted_step_exit': 4085,
                                         'tail_guard_response_cap': 5120}), 5120)
        self.assertEqual(m.effective_cap({'selected_floor': 2, 'predicted_step_exit': 4085}), 4096)

    def test_signed_scientific_notation(self):
        row = m.metrics('\x1b[36m(TaskRunner)\x1b[0m step:1 - actor/ppo_kl:-1.25e-11 - actor/lr:0.0 - timing_s/gen:2.5e+2')[1]
        self.assertEqual(row['actor/ppo_kl'], -1.25e-11)
        self.assertEqual(row['timing_s/gen'], 250.0)

    def test_paper_area_and_rank_duplicate_deduplication(self):
        text = '\n'.join([
            '2026-10-12 10:00:01,000 Elastic EP rank exit start: active_ranks=[8,9,10,11,12,13,14,15]',
            '2026-10-12 10:00:02,000 Elastic EP rank exit start: active_ranks=[8,9,10,11,12,13,14,15]',
            '2026-10-12 10:00:05,000 Elastic EP rank exit start: active_ranks=[12,13,14,15]',
            '2026-10-12 10:00:10,000 rollout_output_time_s: 10'])
        self.assertEqual(m.paper_release(text)[0]['released_rank_s'], 8 * 8 + 4 * 5)

    def test_empty_run_is_pending_not_pass(self):
        import tempfile
        with tempfile.TemporaryDirectory() as path:
            self.assertEqual(m.audit_epoch(path, True)['status'], 'PENDING')

    def test_completed_measurement_keeps_failed_preemption_gate_and_metrics(self):
        import tempfile
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            for name in ('logs', 'rollout_data', 'rollout_length'):
                (root / name).mkdir()
            lines = []
            for step in range(1, 6):
                (root / f'rollout_data/{step}.jsonl').write_text('{}\n' * 512)
                (root / f'rollout_length/length_{step}.txt').write_text('1\n' * 512)
                lines += [f'2026-10-12 10:00:0{step},000 rollout_output_time_s: 1',
                          f'step:{step} - actor/lr:0.0 - critic/rewards/mean:1 - response_length/mean:1 - response_length/max:1 - response/aborted_ratio:0 - timing_s/gen:1 - timing_s/step:2']
            lines += ['Training Progress: 100%', 'After trainer.fit', 'Preempting request 3 for request 1']
            (root / 'logs/test.txt').write_text('\n'.join(lines))
            row = m.audit_epoch(root, True)
            self.assertEqual(row['status'], 'FAIL')
            self.assertTrue(row['measurement_complete'])
            self.assertFalse(row['zero_preemption_passed'])
            self.assertEqual(row['summary']['rollout_tokens_per_s'], 512)


if __name__ == '__main__':
    unittest.main()
