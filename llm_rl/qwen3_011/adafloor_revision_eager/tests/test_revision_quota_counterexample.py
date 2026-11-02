import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis_eval'))
import build_revision_quota_counterexample as cases


class CounterexampleTests(unittest.TestCase):
    def test_prompt_blocks_are_included(self):
        self.assertEqual(cases.kv_peak([[100], [100]], [50, 50], 1, 1000), 512)
        self.assertEqual(cases.kv_peak([[100], [100]], [0, 0], 1, 1000), 256)

    def test_independent_release(self):
        pairs = [[2*i, 2*i+1] for i in range(16)]
        tails = {i: i//2+1 for i in range(32)}
        self.assertEqual(cases.release(pairs, tails, 4), 8*8+4*4)

    def test_binding_kv_case_beats_minskew(self):
        cases.planner._progress = lambda _: None
        result = cases.compare(cases.figure4_rows(), 4, 133120)
        self.assertTrue(result['proxy_optimality_certified'])
        self.assertGreater(result['gain_rank_tokens'], 0)
        for row in result['policies'].values():
            self.assertTrue(row['feasible'])
            self.assertLessEqual(row['guarded_kv_peak_tokens'], 133120)
        self.assertEqual(result['policies']['min_skew']['thresholds_tokens'][0], 5400)
        self.assertEqual(result['policies']['release_area']['thresholds_tokens'][0], 4680)

    def test_unconstrained_neighbor_has_no_gain(self):
        cases.planner._progress = lambda _: None
        result = cases.compare(cases.figure4_rows(scale=80), 4, 133120)
        self.assertEqual(result['gain_rank_tokens'], 0)

    def test_jitter_and_identity_validation(self):
        self.assertEqual(cases.figure4_rows(996), cases.figure4_rows(996))
        bad = cases.figure4_rows()
        bad[-1]['id'] = 0
        with self.assertRaises(ValueError):
            cases.compare(bad, 4, 133120)


if __name__ == '__main__':
    unittest.main()
