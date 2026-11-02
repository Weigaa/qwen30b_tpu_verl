import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis_eval'))
from build_revision_quota_counterexample import compare, figure4_rows, planner
from certify_revision_quota_optimum import certify, profiles_from_requests


class IndependentOptimumTests(unittest.TestCase):
    def data(self):
        rows = figure4_rows(); planner._progress = lambda _: None
        comparison = compare(rows, 4, 133120)
        requests = [dict(id=r['id'], sample_index=s, max_tokens=length,
                         prompt_token_ids=[1]*r['prompt_tokens'])
                    for r in rows for s, length in enumerate(r['lengths'])]
        return dict(comparison=comparison, requests=requests)

    def test_global_optimum_and_best_min_skew_tie(self):
        result = certify(self.data())
        self.assertEqual(result['best_release']['release_rank_tokens'], 36720)
        self.assertEqual(result['best_release_among_all_minimum_skew']['release_rank_tokens'], 30960)
        self.assertEqual(result['strict_objective_gap_rank_tokens'], 5760)
        self.assertEqual(result['minimum_skew']['total_skew_tokens'], 27720)
        for key in ('best_release', 'best_release_among_all_minimum_skew', 'minimum_skew'):
            self.assertEqual(result[key]['mip_gap'], 0)

    def test_duplicate_or_missing_profile_rejected(self):
        data = self.data(); data['requests'].append(copy.deepcopy(data['requests'][0]))
        with self.assertRaises(ValueError): profiles_from_requests(data)
        data['requests'] = data['requests'][1:-1]
        with self.assertRaises(ValueError): profiles_from_requests(data)

    def test_incorrect_reported_optimum_rejected(self):
        data = self.data(); data['comparison']['policies']['release_area']['release_rank_tokens'] += 1
        with self.assertRaises(ValueError): certify(data)


if __name__ == '__main__': unittest.main()
