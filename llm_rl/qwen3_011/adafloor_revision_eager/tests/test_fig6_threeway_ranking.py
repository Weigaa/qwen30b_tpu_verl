from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis_eval'))
import rank_fig6_historical_steps as revised
import rank_fig6_historical_steps_paper_kv as paper


class Figure6RankingTests(unittest.TestCase):
    def test_competition_ties(self):
        self.assertEqual(paper.competition_ranks(dict(ILP=20,MinSkew=20,Fixed4=10)),dict(ILP=1,MinSkew=1,Fixed4=3))
        self.assertEqual(paper.competition_ranks(dict(ILP=20,MinSkew=10,Fixed4=10)),dict(ILP=1,MinSkew=2,Fixed4=2))
        self.assertEqual(paper.competition_ranks(dict(ILP=0,MinSkew=0,Fixed4=0)),dict(ILP=1,MinSkew=1,Fixed4=1))

    def test_no_favored_algorithm_hardcoding(self):
        self.assertEqual(paper.competition_ranks(dict(ILP=1,MinSkew=2,Fixed4=3)),dict(ILP=3,MinSkew=2,Fixed4=1))

    def test_adaptive_equals_quota_but_beats_fixed_floor(self):
        rows=[dict(id=i,prompt_tokens=3,lengths=[100+i]*16) for i in range(32)]
        caps={2:131072,4:280576,8:315648,16:380800}
        for module in (paper,revised):
            result=module.rank_step(rows,caps,3)
            self.assertEqual(result['status'],'RESOLVED')
            self.assertEqual(result['scores'],dict(ILP=168,MinSkew=168,Fixed4=160))
            self.assertEqual(result['ranks'],dict(ILP=1,MinSkew=1,Fixed4=3))

    def test_all_zero_is_tied_first(self):
        rows=[dict(id=i,prompt_tokens=3,lengths=[100]*16) for i in range(32)]
        result=paper.rank_step(rows,{2:131072,4:280576,8:315648},3)
        self.assertTrue(result['all_tied']);self.assertTrue(result['zero_release'])
        self.assertEqual(result['ranks'],dict(ILP=1,MinSkew=1,Fixed4=1))

    def test_each_policy_count_sums_to_total(self):
        records=[dict(comparison=dict(status='RESOLVED',ranks=dict(ILP=1,MinSkew=2,Fixed4=2),
                                      all_tied=False,zero_release=False,ilp_gain_vs_minskew=10)),
                 dict(comparison=dict(status='RESOLVED',ranks=dict(ILP=1,MinSkew=1,Fixed4=1),
                                      all_tied=True,zero_release=True,ilp_gain_vs_minskew=0))]
        result=paper.aggregate(records)
        for policy in paper.POLICIES:self.assertEqual(sum(result['rank_counts'][policy].values()),2)
        self.assertEqual(result['strict_first']['ILP'],1)


if __name__=='__main__':unittest.main()
