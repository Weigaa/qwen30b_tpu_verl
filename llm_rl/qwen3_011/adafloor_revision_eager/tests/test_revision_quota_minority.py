from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis_eval'))
from build_revision_quota_tail_mixture import mixture_rows


class MinorityProfileTests(unittest.TestCase):
    def prompts(self):
        return [dict(prompt_token_ids=[1]*(20+i), prompt_hash=str(i)) for i in range(32)]

    def test_lengths_counts_and_work_preserved(self):
        rows = mixture_rows(self.prompts(), 4, 100)
        self.assertEqual(len(rows), 32)
        self.assertEqual(sum(sum(row['lengths']) for row in rows), 1710400)
        for i, row in enumerate(rows):
            self.assertEqual(len(row['lengths']), 16)
            self.assertEqual(row['lengths'].count(max(row['lengths'])), 4)
            self.assertEqual(row['prompt_tokens'], 20+i)
            self.assertEqual(row['prompt_hash'], str(i))

    def test_undeclared_fraction_rejected(self):
        with self.assertRaises(ValueError): mixture_rows(self.prompts(), 8, 100)
        with self.assertRaises(ValueError): mixture_rows(self.prompts()[:-1], 4, 100)


if __name__ == '__main__': unittest.main()
