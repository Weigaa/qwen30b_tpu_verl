import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis_eval'))
from find_revision_real_step_quota_cases import read_steps,compare_step


class RealStepSearchTests(unittest.TestCase):
    def fixture(self,root):
        (root/'rollout_data').mkdir();(root/'rollout_length').mkdir()
        rows=[];lengths=[];records=[]
        for prompt in range(32):
            values=[100+prompt]*16
            records.append(dict(input='actual'+str(prompt),lengths=values))
            for length in values:
                rows.append(dict(input='actual'+str(prompt),prompts=[151643,151643,11,12,13]))
                lengths.append(str(length))
        (root/'rollout_data/1.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
        (root/'rollout_length/length_1.txt').write_text('\n'.join(lengths)+'\n')
        history=root/'offline_planning_history.json'
        history.write_text(json.dumps(dict(schema_version=1,responses_per_prompt=16,steps=1,prompt_count=32,
            source_files=[dict(rollout_data='rollout_data/1.jsonl',rollout_length='rollout_length/length_1.txt')],records=records)))
        return history

    def test_original_membership_and_prompt_lengths(self):
        with tempfile.TemporaryDirectory() as root:
            h=self.fixture(Path(root));step=read_steps(h)[0]
            self.assertEqual(len(step['profiles']),32)
            self.assertEqual([r['id'] for r in step['profiles']],list(range(32)))
            self.assertEqual(step['profiles'][0]['prompt_tokens'],3)
            self.assertEqual(step['profiles'][-1]['lengths'],[131]*16)
            self.assertEqual(len(step['source_sha256']),2)

    def test_modified_history_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            h=self.fixture(Path(root));data=json.loads(h.read_text())
            data['records'][0]['lengths'][0]=101;h.write_text(json.dumps(data))
            with self.assertRaises(ValueError):read_steps(h)

    def test_complete_graph_has_no_gain(self):
        with tempfile.TemporaryDirectory() as root:
            rows=read_steps(self.fixture(Path(root)))[0]['profiles']
            result=compare_step(rows,4,133120,3)
            self.assertEqual(result['disposition'],'NO_GAIN')
            self.assertEqual(result['gain_rank_tokens'],0)

    def test_unsafe_budget_not_counted_as_preemption(self):
        with tempfile.TemporaryDirectory() as root:
            rows=read_steps(self.fixture(Path(root)))[0]['profiles']
            result=compare_step(rows,4,1,3)
            self.assertEqual(result['disposition'],'KV_INFEASIBLE')


if __name__=='__main__':unittest.main()
