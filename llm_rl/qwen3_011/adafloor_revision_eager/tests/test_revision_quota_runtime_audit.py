import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis_eval'))
from audit_revision_quota_controlled import audit_arm, sha


class RuntimeAuditTests(unittest.TestCase):
    def fixture(self, root):
        work = root / 'workload.json'
        requests = [dict(id=i, sample_index=s, prompt_hash=str(i), request_seed=i*16+s,
                         max_tokens=100) for i in range(32) for s in range(16)]
        owners = {str(r): [2*r, 2*r+1] for r in range(16)}
        work.write_text(json.dumps(dict(requests=requests, comparison=dict(
            policies={'min_skew': {'rank_to_prompt_ids': owners}}))))
        arm = root / 'min_skew'; (arm / 'primary').mkdir(parents=True)
        (arm / 'protocol.env').write_text('exit_code=0\n')
        (arm / 'primary.log').write_text('completed\n')
        for rank in range(16):
            events = [dict(reporter=rank, active_ranks=list(range(8, 16)), start=1, done=2)]
            if rank >= 8:
                events.append(dict(reporter=rank, active_ranks=list(range(12, 16)), start=5, done=6))
            row = dict(status='CONTROLLED_LENGTH_ARM_FINISHED', rank=rank, policy='min_skew',
                       workload_sha256=sha(work), code_sha256='same', weight_audit_sha256='same',
                       runtime_versions={'vllm': 'same0.11', 'vllm_source': 'same-source'},
                       kv_actual_tokens=133120, kv_target_tokens=133120, start=0,
                       generation_end=10, restore_start=11, restore_end=12, prepared_startup_s=30,
                       requests=[dict(r, generated_tokens=r['max_tokens'], finish_reason='length')
                                 for r in requests if r['id'] in owners[str(rank)]], events=events)
            (arm / 'primary' / f'rank_{rank:02d}.json').write_text(json.dumps(row))
        return arm, work

    def mutate_rank(self, arm, rank, change):
        path = arm / 'primary' / f'rank_{rank:02d}.json'
        row = json.loads(path.read_text()); change(row); path.write_text(json.dumps(row))

    def test_actual_release_excludes_transition_and_restore(self):
        with tempfile.TemporaryDirectory() as root:
            arm, work = self.fixture(Path(root)); out, _ = audit_arm(arm, work, 'min_skew')
            self.assertEqual(out['released_rank_s'], 80)
            self.assertEqual(out['rollout_s'], 10)
            self.assertEqual(out['lifecycle_with_restore_s'], 12)
            self.assertEqual(out['generated_tokens'], 51200)
            self.assertEqual(out['released_fraction_of_rollout_rank_time'], .5)

    def test_missing_quorum_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            arm, work = self.fixture(Path(root))
            self.mutate_rank(arm, 8, lambda row: row.update(events=row['events'][:1]))
            with self.assertRaises(ValueError): audit_arm(arm, work, 'min_skew')

    def test_early_length_and_wrong_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            arm, work = self.fixture(Path(root))
            self.mutate_rank(arm, 0, lambda row: row['requests'][0].update(generated_tokens=99))
            with self.assertRaises(ValueError): audit_arm(arm, work, 'min_skew')

    def test_preemption_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            arm, work = self.fixture(Path(root))
            (arm / 'primary.log').write_text('Preempting request\n')
            with self.assertRaises(ValueError): audit_arm(arm, work, 'min_skew')

    def test_stage_order_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            arm, work = self.fixture(Path(root))
            for rank in range(8, 16):
                self.mutate_rank(arm, rank, lambda row: row['events'][1].update(start=.5, done=1))
            with self.assertRaises(ValueError): audit_arm(arm, work, 'min_skew')


if __name__ == '__main__': unittest.main()
