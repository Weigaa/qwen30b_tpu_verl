import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(os.environ.get('ORACLE_EXPERIMENT_ROOT',
                          '/workspace/adafloor_revision_results/oracle_centered_n2_seed606_v1'))
FILE = ROOT / 'paper_runtime_replay/verl/utils/oracle_control.py'
SPEC = importlib.util.spec_from_file_location('replay_control', FILE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class ReplaySamplerTests(unittest.TestCase):
    def setUp(self):
        m._REFERENCES = {'p': [{'token_ids': [2, 4, 6]} for _ in range(16)],
                         'q': [{'token_ids': [7, 8]} for _ in range(16)]}

    def runner(self, order):
        a = NS(output_token_ids=[2], sampling_params=NS(extra_args={
            'adafloor_replay_key': 'p', 'adafloor_replay_sample': 3}))
        b = NS(output_token_ids=[], sampling_params=NS(extra_args={
            'adafloor_replay_key': 'q', 'adafloor_replay_sample': 8}))
        return NS(input_batch=NS(req_ids=order, num_reqs=len(order)), requests={'a': a, 'b': b})

    def test_batch_reordering_and_constrained_logprobs(self):
        runner = self.runner(['b', 'a'])
        logits = m.force_reference_logits(runner, torch.randn(2, 12))
        self.assertEqual(logits.argmax(-1).tolist(), [7, 4])
        self.assertEqual(torch.softmax(logits, -1).sum(-1).tolist(), [1., 1.])
        self.assertEqual(torch.log_softmax(logits, -1)[[0, 1], [7, 4]].tolist(), [0., 0.])

    def test_preemption_resume_uses_completed_prefix_length(self):
        runner = self.runner(['a'])
        runner.requests['a'].output_token_ids = [2, 4]
        self.assertEqual(m.force_reference_logits(runner, torch.randn(1, 12)).argmax(-1).item(), 6)
        runner.requests['a'].output_token_ids.append(6)
        with self.assertRaises(RuntimeError):
            m.force_reference_logits(runner, torch.randn(1, 12))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(RuntimeError):
            m.force_reference_logits(self.runner(['a', 'b']), torch.randn(1, 12))


if __name__ == '__main__':
    unittest.main()
