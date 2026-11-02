import tempfile
from pathlib import Path
import unittest

import run_n4_graph_sidecar as run


class N4LaunchTests(unittest.TestCase):
    def test_only_n4_primary_with_same_graph_sidecar(self):
        h, env, command = run.build_launch(Path('/tmp/n4_sidecar_config_check'))
        self.assertIn('adafloor_n_f4', command)
        self.assertNotIn('adafloor_n_f2', command)
        self.assertEqual(env['FLOOR_KV_CAPS'], '4:280576,8:349952,16:380800')
        self.assertEqual(env['VERL_SIDECAR_WINDOW_GATE_MIN_SECONDS'], '60.0')
        self.assertEqual(env['VERL_SIDECAR_WINDOW_GATE_STRICT_MINIMUM'], '1')
        self.assertEqual(env['VERL_SIDECAR_TASK_QUEUE_ENABLE'], '1')
        self.assertEqual(env['VLLM_ROLLOUT_SLEEP_LEVEL'], '2')
        self.assertEqual(env['FAIR_TOTAL_EPOCHS'], '3')
        self.assertEqual(env['FAIR_FREEZE_ACTOR'], '0')

    def test_reused_graph_parity_proof_matches_runtime(self):
        self.assertEqual(run.validate_parity(run.helper())['status'], 'PASS')


if __name__=='__main__':
    unittest.main()
