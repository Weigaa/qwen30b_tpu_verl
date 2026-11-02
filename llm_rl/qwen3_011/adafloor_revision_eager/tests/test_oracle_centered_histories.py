import copy
import importlib.util
import math
from pathlib import Path
import unittest


FILE = Path(__file__).resolve().parents[1] / 'tools/prepare_oracle_centered_histories.py'
SPEC = importlib.util.spec_from_file_location('oracle_histories', FILE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class OracleHistoryTests(unittest.TestCase):
    def fixtures(self):
        def history(length):
            return {'schema_version': 1, 'responses_per_prompt': 16,
                    'length_definition': 'raw_generated_tokens',
                    'records': [{'input': f'prompt {i}', 'lengths': [length + j for j in range(16)]}
                                for i in range(160)]}
        return history(2000), history(1001)

    def test_scale_oracle_not_current_and_preserve_inputs(self):
        current, oracle = self.fixtures()
        before = copy.deepcopy((current, oracle))
        result = m.derive(current, oracle, seed=606, oracle_scope='reference_trajectory')
        self.assertEqual((current, oracle), before)
        self.assertEqual(result['current']['records'], current['records'])
        self.assertEqual(result['oracle']['records'], oracle['records'])
        self.assertEqual(result['under_0p7']['records'][0]['lengths'][0], math.ceil(1001 * .7))
        self.assertEqual(result['over_1p3']['records'][0]['lengths'][0], math.ceil(1001 * 1.3))
        self.assertEqual(result, m.derive(current, oracle, seed=606, oracle_scope='reference_trajectory'))
        factors = {r['lengths'][0] for r in result['mixed']['records']}
        self.assertEqual(factors, {math.ceil(1001 * .7), math.ceil(1001 * 1.3)})

    def test_identity_and_raw_length_contracts(self):
        current, oracle = self.fixtures()
        oracle['records'][0]['input'] = 'wrong workload'
        with self.assertRaises(ValueError):
            m.derive(current, oracle, seed=606, oracle_scope='reference_trajectory')
        current, oracle = self.fixtures()
        oracle['length_definition'] = 'response_mask'
        with self.assertRaises(ValueError):
            m.derive(current, oracle, seed=606, oracle_scope='reference_trajectory')


if __name__ == '__main__':
    unittest.main()
