import unittest
from code_task import build_training, compute_score, COMMON, DATA, CODE,TRAIN


class CodeProtocolTests(unittest.TestCase):
    def test_only_requested_additional_task_pair(self):
        for policy in ('vanilla', 'planned_f4'):
            env, command = build_training(policy)
            self.assertEqual(env['COMMON_EPOCH0_ROOT'], str(COMMON))
            self.assertEqual(env['TRAIN_FILE'], str(TRAIN))
            self.assertEqual(env['TRAIN_FILE_ORIG'], env['TRAIN_FILE'])
            for key in ('FAIR_DATASET_FRACTION','DYNAMIC_DATASET_FRACTION','DATASET_FRACTION_FOR_ORACLE'):
                self.assertEqual(env[key],'1.0')
            self.assertEqual(env['FAIR_FREEZE_ACTOR'], '1')
            self.assertEqual(env['FAIR_TOTAL_EPOCHS'], '2')
            self.assertEqual(env['VERL_PAIRED_REQUEST_SAMPLING_SEEDS'], '1')
            self.assertEqual(env['TASK_QUEUE_ENABLE'], '1')
            self.assertEqual(env['VLLM_ROLLOUT_SLEEP_LEVEL'], '2')
            self.assertEqual(env['VANILLA_KV_TOKENS_PER_RANK'], '380800')
            self.assertTrue(env['PATH'].startswith(str(CODE/'offline_bin')+':'))
            self.assertEqual(env['MAX_CROSS_STEP_REPAIR_SWAPS'],'32')
            self.assertEqual(env['FAIR_CODE_CHAT_TOKENIZER'],'/data/Qwen3-30B-A3B')
            self.assertIn('actor_rollout_ref.rollout.seed=401', command)
            self.assertIn(f'custom_reward_function.path={CODE / "code_task.py"}', command)
            self.assertEqual([x for x in command if x.startswith('actor_rollout_ref.actor.optim.lr=')][-1],
                             'actor_rollout_ref.actor.optim.lr=0.0')
        with self.assertRaises(ValueError):
            build_training('natural_f2')

    def test_dummy_reward_never_executes_generated_code(self):
        self.assertEqual(compute_score('apps', 'raise RuntimeError("must not run")', 'invalid'), 0.0)


if __name__ == '__main__':
    unittest.main()
