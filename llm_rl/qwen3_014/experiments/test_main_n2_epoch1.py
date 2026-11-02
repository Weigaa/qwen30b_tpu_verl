"""Configuration checks only, never load models or launch training."""
from pathlib import Path
import unittest
from run_main_n2_graph_sidecar_epoch1 import build_launch,CAPS


class FormalEpochTests(unittest.TestCase):
    def test_normal_five_step_training_not_short_fixture(self):
        h,e,c=build_launch(Path('/tmp/n2_epoch1_configuration_only'))
        expected={'FAIR_START_EPOCH':'1','FAIR_TOTAL_EPOCHS':'2','FAIR_TRAIN_STEPS':'5',
                  'FAIR_TRAIN_BATCH_SIZE':'32','FAIR_ROLLOUT_N':'16',
                  'FAIR_MAX_RESPONSE_LENGTH':'16384','FAIR_FREEZE_ACTOR':'0',
                  'VERL_PAIRED_REQUEST_SAMPLING_SEEDS':'0','FLOOR_KV_CAPS':CAPS}
        for k,v in expected.items():self.assertEqual(e[k],v)
        self.assertIn('actor_rollout_ref.actor.optim.lr=1e-6',c)
        self.assertFalse(any(k.startswith('VERL_ELASTIC_TAIL_VALIDATE_') for k in e))

    def test_formal_sidecar_and_native_graph_lifecycle(self):
        _,e,_=build_launch(Path('/tmp/n2_epoch1_configuration_only'))
        for k,v in {'TASK_QUEUE_ENABLE':'1','VLLM_ROLLOUT_SLEEP_LEVEL':'2',
                    'VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE':'1',
                    'VERL_SIDECAR_ENFORCE_EAGER':'0','VERL_SIDECAR_TASK_QUEUE_ENABLE':'1',
                    'VERL_SIDECAR_MAX_TOKENS':'4096','VERL_SIDECAR_MAX_NUM_SEQS':'128',
                    'VERL_SIDECAR_REPEAT_UNTIL_KILLED':'1','VERL_SIDECAR_WINDOW_GATE_ENABLE':'1',
                    'VERL_SIDECAR_WINDOW_GATE_MIN_SECONDS':'60.0',
                    'VERL_SIDECAR_WINDOW_GATE_STRICT_MINIMUM':'1',
                    'VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8':'349952'}.items():
            self.assertEqual(e[k],v)


if __name__=='__main__':unittest.main()
