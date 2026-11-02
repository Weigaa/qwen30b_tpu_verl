"""CPU protocol and model-routing tests, not performance measurements."""
from pathlib import Path
import unittest

import run_n4_sidecar_models as m


class ModelProtocolTests(unittest.TestCase):
    def test_same_n4_primary_for_both_secondary_models(self):
        root=Path('/tmp/n4_model_protocol_cpu_check')
        for name,profile in m.PROFILES.items():
            _,env,command=m.build_launch(root/name,name)
            _,reference,reference_command=m.n4.build_launch(root/name)
            self.assertEqual(command,reference_command)
            changed={k for k in env if env[k]!=reference.get(k)}
            expected={'VERL_SIDECAR_MODEL_PATH','VERL_SIDECAR_SCRIPT',
                      'VERL_SIDECAR_PARALLEL_MODE',
                      'VERL_SIDECAR_REPLICA_COUNT','VERL_SIDECAR_HCCL_DETERMINISTIC',
                      'VERL_SIDECAR_LCCL_DETERMINISTIC'}
            if profile['dp']>1:expected.add('VERL_SIDECAR_DATA_PARALLEL_SIZE')
            if profile['tp']>1:
                expected.add('VERL_SIDECAR_TENSOR_PARALLEL_SIZE')
            if profile['ep']:expected.add('VERL_SIDECAR_ENABLE_EXPERT_PARALLEL')
            self.assertEqual(changed,expected)
            self.assertEqual(profile['tp']*profile['dp']*profile['replicas'],8)
            for key,value in [('FAIR_START_EPOCH','1'),('FAIR_TOTAL_EPOCHS','3'),
                              ('FAIR_TRAIN_STEPS','5'),('FAIR_FREEZE_ACTOR','0'),
                              ('VERL_SIDECAR_WINDOW_GATE_MIN_SECONDS','60.0'),
                              ('VERL_SIDECAR_MAX_NUM_SEQS','128'),
                              ('VERL_SIDECAR_MAX_TOKENS','4096')]:
                self.assertEqual(env[key],value)

    def test_user_layouts_preserve_eight_devices_and_pangu_shards_experts(self):
        import json
        p=Path(m.PROFILES['pangu_moe']['model'])
        size=json.loads((p/'model.safetensors.index.json').read_text())['metadata']['total_size']
        self.assertGreater(size/2,64*1024**3)
        self.assertEqual((m.PROFILES['qwen3_8b']['dp'],m.PROFILES['qwen3_8b']['tp']),(2,4))
        self.assertEqual(m.PROFILES['qwen3_8b']['replicas'],1)
        self.assertEqual((m.PROFILES['pangu_moe']['dp'],m.PROFILES['pangu_moe']['tp']),(8,1))
        self.assertTrue(m.PROFILES['pangu_moe']['ep'])

    def test_pangu_router_matches_reference_and_ties(self):
        import torch
        import sys
        sys.path.insert(0,str(m.SECONDARY))
        from vllm.model_executor.models.pangu_pro_moe_native import pangu_grouped_route
        for dtype in (torch.float32,torch.bfloat16):
            torch.manual_seed(12)
            scale=torch.randn(1,64,dtype=dtype)
            for scores in (torch.randn(9,64,dtype=dtype),torch.zeros(9,64,dtype=dtype)):
                weights,ids=pangu_grouped_route(scores,dtype,scale)
                reference,expected=torch.softmax(scores.float(),dim=-1).reshape(9,8,8).max(-1)
                expected=expected+torch.arange(0,64,8)
                reference=reference.to(dtype)*scale[0,expected]
                self.assertTrue(torch.equal(ids.long(),expected))
                self.assertTrue(torch.equal(weights,reference))


if __name__=='__main__':unittest.main()
