"""Byte accounting, not a relaxed OOM threshold or a bigger memory fraction."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'qwen3_sidecar/vllm_ascend/worker/sidecar_memory.py'
spec=importlib.util.spec_from_file_location('sidecar_memory_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
G=1024**3


class MemoryBudgetTests(unittest.TestCase):
    def compute(self,**changes):
        values=dict(total_bytes=64*G,utilization=.8,baseline_non_torch_bytes=14*G,
            peak_torch_bytes=37*G,current_torch_bytes=31*G,
            global_non_torch_bytes=18*G,physical_free_bytes=15*G)
        values.update(changes)
        return module.instance_kv_budget(**values)

    def test_external_primary_not_charged_as_own_non_torch(self):
        b=self.compute()
        self.assertEqual(b['own_non_torch_bytes'],4*G)
        self.assertEqual(b['instance_limit_bytes'],int(64*G*.8))
        self.assertGreater(b['kv_budget_bytes'],0)
        self.assertLessEqual(b['kv_budget_bytes'],b['physical_kv_limit_bytes'])

    def test_shared_pressure_caps_allocation_even_if_instance_has_budget(self):
        b=self.compute(physical_free_bytes=8*G)
        self.assertLess(b['physical_kv_limit_bytes'],b['instance_kv_limit_bytes'])
        self.assertEqual(b['kv_budget_bytes'],b['physical_kv_limit_bytes'])
        self.assertLessEqual(b['kv_budget_bytes']+b['transient_headroom_bytes']+
            b['physical_headroom_bytes'],8*G)

    def test_no_physical_room_returns_zero_not_a_fabricated_budget(self):
        self.assertEqual(self.compute(physical_free_bytes=6*G)['kv_budget_bytes'],0)

    def test_own_growth_and_baseline_release_are_conservative(self):
        self.assertLessEqual(self.compute(global_non_torch_bytes=22*G)['kv_budget_bytes'],
                             self.compute()['kv_budget_bytes'])
        self.assertEqual(self.compute(global_non_torch_bytes=10*G)['own_non_torch_bytes'],0)


if __name__=='__main__':unittest.main()
