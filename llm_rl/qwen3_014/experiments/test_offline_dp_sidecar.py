import importlib.util
from pathlib import Path
import unittest
import tempfile
import os
import subprocess
import json

path=Path(__file__).resolve().parents[1]/'qwen3_sidecar/tools/launch_offline_dp_sidecar.py'
spec=importlib.util.spec_from_file_location('offline_dp_launcher_test',path)
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


class OfflineDPTests(unittest.TestCase):
    def env(self,limit):
        return dict(VERL_SIDECAR_DATA_PARALLEL_SIZE='8',VERL_SIDECAR_MASTER_ADDR='127.0.0.1',
            VERL_SIDECAR_MASTER_PORT='24300',VERL_SIDECAR_MAX_PROMPTS=str(limit),
            VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA='128',VERL_SIDECAR_OUTPUT_FILE='/tmp/sidecar.jsonl',
            VERL_SIDECAR_TENSOR_PARALLEL_SIZE='1',VERL_SIDECAR_STATE_DIR='/tmp/sidecar_state')

    def test_ep_workers_share_world_and_keep_distinct_data(self):
        ranks=[launcher.worker_env(self.env(1024),rank) for rank in range(8)]
        self.assertEqual({r['VLLM_DP_MASTER_PORT'] for r in ranks},{'24300'})
        self.assertEqual({r['VLLM_DP_SIZE'] for r in ranks},{'8'})
        self.assertEqual({r['VLLM_DP_RANK'] for r in ranks},set(map(str,range(8))))
        self.assertEqual(len({r['VERL_SIDECAR_OUTPUT_FILE'] for r in ranks}),8)
        self.assertEqual({r['VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE'] for r in ranks},{'128'})

    def test_formal_wrapper_isolates_cwd_and_nested_registry(self):
        root=path.parents[1]
        wrapper=root.parent/'experiments/run_n4_sidecar_model.sh'
        primary=root.parent/'qwen3_main'
        with tempfile.TemporaryDirectory() as directory:
            # Intercept only the wrapper's final exec. Execute the real nested
            # registry inspection after its cwd/PYTHONPATH isolation.
            fake=Path(directory)/'bash'
            fake.write_text('#!/bin/bash\nexec python3 "$(dirname "$1")/check_sidecar_runtime.py"\n')
            fake.chmod(0o755)
            env=os.environ.copy()
            env['PATH']=directory+':'+env['PATH']
            env['PYTHONPATH']=str(primary)+':'+env.get('PYTHONPATH','')
            result=subprocess.run(['/bin/bash',str(wrapper)],cwd=primary,env=env,
                                  capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            receipt=json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(receipt['status'],'PASS')
            self.assertTrue(Path(receipt['registry_subprocess']['pangu']).is_relative_to(root))

    def test_preflight_keeps_sixteen_global_queries(self):
        ranks=[launcher.worker_env(self.env(16),rank) for rank in range(8)]
        self.assertEqual(sum(int(r['VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE']) for r in ranks),16)

    def test_internal_dp2_tp4_one_coordinated_world(self):
        base=self.env(16)
        base.update(VERL_SIDECAR_DATA_PARALLEL_SIZE='2',VERL_SIDECAR_TENSOR_PARALLEL_SIZE='4')
        ranks=[launcher.worker_env(base,rank) for rank in range(8)]
        self.assertEqual([r['VLLM_DP_RANK'] for r in ranks],['0']*4+['1']*4)
        self.assertEqual({r['WORLD_SIZE'] for r in ranks},{'8'})
        self.assertEqual({r['VLLM_ENABLE_V1_MULTIPROCESSING'] for r in ranks},{'0'})
        self.assertEqual({r['VERL_SIDECAR_EXTERNAL_LAUNCHER'] for r in ranks},{'1'})
        self.assertEqual(ranks[0]['VERL_SIDECAR_STATE_DIR'],ranks[4]['VERL_SIDECAR_STATE_DIR'])
        self.assertIn('tp_followers',ranks[1]['VERL_SIDECAR_STATE_DIR'])
        self.assertNotEqual(ranks[0]['VERL_SIDECAR_OUTPUT_FILE'],ranks[4]['VERL_SIDECAR_OUTPUT_FILE'])
        self.assertTrue(ranks[0]['VERL_SIDECAR_OUTPUT_FILE'].endswith('.shard0'))
        self.assertTrue(ranks[4]['VERL_SIDECAR_OUTPUT_FILE'].endswith('.shard1'))

    def test_aggregate_matches_watcher_shards_and_survives_merge_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=root/'outputs.jsonl'
            first=Path(str(target)+'.shard0');second=Path(str(target)+'.shard1')
            first.write_text('{"prompt_id":0}\n');second.write_text('{"prompt_id":1}\n')
            launcher.merge_outputs(target,[first,second])
            expected=first.read_bytes()+second.read_bytes()
            self.assertEqual(target.read_bytes(),expected)
            invalid=root/'not_a_file';invalid.mkdir()
            with self.assertRaises(IsADirectoryError):
                launcher.merge_outputs(target,[first,invalid])
            self.assertEqual(target.read_bytes(),expected)
            self.assertFalse(list(root.glob('*.merge.*.tmp')))

    def test_dense_dp_metadata_is_counts_not_expert_semantics(self):
        import sys
        from types import SimpleNamespace
        import torch
        sys.path.insert(0,str(path.parents[1]))
        from vllm.forward_context import DPMetadata
        config=SimpleNamespace(data_parallel_size=2,data_parallel_rank=1,
            is_moe_model=False,distributed_executor_backend='external_launcher')
        metadata=DPMetadata.make(config,6,torch.tensor([5,6]))
        self.assertEqual(metadata.max_tokens_across_dp_cpu.item(),6)
        self.assertEqual(metadata.num_tokens_across_dp_cpu.tolist(),[5,6])
        config.distributed_executor_backend='mp'
        with self.assertRaises(AssertionError):DPMetadata.make(config,6,torch.tensor([5,6]))

    def test_local_completion_keeps_serving_remote_experts(self):
        import sys
        sys.path.insert(0,str(path.parents[1]))
        from vllm_ascend.distributed.sidecar_control import has_stream_work
        class Engine:
            pending=True
            calls=0
            def has_unfinished_requests(self):
                self.calls+=1
                return self.pending
        engine=Engine()
        self.assertTrue(has_stream_work(engine,{},True))
        self.assertEqual(engine.calls,1)
        engine.pending=False
        self.assertFalse(has_stream_work(engine,{},True))
        self.assertFalse(has_stream_work(engine,{},False))
        self.assertTrue(has_stream_work(engine,{'local':'request'},False))


if __name__=='__main__':unittest.main()
