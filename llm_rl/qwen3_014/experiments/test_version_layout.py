"""CPU checks for version isolation and runtime selection after consolidation."""
from pathlib import Path
import unittest

import run_n2_f8_campaign as campaign
import run_n4_sidecar_models as models
import run_n4_graph_sidecar as basic

ROOT = Path(__file__).resolve().parents[1]


class VersionLayoutTests(unittest.TestCase):
    def test_exactly_two_framework_implementations(self):
        frameworks = sorted(p.name for p in ROOT.iterdir() if (p/'vllm').is_dir())
        self.assertEqual(frameworks, ['qwen3_main', 'qwen3_sidecar'])
        for name in frameworks:
            version = (ROOT/name/'vllm/_version.py').read_text()
            self.assertIn("'0.14.1'", version)

    def test_main_contains_graph_eager_and_basic_sidecar(self):
        main = ROOT/'qwen3_main'
        for name in ['run_qwen3_adafloor_eager_v014.sh',
                     'run_qwen3_v014_paper_graph_core3_k8_full_epoch.sh',
                     'internal/run_elastic_sidecar_infer.sh',
                     'verl/utils/sidecar_restore_handshake.py']:
            self.assertTrue((main/name).is_file(), name)
        helper, env, command = basic.build_launch(Path('/tmp/layout_cpu_config_only'))
        self.assertEqual(helper.REPO, main)
        self.assertEqual(env['VERL_SIDECAR_ENABLE'], '1')
        self.assertEqual(env['VERL_SIDECAR_TASK_QUEUE_ENABLE'], '1')
        self.assertEqual(env['FLOOR_KV_CAPS'], '4:280576,8:349952,16:380800')
        self.assertTrue(any(str(main) in part for part in command))

    def test_multimodel_processes_use_secondary_runtime(self):
        self.assertEqual(campaign.MAIN, ROOT/'qwen3_main')
        self.assertEqual(campaign.SIDECAR, ROOT/'qwen3_main')
        self.assertEqual(models.SECONDARY, ROOT/'qwen3_sidecar')
        wrapper = (ROOT/'experiments/run_n4_sidecar_model.sh').read_text()
        self.assertIn(str(ROOT/'qwen3_sidecar'), wrapper)
        self.assertIn('cd "$root"', wrapper)
        self.assertTrue((models.SECONDARY/'vllm/model_executor/models/pangu_pro_moe_native.py').is_file())

    def test_primary_launcher_really_starts_and_cleans_up_basic_sidecar(self):
        source=(ROOT/'qwen3_main/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh').read_text()
        self.assertIn('OUTPUT_ROOT=${OUTPUT_ROOT:-/data/adafloor_graph_results/qwen3_runs}',source)
        self.assertIn('if [[ "$VERL_SIDECAR_ENABLE" == "1" ]]',source)
        self.assertEqual(source.count('internal/watch_elastic_shrink_and_run_sidecar.sh "$logfile"'),1)
        self.assertIn('sidecar_monitor_pid=$!',source)
        self.assertIn('trap cleanup_sidecar_monitor EXIT',source)
        self.assertIn('wait "$sidecar_monitor_pid"',source)

    def test_legacy_parser_path_resolves_without_top_level_aliases(self):
        helper = basic.helper()
        parser = helper.REPO.parent/'../qwen3_011/qwen3_shrink_aware/analysis_eval/plot_e2e_performance.py'
        self.assertTrue(parser.resolve().is_file())
        for name in ['qwen3', 'qwen3_graph_sidecar_v014', 'qwen3_graph_sidecar_models_v014',
                     'qwen3_mlsys_supplement_v014', 'qwen3_mlsys_eager_controls_v014']:
            self.assertFalse((ROOT.parent/name).exists(), name)


if __name__ == '__main__':
    unittest.main()
