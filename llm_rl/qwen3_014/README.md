# AdaFloor vLLM 0.14.1

There are exactly two active framework implementations in this directory.

| Directory | Purpose |
| --- | --- |
| `qwen3_main` | Primary AdaFloor runtime, eager/TQ2, FULL_DECODE_ONLY/TQ1, native sleep2, Natural/Planned shrink and restore, batching, TailGuard and basic sidecar |
| `qwen3_sidecar` | Auxiliary runtime extending basic sidecar with internal DP/TP, PanGu native MoE/EP, colocated memory accounting and robust stop/output persistence |
| `experiments` | Host experiment orchestration, scoring and audit tools, not a third framework implementation |

`qwen3_main` consolidates four former copies: `qwen3`,
`qwen3_graph_sidecar_v014`, `qwen3_mlsys_supplement_v014`, and
`qwen3_mlsys_eager_controls_v014`. The main tree's current paper and analysis
files are retained. The newer graph-sidecar execution changes take precedence
over older sidecar helpers in the supplement copy. Eager versus graph is a
configuration choice, not a reason to duplicate the framework.

`qwen3_sidecar` comes from `qwen3_graph_sidecar_models_v014`. In the current
multi-model experiment workflow, the primary job runs from `qwen3_main` and
only the independent auxiliary processes use `qwen3_sidecar`. Do not put both
framework roots on the same process's import path. Both cwd and PYTHONPATH
must point at the selected implementation for nested model-registry imports.
`experiments/run_n4_sidecar_model.sh` performs this isolation.

## Main entry points

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main

# Configuration inspection only, no NPU experiment
bash run_qwen3_adafloor_eager_v014.sh dry-run

# Existing paper-suite command help
python tools/run_mlsys_fig6.py --help
```

Paper graph launchers, Natural-F2/Planned-F4 scripts, and their existing
configuration options remain in `qwen3_main`. The active paper PDF and source
package are under `qwen3_main/analysis_eval/mlsys2027_two_metrics_v2/`.
Outputs default to `/data/adafloor_graph_results/qwen3_runs` for main legacy
regroup launchers and `qwen3_sidecar_runs` for the auxiliary copy. Explicit
output-root overrides retain precedence.

## Migration and reproducibility

Pre-merge source archives, including the earlier `ref_ziyi_qwen3` development
reference, are preserved in
`/data/adafloor_code_archives/qwen3_layout_before_20261101/`.
`LAYOUT_MIGRATION_20261101.json` records archive hashes, merge decisions and
path changes. Existing v0.11 implementations are under `../qwen3_011/`.
No old top-level source-directory symlinks are required.

Original experimental results, historical protocol hashes, evidence bundles
and common epoch0 checkpoints are not rewritten. Source relocation and
consolidation change the current source fingerprint. Historical controllers
that require exact equality with an old execution fingerprint must continue
to reject the new source. Do not bypass those checks or resume an old pinned
campaign as if the source were unchanged. Use a new result directory and pin
the current sources for new experiments. Original source snapshots are the
authority for historical reproduction.

Migration validation consists of CPU protocol/import tests, path checks,
shell syntax checks and offline paper evidence audits. This restructuring
does not claim a new NPU end-to-end performance or correctness experiment.

## 2026-11-02 real-NPU short regression

The short paired regression has now exercised the archived pre-merge source
and `qwen3_main` on eager baseline, FULL-decode baseline, and Natural-F2
FULL decode with basic graph sidecar. Each arm uses two short inference
steps from the protected common epoch0, with actor updates skipped.

It found and repaired one real consolidation defect. Keeping the primary
launcher wholesale to preserve its `/data` output default had omitted the
basic sidecar watcher start/cleanup block. That block is now restored, and
`experiments/test_version_layout.py` checks its presence and unique launch.
The merge utility now preserves only the output default when overlaying
the complete basic-sidecar launcher.

All1024 primary answers per arm, scores, ordering and lengths match their
corresponding reference. Both Natural-F2 runs shrink16->8->4->2 twice,
restore16 ranks after sidecar exit/ACK, and report zero KV preemptions.
The measurements are short functional/descriptive regressions, not new
paper throughput numbers or proof of statistical performance equivalence.
Auxiliary dynamic batching is not claimed to have exact output parity.

Full report and raw evidence are in
`/data/adafloor_graph_results/merge_regression_20261102T060000Z/RESULTS.md`
and `RAW_AUDIT.json`. `LAYOUT_REGRESSION_20261102.json` records the source
fix and verification. The original failed setup attempts are retained.
