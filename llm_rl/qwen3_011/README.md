# Preserved AdaFloor vLLM 0.11 implementations

The existing implementation names are preserved, without merging their
different historical execution paths.

| Directory | Role |
| --- | --- |
| `qwen3_shrink_aware` | Original AdaFloor eager implementation and historical experiments |
| `adafloor_gragh` | v0.11 FULL decode graph implementation and comparisons |
| `qwen3_shrink_aware_sidecar` | v0.11 sidecar implementation |
| `adafloor_revision_eager` | Paper-revision source snapshot |
| `qwen3_old_machine_v11_bestmode345` | Earlier-machine v0.11 reference configuration |
| `ref_wj_qwen3` | Historical reference implementations and comparisons |
| `qwen3_shrink_aware_cleanup_20260626T182116Z` | Historical cleanup/results archive, not another runtime |

Only source/documentation paths affected by relocation were adjusted.
Existing Git metadata and raw experimental records are retained. Historical
JSON source snapshots keep their original paths and hashes as provenance,
not as promises that the old absolute working directories still exist.

Current v0.14 development is in `../qwen3_014/qwen3_main` and
`../qwen3_014/qwen3_sidecar`. The earlier `ref_ziyi_qwen3` tree was a v0.14
development reference rather than a v0.11 implementation and is archived
outside this directory.
