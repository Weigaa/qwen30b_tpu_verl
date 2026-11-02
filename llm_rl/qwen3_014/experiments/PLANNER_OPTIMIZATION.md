# Exact offline planner acceleration

`optimized_planner.py` wraps the pinned v0.14 planner without editing the model,
runtime, baseline, dataset, KV capacities, floor set, search budget or MILP objective.

The accelerations are:

1. For a quota requiring q finished ranks, reject a candidate if its early edges
   cannot cover 2q distinct planning units.
2. Maximize the number of early edges within a full maximum-cardinality matching.
   If a perfect matching cannot contain q early edges, that quota is impossible.
   Passing each individual bound is not treated as joint quota feasibility.
3. Memoize exact two-unit metrics using frozen semantic inputs and all numeric
   parameters. Original edge order and original MILP remain unchanged.
4. In cross-step repair, after solving the bad batch, use
   `(not feasible_bad, overflow_bad, overflow_bad)` as a lower bound for the
   candidate's first three repair-score fields. If even this ideal-neighbor
   bound cannot improve the current score, skip the neighbor solve. Candidate
   rounds, ordering, acceptance rule and tie-breaking are unchanged.

Validation is CPU-only:

```bash
python3 -m pytest -q test_optimized_planner.py test_cached_planner.py test_code_natural_f2.py
python3 -u validate_optimized_planner.py --output /new/path/math_parity.json
python3 -u validate_code_planner_repair.py --work /new/path/code_work --output /new/path/code_parity.json
```

The first full-plan check compares every field of the five-step mathematical
Natural-F2 plan. The second performs a fresh code-P4 search and compares all
three JSON plans and every reordered parquet row against the historical
31-swap plan. It does not replay the old swap trace.

`run_code_natural_f2.py` pins the optimizer hash, requires both validation
receipts, freezes the new code-N2 plan, and waits for idle devices and other
trainers before NPU execution. Old CPU attempts and their SIGTERM exits remain
available. None of those interrupted attempts generated NPU answers.

These checks establish parity on the tested inputs. The numerical MILP keeps
its original5s time limit and solver behavior. Planning time is measured
separately and must not be silently counted as zero in full-job cost claims.

## Measured results on2026-10-25

- Final mathematical Natural-F2 five-step parity: original63.736s,
  optimized1.567s,40.68x for solving only. Every plan field is identical.
- New code-N2 complete planning:1695.407s (28m15.4s), including31 applied
  cross-step repairs, all160 prompts, no old repair-trace replay. Selected
  floors[4,16,16,16,16], response caps[16384,16384,16384,16384,16384].
- Code-N2 counters:61576 quota calls,54451 vertex-pruned,6894 matching-pruned,
  231 original oracle calls,18477 neighbor solves safely skipped.
- Unit regression:20 tests pass. Fresh code-P4 gold-plan parity is a separate
  CPU regression gate and is required before the new NPU arm starts.
- The original code-N2 run was interrupted before completion, so there is
  no measured whole-plan N2 speedup factor. Do not compare the new28-minute
  N2 plan directly to the old51-minute P4 plan as a controlled speedup.

Current N2 output:
`/workspace/adafloor_graph_results/mlsys_v014_code_natural_f2_seed401_fast_20261025T093400Z`

Durable tmux sessions:
`code_n2_v014_fast_seed401_20261025` (generation controller, waits for gates),
`code_planner_parity_20261025` (fresh code-P4 CPU regression).
