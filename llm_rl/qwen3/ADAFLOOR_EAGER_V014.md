# AdaFloor eager migration on vLLM 0.14

This tree contains the first correctness-oriented port of AdaFloor's eager
rollout path to the local vLLM/vLLM-Ascend 0.14 runtime.

## Preserved baseline

- Git branch: `qwen3-v014-baseline-fullgraph-20260821`
- Source archive: `../../.baseline_snapshots/qwen3_v014_fullgraph_20260821.tar.gz`
- Archive SHA256: `acec139b8bf5fbc32aa00d8178039aa35c463768262d099063f7c30be46af2e0`

The archive is authoritative for the untracked framework source files under
`llm_rl/qwen3`. The branch preserves the surrounding repository state.

## Migrated path

The default-off port adds the following control flow around the pre-existing
0.14 elastic worker implementation:

1. Predict or load response lengths and assign repeated requests to rollout
   ranks before generation.
2. Install a per-rollout survivor plan in every synchronous vLLM worker.
3. Execute either a fixed Planned `16 -> 8 -> 4` safe prefix or Natural staged
   shrinking while unfinished requests remain.
4. Restore the full external-DP world after generation.
5. Restore the original request order before reward and actor computation.

The initial launcher intentionally uses eager rollout, TQ2, synchronous
generation, floor 4, and no TailGuard by default. This isolates shrink, restore,
and output ordering correctness from graph capture and response-cap effects.
TailGuard is available as an explicit opt-in after the no-cap path is validated.

## Entry point

Inspect the effective migration configuration without starting Ray or NPUs:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3
./run_qwen3_adafloor_eager_v014.sh dry-run
```

Run the Planned floor-4 eager configuration:

```bash
./run_qwen3_adafloor_eager_v014.sh run
```

Select the Natural trigger policy with:

```bash
ADAFLOOR_TARGET_POLICY=natural \
  ./run_qwen3_adafloor_eager_v014.sh run
```

Enable the default 4096-token TailGuard for floor-4 short steps with:

```bash
ADAFLOOR_TAIL_GUARD=1 ./run_qwen3_adafloor_eager_v014.sh run
```

An externally generated rank plan can be supplied with
`ADAFLOOR_ASSIGNMENT_POLICY=optimized_rank_plan` and `ADAFLOOR_PLAN_PATH`.

## Verification status

The planner, trigger, assignment, driver ordering, worker metadata propagation,
default-off behavior, launcher contract, and Python syntax are covered by CPU
tests. A 16-NPU end-to-end shrink and restore run is still required before this
port should be treated as performance-ready or paper evidence.
