# Ada-N4 secondary-model extension

Reference: completed Qwen2.5-1.5B graph sidecar arm at
`/data/adafloor_graph_results/mlsys_n4_graph_sidecar_gate60_seed0_20261026T124500Z`.

Exactly two formal arms are scheduled, Qwen3-8B then PanGu-Pro-MoE. Each uses
seed0, the protected common math epoch0, its own updated epoch1 actor/history
for epoch2, and five32×n16 steps each epoch. Primary FULL/TQ1/native sleep2,
lifecycle1/KV-first, N4 F4/F8/F16=280576/349952/380800, TailGuard and K8 buckets
are unchanged. No new epoch0, OFF arm, training seed, or favorable-window selection.

Both use GSM8K train, the exact existing raw-prompt path, greedy n1,
max_response4096, max_model6144, max_num_seqs128, max_batched_tokens65536,
memory_fraction0.8, chunks128 and per-replica pending limit128. Graph capture
sizes remain[1,2,4,8,16,32,64,128], FULL/TQ1. Admission remains history-only >60s.
All startup/capture and process shutdown costs stay in the measured windows.

- Qwen3-8B: explicitly user-confirmed internal DP2 x TP4, one coordinated
  8-rank world, REPLICA_COUNT=1. This is NOT two independent TP4 replicas.
- PanGu-Pro-MoE: user-selected internal DP8 x TP1, EP8 enabled so experts are
  sharded across the eight data-parallel workers. This is not eight independent
  full-model copies. The model's dense/shared components remain replicated.
  EP resolves the per-card memory issue, not a TP4 requirement.
  Both models use the v0.14 torchrun_dp_example.py external_launcher contract.
  LLM receives data_parallel_size and distributed_executor_backend=external_launcher,
  with explicit eight-rank world/local-rank/master environment and V1 multiprocessing
  disabled. Every TP worker schedules identical inputs for its DP shard. Only TP
  leaders' output/state is aggregated, followers persist private resume state.
  Stop is collectively agreed before leaving a decode step.

These replace the initial assumed layouts at the author's explicit request.
The initial Qwen TP1 arm was cancelled, its evidence retained and excluded.
New-layout preflights are required; the old TP1 graph proof is not reused.

The TP4 first-decode mismatch was localized with boundary snapshots. Rotary,
Attention, local GEMM, and TP-reduction inputs were bit-exact. The first
divergence appears in BF16 HCCL all-reduce output, then amplifies over layers.
With HCCL_DETERMINISTIC=true and LCCL_DETERMINISTIC=1, a no-hook two-token
counterfactual passed first-decode hidden/logit/argmax checks on all8 ranks.
Both full eager and graph preflights/formal sidecars use the same communication
settings. This does not enable batch-invariant matmul replacements, change
Attention, relax a threshold, or affect the primary runtime.
Raw boundary evidence: `/data/adafloor_graph_results/n4_tp_graph_boundary_mm_20261027T080500Z`.
Targeted no-hook pass: `/data/adafloor_graph_results/n4_tp_graph_deterministic_20261027T080800Z`.
The short diagnostic is excluded from formal timing. Both final model-specific
16-query checks passed before either formal arm started. Every complete greedy
token sequence matched between eager/TQ2 and FULL/TQ1, 16/16 for each model.
The Qwen3-8B check produced 51,692 output tokens and PanGu produced 12,909.
These are preflight outputs, not the throughput of the formal experiment.

PanGu also required a frontend lifetime fix. A DP rank with no local requests
must continue serving experts until the engine's DP-global unfinished-work
check is false. Otherwise the first finished rank destroys an EP communicator
still needed by its peers. The final PanGu preflight includes this fix and
completed normally. Eight CPU protocol/router/frontend regression tests pass.

The nonelastic sidecar explicitly clears inherited primary elastic DP-sync
settings. Each launch executes a frozen shell snapshot, so editing the source
cannot change a running Bash program. These changes do not alter model math,
the primary runtime, the admission gate, or serving capacities.

The primary repo `qwen3_graph_sidecar_v014` remains unchanged. Only independent
sidecar processes use `qwen3_graph_sidecar_models_v014`, via
`run_n4_sidecar_model.sh`. The isolated copy adds legacy PanGu model registration
and an adapter preserving grouped routing, router_scale, biased attention output,
and the ungated shared expert. It reuses the v0.14 native Attention/MoE graph path,
not the old torchair/GE execution engine. No model weights or configs are edited.

Before each formal arm, a model-specific16-query eager/graph greedy preflight
is required. If free-running trajectories differ, same-input hidden/logit checks
must be exact with diagnostics not changing graph token IDs. Never reuse the
Qwen2.5 proof for another model. Preflight is excluded from formal timing.

Controller and runtime hashes, actual graph events, completions, partials and
physical exit/ACK-before-primary-restore are retained. An error stops the queue
with evidence, without smaller capacities, longer gates or eager fallback.
Valid zero-query/startup-consuming windows are reported rather than replaced.
Own private checkpoints are removed only after original two-epoch validation.
Public epoch0 remains protected. No paper results are changed by this queue.

Run root:
`/data/adafloor_graph_results/mlsys_n4_sidecar_internal_dp_deterministic_20261027T081200Z`

Controller log:
`/workspace/adafloor_graph_results/mlsys_n4_sidecar_deterministic_20261027.controller.log`

Tmux: `mlsys_n4_sidecar_optimal_20261027`.

Status checked at 2026-10-27 08:59 UTC (16:59 China time): the formal Qwen3-8B
arm is running. Its first primary rollout completed at 4,465.701113 tokens/s,
with actual FULL replay after shrink to four ranks and no KV preemptions in
the completed rank audits. This is one step, not an epoch average. PanGu's
formal arm remains queued after Qwen3-8B's two validated epochs. There are no
complete formal results for these two models yet.

Final preflight evidence:
`preflight/qwen3_8b/PARITY_RESULT.json` and
`preflight/pangu_moe/PARITY_RESULT.json` under the run root above.
The raw failed attempts are retained separately and excluded from results.

CPU protocol check:
`python -m unittest -v test_n4_sidecar_models.py test_offline_dp_sidecar.py`

## Stop/checkpoint repair, 2026-10-27 evening

The earlier Qwen3-8B primary finished both five-step epochs with 5,120 nonempty
answers and no audited KV preemption. It is not a passing sidecar result.
The queue stopped during sidecar aggregation, before PanGu's formal arm.
There were 218 leader-shard completions but no published aggregate.

Two concrete frontend defects were found. With periodic tracing disabled,
the streaming loop still retained every cumulative token snapshot, then wrote
that history on stop. Individual partial-history files reached 0.6--1.3 GiB,
including private TP-follower copies. This exhausted the unchanged3s soft-stop
grace in longer windows. In addition, the launcher used `.dp*` output names
while the external lease watcher only merged `.shard*` after process exit.
Killing the launcher before its final aggregate left completed shard data
invisible to the result validator.

The isolated sidecar now retains only the latest unfinished snapshot when
periodic tracing is off, never re-flushes historical traces during stop, names
leader output files `.shard*`, and atomically publishes launcher aggregates.
Model kernels, routing, capacities, layouts, gate and stop deadlines are
unchanged. Eleven CPU regression tests pass, including execution of the real
embedded streaming loop with a fake100-step engine and a failed-merge test.

New queue, PanGu first and repaired Qwen3-8B second:
`/data/adafloor_graph_results/mlsys_n4_sidecar_stopfix_20261027T144200Z`.
Tmux: `mlsys_n4_sidecar_stopfix_20261027`.
Controller log:
`/workspace/adafloor_graph_results/mlsys_n4_sidecar_stopfix_20261027.controller.log`.

Each model must pass graph generation against the already completed frozen
16-query eager reference, then an actual128-request/full-capacity stop check
with all8 worker FULL-replay receipts and the original3s+2s exit deadlines.
These diagnostics are excluded from formal timings. No complete new formal
results are claimed at queue launch. Original error evidence is preserved.
The previous57-GiB final private epoch2 checkpoint was removed only after
the original two-epoch primary validation, with an ownership-checked receipt.
The public epoch0 and all raw sidecar outputs remain untouched.

CPU checks:
`python -m unittest -v test_n4_sidecar_models.py test_offline_dp_sidecar.py test_sidecar_stop_checkpoint.py`

At 2026-10-27 14:51 UTC (22:51 China time), both repaired graph preflights
passed16/16 complete greedy sequences against their own eager references.
Both128-request/full-capacity stop tests also passed, with all8 workers
checkpointed, no remaining inflight requests, and no SIGKILL. Physical
process-group exit took4.079s for PanGu and4.092s for Qwen3-8B after file stop,
using the original SIGTERM escalation after3s. Total latest-partial snapshots
were1.94MB and3.99MB respectively, not GiB-scale cumulative histories.
These early-stop diagnostics intentionally do not wait for query completion
and are not performance measurements. PanGu's formal arm started14:51:47 UTC.
Its two epochs must still finish and pass sidecar validation before any
final result or paper update is claimed. Full source archive and hashes are
saved in the new run root.

At14:57 UTC, PanGu's formal arm's primary RL model was actually advancing
FULL decode beyond iteration512 on16 ranks, without OOM or traceback.
This confirms stable rollout entry, not completion of a formal sidecar window
or either full experiment. `STABLE_LAUNCH_CHECK.json` records this distinction.
