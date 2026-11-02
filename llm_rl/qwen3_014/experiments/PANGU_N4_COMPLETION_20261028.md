# Complete the PanGu Ada-N4 sidecar arm

## Final status: COMPLETE, 2026-10-28 03:42 China time

Both updated-policy epochs and all10 primary training steps finished.
Final controller state COMPLETE, child exit0, original full-arm validation
PASS, independent `FINAL_COMPLETION_AUDIT.json` PASS, and final integrity check
PASS. All5120 primary responses are nonempty,160 rank-step audits show FULL
decode, KV preemptions0, no OOM. Additional eager iterations beyond global
prefill0. Ten actor metric records are finite with nonzero gradient norms.

Primary response-only Q5535.945413 tokens/s, reward0.8333984375, mean response
6004.507031 tokens. Same-seed updated-policy OFF difference+0.2093% is
descriptive, not a frozen causal interference measurement.

Five sidecar starts, one useful window and four startup-only windows.
164 queries before stop,168 including drain. Original strict scoring34/168.
All output/state/prompt/physical exit and ACK-before-restore checks passed.
Cold-start limitation and format-only scoring diagnosis remain explicit in
`PANGU_RESULTS.md`, not hidden by successful-window selection or a new metric.

Private run checkpoint storage was removed only after full validation.
The protected common epoch0 remains present with61,100,638,832 checkpoint bytes.
No NPU processes remain. No paper or Qwen3-8B experiment was changed by this goal.

Report:
`/data/adafloor_graph_results/mlsys_pangu_n4_colocated_20261027T171500Z/PANGU_RESULTS.md`

The remainder is the retained investigation and completion-audit plan.

Goal: one complete, validated PanGu arm, not a standalone smoke test.
Same seed0/common math epoch0, own epoch1 checkpoint/history for epoch2,
two epochs of five32-by-n16 primary steps. Primary FULL/TQ1/native sleep2,
N4 F4/F8/F16 KV capacities280576/349952/380800, TailGuard/K8/history-based
strict>60s sidecar gate unchanged. PanGu uses internal DP8 x TP1 + EP8.
Secondary GSM8K/raw prompts/greedy n1/max4096/model6144/seq128/batch65536/
memory fraction0.8 and original3s soft-stop+2s TERM grace remain unchanged.

## Failure and targeted repair

Previous run:
`/data/adafloor_graph_results/mlsys_n4_sidecar_stopfix_20261027T144200Z`.
It stopped at epoch1 step3. Sidecar parent imported the isolated runtime using
PYTHONPATH, but the nested `python -m vllm.model_executor.models.registry`
inherited the primary RL checkout as cwd. Python prepends cwd ahead of
PYTHONPATH for -m. The child imported the old primary vllm and could not find
`pangu_pro_moe_native`. The sidecar failed before loading weights. The
subsequent restore-handshake timeout was secondary, not the root cause.
Neither NPU OOM nor graph numerical correctness caused this failure.

Both the external model wrapper and frozen-shell launcher now cd to their
own isolated runtime before launching any Python. No primary runtime or model
math was changed. A regression invokes the actual wrapper from the primary
checkout with primary-first inherited PYTHONPATH, then runs a real nested
registry subprocess and checks both processes' vllm/registry/PanGu paths.
All12 CPU regression tests passed. The full-capacity stop diagnostic now
uses the actual formal wrapper/cwd and similarly polluted inherited path,
rather than launching directly from the isolated checkout.

## Current authoritative run

Root:
`/data/adafloor_graph_results/mlsys_pangu_n4_colocated_20261027T171500Z`

Tmux: `mlsys_pangu_n4_colocated_20261028`.

Controller log:
`/workspace/adafloor_graph_results/mlsys_pangu_n4_colocated_20261028.controller.log`

The previous import-fix run documented below was later stopped, not completed.
During epoch1 step4, all8 PanGu ranks returned zero KV capacity after loading
weights. The raw `sidecar/monitor.log` contains `No available memory for the
cache blocks`. This was a memory-attribution failure, not a repeated import
error. Native primary sleep still left13.34--13.65GiB on the loaned devices.

The local CacheConfig documents fraction0.8 as a per-instance limit. The
legacy Ascend profiler charged the other process's existing memory as this
instance's non-Torch overhead. The isolated secondary now measures the
before-load baseline, counts only its own additional non-Torch growth, and
caps KV by BOTH its unchanged per-instance fraction and current physical
free HBM minus observed transient Torch peak and1GiB/2% graph headroom.
No primary source or KV cap was changed. Physical insufficiency still yields0.

The new graph and stop preflights retain16GiB in a separate owned process on
each of the8 devices. At01:16 China time, all8 ranks had actual FULL replay,
with4.02--4.83GB physically bounded KV budgets. The same measurements would
return0 on all8 ranks under the legacy formula. These diagnostics are not
formal performance data. See `SHARED_HBM_COUNTERFACTUAL.json`. Sixteen CPU
regression tests pass. The16-query generation check subsequently passed exact
full token parity, and the128-request stop test with the same16GiB external
residual passed: all8 workers checkpointed, physical group exit3.897s, TERM
but no SIGKILL, no leftover inflight state. All16 diagnostic residual worker
processes from the two sequential checks exited0. `ALL_PREFLIGHTS_PASS.json`
records these checks. The new formal two-epoch/10-step PanGu run started at
01:21:10 China time. These preflight results do not prove formal completion.

Current independent final auditor tmux:
`mlsys_pangu_n4_colocated_audit_20261028`, waiting for controller PID3757468.
Audit log:
`/workspace/adafloor_graph_results/mlsys_pangu_n4_colocated_20261028.audit.log`.

At01:26--01:27, the new formal primary was advancing actual FULL decode
beyond iteration768. Controller PID3757468 and trainer PID3765200 were live,
with no OOM or traceback. `STABLE_LAUNCH_CHECK.json` records this launch.
This new run starts from public epoch0 and does not splice the earlier
failed run's three completed steps into its ten-step result. Both formal
epochs, real sidecar windows and final audit are still pending.

At02:04--02:11, the current colocated run's epoch1 step4 real window passed
the previously failing load/profile point. All8 ranks became ready and replayed
FULL graphs; actual KV budgets were physically bounded around6--7GB, not0.
The window lasted142.258s before stop and completed164 queries before stop,
168 including4 drain completions. All168 output keys matched durable completed
states, all final texts were nonempty, all8 shards had FULL replay proof, and
no inflight requests remained. The process group exited5.822s after trainer
stop request, followed by matching ACK then primary restore. TERM but no
SIGKILL in this useful window. No import error, cache-capacity failure or OOM.
See `FIRST_USEFUL_FORMAL_WINDOW.json`. This is not full-arm completion.

Original strict GSM8K scoring gives34/168. A format-only diagnostic found
152/168 matching last numeric answers; many use boxed or other final-answer
formats instead of the required ####number. The standalone16-query eager
reference also has low strict scoring (2/16, versus13/16 flexible). Flexible
scores are NOT substituted into formal results. Seven common queries across
the standalone small batch and formal batch have differing text trajectories;
these are not a matched full-batch ON/OFF/accuracy comparison. Fixed16-query
eager/graph token parity remains exact. Do not claim universal batch-invariant
generation or full-service solo/concurrent accuracy equivalence from this.
See `SIDECAR_FORMAT_DIAGNOSTIC.json` for the preserved diagnostic scope.

At02:11, four training steps were complete and epoch1 step5 was advancing
FULL decode. Both current controller/trainer PIDs3757468/3765200 remained live.

At02:32:45, epoch1 exited0 after all5 actor updates and its checkpoint save.
Original primary parser passed2560 answers,80 completed FULL rank-step audits,
zero KV preemptions. Response-only Q5454.903045 tokens/s, rollout sum2811.976285s,
mean reward0.834375, mean response5991.819531 tokens. All five sidecar decisions
and both admitted windows passed the original full sidecar validator, with
164 jobs before stop and168 including drain;856 partial requests were durably
checkpointed at epoch end, not counted as completed service.
See `EPOCH1_PRIMARY_VALIDATED.json` and `EPOCH1_SIDECAR_VALIDATED.json`.

At02:38, epoch2 trainer PID3915688 and controller3757468 were live. All16 ranks
had actual successful loads from this run's own epoch1 checkpoint, and all5
epoch2 plans referenced common epoch0 plus this run's own epoch1 response
history. Epoch2 had actual FULL replay beyond iteration1280 with no OOM.
See `EPOCH2_CHAIN_VALIDATED.json` and the timestamped `PROGRESS.json`.
This is5/10 complete primary training steps, NOT full-arm completion. The
second epoch and final independent audit still have to finish.

### Historical import-fix run, before the shared-HBM repair

Root: `/data/adafloor_graph_results/mlsys_pangu_n4_importfix_20261027T161500Z`.

Only PanGu is requested in this queue. Qwen3-8B is outside this completion
goal. Reuse the existing frozen16-query eager reference, rerun graph against
it, then rerun the full-capacity file-stop diagnostic from primary cwd before
starting the complete formal arm. No old formal steps are spliced into new
epochs. Diagnostics are excluded from formal timing. Preserve all old raw
failure artifacts and the protected public epoch0.

At00:22 China time on Oct28, all16 greedy trajectories matched the eager
reference. The full-capacity stop test launched from the primary checkout
through the formal wrapper passed: all8 workers checkpointed, no inflight
work remained, physical group exit4.110s after stop, TERM but no SIGKILL,
latest partial snapshots1.94MB. The formal PanGu arm started00:22:20.
These are preflight/launch results, not formal experiment completion.

At00:27, trainer PID3648229 and controller PID3641330 were live. The formal
primary had80 actual FULL-decode receipts, progressing beyond iteration512,
with no OOM or traceback. First admitted PanGu sidecar window, both five-step
epochs, and final result validation are still pending. See
`STABLE_LAUNCH_CHECK.json` for the launch evidence, not a completion claim.

At00:41, step1 actor update was complete (reward0.837890625, mean response
2072.236328125 tokens). The first two rollouts took233.562221s/350.294929s,
with instantaneous logger rates4667.505704/5321.158967 tokens/s; those logger
rates include prompt tokens and are not the final paper response-only Q.
Step2 actor update was still running, and both initial sidecar gates skipped.
Controller/trainer PIDs3641330/3648229 remained live. `PROGRESS.json` is a
timestamped partial snapshot, not a completion receipt. Raw TaskRunner logs
can be ahead of the aggregated Ray log relay and are retained by the queue.

An independent final auditor waits for this exact controller PID/start-time
in tmux `mlsys_pangu_n4_audit_20261028`, then runs
`audit_pangu_complete.py`. It checks actual per-rank checkpoint load receipts,
own prior-epoch planning history, identical reference prompt multiset, all5120
answers/160FULL audits, and all sidecar raw lifecycle/output/prompt records.
It fails closed while the experiment is incomplete. Its pinned source hash
is in `AUDITOR_LAUNCH.json`. On success it writes `FINAL_COMPLETION_AUDIT.json`
and `PANGU_RESULTS.md`. Log:
`/workspace/adafloor_graph_results/mlsys_pangu_n4_importfix_20261028.audit.log`.
No experimental runtime sources were edited during this wait.

At00:53, the third primary rollout (481.001298s) and its actor update had
completed, giving3/10 complete training steps. The first formal PanGu window, epoch1 step3,
started on physical devices0,1,2,3,4,8,9,10 after full shrink quorum. The
isolated model/EP path initialized without the previous missing-module error
or OOM, but this61.74s lease ended during cold load, before any worker became
ready or query completed. Original stop escalation included SIGKILL. Physical
group exit, matching ACK, then primary restore were checked in that order.
This is retained as a startup-only zero-query window, not successful service.
The observed window and handback proof are in `FIRST_FORMAL_SIDECAR_WINDOW.json`.
The unchanged queue remains live for all remaining steps/windows and epoch2.

## Completion audit

- Formal subprocess exits0 and original two-epoch validation passes.
- Own epoch1 actor/history feeds epoch2, with the protected common epoch0
  used only as the initial state.
- Both epochs have five primary steps,2560 nonempty answers and80 completed
  rank-step runtime audits with actual FULL decode.
- All actual sidecar admission decisions obey the unchanged history-only
  strict>60s gate and use eight physically detached devices with DP8/TP1/EP8.
- Validate startup/capture/replay, final outputs against completed states,
  partial/resume state, no OOM, and physical process exit/ACK before restore.
- Report all admitted windows, including zero-query/startup-consuming windows,
  and actual graceful/TERM/SIGKILL behavior. Never relabel these as completed
  service when no queries finished.
- `TRAINING_SIDECAR_RESULT.json`, `ARM_PASS.json`, `COMPARISON.json` and root
  `STATE.json` must exist and agree before marking the goal complete.
- Report main throughput/reward/length, sidecar completed queries/tokens and
  accuracy, startup/shutdown, and caveats versus Qwen2.5 reference. Cross-model
  tokens are not pooled into one throughput claim. No paper edits in this task.
