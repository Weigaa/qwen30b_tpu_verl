# Mode5 Attempt Ledger 2026-06-04

## Purpose

This file records the major `mode=5` validation / optimization attempts on 2026-06-04, with the goal of keeping a single local source of truth for:

- which runs were valid,
- which runs were invalid or not comparable,
- which runs failed in entry / worker-init / runtime,
- the current fastest end-to-end rollout result,
- the current best-known hotspot-quality timing baseline.

It is intentionally stricter than chat memory:

- `TBE Subprocess[task_distribute] raise error[], main process disappeared!` is **not** treated as a failure signal by itself.
- A run is treated as successful only when there is authoritative evidence such as:
  - `rollout_output_time_s` emitted, and/or
  - `[run] ... exit_code=0`
- A run is treated as failed when there is authoritative evidence such as:
  - `exit_code=1`,
  - `Worker exit type: SYSTEM_ERROR`,
  - `ActorDiedError`,
  - import failure, or
  - HCCL bind failure.

## Current Bests

### Fastest Verified End-to-End Rollout

Current fastest verified `mode=5` rollout observed so far:

- run: [wjeagerqwen30b-a3b-with_draft_breakdown_20260605042224_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260605042224_elastic.txt)
- trainer worker log: [/tmp/ray/session_2026-06-05_04-22-33_888741_1033468/logs/worker-5a9a361a082058367aa00ee2f083bfc2f7808ee2927ec68ab30c9ecf-01000000-1053590.out](/tmp/ray/session_2026-06-05_04-22-33_888741_1033468/logs/worker-5a9a361a082058367aa00ee2f083bfc2f7808ee2927ec68ab30c9ecf-01000000-1053590.out)
- verified rollout times observed:
  - step1: `229.267634`
  - step2: `244.775307`
  - step3: `238.954965`

Current fastest verified value:

- **`229.267634 s`**

Important caveat:

- This is now the strongest verified wall-time sample and also the strongest verified 3-step mainline profile.
- However, the old `20260604154651` hotspot-quality baseline remains useful as a directional reference because its stage=2 timing is still the cleanest known expression of the original overlap objective.

### Best Known Hotspot-Quality Timing Baseline

Current best-known timing baseline for the actual overlap objective:

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604154651_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604154651_elastic.txt)
- `rollout_output_time_s = 270.998779`, see [wjeagerqwen30b-a3b-with_draft_breakdown_20260604154651_elastic.txt:6595](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604154651_elastic.txt:6595)
- `[run] ... exit_code=0`, see [wjeagerqwen30b-a3b-with_draft_breakdown_20260604154651_elastic.txt:6784](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604154651_elastic.txt:6784)

This run remains the best hotspot-quality reference because its `stage=1/2` timing most clearly matches the intended overlap direction:

- `stage=2`
  - `prefetch_total p50/p90 = 7.264 / 7.900 ms`
  - `submit_populate p50/p90 = 1.909 / 2.310 ms`
  - `submit_remote_npu p50/p90 = 1.412 / 1.733 ms`
  - `submit_cpu p50/p90 = 0.106 / 0.123 ms`
- `stage=1`
  - `prefetch_total p50/p90 = 13.691 / 14.287 ms`
  - `submit_populate p50/p90 = 3.086 / 3.569 ms`
  - `submit_remote_npu p50/p90 = 2.454 / 2.807 ms`
  - `submit_cpu p50/p90 = 0.110 / 0.118 ms`

Interpretation:

- `20260605042224` is now the fastest verified wall-time sample and the strongest verified 3-step mainline sample.
- `20260604154651` is still the best overlap-quality timing sample.
- These are **not the same** current best.

## Major Attempt Ledger

Below, "valid" means the run reached a meaningful comparison boundary for the question being asked. "Invalid" means it did not.

### 1. 20260604105119

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604105119_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604105119_elastic.txt)
- type: valid `mode=5 dual_source` success run
- result:
  - `rollout_output_time_s = 295.626066`
  - `[run] ... exit_code=0`
- value:
  - first clean success for the true `dual_source` implementation
  - used as early timing/statistics baseline

### 2. 20260604123327

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604123327_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604123327_elastic.txt)
- type: valid `mode5_runtime_strategy=legacy_cpu_shadow` comparison run
- result:
  - `rollout_output_time_s = 299.227704`
  - `[run] ... exit_code=0`
- value:
  - used to compare `legacy_cpu_shadow` vs true `dual_source`

### 3. 20260604134201

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604134201_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604134201_elastic.txt)
- type: valid intermediate optimization run
- result:
  - not kept as current best
- value:
  - showed that only reordering CPU submit was not enough

### 4. 20260604135215

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604135215_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604135215_elastic.txt)
- type: valid `mode=5 dual_source` success run
- result:
  - `rollout_output_time_s = 277.125605`
  - `[run] ... exit_code=0`
- value:
  - first major drop after:
    - CPU submit pre-move
    - same-package remote preference
    - batched control-message handling

### 5. 20260604141637

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604141637_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604141637_elastic.txt)
- type: invalid / non-comparable optimization branch
- result:
  - early-stage failure pattern
- value:
  - part of the failed `parallel_cpu_remote_submit` / unstable branch exploration

### 6. 20260604145349

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604145349_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604145349_elastic.txt)
- type: interrupted comparison sample
- result:
  - reached `stage_async_pack` path
  - later terminated externally (`SIGTERM` style end), not used as final timing proof
- value:
  - showed the experiment path was alive, but not a clean final comparison sample

### 7. 20260604150456 and nearby short launcher-only runs

- logs include:
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260604150456_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604150456_elastic.txt)
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260604152628_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604152628_elastic.txt)
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260604152836_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604152836_elastic.txt)
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260604153021_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604153021_elastic.txt)
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260604153431_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604153431_elastic.txt)
- type: invalid / shared-machine contamination
- result:
  - mostly launcher-only or HCCL bind / port / shared-resource collision
- value:
  - not trustworthy for runtime comparison

### 8. 20260604154651

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604154651_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604154651_elastic.txt)
- type: valid `mode=5 dual_source` success run
- result:
  - `rollout_output_time_s = 270.998779`
  - `[run] ... exit_code=0`
- value:
  - current best hotspot-quality timing baseline
  - closest known run to the intended `prefetch ~= max(remote, cpu)` shape

### 9. 20260604160448 / 20260604163021 / 20260604164540 / 20260604165114

- logs:
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260604160448_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604160448_elastic.txt)
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260604163021_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604163021_elastic.txt)
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260604164540_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604164540_elastic.txt)
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260604165114_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604165114_elastic.txt)
- type: invalid / unstable single-control-message exploration
- result:
  - some launcher-only runs, some early failures
- value:
  - showed this experimental branch was not ready to become default

### 10. 20260604181240

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604181240_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604181240_elastic.txt)
- type: invalid / unclear short run
- result:
  - not used as trusted comparison evidence

### 11. 20260604181656

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604181656_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604181656_elastic.txt)
- type: invalid / entry-layer HCCL bind failure
- result:
  - `hcclCommInitRootInfoConfig ... error code is 7`
  - `Bind_Failed(EJ0003)`
  - `[run] ... exit_code=1`
- value:
  - useful only as evidence that entry-layer port collisions were still present at that point

### 12. 20260604182806

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604182806_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604182806_elastic.txt)
- ray worker log: [/tmp/ray/session_2026-06-04_18-28-16_126908_3562084/logs/worker-663cadc54d9ec7e7fd9c1029fbc1f56cbcae53062d11ec2f8b1fbc11-01000000-3581720.out](/tmp/ray/session_2026-06-04_18-28-16_126908_3562084/logs/worker-663cadc54d9ec7e7fd9c1029fbc1f56cbcae53062d11ec2f8b1fbc11-01000000-3581720.out)
- type: valid success run
- result:
  - `rollout_output_time_s = 255.078942`
  - `252.435529`
  - `252.556386`
- value:
  - current fastest verified end-to-end rollout wall time
- caveat:
  - hotspot timing was worse than `20260604154651`, so this is not the cleanest overlap-quality run

### 13. 20260604185317

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604185317_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604185317_elastic.txt)
- type: invalid / not used as trusted comparison sample

### 14. 20260604185942

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604185942_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604185942_elastic.txt)
- type: invalid / HCCL bind failure during front-stage retry
- result:
  - `[run] ... exit_code=1`
- value:
  - not a timing sample

### 15. 20260604191245

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604191245_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604191245_elastic.txt)
- type: invalid / entry failure
- result:
  - `[run] ... exit_code=1`
- value:
  - not a timing sample

### 16. 20260604192016

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604192016_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604192016_elastic.txt)
- type: invalid / launcher-only retry
- result:
  - launcher head only, trainer never really settled
- value:
  - not comparable

### 17. 20260604192748

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604192748_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604192748_elastic.txt)
- type: invalid / short unsuccessful retry
- value:
  - not used as trusted evidence

### 18. 20260604193735

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604193735_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604193735_elastic.txt)
- type: valid failure sample
- result:
  - got past entry / WorkerGroup bring-up with patched HCCL base-port block
  - later hit `SYSTEM_ERROR` / `ActorDiedError`
- value:
  - proved that the local `base.py` HCCL block fix was live
  - also showed runtime instability was now beyond launcher / base-port entry

### 19. 20260604195353

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604195353_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604195353_elastic.txt)
- type: valid failure sample
- result:
  - progressed further than entry
  - still died in training-early runtime path
- value:
  - used to test minimal rollback of unstable remote-path changes

### 20. 20260604200421

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604200421_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604200421_elastic.txt)
- type: valid failure sample
- result:
  - entered runtime deeply enough to show real prompt processing and shrink/refresh activity
  - later hit:
    - `Worker exit type: SYSTEM_ERROR`
    - `ActorDiedError`
    - `[run] ... exit_code=1`
- value:
  - important reminder that `TBE Subprocess...` itself was not the failure signal
  - the real failure boundary was `SYSTEM_ERROR + ActorDiedError + exit_code=1`

### 21. 20260604201424

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604201424_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604201424_elastic.txt)
- type: invalid / wrong-mode sample
- result:
  - launcher fell back to mode 4 instead of correct mode 5
- value:
  - discarded for mode5 comparison

### 22. 20260604201535

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604201535_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604201535_elastic.txt)
- type: valid failure sample
- result:
  - correct `mode=5 dual_source + timing_sync=1`
  - `WorkerGroup ... HCCL_IF_BASE_PORT=45681`
  - still failed during worker/model-init region with HCCL bind errors
  - `[run] ... exit_code=1`
- value:
  - proved local `base.py` HCCL base-port fix was being consumed in a true mode5 run

### 23. Control worktree attempt: 20260604202034

- log: [/workspace/cann-recipes-train/llm_rl/qwen3_control_origin_master/wjeagerqwen30b-a3b-with_draft_breakdown_20260604202034_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3_control_origin_master/wjeagerqwen30b-a3b-with_draft_breakdown_20260604202034_elastic.txt)
- type: invalid control experiment
- result:
  - failed with:
    - `ImportError: cannot import name 'reset_moe_comm_method_cache'`
  - `[run] ... exit_code=1`
- meaning:
  - this was **not** a valid baseline runtime comparison
  - `origin/master` control tree was internally inconsistent for this test because:
    - `worker_v1.py` imported `reset_moe_comm_method_cache`
    - but the control tree `moe_comm_method.py` did not define it
- consequence:
  - this run cannot be used to claim that baseline `origin/master` reproduces the same mode5 runtime crash

### 24. Corrected control worktree attempt: 20260604203613

- log: [/workspace/cann-recipes-train/llm_rl/qwen3_control_origin_master/wjeagerqwen30b-a3b-with_draft_breakdown_20260604203613_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3_control_origin_master/wjeagerqwen30b-a3b-with_draft_breakdown_20260604203613_elastic.txt)
- type: invalid / pending-classification corrected control attempt
- result:
  - launched from corrected control tree:
    - `origin/master worker_v1.py`
    - plus only required support patches in `base.py` and `moe_comm_method.py`
  - file currently contains only launcher / plugin-init header
  - no confirmed:
    - `WorkerGroup ... HCCL_IF_BASE_PORT=...`
    - `TaskRunnerTiming init_workers_done`
    - `Before trainer.fit()`
    - `Mode5 timing`
    - `rollout_output_time_s`
    - `exit_code`
- meaning:
  - this corrected control attempt fixed the earlier `ImportError` problem
  - but it still did **not** produce a usable runtime baseline result
  - so it cannot yet be used to prove whether `origin/master worker_v1.py` shares the same runtime crash boundary as the local worktree

### 25. Corrected control worktree attempt: 20260604204633

- log: [/workspace/cann-recipes-train/llm_rl/qwen3_control_origin_master/wjeagerqwen30b-a3b-with_draft_breakdown_20260604204633_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3_control_origin_master/wjeagerqwen30b-a3b-with_draft_breakdown_20260604204633_elastic.txt)
- type: valid corrected-control failure sample
- setup:
  - control worktree kept `origin/master worker_v1.py`
  - but used the required support patches from the main worktree in:
    - `verl/single_controller/ray/base.py`
    - `vllm_ascend/ops/moe/moe_comm_method.py`
- result:
  - control run got past launcher and true trainer bring-up
  - authoritative evidence includes:
    - `WorkerGroup obVEDA uses MASTER_PORT=46169 HCCL_IF_BASE_PORT=45801`
    - `TaskRunnerTiming init_workers_done`
    - `TaskRunner Before trainer.fit()`
    - multiple `Mode5 hybrid refresh aggregate` lines at `stage=8`
    - multiple workers reaching `Processed prompts: 100%`
  - later failed with:
    - `Worker exit type: SYSTEM_ERROR`
    - `ActorDiedError`
    - `[run] ... exit_code=1`
- meaning:
  - this is the first corrected control run that is actually comparable at runtime depth
  - it shows that a baseline tree with `origin/master worker_v1.py`, once made runnable with only the required support patches, can also:
    - enter real `mode=5 dual_source` runtime,
    - reach `stage=8` refresh and initial prompt processing,
    - and still die in the same broad early-runtime region
  - consequence:
    - the remaining small local `worker_v1.py` diff is **not** enough by itself to explain the instability boundary
    - the failure boundary is at least partly shared with the corrected control baseline

## Current Practical Reading

As of this document version, the most useful operational summary is:

1. Fastest verified wall time so far:
- **`252.435529 s`**
- run family: `20260604182806`

2. Best hotspot-quality overlap baseline so far:
- **`20260604154651`**
- `rollout_output_time_s = 270.998779`
- still the cleanest reference for lower `stage=1/2 submit_populate_us` and `submit_remote_npu_us`

3. Current debugging boundary:
- late-night attempts after `20260604182806` mostly split into:
  - entry-layer HCCL bind failures,
  - worker early `SYSTEM_ERROR / ActorDiedError`,
  - invalid control experiments,
  - or corrected control attempts that reproduced the same broad early-runtime crash boundary after entering real `mode=5` runtime
- therefore:
  - no later run has yet replaced `20260604154651` as the best hotspot-quality timing reference
  - and the current evidence no longer supports blaming only the remaining small local `worker_v1.py` diff

### 26. 20260604205903

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604205903_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604205903_elastic.txt)
- type: valid failure sample
- setup:
  - current main worktree
  - retained local `base.py` HCCL base-port fix
  - retained current local `worker_v1.py` small conservative diffs
- result:
  - reached true runtime execution boundary, not launcher-only failure
  - crossed into:
    - `trainer.fit()`
    - `generate_sequences(...)`
    - elastic shrink / refresh runtime path
  - later failed with:
    - `Worker exit type: SYSTEM_ERROR`
    - `ActorDiedError`
    - `[run] ... exit_code=1`
- meaning:
  - this rerun reproduces the same broad early-runtime crash boundary as the corrected control baseline
  - it further weakens the theory that the remaining tiny local `worker_v1.py` diff alone explains instability

### 27. 20260604210936

- log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604210936_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604210936_elastic.txt)
- type: valid failure sample
- setup:
  - current main worktree
  - local `base.py` HCCL base-port fix retained
  - `moe_comm_method.py` setup-signature cache rolled back to conservative always-setup behavior
- result:
  - run really launched and occupied all 16 NPUs
  - later still failed with:
    - `Worker exit type: SYSTEM_ERROR`
    - `ActorDiedError`
    - `[run] ... exit_code=1`
- meaning:
  - rolling back the `moe_comm_method.py` setup cache did **not** materially change the early-runtime failure boundary
  - therefore that cache is unlikely to be the decisive shared root cause

## Updated Practical Reading

The current reading should now be interpreted as:

1. Fastest verified wall time still remains:
- **`252.435529 s`**
- run family: `20260604182806`

2. Best hotspot-quality overlap baseline still remains:
- **`20260604154651`**
- `rollout_output_time_s = 270.998779`

3. Multiple later retries now point to the same broader conclusion:
- corrected-control and current-main-tree runs can both enter true `mode=5 dual_source` runtime
- both can reach the early execution / prompt-processing region
- both still die with the same broad `SYSTEM_ERROR / ActorDiedError` boundary

4. What has been weakened by recent evidence:
- the hypothesis that the remaining tiny local `worker_v1.py` diff is the main cause
- the hypothesis that the `moe_comm_method.py` setup-signature cache is the decisive shared root cause

5. What is now more likely true:
- the next meaningful baseline should come from a historically self-consistent code state, not from assuming current `origin/master` or a stitched control tree is automatically authoritative

## Updating Rule

When adding future runs to this ledger, classify each attempt into one of four buckets:

- valid success
- valid failure
- invalid / not comparable
- control experiment invalid due to code-state mismatch

And always keep **both** of these top lines current:

- fastest verified end-to-end wall time
- best known hotspot-quality overlap baseline

### 20260604213306
- Code state: historical self-consistent baseline candidate exported from `d675bf4`, with only `verl/single_controller/ray/base.py` minimally patched to honor `VERL_HCCL_IF_BASE_PORT_START` / `HCCL_IF_BASE_PORT` for `WorkerGroup` HCCL base-port allocation.
- Log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604213306_elastic.txt](./qwen3_mode5_hist_base_d675bf4/wjeagerqwen30b-a3b-with_draft_breakdown_20260604213306_elastic.txt)
- Classification: valid failure sample; historically self-consistent baseline enters true `mode=5 dual_source` runtime and fails on a concrete shared runtime bug rather than launcher-only issues.
- Key evidence:
  - NPU usage rose to deep runtime occupancy across all 16 devices, so this was not a launcher-only false start.
  - Failure was not the trailing `TBE Subprocess...` noise. The first hard root signal was:
    - `RuntimeError: Invalid DP metadata sync result: mode=5 ... use_cpu_sync=False ... max_tokens_across_dp=~1.0e9 ... This would otherwise force a bogus post-shrink AllToAll path.`
    - thrown from `vllm_ascend/worker/model_runner_v1.py::_sync_metadata_across_dp(...)`.
  - This shows the historically cleaner baseline still carries a shared mode5 DP metadata sync bug: mode3/4 default to CPU-group sync, but mode5 still uses NPU all-reduce and can return corrupted token counts after shrink.
- Practical reading:
  - This is a stronger root-cause boundary than earlier `SYSTEM_ERROR/ActorDiedError` samples. It points to a concrete shared fix direction in `model_runner_v1.py` rather than more `worker_v1.py` remote-path speculation.
  - It also means the next runtime experiment should first enable/force CPU DP metadata sync for mode5 before judging any prefetch overlap optimization.


### 20260604214331
- Code state: current main worktree plus `mode=5` CPU DP metadata sync enabled/fixed in `vllm_ascend/worker/model_runner_v1.py` via `VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC=1`.
- Log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604214331_elastic.txt](./wjeagerqwen30b-a3b-with_draft_breakdown_20260604214331_elastic.txt)
- Classification: valid failure sample; important positive regression check.
- Key evidence:
  - This run no longer failed with `RuntimeError: Invalid DP metadata sync result ... use_cpu_sync=False ... bogus post-shrink AllToAll path`, which was the new shared root cause exposed by the historical baseline run `20260604213306`.
  - The run progressed further and then failed later in `trainer.fit() -> generate_sequences(...)` with `ActorDiedError` / `Worker exit type: SYSTEM_ERROR`.
  - Therefore `mode5 CPU DP metadata sync` is not a cosmetic change; it removes one shared crash boundary and exposes the next deeper blocker.
- Practical reading:
  - This is real forward progress even though the run still ends with `exit_code=1`.
  - The next debugging layer should focus on the worker/runtime path reached after the DP metadata sync is made trustworthy, not on the old bogus synced-token-count failure.


### 20260604214331 deeper failure boundary refinement
- Same run family as above:
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260604214331_elastic.txt](./wjeagerqwen30b-a3b-with_draft_breakdown_20260604214331_elastic.txt)
  - Ray session:
    - `/tmp/ray/session_2026-06-04_21-43-40_499141_123615`
- Classification:
  - valid deeper runtime failure analysis
- Key evidence:
  - Multiple workers reached:
    - `Processed prompts: 100%`
  - Then the first hard failure became:
    - `Unhandled exception: N4gloo13EnforceNotMetE`
    - `what(): [enforce fail at ... gloo/transport/tcp/pair.cc:446] op.preamble.length <= op.nbytes. 3096 vs 16`
  - Representative files:
    - `/tmp/ray/session_2026-06-04_21-43-40_499141_123615/logs/worker-6e395264c013c1c7fd20811901e7cdec0d2c5cd12ff6bc26dc6f470e-01000000-146146.err`
    - `/tmp/ray/session_2026-06-04_21-43-40_499141_123615/logs/worker-0f0fd2ac27b724e5361fb2489320f55ff14b6981315d39dc466dd3d7-01000000-146140.err`
    - `/tmp/ray/session_2026-06-04_21-43-40_499141_123615/logs/worker-1efc2e069ac3135fc88e541828446b2e7c15fd272191b44857e8fa3e-01000000-146143.err`
- Root-cause interpretation:
  - `3096 = 129 * 3 * 8`
  - `16 = 2 * 8`
  - This matches a protocol mismatch between:
    - the experimental mode5 single `control` message format:
      - `(129, 3)` `int64`
    - and the legacy two-message control path shape header:
      - `(2,)` `int64`
  - The bug was in `vllm_ascend/worker/worker_v1.py`:
    - mode5 owner-side send path was unconditionally sending single `control` messages,
    - while the cache-rank receive path still honored `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE`
      and default launcher configuration kept that flag at `0`.
  - Result:
    - sender could emit `(129, 3)` while receiver still expected `(2,)`,
    - producing the exact Gloo length mismatch `3096 vs 16`.
- Fix applied in current main worktree:
  - `worker_v1.py` owner-side control send now uses:
    - `if execution_mode == 5 and _mode5_single_control_message_remote():`
    - instead of unconditionally switching all mode5 sends to the new protocol.
  - Mode5 stale-cache self-sentinel was also aligned to the same protocol gate:
    - `(129, 3)` when single-control-message mode is enabled
    - `(2,)` otherwise
- Practical reading:
  - This was a real shared protocol bug, not a generic “Gloo instability”.
  - Until the experimental single-control-message path is intentionally re-enabled and revalidated,
    stable mode5 mainline validation should keep:
    - `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=0`

## Current recommended stable validation config

For the next clean mainline mode5 validation, prefer:

- `VLLM_ASCEND_ELASTIC_EXECUTION_MODE=5`
- `mode5_runtime_strategy=dual_source`
- `VLLM_ASCEND_MODE3_TIMING_LOG=1`
- `VLLM_ASCEND_MODE3_TIMING_SYNC=1`
- `VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC=1`
- `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=0`
- fresh:
  - `HCCL_IF_BASE_PORT`
  - `MASTER_PORT`
  - `VERL_HCCL_IF_BASE_PORT_START`

Interpretation:

- `VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC=1` is currently required to avoid the bogus synced-token-count / invalid DP metadata sync failure exposed by `20260604213306`.
- `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=0` should remain the stable default until the single-control-message experimental path is revalidated end-to-end.


### 20260604234736
- Code state: current main worktree with the two shared stability fixes retained:
  - `vllm_ascend/worker/model_runner_v1.py`: mode5 CPU DP metadata sync enabled by default via `VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC=1`
  - `vllm_ascend/worker/worker_v1.py`: mode5 remote control protocol kept on the stable double-message mainline via `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=0`
  - plus one focused owner-side remote-fetch submit change intended to combine the two earlier bests:
    - keep the faster stage-refresh/build-sources path seen in `20260604182806`
    - while rolling back the over-serialized owner-side `wait-all-sends then post-all-irecv` behavior that likely worsened stage=2/1 submit timing.
- Log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604234736_elastic.txt](./wjeagerqwen30b-a3b-with_draft_breakdown_20260604234736_elastic.txt)
- Classification: valid success sample for stability and timing visibility; not a new performance best.
- Key evidence:
  - This run cleanly crossed the recent shared stability blockers:
    - no `Invalid DP metadata sync result`
    - no `3096 vs 16`
    - no immediate `SYSTEM_ERROR / ActorDiedError`
  - It entered true mode5 runtime and emitted stage 8/4/2/1 timing lines.
  - First verified rollout result:
    - `rollout_output_time_s = 280.873270`
  - First step summary:
    - `timing_s/gen = 280.869259`
    - `timing_s/old_log_prob = 27.293029`
    - `timing_s/ref = 15.714460`
    - `timing_s/update_actor = 108.168324`
    - `timing_s/step = 435.048674`
  - Stage=2 hotspot summary from the emitted timing lines:
    - `submit_populate_us p50/p90 = 2365.8 / 2699.04`
    - `submit_remote_npu_us p50/p90 = 1870.3 / 2114.82`
    - `prefetch_dev_ms p50/p90 = 5.361 / 5.749`
  - Stage=1 hotspot summary:
    - `submit_populate_us p50/p90 = 4450.2 / 4657.76`
    - `submit_remote_npu_us p50/p90 = 3810.2 / 4016.62`
    - `prefetch_dev_ms p50/p90 = 11.150 / 11.197`
  - Stage-refresh aggregate means from this run:
    - `stage 8 total_ms = 610.39`
    - `stage 4 total_ms = 985.50`
    - `stage 2 total_ms = 1683.40`
    - `stage 1 total_ms = 3993.53`
- Practical reading:
  - This run is important because it shows the mainline is materially more stable and can again produce valid mode5 timing and rollout samples after the DP-metadata-sync and control-protocol fixes.
  - The **first** verified rollout is not a performance win: `rollout_output_time_s = 280.873270`, and the first-cycle stage=2/1 hotspot metrics are still worse than the overlap-quality baseline `20260604154651`.
  - However, later timing evidence inside the same run shows a strong warm-state effect: a second refresh cycle appears with much faster aggregate timings than the first cycle, including:
    - `stage 8 total_ms = 399.31`
    - `stage 4 total_ms = 565.62`
    - `stage 2 total_ms = 749.42`
    - `stage 1 total_ms = 1900.76`
    which are materially better than both the first cycle of this run and the earlier `20260604182806` aggregate totals.
  - Because the run was manually interrupted after the first verified rollout, it is **not yet valid** to conclude that this code state is performance-negative overall. The more accurate reading is:
    - stability-positive,
    - first-rollout slower,
    - but with strong evidence that subsequent warm cycles may preserve the fast stage-refresh/build-sources path.
  - Therefore this attempt is not a `best_mode5` candidate yet, but it should also not be treated as a clean negative result until a full multi-step rerun verifies whether the later-cycle aggregate gains translate into faster `gen`/`rollout_output_time_s`.

### 20260605001622
- Run log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260605001622_elastic.txt](./wjeagerqwen30b-a3b-with_draft_breakdown_20260605001622_elastic.txt)
- Classification: valid completed mixed-result sample
- Config:
  - `mode=5`
  - `dual_source`
  - `VLLM_ASCEND_MODE3_TIMING_SYNC=1`
  - `VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC=1`
  - `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=0`
- Verified rollout samples:
  - step1 `266.758895 s`
  - step2 `270.110607 s`
  - step3 `293.717419 s`
- Step-level interpretation:
  - step1 beats the `270.998779 s` baseline by `4.239884 s`
  - step2 beats the `270.998779 s` baseline by `0.888172 s`
  - step3 regresses above the baseline by `22.718640 s`
- Gen/step buckets relative to the `270.998779 s` baseline:
  - step1: `timing_s/gen 266.755196`, `timing_s/step 427.492554`
  - step2: `timing_s/gen 270.106649`, `timing_s/step 417.512562`, `old_log_prob 23.184293`, `update_actor 105.778717`
  - step3: `timing_s/gen 293.713285`, `timing_s/step 438.455815`
- Hotspot summary for this run:
  - stage2 `submit_populate_us p50/p90 = 2204.95 / 2427.19`, `submit_remote_npu_us p50/p90 = 1741.95 / 1937.80`, `prefetch_dev_ms p50/p90 = 5.351 / 5.561`
  - stage1 `submit_populate_us p50/p90 = 4297.20 / 4964.56`, `submit_remote_npu_us p50/p90 = 3681.80 / 4171.42`, `prefetch_dev_ms p50/p90 = 11.108 / 11.217`
- Aggregate interpretation:
  - later stage-8 refresh aggregate enters a very strong `337-389 ms` band, preserving the fast refresh/build_sources behavior we wanted from the `252 s` family
  - stage-4 aggregate later sits around `501-593 ms`, stage-2 around `653-698 ms`, and stage-1 around `1333 ms` on the hottest rank, which is also materially stronger than the older cold-cycle samples
  - despite that, the later rollout still regresses, so the remaining slowdown is not explained by stage-8/4 refresh alone
- Conclusion:
  - this code state is stability-positive and preserves the fast stage-8/4 path
  - it is not yet a new best because later-step `gen` regresses sharply
  - the next iteration should focus on the later-step stage-2/1 or post-shrink path, not on stage-8/4 refresh

### 20260605013258
- Run log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260605013258_elastic.txt](./wjeagerqwen30b-a3b-with_draft_breakdown_20260605013258_elastic.txt)
- Classification: valid three-step positive sample on the current stable mainline and submit-path patch; stronger than the old `270.998779` overlap-quality baseline on all three verified rollout steps, but still not the historical wall-time best.
- Config:
  - `mode=5`
  - `dual_source`
  - `VLLM_ASCEND_MODE3_TIMING_SYNC=1`
  - `VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC=1`
  - `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=0`
  - `VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION=0.75`
  - `VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY=fixed`
- Verified source mix recovery:
  - import partition summaries match the `270.998779` overlap-quality baseline again:
    - floor8 `remote_selected=48 cpu_selected=16`
    - floor4 `remote_selected=72 cpu_selected=24`
    - floor2 `remote_selected=84 cpu_selected=28`
  - first timing source mix also matches baseline:
    - stage2 `source_from_remote_npu=42 source_from_cpu=14`
    - stage1 `source_from_remote_npu=90 source_from_cpu=30`
- Verified rollout results:
  - step1 `rollout_output_time_s = 269.776622`
  - step2 `rollout_output_time_s = 266.018334`
  - step3 `rollout_output_time_s = 262.421108`
- Verified step summaries:
  - step1:
    - `timing_s/gen = 269.773030`
    - `timing_s/old_log_prob = 28.030720`
    - `timing_s/ref = 15.354492`
    - `timing_s/update_actor = 105.991481`
    - `timing_s/step = 422.078180`
  - step2:
    - `timing_s/gen = 266.014432`
    - `timing_s/old_log_prob = 23.203719`
    - `timing_s/ref = 13.695731`
    - `timing_s/update_actor = 104.878858`
    - `timing_s/step = 410.692608`
  - step3:
    - `timing_s/gen = 262.417666`
    - `timing_s/old_log_prob = 22.707037`
    - `timing_s/ref = 14.483774`
    - `timing_s/update_actor = 105.066645`
    - `timing_s/step = 407.790079`
- Interpretation relative to the `270.998779` baseline:
  - step1 beats baseline by `1.222157 s`
  - step2 beats baseline by `4.980445 s`
  - step3 beats baseline by `8.577671 s`
  - this is the first current-mainline sample where all three verified rollout steps beat the old overlap-quality baseline while keeping the recovered `fixed/0.75` source mix.
- Remaining caveat:
  - stage=2/1 submit-path timing is still visibly heavier than the old `20260604154651` hotspot baseline, so this is not yet proof that the original stage=2/1 overlap-quality metrics have been fully recovered.
  - treat this run as a strong new stable-mainline milestone for the current submit-path patch. It is a better canonical baseline than `270.998779` for the current code path, but it still does not beat the historical wall-time best `252.435529`.

### 20260605021206
- Run log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260605021206_elastic.txt](./wjeagerqwen30b-a3b-with_draft_breakdown_20260605021206_elastic.txt)
- Ray session:
  - `/tmp/ray/session_2026-06-05_02-12-16_209900_714262`
- Classification: valid positive sample with `single_control_message_remote=1`; current strongest candidate for combining the old `270s` stage=2/1 direction with the faster wall-time path from the `252s` family.
- Config:
  - `mode=5`
  - `dual_source`
  - `VLLM_ASCEND_MODE3_TIMING_SYNC=1`
  - `VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC=1`
  - `VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION=0.75`
  - `VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY=fixed`
  - `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=1`
- Verified rollout results so far:
  - step1 `rollout_output_time_s = 254.491324`
  - step2 `rollout_output_time_s = 264.681957`
- Verified step summaries:
  - step1:
    - `timing_s/gen = 254.487689`
    - `timing_s/old_log_prob = 28.941315`
    - `timing_s/ref = 14.691197`
    - `timing_s/update_actor = 113.994052`
    - `timing_s/step = 415.024120`
  - step2:
    - `timing_s/gen = 264.678575`
    - `timing_s/old_log_prob = 25.340860`
    - `timing_s/ref = 15.524812`
    - `timing_s/update_actor = 111.496597`
    - `timing_s/step = 419.897830`
- Interpretation relative to earlier references:
  - versus the old overlap-quality baseline `20260604154651 (270.998779)`:
    - step1 improves by `16.507455 s`
    - step2 improves by `6.316822 s`
  - versus the newer stable-mainline baseline `20260605013258`:
    - step1 improves by `15.285298 s`
    - step2 improves by `1.336377 s`
  - step1 is also very close to the historical wall-time best family `20260604182806`, landing only about `2.055795 s` above the absolute best `252.435529`.
- Stage-refresh aggregate summary:
  - `stage8 total_mean = 430.91 ms`
  - `stage4 total_mean = 665.46 ms`
  - `stage2 total_mean = 927.25 ms`
  - `stage1 total_mean = 2035.63 ms`
- Hotspot summary from emitted timing lines:
  - stage2:
    - `submit_populate_us p50/p90 = 2195.25 / 2235.80`
    - `submit_remote_npu_us p50/p90 = 1745.00 / 1776.39`
    - `prefetch_submit_us p50/p90 = 2318.65 / 2359.25`
    - `prefetch_dev_ms p50/p90 = 4.94 / 5.03`
  - stage1:
    - `submit_populate_us p50/p90 = 4725.30 / 5055.66`
    - `submit_remote_npu_us p50/p90 = 3979.10 / 4217.72`
    - `prefetch_submit_us p50/p90 = 4900.80 / 5229.92`
    - `prefetch_dev_ms p50/p90 = 10.35 / 10.41`
- Practical reading:
  - Re-opening the single-control-message path is now a valid mainline optimization, not just an unstable experiment, because the protocol mismatch bug was fixed earlier and the run has re-entered true mode5 runtime cleanly.
  - This change materially improves wall time and strongly improves the stage2 submit path compared with `20260605013258`:
    - stage2 `submit_remote_npu_us` drops from `2274.05 / 2580.48` to `1745.00 / 1776.39`
    - stage2 `submit_populate_us` drops from `2823.00 / 3213.64` to `2195.25 / 2235.80`
    - stage2 `prefetch_submit_us` drops from `2972.05 / 3374.36` to `2318.65 / 2359.25`
  - Stage1 remains noticeably heavier than the old `20260604154651` overlap-quality baseline, so this is not yet the final answer to the original objective.
  - However, this run is the strongest evidence so far that the code can preserve the faster non-stage2/1 path while recovering a substantial portion of the desired stage2 behavior. The next narrowing step should focus only on stage1 submit-path overhead inside `worker_v1.py`, not on source mix or stage8/4 refresh.


### 20260605023723
- Run log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260605023723_elastic.txt](./wjeagerqwen30b-a3b-with_draft_breakdown_20260605023723_elastic.txt)
- Ray session:
  - `/tmp/ray/session_2026-06-05_02-37-32_767609_780790`
- Classification: valid failed single-variable experiment.
- Setup:
  - kept the strong `20260605021206` mainline:
    - `mode=5`
    - `dual_source`
    - `VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC=1`
    - `VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION=0.75`
    - `VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY=fixed`
    - `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=1`
  - added one new owner-side submit-path patch in `worker_v1.py` that deferred CPU control-send waits until after payload copy.
- Result:
  - run reached `Elastic first live step: entering engine_core.get_output`
  - but never produced a valid `rollout_output_time_s` / `timing_s/gen`
  - authoritative failure evidence later showed:
    - `Aborted (core dumped)`
    - `[run] ... exit_code=134`
- Meaning:
  - this was not merely a slow sample; it was a true failed patch branch.
  - the deferred send-wait patch should not stay on the mainline.

### 20260605030829
- Run log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260605030829_elastic.txt](./wjeagerqwen30b-a3b-with_draft_breakdown_20260605030829_elastic.txt)
- Ray session:
  - `/tmp/ray/session_2026-06-05_03-08-38_682771_863895`
- Classification: valid strongest positive rerun after rolling back the failed `20260605023723` patch; current strongest evidence that the `a12f64a` mainline really combines the fast wall-time path with a substantially improved stage1 path.
- Config:
  - `mode=5`
  - `dual_source`
  - `VLLM_ASCEND_MODE3_TIMING_SYNC=1`
  - `VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC=1`
  - `VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION=0.75`
  - `VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY=fixed`
  - `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=1`
- Verified rollout results:
  - step1 `rollout_output_time_s = 252.091123`
  - step2 `rollout_output_time_s = 255.107707`
  - step3 `rollout_output_time_s = 249.546084`
- Verified step summaries:
  - step1:
    - `timing_s/gen = 252.087249`
    - `timing_s/old_log_prob = 27.236369`
    - `timing_s/ref = 14.688806`
    - `timing_s/update_actor = 105.345235`
    - `timing_s/step = 402.412223`
  - step2:
    - `timing_s/gen = 255.104085`
    - `timing_s/old_log_prob = 20.986882`
    - `timing_s/ref = 13.803832`
    - `timing_s/update_actor = 99.966502`
    - `timing_s/step = 392.856732`
  - step3:
    - `timing_s/gen = 249.538605`
    - `timing_s/old_log_prob = 22.124644`
    - `timing_s/ref = 13.276346`
    - `timing_s/update_actor = 101.223105`
    - `timing_s/step = 389.295643`
- Interpretation relative to earlier references:
  - versus the historical wall-time best family `20260604182806`:
    - step1 beats `255.078942` by `2.987819 s`
    - step2 is `2.672178 s` slower than the family best `252.435529`
    - step3 beats `252.556386` by `3.010302 s`
  - versus the previous strong mainline `20260605021206`:
    - step1 improves by `2.400201 s`
    - step2 improves by `9.574250 s`
    - step3 improves by `8.567600 s`
  - versus the old overlap-quality baseline `20260604154651 (270.998779)`:
    - step1 improves by `18.907656 s`
    - step2 improves by `15.891072 s`
    - step3 improves by `21.452695 s`
- Current stage-refresh aggregate summary from emitted lines:
  - `stage8 total_mean ≈ 403.55 ms`
  - `stage4 total_mean ≈ 664.35 ms`
  - `stage2 total_mean ≈ 939.60 ms`
  - `stage1 total_mean ≈ 1755.82 ms`
- Current hotspot summary from emitted timing lines:
  - stage2:
    - `submit_populate_us p50/p90 = 2289.35 / 2436.00`
    - `submit_remote_npu_us p50/p90 = 1809.70 / 1918.89`
    - `prefetch_submit_us p50/p90 = 2417.05 / 2566.26`
    - `prefetch_dev_ms p50/p90 = 5.08 / 5.14`
  - stage1:
    - `submit_populate_us p50/p90 = 3972.50 / 4143.58`
    - `submit_remote_npu_us p50/p90 = 3357.70 / 3498.32`
    - `prefetch_submit_us p50/p90 = 4102.60 / 4281.88`
    - `prefetch_dev_ms p50/p90 = 10.33 / 10.37`
- Practical reading:
  - This run is materially stronger than `20260605021206` on all three verified rollout steps.
  - It keeps the stage8/4 fast path while significantly improving the remaining stage1 path compared with `20260605021206`:
    - stage1 `submit_populate_us` improves by about `752.8 us` at p50
    - stage1 `submit_remote_npu_us` improves by about `621.4 us` at p50
    - stage1 `prefetch_submit_us` improves by about `798.2 us` at p50
  - stage2 is slightly worse than `20260605021206`, but still dramatically better than `20260605013258` and far closer to the original `270.998779` overlap-direction baseline than earlier stable mainline runs.
  - As of the currently verified evidence, this is the best real combination yet of:
    - the old `270s` direction on stage2/1,
    - and the faster wall-time path from the `252s` family.
  - This run also establishes a new fastest verified rollout value of **`249.546084 s`**.

### 20260605034323
- Run log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260605034323_elastic.txt](./wjeagerqwen30b-a3b-with_draft_breakdown_20260605034323_elastic.txt)
- Ray session:
  - `/tmp/ray/session_2026-06-05_03-43-32_630226_942076`
- Classification: valid clean single-variable follow-up on top of the `a3cfdc2` mainline; clean exit with `exit_code=0`.
- Config:
  - kept the validated mainline:
    - `mode=5`
    - `dual_source`
    - `VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC=1`
    - `VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION=0.75`
    - `VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY=fixed`
    - `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=1`
  - added one new owner-side submit-path patch in `worker_v1.py` that only changes the `parallel_remote_fetch=1` path:
    - queue control sends
    - post payload `irecv` earlier
    - delay the corresponding CPU send waits until after payload copy
    - leave `parallel=0` behavior unchanged
- Verified rollout results:
  - step1 `rollout_output_time_s = 255.670438`
  - step2 `rollout_output_time_s = 249.003097`
  - step3 `rollout_output_time_s = 250.230129`
- Verified step summaries:
  - step1:
    - `timing_s/gen = 255.666954`
    - `timing_s/old_log_prob = 29.405523`
    - `timing_s/ref = 15.759657`
    - `timing_s/update_actor = 104.363073`
    - `timing_s/step = 408.309690`
  - step2:
    - `timing_s/gen = 248.999479`
    - `timing_s/old_log_prob = 21.873047`
    - `timing_s/ref = 14.775532`
    - `timing_s/update_actor = 101.514047`
    - `timing_s/step = 390.282393`
  - step3:
    - `timing_s/gen = 250.226310`
    - `timing_s/old_log_prob = 22.134977`
    - `timing_s/ref = 13.852267`
    - `timing_s/update_actor = 101.537359`
    - `timing_s/step = 390.792512`
- Comparison vs previous best mainline `20260605030829`:
  - previous three steps:
    - `252.091123`
    - `255.107707`
    - `249.546084`
  - current three steps:
    - `255.670438`
    - `249.003097`
    - `250.230129`
  - net effect:
    - step1 is worse by `3.579315 s`
    - step2 is better by `6.104610 s`
    - step3 is worse by `0.684045 s`
    - 3-step total improves from `756.744914 s` to `754.903664 s`
    - 3-step average improves by about `0.613750 s`
- Practical reading:
  - This patch is not a simple cold-start win; it trades a slower first step for a materially stronger step2 and a slightly improved 3-step total.
  - The clean `exit_code=0` matters: unlike the earlier failed `20260605023723` branch, this change stays on the valid mainline path.
  - The most defensible interpretation is that the patch improves the parallel owner-side submit path enough to matter in steady-state or later-step behavior, even though step1 regresses.
  - Because the full 3-step total now beats the previous `a3cfdc2` mainline, this run should be treated as the new strongest verified mainline snapshot for ongoing work, while still noting that the cold first step is worse.

### 20260605042224
- Run log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260605042224_elastic.txt](./wjeagerqwen30b-a3b-with_draft_breakdown_20260605042224_elastic.txt)
- Ray session:
  - `/tmp/ray/session_2026-06-05_04-22-33_888741_1033468`
- Classification: valid clean hybrid follow-up on top of the `20260605034323` idea; clean exit with `exit_code=0`.
- Patch intent:
  - preserve the later-step benefit of the queued `parallel_remote_fetch=1` owner path
  - but restore the older wait-before-irecv order on the first cold fetch key only
  - practical meaning: first cold step uses the old conservative order; later same-key fetches use the newer lower-overhead queued order
- Verified rollout results:
  - step1 `rollout_output_time_s = 229.267634`
  - step2 `rollout_output_time_s = 244.775307`
  - step3 `rollout_output_time_s = 238.954965`
- Verified step summaries:
  - step1:
    - `timing_s/gen = 229.264582`
    - `timing_s/old_log_prob = 26.765794`
    - `timing_s/ref = 13.748301`
    - `timing_s/update_actor = 101.213635`
    - `timing_s/step = 373.961911`
  - step2:
    - `timing_s/gen = 244.772300`
    - `timing_s/old_log_prob = 21.359867`
    - `timing_s/ref = 14.268682`
    - `timing_s/update_actor = 103.327739`
    - `timing_s/step = 386.852041`
  - step3:
    - `timing_s/gen = 238.951195`
    - `timing_s/old_log_prob = 21.911121`
    - `timing_s/ref = 13.794288`
    - `timing_s/update_actor = 100.558493`
    - `timing_s/step = 378.207749`
- Clean exit evidence:
  - elastic log records `[run] end_time=2026-06-05T04:43:43+0800 exit_code=0`
  - no `Aborted`, `ActorDiedError`, or `SYSTEM_ERROR` markers appear in the run log
- Comparison vs `20260605034323`:
  - `20260605034323` three steps:
    - `255.670438`
    - `249.003097`
    - `250.230129`
  - current three steps:
    - `229.267634`
    - `244.775307`
    - `238.954965`
  - net effect:
    - 3-step total improves from `754.903664 s` to `712.997906 s`
    - 3-step average improves from about `251.634555 s` to about `237.665969 s`
    - total gain is `41.905758 s`
- Comparison vs `20260605030829`:
  - `20260605030829` three steps:
    - `252.091123`
    - `255.107707`
    - `249.546084`
  - current three steps:
    - `229.267634`
    - `244.775307`
    - `238.954965`
  - net effect:
    - 3-step total improves from `756.744914 s` to `712.997906 s`
    - total gain is `43.747008 s`
- Single-step comparison:
  - fastest previous verified single rollout was `249.546084`
  - current step1 `229.267634` is faster by `20.278450 s`
- Practical reading:
  - This is no longer just a promising first-step sample; it is the strongest verified multi-step profile seen so far on this mode5 line.
  - The hybrid ordering successfully combines the cold-step strength that was missing in `20260605034323` with even stronger later-step behavior.
  - As of the current verified evidence, this is the new mainline best candidate and should replace the prior `best_mode5` snapshot.

### 20260605050337
- Run log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260605050337_elastic.txt](./wjeagerqwen30b-a3b-with_draft_breakdown_20260605050337_elastic.txt)
- Classification: valid mode=5 runtime sample, but failed as a promotable branch because it hung before the first `rollout_output_time_s / timing_s/gen` ever landed.
- Patch intent:
  - keep the `20260605042224` mainline settings
  - but narrow the cold-start legacy ordering so only larger-fanout paths stay conservative
  - practical meaning at the time: let both stage2- and stage4-style first-hit parallel remote fetches go queued-first
- Verified useful evidence:
  - live stage2 remote submit timing improved relative to `20260605042224`
  - the first larger queued-first stage after the warm stage8 path showed `remote_rank_count=6`, confirming this branch really exercised the intended stage2-style fanout
- Failure boundary:
  - main elastic log and key worker logs stopped advancing at `2026-06-05 05:08:43 +0800`
  - no first `rollout_output_time_s`
  - no `timing_s/gen`
  - no first step summary
  - trainer/raylet/TaskRunner processes and NPU occupancy remained alive afterward
- Practical reading:
  - this branch did not merely run slower; it entered a deep live-step hang
  - the stage2 hotspot direction looked better, but the overall runtime was not viable
  - do not promote this branch; its only value is evidence that a broader queued-first cold path can improve stage2 while still destabilizing the full mode=5 flow

### 20260605053619
- Run log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260605053619_elastic.txt](./wjeagerqwen30b-a3b-with_draft_breakdown_20260605053619_elastic.txt)
- Ray session:
  - `/tmp/ray/session_2026-06-05_05-36-28_683190_1212146`
- Classification: valid mode=5 runtime sample, but another failed branch because it again hung before producing the first `rollout_output_time_s / timing_s/gen`.
- Patch intent:
  - keep `ab6e3f6 / 20260605042224` mainline behavior
  - but restrict queued-first cold fetch to the stage2-style fanout only
  - keep stage1 and stage4 on the older conservative cold ordering
- Implementation note:
  - this version used `VLLM_ASCEND_MODE5_STAGE2_COLD_WAIT_REMOTE_RANKS=6`
  - live logs confirmed the intended fanout mapping was actually exercised:
    - stage4 cold path still showed `remote_rank_count=3` with conservative `request sent`
    - a later deeper path showed `remote_rank_count=6` with queued-first behavior
- Verified useful evidence:
  - stage8 timing landed normally
  - the stage2-style path again showed better submit behavior than the current `ab6e3f6` best mainline
- Failure boundary:
  - elastic log timestamps stopped at `2026-06-05 05:40:06 +0800`
  - no `rollout_output_time_s`
  - no `timing_s/gen`
  - no first step summary
  - trainer/raylet/TaskRunner processes and NPU occupancy remained alive until manual cleanup
- Practical reading:
  - this is stronger evidence than `20260605050337` that even a stage2-only queued-first cold path still destabilizes the current mode=5 runtime before the first completed rollout
  - the branch should be treated as:
    - stage2-hotspot-positive
    - overall-runtime-negative
  - the worktree should return to the validated `ab6e3f6` mainline after recording this result
