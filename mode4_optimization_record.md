# Mode=4 Remote-NPU Double Buffer Optimization Record

This note records the mode=4 optimization attempts and the current decision rules.
Read this before starting a new mode=4 optimization so we do not repeat known-bad directions.

## Goal

Mode=4 should implement remote-NPU double buffering:

- Before first shrink: no double buffer is needed.
- After `16 -> 8`: active ranks keep rollout compute; inactive ranks keep MoE expert weights as remote NPU cache.
- Each active rank should load the current runtime buffer from:
  - local NPU resident experts: 8 experts per layer at `16 -> 8`.
  - remote NPU cache experts: 8 experts per layer at `16 -> 8`.
  - CPU source count must stay `0`.
- For block prefetch, current experiment uses `VLLM_ASCEND_MODE4_BLOCK_PREFETCH_LAYERS=8`, so one prefetch block contains `8 layers * 16 experts = 128 expert rows`.
- Threshold validation target is roughly `~90s/step`; if a one-step threshold test is clearly slower, stop and optimize before running 3-step validation.

## Test Command

Use one-step threshold validation first:

```bash
TRAINER_TOTAL_EPOCHS=1 \
VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896 \
VLLM_ASCEND_MODE4_BLOCK_PREFETCH_LAYERS=8 \
bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh
```

Only after the one-step result is close enough should we run 3 steps:

```bash
TRAINER_TOTAL_EPOCHS=3 \
VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896 \
VLLM_ASCEND_MODE4_BLOCK_PREFETCH_LAYERS=8 \
bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh
```

Full validation should remove `VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS` only after threshold mode is stable.

## Parse Command

```bash
python - <<'PY'
from pathlib import Path
import re, statistics as st
p = Path('REPLACE_WITH_LOG.txt')
text = p.read_text(errors='ignore')
print('rollout_output_time_s', re.findall(r'rollout_output_time_s: ([0-9.]+)', text))
print('warmup done', re.findall(r'Elastic post-shrink MoE dispatch warmup done:.*tokens=([0-9]+) total_ms=([0-9.]+)', text)[:12])
print('remote payload warmup ms', re.findall(r'Mode4 remote payload warmup done:.*total_ms=([0-9.]+)', text)[:12])
print('next-block prime ms', re.findall(r'Mode4 next-block slot primed:.*total_ms=([0-9.]+)', text)[:12])
print('shrink done', re.findall(r'Elastic parallel shrink done:.*rebuild_ms=([0-9.]+).*refresh_ms=([0-9.]+).*warmup_ms=([0-9.]+).*total_ms=([0-9.]+)', text)[:12])
lines = [l for l in text.splitlines() if 'Mode4 timing fused-experts' in l]
print('mode4 timing lines', len(lines))
for key in ['prefetch_remote_npu_dev_ms', 'prefetch_local_npu_dev_ms', 'prefetch_dev_ms', 'current_compute_dev_ms', 'current_compute_wall_ms', 'bind_wait_us', 'ready_wait_dev_ms', 'submit_remote_npu_us', 'submit_local_npu_us', 'prefetch_submit_us']:
    vals = []
    for l in lines:
        m = re.search(r'\b' + re.escape(key) + r'=(-?[0-9.]+)', l)
        if m:
            v = float(m.group(1))
            if v >= 0:
                vals.append(v)
    if vals:
        vals = sorted(vals)
        def pct(q): return vals[min(len(vals) - 1, int(len(vals) * q))]
        print(key, 'n', len(vals), 'avg', round(sum(vals) / len(vals), 3), 'p50', round(st.median(vals), 3), 'p90', round(pct(.9), 3), 'p95', round(pct(.95), 3), 'max', round(max(vals), 3))
PY
```

## Baseline And Recent Runs

### Prepacked payload only

Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260527215602_elastic.txt`

Observed:

- `rollout_output_time_s = 142.567261`.
- Shrink total max around `9.42s`; rebuild around `7.68s`.
- `prefetch_remote_npu_dev_ms`: p50 around `13.19ms`, max around `1.41s`.
- `current_compute_dev_ms`: p50 around `1.14ms`, max around `9.43s`.
- Bad direction: still has layer0 cold-start outliers.

### Removed `prefetch_stream.wait_stream(current_stream)`

Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260527221632_elastic.txt`

Observed:

- `rollout_output_time_s = 151.173856`, worse.
- `submit_prefetch_wait_stream_us` became near zero, but speed did not improve.
- `current_compute_dev_ms` max still around `10.76s`.
- Bad direction: removing stream wait does not solve the real bottleneck and can make layer0 scheduling worse.
- Rule: keep `prefetch_stream.wait_stream(current_stream)` unless a new test proves otherwise.

### Mode4 MoE-only warmup

Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260527224437_elastic.txt`

Observed:

- `rollout_output_time_s = 155.624358`, worse.
- New `Mode4 remote-NPU MoE-only warmup` cost about `10.86s - 11.36s` per active rank.
- Shrink total grew to about `30.3s` because warmup added about `11.2s`, and rebuild also rose to about `17.6s` in that run.
- It did not eliminate all layer0 outliers.
- Bad direction: broad MoE MLP warmup after shrink is too expensive for the current target.
- Rule: `VLLM_ASCEND_MODE4_POST_SHRINK_MOE_WARMUP` should default to `0`.

## Current Intended Direction

Prefer narrow, targeted warmup/priming instead of synthetic compute warmup:

- Keep remote payload prepacking in the cache service.
- Keep remote payload warmup if it is cheap enough; recent warmup was about `130ms`.
- Prime the actual next-block runtime slot after shrink:
  - cache service starts;
  - active rank synchronously fills block8 slot using the same mode4 remote-NPU fetch path;
  - live layer0 should not be the first point that fetches block8.
- This should avoid the first-block remote fetch/compute outlier without adding an 11s MLP warmup.

Current new log marker to check:

```text
Mode4 next-block slot primed: rank=... block_start=8 block_layers=8 total_ms=...
```

## Known Pitfalls

Do not repeat these unless intentionally re-testing:

- Do not enable generic mode4 MoE warmup by default; it was too expensive.
- Do not remove `prefetch_stream.wait_stream(current_stream)` as a blanket optimization; measured result was worse.
- Do not use all2all warmup for mode=4; shrink-side decode uses MC2 and all2all warmup is not representative.
- Do not judge only from process exit code. A run can exit cleanly while being too slow.
- Do not conflate local NPU copy and remote NPU P2P copy. Local copy is usually ~1-2ms for the 8-layer block; remote P2P is the main variable.
- Do not accept logs that say `source_from_cpu > 0` for mode=4. That means the implementation is no longer validating remote-NPU double buffering.
- Do not proceed to 3-step validation if the first threshold step is obviously slow.

## Acceptance Checklist Before 3-Step Validation

For a one-step threshold run:

- `exit_code=0`.
- `source_from_cpu=0` in mode4 timing.
- `source_from_local_npu=8` and `source_from_remote_npu=8` after `16 -> 8`.
- `rollout_output_time_s` is close to the target range, not ~140-155s.
- No multi-second layer0 `current_compute_dev_ms` outlier.
- No multi-second first block `prefetch_remote_npu_dev_ms` outlier.
- Shrink `total_ms` should not regress to ~30s.
- If `Mode4 next-block slot primed` is enabled, its cost must be substantially less than the removed 11s MoE warmup and should reduce live layer0 outliers.

## Current Files To Inspect First

- `vllm_ascend/ops/fused_moe.py`
  - `Mode3DoubleBufferManager`
  - `_populate_mode4_block_slot`
  - `prepare_mode4_block_slot`
  - `prefetch_next_layer`
  - mode4 timing log fields
- `vllm_ascend/worker/worker_v1.py`
  - `_mode4_fetch_remote_experts_to_slot`
  - `_mode4_remote_cache_service_loop`
  - `_mode4_build_prepacked_remote_payloads`
  - `_mode4_warmup_remote_payload_fetch`
  - `_mode4_prime_next_block_runtime_slot`
  - `_warmup_post_shrink_moe_dispatch`
- `vllm_ascend/worker/model_runner_v1.py`
  - only use mode4 MoE warmup helpers if explicitly enabled for diagnosis.
- `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`
  - mode4 env defaults.

## Current Hypothesis

The primary remaining cost is not steady-state remote-NPU bandwidth; steady-state remote prefetch p50 can be single-digit milliseconds. The bad runs are dominated by cold-start/outlier behavior at the first live block after shrink and by shrink/rebuild overhead. The next useful optimization should reduce first-block outliers without introducing broad synthetic warmup cost.

### Next-block slot prime after cheap remote payload warmup

Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260527230608_elastic.txt`

Observed:

- `rollout_output_time_s = 143.138064`, still much slower than the `~90s` target.
- Mode4 source semantics are correct:
  - `source_from_cpu = 0`.
  - `source_from_local_npu = 8`.
  - `source_from_remote_npu = 8`.
- Cheap remote payload warmup stayed cheap: about `125ms - 131ms`.
- `Mode4 next-block slot primed` cost only about `29ms - 35ms`, so this is not a bad direction by itself.
- Shrink total remained about `9.1s - 9.5s`; no broad warmup regression.
- Remaining bottleneck is still remote-NPU submit/P2P outliers:
  - `prefetch_remote_npu_dev_ms`: p50 `7.203ms`, p90 `1072.179ms`, p95 `1130.106ms`, max `1163.794ms`.
  - `submit_remote_npu_us`: p50 `2184.55us`, p90 `1067234.4us`, max `1158771.2us`.
  - `current_compute_dev_ms`: p50 `0.957ms`, max `3403.472ms`.
- Interpretation: the steady state is fast, but a small number of remote submit/P2P spikes still dominate wall time. The next optimization should attack remote submit/P2P outliers directly, not add broad compute warmup.

Rule update:

- Keep next-block prime unless a later run proves it hurts; its cost is tiny.
- Do not expect next-block prime alone to reach the target.
- Next direction should reduce remote P2P submit frequency or make remote payload transfer more deterministic, for example by coalescing larger contiguous remote payloads, reusing request metadata, or avoiding repeated per-block rendezvous on the hot path.

## 2026-05-29 distribution fix attempt: avoid double log2phy remap

Observation: full mode4 run `wjeagerqwen30b-a3b-with_draft_breakdown_20260529022302_elastic.txt` completed but `global_token_num` distribution drifted from `largedataset-baseline.txt`: step1 sum 3,395,700 vs baseline 3,286,428; >=16400 count 57 vs 19. Source path still showed `source_from_cpu=0` and shrink only reached 16->8, so the issue is semantic path drift, not CPU fallback or preemption.

Fix: in `_execute_single_wave`, once logical topk ids are remapped by `log2phy`, pass `log2phy=None` into downstream `moe_comm_method.fused_experts`. MC2 token_dispatch also applies `log2phy`; passing the same map after local remap can remap expert ids twice and perturb tail EOS behavior.

Validation plan: first run one real full batch with `TRAINER_TOTAL_EPOCHS=1` and `VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=`. Compare `global_token_num` against `largedataset-baseline.txt`; target is close to baseline `>=16400 ~= 19-20`, not the bad mode4 value 57.

## 2026-06-01 floor-control and logging cleanup

Findings:

- `VLLM_ASCEND_MODE4_ASYNC_PREFETCH_MIN_ACTIVE_RANKS` was a workaround that made stage=2/1 fall back to synchronous current-layer fetch. That no longer matches the intended mode=4 semantics, so the low-floor async-prefetch deferral path was removed.
- Mode=3 and mode=4 timing logs now use explicit source names: `source_from_local_npu`, `source_from_remote_npu`, and `source_from_cpu`.
- Verbose one-time logs (`prefetch scheduled`, `slot binding`) are now controlled by `VLLM_ASCEND_MODE3_VERBOSE_TRANSFER_LOG`; normal timing runs should keep it off and rely on `ModeX timing ...` records.
- The script no longer silently forces mode=4/5 configured floor to 8 by default. For mode=4, floor controls double-buffer slot capacity (16/32/64/128 rows), not full mode=1 expert redundancy. If a run intentionally wants force-floor behavior, set `VLLM_ASCEND_MODE4_STABILITY_FORCE_FLOOR` explicitly.

Useful parser:

```bash
python internal/analyze_mode_prefetch_timing.py LOG.txt
```

The parser drops the first row for each `(mode, stage, layer, rank)` by default to avoid cold-start outliers, then reports p50/p90/mean for communication, submit, transfer, and compute.
