# Mode1 vs Mode4 High-KV Pressure Compare

This repo keeps the comparison harness separate from the existing mode-specific perf scripts.

## What It Tests

For each floor in `2,4,8`, the harness uses the existing repeated-halving response cap hook so one target rank requests enough response tokens to approach the observed mode=4 KV-cache limit, while the other ranks stay at a low cap.

By default:

- `COMPARE_TARGET_RANK=15`
- target rank cap is computed from the observed mode=4 KV limit and `--scale`.
- all non-target buckets stay at one low cap (`COMPARE_LOW_CAP`, default `256`), so the cap list is always `256,256,256,256,<high>`.

The existing `VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS` path maps 16 ranks into buckets `[0-7], [8-11], [12-13], [14], [15]`. The harness puts the high cap only in the last bucket, so rank 15 is the single high-pressure rank without adding a new rollout code path.

The temporary launcher copy also sets `ROLLOUT_IGNORE_EOS=True`, so the cap actually turns into KV pressure instead of being hidden by early EOS.

Normal rollout runs keep their existing behavior. The harness only copies and patches the launcher under its run directory.

The launcher still executes from the repository root. Its `HOME=$(pwd)` must
remain the code root because it is also used to locate `verl/trainer/config`.
The harness isolates outputs with `LOG_HOME=<case_dir>` and
`RECORD_DIR=<case_dir>/record` instead of changing `HOME`.

A comparison is valid only when:

- mode=4 completes with `preempt_count=0`.
- mode=1 under the same pressure has `preempt_count>0`.
- `record target mean/max` from `record/1.jsonl` is close to the high cap. The trainer `response_length/max` metric can stop at the first `151643` marker and under-report forced-tail tokens when `ignore_eos=True`.

## Run

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_true_mode5_a3cfdc2
./run_mode1_mode4_pressure_compare.sh --floors 2,4,8 --scale 0.96
```

Recommended calibration flow:

```bash
# 1. Run mode4 first. This checks whether the cap is close to the mode4 limit
#    while still avoiding preemption.
./run_mode1_mode4_pressure_compare.sh --floors 2,4,8 --modes 4 --scale 0.96

# 2. Reuse the same run directory and run mode1 under the same cap list.
COMPARE_RUN_DIR=/path/to/mode1_mode4_pressure_compare_runs/<timestamp> \
  ./run_mode1_mode4_pressure_compare.sh --floors 2,4,8 --modes 1 --scale 0.96
```

Dry-run the generated caps and directories without launching training:

```bash
./run_mode1_mode4_pressure_compare.sh --floors 2,4,8 --scale 0.96 --dry-run
```

Regenerate summaries for an existing run directory, including interrupted runs:

```bash
COMPARE_RUN_DIR=/path/to/mode1_mode4_pressure_compare_runs/<timestamp> \
  ./run_mode1_mode4_pressure_compare.sh --analyze-only
```

Override one floor's cap if mode=4 preempts or does not get close enough:

```bash
COMPARE_RESPONSE_CAP_FLOOR2=10800 ./run_mode1_mode4_pressure_compare.sh --floors 2
```

The harness resumes completed cases by default. Use `--force` to rerun an existing
case, or `--no-resume` to disable skip logic.

HCCL and master ports are assigned deterministically from `(floor, mode)`, so a
case uses the same ports whether it is run alone or as part of a larger sweep.

## Outputs

Each run writes to:

```text
mode1_mode4_pressure_compare_runs/<timestamp>/
```

Important files:

- `summary.md`: human-readable comparison result.
- `summary.csv`: parseable metrics.
- `suggested_overrides.env`: parser-generated cap suggestions for the next calibration run.
- `run_config.env`: global sweep settings, including the bucket topology used to isolate rank 15.
- `floor{N}_mode{M}/launcher.log`: raw stdout/stderr for one case.
- `floor{N}_mode{M}/case.env`: exact cap list and metadata for one case.

The generated `case.env` and summaries include:

- `estimated_high_rank_tokens = max_num_seqs * (prompt_length + high_response_cap)`.
- `estimated_pressure_ratio_to_mode4_kv = estimated_high_rank_tokens / target_mode4_kv_tokens`.

With the default `--scale 0.96`, the intended pressure ratio is about `0.96`.
This is the first sanity check that the cap is actually close to the mode=4 KV
upper bound before looking at preemption behavior.

The default caps are also compared against observed high-KV mode=1 capacity.
This table uses the conservative configured prompt length. After a real run,
prefer the parser's `record/1.jsonl`-based `suggested_overrides.env` because it
uses the actual target-rank prompt lengths:

| floor | mode4 KV target | observed mode1 KV | default high cap | estimated rank15 tokens | vs mode4 | mode1 margin |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 397,824 | 174,208 | 10,910 | 381,888 | 0.960 | +207,680 |
| 4 | 414,848 | 321,408 | 11,421 | 398,240 | 0.960 | +76,832 |
| 8 | 431,872 | 395,520 | 11,932 | 414,592 | 0.960 | +19,072 |

So the default pressure point is designed to be below mode=4 capacity and above
mode=1 capacity. Floor 8 has the narrowest margin; if mode1 does not preempt
there, increase `COMPARE_RESPONSE_CAP_FLOOR8` slightly while verifying mode4
still has `preempt_count=0`.

`summary.md` includes a `Calibration Advice` table. Use it as the next-action
guide:

- If mode4 preempts or fails, lower that floor's high cap.
- If mode4 is valid but actual `record mode4 pressure` is below the target
  scale, increase that floor's high cap.
- If mode4 is valid and mode1 does not preempt, increase the high cap until
  mode1 preempts while mode4 remains no-preempt.
- Once mode4 is valid and mode1 preempts, compare rollout time for that floor.

The top `Final Verdict` table is the only table intended for the final
answer. It marks a floor valid only when mode4 is no-preempt, mode1 preempts,
both complete, and both hit the requested high-pressure response cap.

For faster iteration, source the generated suggestions before rerunning:

```bash
source mode1_mode4_pressure_compare_runs/<timestamp>/suggested_overrides.env
./run_mode1_mode4_pressure_compare.sh --floors 2,4,8 --modes 4 --force
```

## Result Validity

A floor is valid only when:

- mode4 completed with `preempt_count=0`.
- mode1 completed or failed in a parseable way with `preempt_count>0`.
- `record target mean/max` is close to the high cap shown in `summary.csv`.

`traceback_count` can be nonzero in otherwise successful runs because the TBE
forkserver may emit shutdown tracebacks after the main process exits. Use
`fatal_error_count`, `exit_code`, and `rollout_output_time_s` for the actual
pass/fail judgment.
