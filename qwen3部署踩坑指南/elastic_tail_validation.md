# Elastic Tail Validation

This repo now has a validation-only tail-shaping mode for rollout generation.

## Goal

Make `active_ranks` shrink in a more deterministic repeated-halving pattern so
we can verify:

- `16 -> 8`
- `8 -> 4`
- `4 -> 2`

and then confirm that the runtime has truly reached a single surviving rank,
even if `2 -> 1` is still blocked by the current EP floor.

## How It Works

`vllm_rollout_spmd.py` now reads:

- `VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS`

The value is a comma-separated list of token caps, for example:

```bash
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,1024,2048,4096
```

For a 16-rank rollout worker group, ranks are bucketed as:

- first 8 ranks: cap 256
- next 4 ranks: cap 512
- next 2 ranks: cap 1024
- next 1 rank: cap 2048
- final 1 rank: cap 4096

This does not change model correctness logic. It only shortens generation for
earlier buckets so the decode tail becomes easier to reproduce.

## Logs To Watch

Per-rank cap shaping:

```text
Elastic tail validation cap override: rank=... world_size=... bucket=... max_tokens=...
```

If the system reaches a true single-rank tail but cannot shrink further because
the current MC2 EP floor is still 2:

```text
Elastic single-rank tail reached but blocked by min EP floor: ...
```

## Expected Behavior

With the env var enabled, you should see:

1. Earlier ranks finish sooner.
2. `active_ranks` shrink more predictably.
3. A much better chance of observing `4 -> 2`.
4. If only one rank remains, an explicit log that single-rank tail was reached.

## Important Limitation

This mode helps validate `4 -> 2` today.

It does **not** by itself implement `2 -> 1`. The current runtime still uses a
minimum EP floor of 2, so single-rank completion is only logged, not executed
as a true no-EP fallback yet.
