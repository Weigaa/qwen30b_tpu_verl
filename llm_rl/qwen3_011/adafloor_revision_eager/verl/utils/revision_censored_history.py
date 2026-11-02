"""Conservative prompt-tail updates for right-censored generation lengths.

This does not reconstruct unseen continuations or prove unbiased GRPO learning.
It only prevents capped observations from spuriously lowering the tail estimate.
Callers must supply normalized GENERATED lengths and explicit finish reasons.
"""
import math


def update_tail(previous, generated_lengths, censored, caps, alpha=0.3):
    if (not generated_lengths or len(generated_lengths) != len(censored)
            or len(caps) != len(censored) or not 0 <= alpha <= 1):
        raise ValueError("invalid aligned censor observations")
    if any(not math.isfinite(float(x)) or x < 0 for x in generated_lengths):
        raise ValueError("invalid generated token length")
    if previous is not None and (not math.isfinite(previous) or previous < 0):
        raise ValueError("invalid previous estimate")
    for length, flag, cap in zip(generated_lengths, censored, caps):
        if not isinstance(flag, bool) or cap <= 0 or length > cap or flag and length != cap:
            raise ValueError("censor flags must describe normalized lengths at their actual sampling cap")
    observed = max(generated_lengths)
    if any(censored):
        lower_bound = max(cap for cap, flag in zip(caps, censored) if flag)
        estimate = max(previous or 0, observed, lower_bound)
        kind = "right_censored_lower_bound_preserve_prior"
    else:
        lower_bound = observed
        estimate = observed if previous is None else alpha*previous+(1-alpha)*observed
        kind = "uncensored_ema"
    return {"tail_prediction": estimate, "update_kind": kind,
            "observed_max": observed, "tail_lower_bound": lower_bound,
            "censored_count": sum(censored), "observation_count": len(censored)}
