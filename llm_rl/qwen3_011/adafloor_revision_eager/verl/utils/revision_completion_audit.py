"""Observe completion semantics without changing tokens, masks or sampling."""


def completion_record(token_ids, finish_reason, stop_reason, prompt_tokens, configured_cap, max_model_len):
    if configured_cap <= 0 or prompt_tokens < 0 or max_model_len <= prompt_tokens:
        raise ValueError("invalid effective completion budget")
    cap = min(int(configured_cap), int(max_model_len) - int(prompt_tokens))
    count = len(token_ids)
    if count > cap or finish_reason not in ("stop", "length"):
        raise ValueError("aborted, incomplete or over-budget completion; retain failed evidence")
    if finish_reason == "length" and count != cap:
        raise ValueError("length finish does not match an audited cap/context limit")
    return {"rollout_generated_tokens": count, "rollout_finish_reason": finish_reason,
            "rollout_stop_reason": stop_reason, "rollout_sampling_cap": cap,
            "rollout_configured_cap": int(configured_cap),
            "rollout_censored": finish_reason == "length"}
