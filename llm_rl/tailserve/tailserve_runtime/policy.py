"""CPU-only admission and feedback policy. No claim of a hard slowdown bound."""
from dataclasses import dataclass
import math
import os
from statistics import median


@dataclass(frozen=True)
class Config:
    # Match the RL engine's max_num_batched_tokens on each DP rank. The
    # aggregate budget must not divide that capacity down as more ranks idle.
    global_tokens: int = 16 * 17408
    per_rank_tokens: int = 17408
    max_aux_ranks: int = 16
    warmup_rounds: int = 8
    slowdown_target: float = .05
    violation_rounds: int = 3
    max_prompt_tokens: int = 1024
    max_new_tokens: int = 16384
    kv_reserve_blocks: int = 32
    aux_interval_rounds: int = 1
    interference_feedback: int = 0

    def __post_init__(self):
        for key in ['global_tokens', 'per_rank_tokens', 'max_aux_ranks', 'warmup_rounds',
                    'violation_rounds', 'max_prompt_tokens', 'max_new_tokens', 'aux_interval_rounds']:
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if self.kv_reserve_blocks < 0 or not math.isfinite(self.slowdown_target) or not 0 < self.slowdown_target < 1:
            raise ValueError('invalid headroom/slowdown target')
        if type(self.interference_feedback) is not int or self.interference_feedback not in (0, 1):
            raise ValueError('interference_feedback must be 0 or 1')
        if not self.interference_feedback and self.aux_interval_rounds != 1:
            raise ValueError('continuous execution requires aux_interval_rounds=1; pacing requires explicit interference_feedback=1')

    @classmethod
    def from_env(cls):
        values = {}
        for key, field in cls.__dataclass_fields__.items():
            raw = os.getenv('TAILSERVE_' + key.upper())
            if raw is not None:
                values[key] = float(raw) if key == 'slowdown_target' else int(raw)
        return cls(**values)


@dataclass(frozen=True)
class Observation:
    primary: int
    auxiliary: int
    queued: int
    elapsed: float = 0.


@dataclass(frozen=True)
class Decision:
    stop: bool
    ranks: tuple[int, ...]
    tokens_per_rank: int
    reason: str


class Policy:
    def __init__(self, config):
        self.config = config
        self.signature = None
        self.clean = []
        self.baseline = None
        self.last_had_aux = False
        self.violations = 0
        self.blocked = False
        self.round = 0

    def decide(self, observations):
        self.round += 1
        counts = tuple(o.primary for o in observations)
        if sum(counts) == 0:
            return Decision(True, (), 0, 'primary_complete')
        if not self.config.interference_feedback:
            # A drained rank becomes a real inference producer immediately.
            # Neither calibration, timing spikes nor primary-count changes
            # may silently turn its inference rounds back into dummy rounds.
            return self._select_eligible(observations, continuous=True)
        if counts != self.signature:
            self.signature = counts
            self.clean = []
            self.baseline = None
            self.violations = 0
            self.blocked = False
        elapsed = max(o.elapsed for o in observations)
        if self.last_had_aux and self.baseline:
            # A single bounded auxiliary quantum is amortized over its pacing
            # interval. This remains feedback, not an end-to-end guarantee.
            if elapsed > self.baseline * (1 + self.config.slowdown_target * self.config.aux_interval_rounds):
                self.violations += 1
            else:
                self.violations = 0
        if self.violations >= self.config.violation_rounds:
            # Do not oscillate and repeatedly cold-prefill at the same RL load.
            self.blocked = True
        if not self.last_had_aux and elapsed > 0:
            # Paused auxiliary requests keep KV, but run only the original EP
            # dummy participation. They are clean *compute* timing samples.
            self.clean.append(elapsed)
            if len(self.clean) >= self.config.warmup_rounds:
                self.baseline = median(self.clean[-self.config.warmup_rounds:])
        if self.blocked or self.baseline is None:
            self.last_had_aux = False
            return Decision(False, (), 0, 'feedback_backoff' if self.blocked else 'calibrating')
        if self.round % self.config.aux_interval_rounds:
            self.last_had_aux = False
            return Decision(False, (), 0, 'paced_auxiliary_pause')
        return self._select_eligible(observations)

    def _select_eligible(self, observations, continuous=False):
        eligible = [i for i, o in enumerate(observations) if o.primary == 0 and (o.auxiliary or o.queued)]
        # Preserve active leases rather than cancelling them for fairness rotation.
        eligible.sort(key=lambda i: (-observations[i].auxiliary, i))
        count = min(len(eligible), self.config.max_aux_ranks, self.config.global_tokens)
        ranks = tuple(eligible[:count])
        tokens = min(self.config.per_rank_tokens, self.config.global_tokens // count) if count else 0
        self.last_had_aux = bool(ranks)
        reason = 'continuous_auxiliary' if continuous else 'admit'
        return Decision(False, ranks, tokens, reason if ranks else 'no_eligible_local_tail')
