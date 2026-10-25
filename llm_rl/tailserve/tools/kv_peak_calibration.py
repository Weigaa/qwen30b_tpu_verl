"""Calibrate synchronized KV-peak predictions from completed epoch pairs."""
import math
from fractions import Fraction
from functools import lru_cache


def live_kv_peak(response_lengths, prompt_lengths, *, block_size_tokens):
    if block_size_tokens <= 0 or len(response_lengths) != len(prompt_lengths):
        raise ValueError('invalid KV block size or length counts')
    if any(not math.isfinite(x) or x <= 0 for x in response_lengths):
        raise ValueError('response lengths must be finite and positive')
    if any(not math.isfinite(x) or x < 0 for x in prompt_lengths):
        raise ValueError('prompt lengths must be finite and nonnegative')
    return float(max((sum(
        math.ceil((p + t) / block_size_tokens) * block_size_tokens
        for length, p in zip(response_lengths, prompt_lengths, strict=True)
        if length >= t
    ) for t in sorted(set(response_lengths))), default=0))


def calibrate_peak_factor(histories, groups_by_epoch, *, initial=1.3,
                          minimum=1.2, ratio_guard=1.02, window=3,
                          quantum=0.01, previous_factor=None,
                          max_decrease=0.05, block_size_tokens=128):
    """Each history maps stable source indices to (response lengths, prompt len).

    For completed epoch j's actual assignment, reconstruct its uninflated
    prediction using only j-1 lengths, and compare with j's realized lengths.
    No candidate/current epoch outcomes enter this function. Ratios are of
    reconstructed live-KV peaks, not allocator telemetry or release-area proxies.
    """
    if (not histories or len(histories) != len(groups_by_epoch) or window < 1
            or any(not math.isfinite(x) for x in
                   (initial, minimum, ratio_guard, quantum, max_decrease))
            or initial < minimum or minimum < 1 or ratio_guard < 1
            or quantum <= 0 or max_decrease < 0):
        raise ValueError('invalid peak calibration settings')
    records = []
    start = max(1, len(histories) - window)
    for j in range(start, len(histories)):
        groups = groups_by_epoch[j]
        if not groups:
            raise ValueError(f'missing completed epoch {j} rank assignments')
        for group in groups:
            ids = group['source_indices']
            if not ids or len(set(ids)) != len(ids):
                raise ValueError('empty or duplicate-source rank assignment')
            predicted, actual, prompts = [], [], []
            for idx in ids:
                old, old_p = histories[j-1][idx]
                new, new_p = histories[j][idx]
                if not old or len(old) != len(new) or old_p != new_p or old_p <= 0:
                    raise ValueError('history response count or prompt length mismatch')
                predicted.extend(old)
                actual.extend(new)
                prompts.extend([old_p] * len(old))
            pred = live_kv_peak(predicted, prompts, block_size_tokens=block_size_tokens)
            act = live_kv_peak(actual, prompts, block_size_tokens=block_size_tokens)
            records.append(dict(history_index=j, step=group['step'], rank=group['rank'],
                                source_indices=ids, predicted_peak_tokens=pred,
                                actual_peak_tokens=act, ratio=act/pred))
    if not records:
        target, mode = initial, 'cold_start'
    else:
        target = max(minimum, ratio_guard * max(r['ratio'] for r in records))
        previous = initial if previous_factor is None else previous_factor
        if not math.isfinite(previous) or previous < 1:
            raise ValueError('invalid previous peak factor')
        target = max(target, previous - max_decrease)
        mode = 'completed_epoch_peak_ratios'
    factor = math.ceil((target - 1e-12) / quantum) * quantum
    return dict(schema_version=1, mode=mode, factor=round(factor, 12),
                initial=initial, minimum=minimum, ratio_guard=ratio_guard,
                window=window, quantum=quantum, max_decrease=max_decrease,
                previous_factor=previous_factor, sample_count=len(records),
                max_observed_ratio=max((r['ratio'] for r in records), default=None),
                denominator='uninflated synchronized prompt-plus-response block-rounded peak',
                numerator='reconstructed realized peak for the same prompt-to-rank assignment',
                records=records)


def calibrate_prompt_factors(histories, public, *, previous_factors=None,
                             prior_strength=3.0, minimum=1.10,
                             block_size_tokens=128):
    """One observation per completed epoch transition, not per response.

    Sparse local estimates shrink toward the public factor. A locally observed
    underprediction is never averaged below its guarded ratio. Downward moves
    are limited using the factor actually used in the preceding epoch.
    """
    if (not histories or not math.isfinite(prior_strength) or prior_strength <= 0
            or not math.isfinite(minimum) or minimum < 1
            or minimum > public['initial']):
        raise ValueError('invalid prompt calibration settings')
    previous = {int(k): v for k, v in (previous_factors or {}).items()}
    result = {}
    start = max(1, len(histories) - public['window'])
    for idx in sorted(histories[-1]):
        ratios = []
        for j in range(start, len(histories)):
            if idx not in histories[j-1] or idx not in histories[j]:
                continue
            old, p = histories[j-1][idx]
            new, q = histories[j][idx]
            if p != q or p <= 0 or not old or len(old) != len(new):
                raise ValueError('prompt calibration history mismatch')
            predicted = live_kv_peak(old, [p]*len(old), block_size_tokens=block_size_tokens)
            actual = live_kv_peak(new, [p]*len(new), block_size_tokens=block_size_tokens)
            ratios.append(dict(history_index=j, predicted_peak_tokens=predicted,
                               actual_peak_tokens=actual, ratio=actual/predicted))
        count = len(ratios)
        weight = count / (count + prior_strength)
        local = max(minimum, public['ratio_guard'] * max(
            (r['ratio'] for r in ratios), default=1.0))
        previous_public = public.get('previous_factor')
        prior_factor = previous.get(idx, public['initial'] if previous_public is None else previous_public)
        if not math.isfinite(prior_factor) or prior_factor < 1:
            raise ValueError('invalid preceding prompt factor')
        target = (max(local, (1-weight)*public['factor'] + weight*local,
                      prior_factor-public['max_decrease']) if count else public['factor'])
        factor = math.ceil((target-1e-12)/public['quantum']) * public['quantum']
        result[str(idx)] = dict(factor=round(factor,12), sample_count=count,
                               local_weight=weight, local_target=local,
                               previous_factor=prior_factor, observations=ratios)
    return dict(prompt_factors={idx:r['factor'] for idx,r in result.items()},
                prompt_calibration=result, prompt_prior_strength=prior_strength,
                prompt_minimum=minimum)


@lru_cache(maxsize=32768)
def _weighted_peak(prompts, factors, block_size_tokens, max_response_len):
    """Weighted sum of live curves at the same decode position, then max.

    Rational weights avoid a spurious extra KV block at floating-point ties.
    Each prompt is (response_lengths_tuple, prompt_token_length).
    """
    if len(prompts) != len(factors) or not prompts or block_size_tokens <= 0:
        raise ValueError('invalid weighted KV curve inputs')
    weights = [Fraction(str(a)) for a in factors]
    if any(a < 1 for a in weights):
        raise ValueError('KV factors must be at least one')
    denominator = math.lcm(*(w.denominator for w in weights))
    integer_weights = [w.numerator * (denominator//w.denominator) for w in weights]
    sequences = []
    for (lengths, p), w in zip(prompts, integer_weights, strict=True):
        if p <= 0 or not lengths or any(not math.isfinite(v) or v <= 0 for v in lengths):
            raise ValueError('invalid historical prompt or response length')
        sequences.extend((min(v,max_response_len),p,w) for v in lengths)
    peak = max(sum(w*math.ceil((p+t)/block_size_tokens)*block_size_tokens
                   for length,p,w in sequences if length >= t)
               for t in sorted({length for length,_,_ in sequences}))
    return Fraction(peak,denominator)


class PromptAdaptivePeakModel:
    """Per-candidate residual check using completed histories only.

    Unseen rank pairings are checked counterfactually from each prompt's saved
    lengths. They are not presented as measured simultaneous allocator traces.
    """
    def __init__(self, histories, calibration, *, block_size_tokens=128):
        self.histories = tuple({int(i):(tuple(lengths),p) for i,(lengths,p) in h.items()}
                               for h in histories)
        self.factors = {int(k):v for k,v in calibration['prompt_factors'].items()}
        self.window = calibration['window']
        self.guard = calibration['ratio_guard']
        self.quantum = calibration['quantum']
        self.block = block_size_tokens

    def evaluate(self, ids, max_response_len):
        return self._evaluate(tuple(sorted(ids)),max_response_len)

    def single_lower_bound(self, idx, max_response_len):
        raw = _weighted_peak((self.histories[-1][idx],), (self.factors[idx],),
                              self.block,max_response_len)
        return math.ceil(raw/self.block)*self.block

    @lru_cache(maxsize=16384)
    def _evaluate(self, ids, max_response_len):
        if (not ids or len(set(ids)) != len(ids)
                or not math.isfinite(max_response_len) or max_response_len <= 0):
            raise ValueError('invalid candidate prompt identities or response cap')
        factors = tuple(self.factors[i] for i in ids)
        weighted = _weighted_peak(tuple(self.histories[-1][i] for i in ids),
                                   factors,self.block,max_response_len)
        records = []
        for j in range(max(1,len(self.histories)-self.window),len(self.histories)):
            if any(i not in self.histories[j-1] or i not in self.histories[j] for i in ids):
                continue
            predicted = _weighted_peak(tuple(self.histories[j-1][i] for i in ids),
                                       factors,self.block,max_response_len)
            actual = _weighted_peak(tuple(self.histories[j][i] for i in ids),
                                    (1,)*len(ids),self.block,max_response_len)
            records.append(dict(history_index=j, weighted_prediction=float(predicted),
                                reconstructed_actual=float(actual), ratio=float(actual/predicted)))
        target = max(1.0,self.guard*max((r['ratio'] for r in records),default=0.0))
        beta = round(math.ceil((target-1e-12)/self.quantum)*self.quantum,12)
        admission = math.ceil(weighted*Fraction(str(beta))/self.block)*self.block
        return dict(source_indices=list(ids), prompt_factors=list(factors),
                    weighted_peak_tokens=float(weighted), pair_residual_factor=beta,
                    admission_peak_tokens=admission, calibration_samples=len(records),
                    pair_history_checks=records)
