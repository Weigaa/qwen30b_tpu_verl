"""Backend-independent execution loop. Only original primary outputs are returned."""
import time
from .policy import Observation, Policy


def run(adapter, queue, config, version):
    policy = Policy(config)
    primary_outputs = []
    elapsed = 0.
    active = {}
    rounds = 0
    local_drained = False
    previous_control = None
    adapter.check_collective_contract(config, version)
    queue.emit('generation_open', policy_version=version, primary_requests=adapter.primary_count())
    try:
        while True:
            rounds += 1
            error = None
            try:
                queue.poll()
                candidate = queue.candidate(version)
            except Exception as exc:
                candidate, error = None, str(exc)
            # Every rank participates, including ranks with no primary request.
            adapter.consensus_error(error)
            observations = adapter.gather(Observation(adapter.primary_count(), len(active), int(candidate is not None), elapsed))
            if not local_drained and observations[adapter.rank].primary == 0:
                local_drained = True
                queue.emit('local_primary_drained', policy_version=version, round=rounds)
            decision = policy.decide(observations)
            if decision.stop:
                boundary = time.perf_counter()
                queue.emit('global_primary_complete', policy_version=version, round=rounds,
                           auxiliary_in_flight=len(active))
                adapter.cancel_auxiliary(active, queue, version, 'primary_complete')
                active.clear()
                adapter.verify_drained_collectively()
                queue.emit('generation_closed', policy_version=version, rounds=rounds,
                           primary_outputs=len(primary_outputs),
                           primary_responses=sum(len(getattr(o, 'outputs', ())) for o in primary_outputs),
                           primary_tokens=sum(len(a.token_ids) for o in primary_outputs
                                              for a in getattr(o, 'outputs', ())),
                           stop_cleanup_s=time.perf_counter()-boundary,
                           auxiliary_drained=True)
                break
            error = None
            try:
                control = (decision.reason, decision.ranks, decision.tokens_per_rank)
                if control != previous_control:
                    # One global decision record is enough. Fsync on all EP
                    # ranks would itself add avoidable per-round interference.
                    if adapter.rank == 0:
                        queue.emit('control', policy_version=version, reason=decision.reason,
                                   permitted_ranks=list(decision.ranks), tokens_per_rank=decision.tokens_per_rank,
                                   clean_round_baseline_s=policy.baseline,
                                   slowdown_target=config.slowdown_target,
                                   aux_interval_rounds=config.aux_interval_rounds,
                                   interference_feedback=bool(config.interference_feedback))
                    previous_control = control
                permitted = adapter.rank in decision.ranks
                if active and adapter.primary_count():
                    raise RuntimeError('local RL work appeared after auxiliary admission')
                if permitted and not active and candidate and adapter.validate_candidate(candidate):
                    job = queue.take(version)
                    queue.emit('rejected_input', job_id=job['id'], policy_version=version,
                               error=adapter.validate_candidate(candidate))
                elif permitted and not active and candidate and adapter.can_admit(candidate, config):
                    job = queue.take(version)
                    request_id = adapter.admit(job, version)
                    active[request_id] = job
                    queue.emit('admitted', job_id=job['id'], request_id=request_id, policy_version=version)
                # Zero means pause without scheduling, not cancel/re-prefill.
                # The adapter executes normal EP dummy participation instead.
                adapter.set_auxiliary_token_budget((decision.tokens_per_rank if permitted else 0) if active else None)
            except Exception as exc:
                error = str(exc)
            adapter.consensus_error(error)
            start = time.perf_counter()
            outputs = adapter.step()
            elapsed = time.perf_counter() - start
            for output in outputs:
                if adapter.is_primary(output):
                    if output.finished:
                        primary_outputs.append(output)
                elif output.request_id in active:
                    adapter.note_auxiliary_output(output)
                    if output.finished:
                        job = active.pop(output.request_id)
                        queue.emit('completed', job_id=job['id'], policy_version=version,
                                   **adapter.serialize_output(output))
                else:
                    raise RuntimeError('unclassified output must not enter RL data')
    finally:
        # An exception is fatal to this generation, not permission to update weights.
        try:
            adapter.cancel_auxiliary(active, queue, version, 'generation_cleanup')
        finally:
            adapter.set_auxiliary_token_budget(None)
            adapter.restore_primary_rng()
    return sorted(primary_outputs, key=lambda output: int(output.request_id))
