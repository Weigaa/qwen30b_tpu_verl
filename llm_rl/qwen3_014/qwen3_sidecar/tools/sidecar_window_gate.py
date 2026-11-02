#!/usr/bin/env python3
"""Causal, fail-closed sidecar window forecast; never reads current outputs."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

ANSI=re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')
FORWARD=re.compile(r'rollout_forward_begin dp_rank=(\d+) iteration=(\d+).*global_prefill=False graph_mode=FULL')
STAMP=re.compile(r'(?:\b(\d{4})-)?(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})(?:,(\d{3}))?')


def calibrate_exit_from_prior_steps(
        steps, current_step, predicted_exit, length_dir,
        hard_cap=16384, history_window=3, quantile=0.95, expected_responses=512):
    if (not 0 <= quantile <= 1 or history_window < 1 or hard_cap < 1
            or not math.isfinite(predicted_exit) or predicted_exit <= 0):
        raise ValueError('Invalid causal calibration configuration')
    sources=[];ratios=[]
    by_step={int(step['step']):step for step in steps}
    # Strict inequality is intentional: never open current/future lengths.
    for index in range(max(1,current_step-history_window),current_step):
        path=Path(length_dir)/f'length_{index}.txt'
        data=path.read_bytes()
        lengths=[int(value) for value in data.decode().split()]
        prior=by_step[index]
        prior_cap=int(prior.get('tail_guard_response_cap',16384))
        short_guard=bool(prior.get('tail_guard_enabled',False)) and prior_cap<16384
        # The main Qwen3 response-mask convention includes the first padded
        # EOS after a short TailGuard cap. The primary task is unchanged;
        # only the observed-history token accounting is corrected here.
        limit=prior_cap+int(short_guard)
        if len(lengths)!=expected_responses or any(not 0<value<=limit for value in lengths):
            raise ValueError(f'Incomplete/invalid prior completed step {index}')
        raw_max=max(lengths)
        excluded=sum(value==prior_cap+1 for value in lengths) if short_guard else 0
        if short_guard:lengths=[value-1 if value==prior_cap+1 else value for value in lengths]
        prediction=max(float(value) for value in by_step[index]['rank_loads'].values())
        if not math.isfinite(prediction) or prediction<=0:
            raise ValueError(f'Invalid historical prediction for prior step {index}')
        actual=max(lengths);ratio=actual/prediction;ratios.append(ratio)
        sources.append(dict(step=index,path=str(path),sha256=hashlib.sha256(data).hexdigest(),
                            responses=len(lengths),predicted_max=prediction,actual_max=actual,ratio=ratio))
        if short_guard:sources[-1].update(raw_mask_max=raw_max,prior_response_cap=prior_cap,
                                         synthetic_first_eos_excluded=excluded)
    ordered=sorted(ratios)
    if ordered:
        position=quantile*(len(ordered)-1)
        left=math.floor(position);right=math.ceil(position)
        correction=ordered[left]+(ordered[right]-ordered[left])*(position-left)
    else:
        correction=1.0
    correction=max(1.0,correction)
    calibrated=min(float(hard_cap),float(predicted_exit)*correction)
    return calibrated,dict(method='prior_completed_step_exit_ratio_q95_v1',
        history_window=history_window,quantile=quantile,ratio=correction,sources=sources,
        uncalibrated_exit_tokens=predicted_exit,calibrated_exit_tokens=calibrated,
        current_or_future_step_lengths_read=False,forecast_is_statistical_lower_bound=False)


def forecast(plan,text,active,now,minimum=80.0,lease_id=1,prior_lengths=None,strict_minimum=False):
    if not math.isfinite(minimum) or minimum<=0:
        raise ValueError('Minimum window must be finite and positive')
    result=dict(schema_version=1,lease_id=lease_id,decision_time=now,
                minimum_window_seconds=minimum,minimum_comparison='>' if strict_minimum else '>=',active_ranks=active,
                action='skip',reason='prediction_unavailable',predicted_window_seconds=None,
                forecast_is_statistical_lower_bound=False,
                source='frozen_epoch_plan_rank_exit_predictions_and_observed_pre_shrink_FULL_decode_logs',
                current_response_outputs_read=False)
    try:
        steps=plan if isinstance(plan,list) else plan['steps']
        cleaned=ANSI.sub('',text)
        starts=list(re.finditer(r'driver_generate_start step=(\d+)',cleaned))
        if not starts:
            raise ValueError('Require at least one executed generation step')
        # A full epoch contains several generation steps.  The watcher calls
        # the gate immediately after the current shrink quorum, so use the
        # latest start and only the decode prefix belonging to that step.
        current_step=int(starts[-1].group(1))
        result['step']=current_step
        step=next(row for row in steps if int(row['step'])==current_step)
        current_start=starts[-1].start()
        step_text=cleaned[current_start:]
        predicted_exit=max(float(step['rank_loads'][str(rank)]) for rank in active)
        if not math.isfinite(predicted_exit) or predicted_exit<=0:
            raise ValueError('Invalid frozen exit prediction')
        if prior_lengths is not None:
            predicted_exit,calibration=calibrate_exit_from_prior_steps(
                steps,current_step,predicted_exit,prior_lengths,
                hard_cap=int(step.get('tail_guard_response_cap',16384)))
            result.update(schema_version=2,calibration=calibration,
                source='epoch_plan_predictions_prior_completed_step_exit_calibration_and_pre_shrink_FULL_logs')
        # Topology capture is not decode. Ignore every record after first
        # shrink begins, including any graph recapture/warmup submissions.
        boundaries=[step_text.find(key) for key in ('Elastic EP rank exit start:',
                    'Elastic parallel shrink rpc done:', 'stage=before_shrink_to_')]
        boundaries=[value for value in boundaries if value>=0]
        if not boundaries:
            raise ValueError('No shrink boundary')
        prefix=step_text[:min(boundaries)]
        points={rank:[] for rank in active}
        year=datetime.fromtimestamp(now,timezone(timedelta(hours=8))).year
        for line in prefix.splitlines():
            match=FORWARD.search(line);stamp=STAMP.search(line)
            if not match or not stamp or int(match[1]) not in points:
                continue
            y,mo,day,hour,minute,second,ms=stamp.groups()
            moment=datetime(int(y or year),int(mo),int(day),int(hour),int(minute),int(second),
                            int(ms or 0)*1000,tzinfo=timezone(timedelta(hours=8))).timestamp()
            if moment>now:
                raise ValueError('Future timestamp in input')
            points[int(match[1])].append((int(match[2]),moment))
        rates=[];anchors=[]
        for rank,records in points.items():
            if not records:
                raise ValueError(f'No FULL observation for survivor {rank}')
            records=sorted(set(records))
            anchors.append(dict(rank=rank,iteration=records[-1][0],wall_time=records[-1][1]))
            for (a,ta),(b,tb) in zip(records,records[1:]):
                if b-a>=128 and tb>ta:
                    rates.append((tb-ta)/(b-a))
        if not rates:
            raise ValueError('Insufficient positive decode-rate observations')
        fastest_seconds_per_token=min(rates)
        progress=max(row['iteration'] for row in anchors)
        newest_observation=max(row['wall_time'] for row in anchors)
        stale_seconds=max(0,now-newest_observation)
        remaining_tokens=max(0,predicted_exit-progress)
        # Group rebuild/capture blocks decode between this observation and
        # the quorum. Do not extrapolate generation through that pause.
        estimate=remaining_tokens*fastest_seconds_per_token
        eligible=estimate>minimum if strict_minimum else estimate>=minimum
        result.update(action='start' if eligible else 'skip',
                      reason='predicted_window_sufficient' if eligible else 'predicted_window_too_short',
                      predicted_window_seconds=estimate,predicted_exit_tokens=predicted_exit,
                      observed_progress_iteration=progress,fastest_observed_seconds_per_token=fastest_seconds_per_token,
                      observation_age_seconds=stale_seconds,anchors=anchors,decode_rate_samples=len(rates))
    except (OSError,ValueError,KeyError,StopIteration,TypeError) as error:
        result['forecast_error']=str(error)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--active',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--minimum',type=float,default=80);parser.add_argument('--lease-id',type=int,required=True)
    parser.add_argument('--calibrate-prior-steps',action='store_true')
    parser.add_argument('--strict-minimum',action='store_true',help='Start only above, not equal to, the threshold')
    args=parser.parse_args();now=time.time()
    plan_path=args.log.parent.parent/'oracle/length_sorted_rank_plan.json'
    text=args.log.read_text(errors='replace')
    try:
        plan=json.loads(plan_path.read_text());plan_sha=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    except (OSError,ValueError):plan=[];plan_sha=None
    report=forecast(plan,text,[int(x) for x in args.active.split(',') if x.strip()],now,args.minimum,args.lease_id,
                    prior_lengths=(args.log.parent.parent/'rollout_length' if args.calibrate_prior_steps else None),
                    strict_minimum=args.strict_minimum)
    report.update(plan_path=str(plan_path),plan_sha256=plan_sha,
                  observed_log_snapshot_sha256=hashlib.sha256(text.encode()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(report['action'])


if __name__=='__main__':main()
