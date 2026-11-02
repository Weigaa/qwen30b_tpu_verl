#!/usr/bin/env python3
"""Apply the paper planner using ONLY public historical lengths, never results."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent))
from build_mode1_length_sorted_e2e_plan import _solve_one_batch,_stats_from_lengths


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workload',type=Path,required=True)
    ap.add_argument('--history',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise ValueError('refusing overwrite')
    data=json.loads(args.workload.read_text());history={r['input']:r['lengths'] for r in json.loads(args.history.read_text())['records']}
    by_hash={};groups={}
    for row in data['requests']:
        by_hash.setdefault(row['prompt_hash'],row);groups.setdefault(row['prompt_hash'],[]).append(row)
    hashes=sorted(by_hash);stats=[replace(_stats_from_lengths(tuple(history[by_hash[key]['input']][:8])),source_idx=index) for index,key in enumerate(hashes)]
    plan=_solve_one_batch(stats,max_rank_peak_tokens=380800,floor_kv_caps={2:0,4:133120,8:262656,16:380800},
        adaptive_floor=True,min_adaptive_floor=4,force_selected_floor=4,rank_matching_policy='release_area',
        active_peak_safety_factor=1.16,max_response_len=4608)
    if not plan.feasible:raise ValueError('historical planner cannot admit fixed P4 cohort')
    assignments={}
    for rank,indices in plan.rank_to_prompt_indices.items():
        assignments[rank]=[row for index in indices for row in sorted(groups[hashes[index]],key=lambda r:r['sample_index'])]
    if any(len(rows)!=16 for rows in assignments.values()):raise ValueError('not16 real responses/rank')
    payload={'status':'HISTORY_ONLY_PAPER_PLANNER_LAYOUT','selected_floor':plan.selected_floor,
        'rank_requests':{str(rank):rows for rank,rows in assignments.items()},'predicted_rank_exit':plan.rank_loads,
        'predicted_release_rank_tokens':plan.release_area,'forecast_safety_factor':1.16,'actual_results_used':False,
        'input_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.workload,args.history,Path(__file__))}}
    args.output.write_text(json.dumps(payload,sort_keys=True)+'\n');print(json.dumps({'selected_floor':plan.selected_floor,'predicted_release_rank_tokens':plan.release_area},sort_keys=True))


if __name__=='__main__':main()
