#!/usr/bin/env python3
"""Canonical row identities only; preserve every original code question/label."""
import copy
import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE=Path('/data/eurus2_rl_code/validation_code_paired_160/train.parquet')
DEST=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z/code_prepared/train.parquet')


def normalize(rows):
    output=copy.deepcopy(rows)
    for i,row in enumerate(output):
        info=dict(row.get('extra_info') or {})
        for key in ('index','prompt_occurrence_ordinal'):
            if key in info:info['original_'+key]=info[key]
            info[key]=i
        row['extra_info']=info
    return output


def semantic(rows):
    return [{k:v for k,v in row.items() if k!='extra_info'} for row in rows]


def main():
    if DEST.exists():raise RuntimeError('Refusing existing canonical code view')
    original=pq.read_table(SOURCE).to_pylist();rows=normalize(original)
    if len(rows)!=160 or semantic(rows)!=semantic(original):raise RuntimeError('Code data changed beyond identities')
    DEST.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(rows),DEST)
    actual=pq.read_table(DEST).to_pylist()
    if actual!=rows:raise RuntimeError('Canonical parquet roundtrip changed data')
    manifest=dict(status='ALL160_CODE_SEMANTICS_UNCHANGED_IDENTITIES_CANONICAL',rows=160,
        source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        path=str(DEST),sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
        semantic_sha256=hashlib.sha256(json.dumps(semantic(rows),sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        scope='Only extra_info row identities normalized for v0.14; original identities preserved. Same ordered160 prompts, ground truths and other columns; no filtering/replacement/cap change.')
    (DEST.parent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))


if __name__=='__main__':main()
