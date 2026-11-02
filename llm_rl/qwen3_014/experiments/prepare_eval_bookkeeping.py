#!/usr/bin/env python3
"""A tiny train-split loader fixture for val_only, never optimized/evaluated."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from prepare_heldout import plain

ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')
SOURCE=Path('/data/deepscaler/train.parquet')
DEST=ROOT/'heldout_prepared/train_bookkeeping32.parquet'


def main():
    if DEST.exists():raise RuntimeError('Refusing existing evaluation bookkeeping view')
    tokenizer=AutoTokenizer.from_pretrained('/data/Qwen3-30B-A3B',local_files_only=True,trust_remote_code=False)
    frame=pd.read_parquet(SOURCE);indices=[]
    for i,row in frame.iterrows():
        if not str(row['reward_model'].get('ground_truth','')).strip():continue
        tokens=tokenizer.apply_chat_template(plain(row['prompt']),tokenize=True,add_generation_prompt=True)
        if len(tokens)<=1024:indices.append(int(i))
        if len(indices)==32:break
    if len(indices)!=32:raise RuntimeError('No valid32-row bookkeeping loader fixture')
    frame.iloc[indices].to_parquet(DEST,index=False)
    manifest=dict(status='TRAIN_SPLIT_BOOKKEEPING_ONLY_NOT_OPTIMIZED',rows=32,
        source=str(SOURCE),source_indices=indices,path=str(DEST),sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
        scope='Only satisfies the unused train loader in trainer.val_only. Not validation questions, not an actor update; actual held-out256 unchanged. Evaluation dataset_fraction must be1.0.')
    (ROOT/'heldout_prepared/train_bookkeeping32.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':main()
