#!/usr/bin/env python3
"""CPU-only actual pinned VERL dataset check, not model evaluation."""
import hashlib
import json
from pathlib import Path
import sys

SOURCE=Path('/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main')
ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')
sys.path.insert(0,str(SOURCE))


def main():
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    tokenizer=AutoTokenizer.from_pretrained('/data/Qwen3-30B-A3B',local_files_only=True,trust_remote_code=False)
    output=ROOT/'evaluation_dataset_cpu_check';output.mkdir(exist_ok=True)
    config=OmegaConf.create(dict(dataset_fraction=1.0,max_prompt_length=1024,
        max_samples=-1,shuffle=False,truncation='error',filter_overlong_prompts=True,
        filter_overlong_prompts_workers=1,cache_dir=str(output/'cache')))
    manifest=json.loads((ROOT/'heldout_prepared/manifest.json').read_text())
    validation=RLHFDataset(str(ROOT/'heldout_prepared/heldout256.parquet'),tokenizer,config)
    train=RLHFDataset(str(ROOT/'heldout_prepared/train_bookkeeping32.parquet'),tokenizer,config)
    if len(validation)!=256 or len(train)!=32:raise RuntimeError('Actual dataset loader changed planned counts')
    for i,expected in enumerate(manifest['examples']):
        actual=validation[i]
        ids=actual['input_ids'][actual['attention_mask'].bool()].tolist()
        if tokenizer.decode(ids,skip_special_tokens=True)!=expected['expected_decoded_input']:
            raise RuntimeError('Actual VERL-rendered validation input differs from prespecified manifest')
        if str(actual['reward_model']['ground_truth'])!=str(expected['ground_truth']):
            raise RuntimeError('Actual validation ground truth differs')
    report=dict(status='ACTUAL_VERL_HELDOUT_LOADER256_PROMPTS_GTS_MATCH_CPU_ONLY',
        validation_rows=256,unused_train_loader_rows=32,actor_or_NPU_model_loaded=False,
        actual_dataset_source=str(SOURCE/'verl/utils/dataset/rl_dataset.py'),
        dataset_source_sha256=hashlib.sha256((SOURCE/'verl/utils/dataset/rl_dataset.py').read_bytes()).hexdigest())
    (output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
