#!/usr/bin/env python3
"""Commit the requested version layout without touching the user's index."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
SUFFIXES={'.py','.pyi','.sh','.c','.cc','.cpp','.h','.hpp','.cu','.cuh','.cmake',
          '.yaml','.yml','.toml','.cfg','.ini','.json','.patch','.md','.rst','.proto',
          '.in','.xml','.html','.js','.css','.jinja','.jinja2','.j2','.txt'}
SKIP={'.git','.codex','.agents','.vscode','__pycache__','.pytest_cache','.cache',
      '.config','.triton','.tvm_test_data','Ascend','atc_data','extra-info','kernel_meta',
      'outputs','record','record_old011','tensorboard_log','save4analyse_ingraphmode',
      'logs','rollout_data','rollout_length','checkpoints','validation_results',
      'qwen3_shrink_aware_cleanup_20260626T182116Z','analysis_eval','evidence',
      'node_modules','.build','build','dist'}


def git(*args,env=None,input=None):
    return subprocess.run(['git',*args],cwd=ROOT,env=env,input=input,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout


def selected(path):
    rel=path.relative_to(ROOT)
    if set(rel.parts)&SKIP:
        # The small calibrated capacity contract is needed by runtime tools.
        return (str(rel).endswith('/analysis_eval/MLSYS_FIG6_CAPACITY_20261017.json') or
                str(rel).endswith('/qwen3_shrink_aware/analysis_eval/plot_e2e_performance.py'))
    if any(p.startswith(('resample_result','noresample_result','mode0_','mode1_')) for p in rel.parts[:-1]):
        return False
    if path.is_symlink() or not path.is_file() or path.stat().st_size>50*1024**2:
        return False
    name=path.name
    if name.startswith(('.env','PAPER_SOURCE_SNAPSHOT')) or name.endswith('.log'):
        return False
    if path.suffix=='.txt':
        return name in {'CMakeLists.txt','README.txt'} or name.startswith(('requirements','constraints'))
    return path.suffix in SUFFIXES or name in {'LICENSE','COPYING','NOTICE','Makefile',
        'Dockerfile','.gitignore','.gitmodules','python3'} or name.startswith('Dockerfile.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch',default='014_MLSYS_adafloor')
    p.add_argument('--commit',action='store_true')
    p.add_argument('--update-local',action='store_true',help='Fast-forward the newly prepared local branch before publication')
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    files=[]
    for folder in ['llm_rl/qwen3_014','llm_rl/qwen3_011']:
        files.extend(path for path in (ROOT/folder).rglob('*') if selected(path))
    files.extend(ROOT/name for name in ['llm_rl/README.md','container_image/reorganize_qwen3_layout.py',
                                       'container_image/prepare_qwen3_source_commit.py'])
    files=sorted(set(files))
    # Include the independent prototype's existing recipe code at its current
    # location only if it was already tracked, never publish a new project.
    # Read HEAD, not the user's partly staged index. Otherwise pre-staged
    # deletions would leave old implementation paths in the new commit.
    removed=git('ls-tree','-rz','--name-only','HEAD','llm_rl/qwen3','llm_rl/adafloor_gragh').split(b'\0')
    removed=[path for path in removed if path]
    summary=dict(branch=a.branch,source_files=len(files),bytes=sum(p.stat().st_size for p in files),
                 old_tracked_paths_removed=len(removed),max_file_bytes=max(p.stat().st_size for p in files),
                 excluded='Runtime artifacts, caches, weights, binaries, logs, raw results and unpublished paper sources/PDFs')
    if not a.commit:
        print(json.dumps(summary,indent=2))
        return
    existing=subprocess.run(['git','show-ref','--verify','--quiet','refs/heads/'+a.branch],cwd=ROOT)
    if existing.returncode==0 and not a.update_local:
        raise RuntimeError('Local branch already exists, refusing to overwrite')
    if existing.returncode!=0 and a.update_local:
        raise RuntimeError('No local branch to advance')
    index=ROOT/'.git/index'
    before=hashlib.sha256(index.read_bytes()).hexdigest()
    parent=git('rev-parse','refs/heads/'+a.branch if a.update_local else 'HEAD').decode().strip()
    with tempfile.TemporaryDirectory(prefix='qwen3_publish_') as td:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(td)/'index'))
        git('read-tree',parent,env=env)
        git('update-index','--force-remove','-z','--stdin',env=env,
            input=b'\0'.join(removed)+b'\0')
        paths=Path(td)/'paths'
        paths.write_bytes(b'\0'.join(str(p.relative_to(ROOT)).encode() for p in files)+b'\0')
        git('add','-f','--pathspec-from-file='+str(paths),'--pathspec-file-nul',env=env)
        staged=git('diff','--cached','--stat',env=env).decode()
        tree=git('write-tree',env=env).decode().strip()
        commit=git('commit-tree',tree,'-p',parent,'-m',
                   'Consolidate AdaFloor v0.14 graph/eager/sidecar sources for MLSys',env=env).decode().strip()
        if a.update_local:
            git('update-ref','refs/heads/'+a.branch,commit,parent)
        else:
            git('branch',a.branch,commit)
    assert hashlib.sha256(index.read_bytes()).hexdigest()==before
    report=dict(summary,commit=commit,parent=parent,user_index_preserved_sha256=before,
                current_worktree_branch_unchanged=True,pushed=False,
                source_paths=[str(p.relative_to(ROOT)) for p in files])
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_paths'},indent=2))


if __name__=='__main__':main()
