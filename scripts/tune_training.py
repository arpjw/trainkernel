"""Exploratory A100 tuning; final validation uses separate seeds and baselines."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import torch
from benchmarks.train_loop import make_batches, step_master, _trajectory, _capture, _close, _paired_time
from megakernel import compile as mk_compile

CASES = {'tiny-online': (4,32,64,16), 'online': (1,128,128,16), 'small': (32,128,256,64), 'medium': (128,512,1024,256)}
CONFIGS = {
 'tiny-online': [{'max_blocks':1}, {'max_blocks':4}, {'max_blocks':1,'threads_per_block':512}, {'max_blocks':1,'threads_per_block':1024}],
 'online': [{'max_blocks':1}, {'max_blocks':4}, {'max_blocks':16}, {'max_blocks':1,'split_k_gemv':True}, {'max_blocks':4,'split_k_gemv':True}, {'max_blocks':16,'split_k_gemv':True}, {'max_blocks':1,'threads_per_block':512,'split_k_gemv':True}, {'max_blocks':1,'threads_per_block':1024,'split_k_gemv':True}, {'max_blocks':1,'threads_per_block':512}, {'max_blocks':1,'threads_per_block':1024}],
 'small': [{'max_blocks':108}, {'max_blocks':108,'shared_skew':16}, {'max_blocks':108,'shared_skew':16,'gemm_k_tile':16}, {'max_blocks':108,'gemm_tile_n':64}, {'max_blocks':216,'blocks_per_sm':2,'shared_skew':16}],
 'medium': [{'max_blocks':108}, {'max_blocks':108,'shared_skew':16}, {'max_blocks':108,'shared_skew':16,'gemm_k_tile':16}, {'max_blocks':108,'gemm_tile_n':64}, {'max_blocks':216,'blocks_per_sm':2,'shared_skew':16}, {'max_blocks':216,'blocks_per_sm':2,'gemm_tile_n':64}],
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=CASES,required=True)
    p.add_argument('--json',type=Path,required=True)
    p.add_argument('--only',type=int)
    a=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    params,batches=make_batches(*CASES[a.case],torch.bfloat16,'cuda',seed=20261025)
    params=tuple(v.float() for v in params)
    fn=lambda pp,x,y:step_master(pp,x,y,1e-3)
    root=Path(__file__).resolve().parents[1]
    sources={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((root/'megakernel').glob('*.py'))}
    result={'purpose':'exploratory tuning, not held-out evidence','case':a.case,'shape':CASES[a.case],'seed':20261025,'source_sha256':sources,'trials':[]}
    configs=CONFIGS[a.case]
    if a.only is not None and not 0 <= a.only < len(configs):
        p.error(f'--only must be between 0 and {len(configs) - 1}')
    if a.json.exists():
        prior=json.loads(a.json.read_text())
        if prior.get('source_sha256') != sources or prior.get('case') != a.case:
            p.error('existing result has a different case or source hash; choose a new --json path')
        result=prior
    for index,options in enumerate(configs):
        if a.only is not None and index != a.only:
            continue
        if any(t['index']==index for t in result['trials']):
            continue
        trial={'index':index,'compiler_options':options}
        try:
            start=time.perf_counter()
            mega=mk_compile(fn,(params,*batches[0]),inplace_outputs={5:0,6:1,7:2,8:3},**options)
            trial['compile_seconds']=time.perf_counter()-start
            trial['resources']=mega.runner.resource_usage()
            trial['cuda_sha256']=hashlib.sha256(mega.source.encode()).hexdigest()
            losses,_=_trajectory(mega,params,batches,100)
            trial['trajectory']={'passed':True,'steps':100,'initial_loss':losses[0],'final_loss':losses[-1]}
            state=tuple(v.clone() for v in params)
            graph=_capture(mega,state,batches,20)
            for dst,src in zip(state,params):dst.copy_(src)
            graph.replay()
            _,expected=_trajectory(fn,params,batches,20)
            _close(state,expected,torch.float32)
            trial['replay_correctness']={'passed':True}
            trial['timing']=_paired_time([(graph,state)],params,20,30,10)[0]
            trial['status']='ok'
        except Exception as exc:
            trial.update(status='error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        result['trials'].append(trial)
        a.json.parent.mkdir(parents=True,exist_ok=True)
        a.json.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(trial),flush=True)
    passing=[t for t in result['trials'] if t['status']=='ok']
    if passing:
        result['best_index']=min(passing,key=lambda t:t['timing']['median_ms'])['index']
    a.json.write_text(json.dumps(result,indent=2)+'\n')
    requested = [t for t in result['trials'] if a.only is None or t['index'] == a.only]
    return int(not requested or any(t['status'] != 'ok' for t in requested))

if __name__=='__main__': raise SystemExit(main())
