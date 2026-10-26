"""Diagnostic global-timer measurements of cooperative synchronization stages.

Instrumentation changes the kernel. Use uninstrumented CUDA-event timings for
performance claims. Hardware counters are not required.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import statistics
import torch
from benchmarks.train_mlp import make_inputs, step, compare
from benchmarks.train_loop import make_batches, step_master, _close_training
import megakernel.runtime as runtime
from megakernel import compile as mk_compile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape', default='128,512,1024,256')
    p.add_argument('--max-blocks', type=int, default=108)
    p.add_argument('--threads-per-block', type=int, default=128)
    p.add_argument('--blocks-per-sm', type=int, default=1)
    p.add_argument('--gemm-k-tile', type=int, default=32)
    p.add_argument('--gemm-tile-n', type=int, default=32)
    p.add_argument('--shared-skew', type=int, default=0)
    p.add_argument('--split-k-gemv', action='store_true')
    p.add_argument('--training', action='store_true', help='FP32 masters and loss, BF16 compute, state commit')
    p.add_argument('--json', type=Path, required=True)
    a = p.parse_args()
    original = runtime.generate
    labels, offsets = [], []
    def instrument(program, **kwargs):
        g = original(program, **kwargs)
        offset = (g.workspace_elements + 1) // 2 * 2
        offsets.append(offset)
        source = g.source.replace('auto grid = cg::this_grid();', f'auto grid = cg::this_grid(); uint64_t* stamps = reinterpret_cast<uint64_t*>(workspace + {offset}); if(blockIdx.x == 0 && threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamps[0]));')
        chunks = source.split('mk_grid_sync(grid);')
        source = chunks[0]
        for i, chunk in enumerate(chunks[1:], 1):
            labels.append(re.findall(r'// (\w+): (\w+)', chunks[i-1]))
            source += f'mk_grid_sync(grid); if(blockIdx.x == 0 && threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamps[{i}]));' + chunk
        # Include commit/copy tail time through the end of the kernel body.
        match = re.search(r'\n}\s*\nvoid launch\(', source)
        if match is None:
            raise RuntimeError('cannot find generated kernel boundary')
        boundary = match.start()
        source = source[:boundary] + f'\nmk_grid_sync(grid); if(blockIdx.x == 0 && threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamps[{len(chunks)}]));' + source[boundary:]
        labels.append(re.findall(r'// (\w+): (\w+)', chunks[-1]) or [('kernel_tail', 'completion')])
        return replace(g, source=source, workspace_elements=offset + 2 * (len(chunks)+1))
    runtime.generate = instrument
    shape = tuple(map(int, a.shape.split(',')))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if a.training:
        params, batches = make_batches(*shape, torch.bfloat16, 'cuda', seed=20261025)
        params = tuple(x.float() for x in params)
        x, y = batches[0]
        fn = lambda pp, xx, yy: step_master(pp, xx, yy, 1e-3)
    else:
        params, x, y = make_inputs(*shape, torch.bfloat16, 'cuda')
        fn = lambda pp, xx, yy: step(pp, xx, yy, 1e-3)
    initial = tuple(v.clone() for v in params)
    kwargs = dict(max_blocks=a.max_blocks, threads_per_block=a.threads_per_block, blocks_per_sm=a.blocks_per_sm, gemm_k_tile=a.gemm_k_tile, gemm_tile_n=a.gemm_tile_n, shared_skew=a.shared_skew, split_k_gemv=a.split_k_gemv)
    if a.training: kwargs['inplace_outputs'] = {5:0,6:1,7:2,8:3}
    mega = mk_compile(fn, (params, x, y), **kwargs)
    expected = fn(params,x,y)
    actual = mega(params,x,y)
    if a.training: _close_training(actual,expected)
    else: compare(actual,expected,torch.bfloat16)
    flat = (*params,x,y)
    buffers = mega.runner.allocate(*flat)
    durations=[]
    for i in range(30):
        for dst,src in zip(params,initial): dst.copy_(src)
        mega.runner.run_into(*flat,buffers=buffers)
        torch.cuda.synchronize()
        stamps=buffers.workspace[offsets[0]:].view(torch.int64).cpu().tolist()
        if i>=10: durations.append([(b-a)/1000 for a,b in zip(stamps,stamps[1:])])
    result={'shape':shape,'options':kwargs,'instrumented':True,'resources':mega.runner.resource_usage(), 'stages':[{'operations':label,'median_us':statistics.median(row[i] for row in durations)} for i,label in enumerate(labels)]}
    a.json.parent.mkdir(parents=True,exist_ok=True)
    a.json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
