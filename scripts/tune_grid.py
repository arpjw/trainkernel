"""Exploratory grid sweep. Tuning results are not held-out benchmark evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from benchmarks.train_mlp import make_inputs, step, compare, _capture, _measure
from megakernel import compile as mk_compile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape', default='4,32,64,16')
    p.add_argument('--grids', default='1,4,16,108')
    p.add_argument('--json', type=Path, required=True)
    a = p.parse_args()
    shape = tuple(map(int, a.shape.split(',')))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    params, x, y = make_inputs(*shape, torch.bfloat16, 'cuda')
    fn = lambda pp, xx, yy: step(pp, xx, yy, 1e-3)
    expected = fn(params, x, y)
    result = {'shape': shape, 'purpose': 'exploratory tuning', 'trials': []}
    for grid in map(int, a.grids.split(',')):
        trial = {'max_blocks': grid}
        try:
            start = time.perf_counter()
            mega = mk_compile(fn, (params, x, y), max_blocks=grid)
            trial['compile_seconds'] = time.perf_counter() - start
            trial['correctness'] = compare(mega(params, x, y), expected, torch.bfloat16)
            graph, outputs = _capture(lambda: mega(params, x, y))
            graph.replay()
            compare(outputs, expected, torch.bfloat16)
            trial['timing'] = _measure(graph.replay, warmup=10, samples=30, inner=20)
            trial['status'] = 'ok'
        except Exception as exc:
            trial.update(status='error', error=str(exc))
        result['trials'].append(trial)
        print(json.dumps(trial), flush=True)
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=2) + '\n')

if __name__ == '__main__':
    main()
