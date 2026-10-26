"""Profile one stateful FP32-master/BF16-compute training step.

This is diagnostic tooling: CUDA graph launch records are reported separately
from actual device kernel records and are never counted as kernels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from benchmarks.train_loop import make_batches, step_master


def _events(profile):
    cuda = [e for e in profile.events() if e.device_type == torch.profiler.DeviceType.CUDA]
    graph = [e for e in cuda if "cudaGraphLaunch" in e.name or "CUDAGraph" in e.name]
    kernels = [e for e in cuda if e not in graph and getattr(e, "device_time_total", 0) > 0]
    return {
        "kernel_count": len(kernels),
        "graph_launch_count": len(graph),
        "kernels": [{"name": e.name, "duration_us": e.device_time_total} for e in kernels],
        "graph_launches": [{"name": e.name, "duration_us": e.device_time_total} for e in graph],
    }


def _profile(fn, state, batch, warmup):
    for _ in range(warmup):
        fn(state, *batch)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
        out = fn(state, *batch)
        torch.cuda.synchronize()
    return out, _events(p)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (("batch", 128), ("input", 512), ("hidden", 1024), ("output", 256), ("warmup", 3)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--maxblocks", "--max-blocks", dest="maxblocks", type=int, default=108)
    p.add_argument("--threads-per-block", type=int, choices=(128, 256, 512, 1024), default=128)
    p.add_argument("--split-k-gemv", action="store_true")
    p.add_argument("--blocks-per-sm", type=int, default=1)
    p.add_argument("--gemm-k-tile", type=int, choices=(16, 32), default=32)
    p.add_argument("--gemm-tile-n", type=int, choices=(32, 64), default=32)
    p.add_argument("--shared-skew", type=int, choices=(0, 16), default=0)
    p.add_argument("--cuda-path", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument("--seed", type=int, default=20261025)
    args = p.parse_args(argv)
    result = {"config": vars(args), "status": "pending"}
    result["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in result["config"].items()}
    if not torch.cuda.is_available():
        result.update(status="skipped", reason="CUDA unavailable")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda")
        raw, batches = make_batches(args.batch, args.input, args.hidden, args.output, torch.bfloat16, device, 1, args.seed)
        initial = tuple(x.float() for x in raw)
        batch = batches[0]
        from trainkernel import compile as mk_compile
        options = {"max_blocks": args.maxblocks, "blocks_per_sm": args.blocks_per_sm, "threads_per_block": args.threads_per_block,
                   "split_k_gemv": args.split_k_gemv, "gemm_k_tile": args.gemm_k_tile, "gemm_tile_n": args.gemm_tile_n, "shared_skew": args.shared_skew}
        t0 = time.perf_counter(); mega = mk_compile(lambda q, x, y: step_master(q, x, y, 1e-3), (initial, *batch), inplace_outputs={5: 0, 6: 1, 7: 2, 8: 3}, **options); compile_s = time.perf_counter() - t0
        state = tuple(x.clone() for x in initial)
        out, mega_events = _profile(mega, state, batch, args.warmup)
        result.update(status="ok", compile_seconds=compile_s, megakernel=mega_events, loss=float(out[0]))
        source = getattr(mega, "source", "")
        result["megakernel_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        if source and args.cuda_path:
            args.cuda_path.parent.mkdir(parents=True, exist_ok=True)
            args.cuda_path.write_text(source)
            result["cuda_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        # Stateful Inductor is profiled without CUDA graph capture.
        def stateful(q, x, y):
            value = step_master(q, x, y, 1e-3)
            for dst, src in zip(q, value[2]): dst.copy_(src)
            return value
        ind = torch.compile(stateful, mode="max-autotune-no-cudagraphs", fullgraph=True)
        istate = tuple(x.clone() for x in initial)
        _, ind_events = _profile(ind, istate, batch, args.warmup)
        result["inductor_stateful"] = ind_events
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True); args.json.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
