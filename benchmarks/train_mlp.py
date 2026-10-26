"""Reproducible full-step correctness and fixed-input GPU latency benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import torch
from torch.utils._pytree import tree_flatten


def step(params, x, target, lr):
    def loss_fn(p):
        hidden = torch.relu(x @ p[0] + p[1])
        prediction = hidden @ p[2] + p[3]
        return ((prediction - target) ** 2).mean()
    grads, loss = torch.func.grad_and_value(loss_fn)(params)
    updated = tuple(p - lr * g for p, g in zip(params, grads))
    return loss, grads, updated


def make_inputs(batch, input_size, hidden, output_size, dtype, device):
    generator = torch.Generator(device="cpu").manual_seed(17)
    def rand(shape):
        return torch.randn(shape, generator=generator, dtype=dtype).to(device)
    params = (rand((input_size, hidden)) / math.sqrt(input_size), rand((hidden,)),
              rand((hidden, output_size)) / math.sqrt(hidden), rand((output_size,)))
    return params, rand((batch, input_size)), rand((batch, output_size))


def tolerances(dtype):
    if dtype == torch.bfloat16:
        return {"rtol": 0.04, "atol": 0.004}
    if dtype == torch.float16:
        return {"rtol": 0.008, "atol": 0.0005}
    return {"rtol": 0.0002, "atol": 0.00002}


def compare(actual, expected, dtype, *, enforce=True):
    got, got_tree = tree_flatten(actual)
    want, want_tree = tree_flatten(expected)
    assert got_tree == want_tree
    errors, failures = [], []
    for index, (a, b) in enumerate(zip(got, want)):
        errors.append(float((a.detach().float() - b.detach().float()).abs().max()) if a.numel() else 0.0)
        try:
            torch.testing.assert_close(a, b, **tolerances(dtype))
        except AssertionError as exc:
            if enforce:
                raise
            failures.append({"output": index, "error": str(exc)})
    return {"passed": not failures, "max_absolute_error_per_output": errors, "tolerances": tolerances(dtype), "failures": failures}


def trajectory(fn, params, x, target, steps, dtype):
    eager_params = params
    actual_params = params
    losses = []
    for _ in range(steps):
        expected = step(eager_params, x, target, 1e-3)
        actual = fn(actual_params, x, target)
        compare(actual, expected, dtype)
        losses.append({"eager": float(expected[0]), "compiled": float(actual[0])})
        eager_params, actual_params = expected[2], actual[2]
    return losses


def _measure(fn, *, warmup, samples, inner):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / inner)
    ordered = sorted(timings)
    return {"median_ms": statistics.median(ordered), "p10_ms": ordered[int((len(ordered)-1)*0.1)],
            "p90_ms": ordered[int((len(ordered)-1)*0.9)], "samples": samples, "steps_per_sample": inner}


def _capture(fn):
    # Retain captured outputs so their backing storage stays owned for every replay.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    return graph, outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    for name, default in (("batch", 32), ("input", 128), ("hidden", 256), ("output", 64), ("steps", 5), ("warmup", 10), ("repeats", 20), ("inner", 10)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--dump-cuda", type=Path)
    parser.add_argument("--skip-inductor", action="store_true")
    parser.add_argument("--blocks-per-sm", type=int, default=1)
    args = parser.parse_args(argv)
    if any(getattr(args, name) < 1 for name in ("batch", "input", "hidden", "output", "steps", "warmup", "repeats", "inner")):
        parser.error("shapes and iteration counts must be positive")
    result: dict[str, Any] = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "status": "pending"}
    if not torch.cuda.is_available():
        result.update(status="skipped", reason="CUDA unavailable")
    else:
        try:
            from trainkernel import compile as mk_compile
            device, dtype = torch.device("cuda", torch.cuda.current_device()), getattr(torch, args.dtype)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            params, x, target = make_inputs(args.batch, args.input, args.hidden, args.output, dtype, device)
            result.update(device=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)))
            eager = lambda p, xx, yy: step(p, xx, yy, 1e-3)
            start = time.perf_counter()
            mega = mk_compile(eager, (params, x, target), blocks_per_sm=args.blocks_per_sm)
            result["megakernel_compile_seconds"] = time.perf_counter() - start
            generated = mega.runner.generated
            result["kernel"] = {"launches_per_step": 1, "materialized_operations": generated.materialized_operations,
                                "inlined_operations": generated.inlined_operations, "tensor_core_operations": generated.tensor_core_operations,
                                "workspace_bytes": generated.workspace_elements * 4, "blocks_per_sm_limit": generated.blocks_per_sm, "max_work_blocks": generated.max_work_blocks, "grid_barriers": generated.grid_barriers, "direct_outputs": generated.direct_outputs}
            if args.dump_cuda:
                args.dump_cuda.parent.mkdir(parents=True, exist_ok=True)
                args.dump_cuda.write_text(mega.source)
            reference = eager(params, x, target)
            result["correctness"] = compare(mega(params, x, target), reference, dtype)
            result["trajectory"] = trajectory(mega, params, x, target, args.steps, dtype)
            methods = {"eager": lambda: eager(params, x, target), "megakernel": lambda: mega(params, x, target)}
            result["timings"] = {}
            for name, fn in list(methods.items()):
                graph, outputs = _capture(fn)
                graph.replay()
                compare(outputs, reference, dtype)
                methods[f"{name}_cuda_graph"] = graph.replay
            if not args.skip_inductor:
                compiled = torch.compile(eager, mode="max-autotune", fullgraph=True)
                start = time.perf_counter()
                inductor_outputs = compiled(params, x, target)
                torch.cuda.synchronize()
                result["inductor_compile_seconds"] = time.perf_counter() - start
                result["inductor_correctness"] = compare(inductor_outputs, reference, dtype, enforce=False)
                del inductor_outputs
                methods["inductor_max_autotune"] = lambda: compiled(params, x, target)
                result["inductor_cuda_graphs_requested"] = True
                # Capture optimized Inductor kernels directly as an additional
                # baseline, eliminating its Python wrapper from GPU comparisons.
                graphable = torch.compile(eager, mode="max-autotune-no-cudagraphs", fullgraph=True)
                graph, outputs = _capture(lambda: graphable(params, x, target))
                graph.replay()
                result["inductor_graph_correctness"] = compare(outputs, reference, dtype, enforce=False)
                methods["inductor_cuda_graph"] = graph.replay
                from torch._inductor import list_mode_options
                strict_options = dict(list_mode_options("max-autotune-no-cudagraphs"))
                strict_options["emulate_precision_casts"] = True
                strict = torch.compile(eager, options=strict_options, fullgraph=True)
                strict_graph, strict_outputs = _capture(lambda: strict(params, x, target))
                strict_graph.replay()
                result["inductor_eager_rounding_correctness"] = compare(strict_outputs, reference, dtype, enforce=False)
                result["inductor_eager_rounding_options"] = strict_options
                methods["inductor_eager_rounding_cuda_graph"] = strict_graph.replay
            for name, fn in methods.items():
                result["timings"][name] = _measure(fn, warmup=args.warmup, samples=args.repeats, inner=args.inner)
            result["status"] = "ok"
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            import traceback
            result["traceback"] = traceback.format_exc()
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
