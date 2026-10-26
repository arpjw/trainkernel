"""Stateful training-loop benchmark for the static megakernel subset.

The timed region is a CUDA graph containing stateful training steps. Pure
implementations copy returned updates to state; in-place implementations
write state during the step. Inputs are cycled across
several fixed batches, so results cover parameter feedback rather than a
single replay with frozen weights.  Compilation and graph capture are reported
separately and never included in step latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


def step(params, x, target, lr: float):
    """Pure BF16/FP32 SGD step.  Updates are returned, never mutated."""
    def loss_fn(p):
        hidden = torch.relu(x @ p[0] + p[1])
        return ((hidden @ p[2] + p[3] - target) ** 2).mean()

    grads, loss = torch.func.grad_and_value(loss_fn)(params)
    updated = tuple(p - lr * g for p, g in zip(params, grads))
    return loss, grads, updated


def step_master(params, x, target, lr: float):
    """FP32 master weights with BF16 matmul/activation compute."""
    def loss_fn(p):
        q = tuple(v.to(torch.bfloat16) for v in p)
        hidden = torch.relu(x @ q[0] + q[1])
        prediction = hidden @ q[2] + q[3]
        return ((prediction.float() - target.float()) ** 2).mean()

    grads, loss = torch.func.grad_and_value(loss_fn)(params)
    updated = tuple(p - lr * g for p, g in zip(params, grads))
    return loss, grads, updated


def make_batches(batch: int, input_size: int, hidden: int, output: int,
                 dtype: torch.dtype, device: torch.device, count: int = 4,
                 seed: int = 20261025, task: str = "regression"):
    if task not in {"regression", "classification-head"}:
        raise ValueError("task must be regression or classification-head")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    def rand(shape):
        return torch.randn(shape, generator=generator, dtype=dtype).to(device)

    params = (rand((input_size, hidden)) / math.sqrt(input_size), rand((hidden,)),
              rand((hidden, output)) / math.sqrt(hidden), rand((output,)))
    teacher = (rand((input_size, hidden)) / math.sqrt(input_size), rand((hidden,)),
               rand((hidden, output)) / math.sqrt(hidden), rand((output,)))
    batches = []
    for _ in range(count):
        x = rand((batch, input_size))
        with torch.no_grad():
            target = torch.relu(x @ teacher[0] + teacher[1]) @ teacher[2] + teacher[3]
            if task == "classification-head":
                labels = target.argmax(dim=-1)
                target = torch.nn.functional.one_hot(labels, num_classes=output).to(dtype)
        batches.append((x, target))
    return params, batches


def _close(actual, expected, dtype):
    assert len(actual) == len(expected), "incorrect tensor count"
    tol = {torch.float32: (2e-4, 2e-5), torch.bfloat16: (0.04, 0.004), torch.float16: (0.008, 5e-4)}[dtype]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=tol[0], atol=tol[1])


def _close_training(actual, expected):
    assert len(actual) == len(expected) == 3, "expected loss, gradients, and updates"
    assert len(actual[1]) == len(expected[1]), "incorrect gradient count"
    assert len(actual[2]) == len(expected[2]), "incorrect parameter count"
    # Loss and gradients are BF16-compute results; master updates are FP32.
    _close((actual[0],), (expected[0],), torch.float32)
    _close(actual[1], expected[1], torch.bfloat16)
    _close(actual[2], expected[2], torch.float32)


def _capture(method: Callable, state, batches, steps: int):
    """Capture state feedback, skipping copies for outputs already aliased to state."""
    for i in range(3):
        x, target = batches[i % len(batches)]
        outputs = method(state, x, target)
        for dst, src in zip(state, outputs[2]):
            if dst.data_ptr() != src.data_ptr():
                dst.copy_(src)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(steps):
            x, target = batches[i % len(batches)]
            outputs = method(state, x, target)
            for dst, src in zip(state, outputs[2]):
                if dst.data_ptr() != src.data_ptr():
                    dst.copy_(src)
    return graph


def _trajectory(method, initial, batches, steps, *, enforce=True, report=False):
    eager_state = tuple(p.clone() for p in initial)
    actual_state = tuple(p.clone() for p in initial)
    losses = []
    failures = []
    max_errors = {"loss": 0.0, "gradients": 0.0, "updates": 0.0}
    finite = True
    for i in range(steps):
        x, target = batches[i % len(batches)]
        expected = step_master(eager_state, x, target, 1e-3)
        actual = method(actual_state, x, target)
        loss_error = float((actual[0].float() - expected[0].float()).abs().max())
        gradient_errors = [float((a.float() - b.float()).abs().max()) for a, b in zip(actual[1], expected[1])]
        update_errors = [float((a.float() - b.float()).abs().max()) for a, b in zip(actual[2], expected[2])]
        max_errors["loss"] = max(max_errors["loss"], loss_error)
        max_errors["gradients"] = max(max_errors["gradients"], *gradient_errors)
        max_errors["updates"] = max(max_errors["updates"], *update_errors)
        finite = finite and all(bool(torch.isfinite(t).all()) for group in (actual[1], actual[2]) for t in group) and bool(torch.isfinite(actual[0]))
        try:
            _close_training(actual, expected)
        except AssertionError as exc:
            if enforce:
                raise AssertionError(f"trajectory step {i}: {exc}") from exc
            failures.append({"step": i, "error": str(exc), "max_absolute_error": max(loss_error, *gradient_errors, *update_errors)})
        losses.append(float(actual[0]))
        eager_state, actual_state = expected[2], actual[2]
    if report:
        return losses, actual_state, {"passed": not failures and finite, "failure_count": len(failures), "first_failure_step": failures[0]["step"] if failures else None, "max_absolute_error": max(max_errors.values()), "max_absolute_error_by_kind": max_errors, "all_finite": finite, "failures": failures}
    return losses, actual_state


def _paired_time(pairs, initial, steps, samples, warmup):
    """Interleave methods; reset each graph's state outside its timed replay."""
    for _ in range(warmup):
        for graph, state in pairs:
            for dst, src in zip(state, initial):
                dst.copy_(src)
            graph.replay()
    torch.cuda.synchronize()
    values = [[] for _ in pairs]
    for _ in range(samples):
        order = list(range(len(pairs)))
        if _ % 2:
            order.reverse()
        for index in order:
            graph, state = pairs[index]
            for dst, src in zip(state, initial):
                dst.copy_(src)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); graph.replay(); end.record(); end.synchronize()
            values[index].append(start.elapsed_time(end) / steps)
    return [{"median_ms": statistics.median(sorted(v)), "p10_ms": sorted(v)[int(.1 * (len(v) - 1))],
             "p90_ms": sorted(v)[int(.9 * (len(v) - 1))], "samples": samples, "steps_per_replay": steps,
             "raw_ms": v}
            for v in values]


def _bootstrap_speedup_ci(numerator, denominator, draws=2000, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a, b = torch.tensor(numerator), torch.tensor(denominator)
    ratios = []
    for _ in range(draws):
        indices = torch.randint(len(a), (len(a),), generator=generator)
        ratios.append(float(torch.quantile(a[indices], 0.5) / torch.quantile(b[indices], 0.5)))
    ratios.sort()
    return {"median_ratio": statistics.median(ratios), "ci95_ratio": [ratios[int(.025 * (draws - 1))], ratios[int(.975 * (draws - 1))]], "draws": draws, "seed": seed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--input", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--output", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--trajectory-steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--inductor-mode", choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"), default="max-autotune-no-cudagraphs")
    parser.add_argument("--task", choices=("regression", "classification-head"), default="regression")
    parser.add_argument("--max-blocks", type=int, default=108)
    parser.add_argument("--blocks-per-sm", type=int, default=1)
    parser.add_argument("--gemm-k-tile", type=int, choices=(16, 32), default=32)
    parser.add_argument("--gemm-tile-n", type=int, choices=(32, 64), default=32)
    parser.add_argument("--shared-skew", type=int, choices=(0, 16), default=0)
    parser.add_argument("--split-k-gemv", action="store_true")
    parser.add_argument("--threads-per-block", type=int, choices=(128,256,512,1024), default=128)
    parser.add_argument("--no-inline-input-casts", action="store_true")
    args = parser.parse_args(argv)
    if any(value < 1 for value in (args.batch, args.input, args.hidden, args.output,
                                   args.batches, args.steps, args.trajectory_steps,
                                   args.warmup, args.samples, args.max_blocks, args.blocks_per_sm)):
        parser.error("all sizes and iteration counts must be positive")
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    result = {"status": "pending", "config": config, "optimizer": "sgd",
              "task": args.task,
              "numerical_policy": "FP32 master weights; BF16 compute casts; all losses, gradients, and updates checked",
              "numerical_tolerances": {"loss_fp32": {"rtol": 0.0002, "atol": 0.00002}, "gradients_bf16_compute": {"rtol": 0.04, "atol": 0.004}, "updates_fp32": {"rtol": 0.0002, "atol": 0.00002}}}
    if not torch.cuda.is_available():
        result.update(status="skipped", reason="CUDA unavailable")
    else:
        try:
            device = torch.device("cuda", torch.cuda.current_device())
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            # Master state is FP32; activations and inputs are BF16.
            bf_params, batches = make_batches(args.batch, args.input, args.hidden, args.output, torch.bfloat16, device, args.batches, args.seed, args.task)
            params = tuple(p.float() for p in bf_params)
            from trainkernel import compile as mk_compile
            compile_start = time.perf_counter()
            compile_options = {"max_blocks": args.max_blocks, "blocks_per_sm": args.blocks_per_sm, "gemm_k_tile": args.gemm_k_tile, "gemm_tile_n": args.gemm_tile_n, "shared_skew": args.shared_skew, "split_k_gemv": args.split_k_gemv, "threads_per_block": args.threads_per_block, "inline_input_casts": not args.no_inline_input_casts}
            mega = mk_compile(lambda p, x, y: step_master(p, x, y, 1e-3), (params, *batches[0]), **compile_options)
            mega_compile_seconds = time.perf_counter() - compile_start
            reference_loss, reference_grads, reference_updates = step_master(params, *batches[0], 1e-3)
            actual = mega(params, *batches[0])
            _close_training(actual, (reference_loss, reference_grads, reference_updates))
            methods = {"megakernel": mega, "eager_graph": lambda p, x, y: step_master(p, x, y, 1e-3)}
            compile_start = time.perf_counter()
            mega_inplace = mk_compile(lambda p, x, y: step_master(p, x, y, 1e-3), (params, *batches[0]), **compile_options, inplace_outputs={5: 0, 6: 1, 7: 2, 8: 3})
            mega_inplace_compile_seconds = time.perf_counter() - compile_start
            cloned = tuple(p.clone() for p in params)
            inplace_actual = mega_inplace(cloned, *batches[0])
            _close_training(inplace_actual, (reference_loss, reference_grads, reference_updates))
            methods["mega_inplace"] = mega_inplace
            graphable = torch.compile(lambda p, x, y: step_master(p, x, y, 1e-3), mode=args.inductor_mode, fullgraph=True)
            # Compile once outside the timed region, then capture raw Inductor kernels.
            compile_start = time.perf_counter()
            graphable(params, *batches[0]); torch.cuda.synchronize()
            inductor_compile_seconds = time.perf_counter() - compile_start
            methods["inductor"] = graphable
            from torch._inductor import list_mode_options
            strict_options = dict(list_mode_options(args.inductor_mode))
            strict_options["emulate_precision_casts"] = True
            strict = torch.compile(lambda p, x, y: step_master(p, x, y, 1e-3), options=strict_options, fullgraph=True)
            compile_start = time.perf_counter()
            strict(params, *batches[0]); torch.cuda.synchronize()
            strict_compile_seconds = time.perf_counter() - compile_start
            methods["inductor_strict"] = strict
            def stateful(p, x, y):
                outputs = step_master(p, x, y, 1e-3)
                for dst, src in zip(p, outputs[2]):
                    dst.copy_(src)
                return outputs[0], outputs[1], p

            inplace = torch.compile(stateful, mode=args.inductor_mode, fullgraph=True)
            compile_start = time.perf_counter()
            inplace(tuple(p.clone() for p in params), *batches[0]); torch.cuda.synchronize()
            inplace_compile_seconds = time.perf_counter() - compile_start
            methods["inductor_inplace"] = inplace
            inplace_strict = torch.compile(stateful, options=strict_options, fullgraph=True)
            compile_start = time.perf_counter()
            inplace_strict(tuple(p.clone() for p in params), *batches[0]); torch.cuda.synchronize()
            inplace_strict_compile_seconds = time.perf_counter() - compile_start
            methods["inductor_inplace_strict"] = inplace_strict
            result["device"] = torch.cuda.get_device_name(device)
            result["compilation_seconds"] = {"megakernel": mega_compile_seconds, "mega_inplace": mega_inplace_compile_seconds, "eager_graph": 0.0, "inductor": inductor_compile_seconds, "inductor_strict": strict_compile_seconds, "inductor_inplace": inplace_compile_seconds, "inductor_inplace_strict": inplace_strict_compile_seconds}
            result["resource_usage"] = {"megakernel": mega.runner.resource_usage(), "mega_inplace": mega_inplace.runner.resource_usage()}
            result["kernel_config"] = compile_options
            result["kernel"] = {name: {key: getattr(compiled.runner.generated, key) for key in ("materialized_operations", "inlined_operations", "grid_barriers", "max_work_blocks", "workspace_elements", "tensor_core_operations", "direct_state_commits")} for name, compiled in (("megakernel", mega), ("mega_inplace", mega_inplace))}
            source_root = Path(__file__).resolve().parents[1] / "trainkernel"
            files = sorted(source_root.glob("*.py")) + [Path(__file__).resolve()]
            result["source_sha256"] = {str(path.relative_to(source_root.parents[0])): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
            result["cuda_source_sha256"] = {"megakernel": hashlib.sha256(mega.source.encode()).hexdigest(), "mega_inplace": hashlib.sha256(mega_inplace.source.encode()).hexdigest()}
            result["software"] = {"torch": torch.__version__, "cuda": torch.version.cuda}
            result["strict_inductor_options"] = strict_options
            result["trajectory"] = {}
            for name, method in methods.items():
                try:
                    enforce = name.startswith("mega")
                    losses, final_state, diagnostics = _trajectory(method, params, batches, max(args.trajectory_steps, 100), enforce=enforce, report=True)
                    window = max(1, min(len(losses) // 5, len(batches)))
                    result["trajectory"][name] = {**diagnostics, "losses": losses, "initial_cycle_loss": statistics.mean(losses[:window]), "final_cycle_loss": statistics.mean(losses[-window:]), "loss_decreased": statistics.mean(losses[-window:]) < statistics.mean(losses[:window]), "final_parameter_norm": float(torch.cat([p.float().flatten() for p in final_state]).norm())}
                except AssertionError as exc:
                    # Inductor may legally elide low precision casts. Keep its
                    # timing, but preserve the numerical discrepancy in JSON.
                    result["trajectory"][name] = {"passed": False, "failure_count": None, "first_failure_step": None, "max_absolute_error": None, "error": str(exc)}
                    if name.startswith("mega"):
                        raise
                state = tuple(p.clone() for p in params)
                graph = _capture(method, state, batches, args.steps)
                result.setdefault("graphs", {})[name] = {"graph": graph, "state": state}
            # Validate the captured state-feedback path independently from the
            # pure-call trajectory check. Reset is outside the replay timing.
            _, expected_replay_state = _trajectory(lambda p, x, y: step_master(p, x, y, 1e-3), params, batches, args.steps)
            result["replay_correctness"] = {}
            for name in methods:
                state = result["graphs"][name]["state"]
                for dst, src in zip(state, params):
                    dst.copy_(src)
                result["graphs"][name]["graph"].replay()
                try:
                    _close(state, expected_replay_state, torch.float32)
                    result["replay_correctness"][name] = {"passed": True}
                except AssertionError as exc:
                    result["replay_correctness"][name] = {"passed": False, "error": str(exc)}
                    if name.startswith("mega"):
                        raise
            initial = tuple(p.clone() for p in params)
            pairs = [(result["graphs"][name]["graph"], result["graphs"][name]["state"]) for name in methods]
            timings = _paired_time(pairs, initial, args.steps, args.samples, args.warmup)
            result["timings"] = {name: timing for name, timing in zip(methods, timings)}
            eligible = [name for name in result["timings"] if name.startswith("inductor") and result["trajectory"].get(name, {}).get("passed") and result["replay_correctness"].get(name, {}).get("passed")]
            fastest = min(eligible, key=lambda name: result["timings"][name]["median_ms"]) if eligible else None
            result["fastest_passing_inductor"] = fastest
            result["speed_ratio_vs_fastest_passing_inductor"] = ({name: timing["median_ms"] / result["timings"][fastest]["median_ms"] for name, timing in result["timings"].items()} if fastest else None)
            passing_references = [name for name in result["timings"] if not name.startswith("mega") and result["trajectory"].get(name, {}).get("passed") and result["replay_correctness"].get(name, {}).get("passed")]
            fastest_passing_reference = min(passing_references, key=lambda name: result["timings"][name]["median_ms"]) if passing_references else None
            result["fastest_passing_reference"] = fastest_passing_reference
            result["speed_ratio_vs_fastest_passing_reference"] = ({name: timing["median_ms"] / result["timings"][fastest_passing_reference]["median_ms"] for name, timing in result["timings"].items()} if fastest_passing_reference else None)
            all_inductor = min((name for name in result["timings"] if name.startswith("inductor")), key=lambda name: result["timings"][name]["median_ms"])
            result["fastest_all_inductor"] = all_inductor
            result["speed_ratio_vs_fastest_all_inductor"] = {name: timing["median_ms"] / result["timings"][all_inductor]["median_ms"] for name, timing in result["timings"].items()}
            result["bootstrap_vs_fastest_all_inductor"] = _bootstrap_speedup_ci(result["timings"].get("mega_inplace", result["timings"]["megakernel"])["raw_ms"], result["timings"][all_inductor]["raw_ms"])
            result["bootstrap_vs_fastest_passing_inductor"] = (_bootstrap_speedup_ci(result["timings"].get("mega_inplace", result["timings"]["megakernel"])["raw_ms"], result["timings"][fastest]["raw_ms"]) if fastest else None)
            result["bootstrap_vs_fastest_passing_reference"] = (_bootstrap_speedup_ci(result["timings"]["mega_inplace"]["raw_ms"], result["timings"][fastest_passing_reference]["raw_ms"]) if fastest_passing_reference else None)
            result["amortization"] = {}
            if fastest_passing_reference:
                reference_ms = result["timings"][fastest_passing_reference]["median_ms"]
                for name in ("megakernel", "mega_inplace"):
                    compiler_s = result["compilation_seconds"].get(name, 0.0)
                    savings_ms = reference_ms - result["timings"][name]["median_ms"]
                    result["amortization"][name] = {
                        "reference": fastest_passing_reference,
                        "compile_seconds": compiler_s,
                        "steady_state_savings_ms_per_step": savings_ms,
                        "break_even_steps": (compiler_s * 1000.0 / savings_ms if savings_ms > 0 else None),
                        "break_even_defined": savings_ms > 0,
                    }
            if "mega_inplace" in result["timings"] and fastest:
                result["mega_inplace_bootstrap_vs_fastest_inductor"] = _bootstrap_speedup_ci(result["timings"]["mega_inplace"]["raw_ms"], result["timings"][fastest]["raw_ms"])
            result.pop("graphs")
            if all("losses" in result["trajectory"].get(name, {}) for name in ("megakernel", "inductor")):
                result["trajectory"]["loss_match"] = max(abs(a - b) for a, b in zip(result["trajectory"]["megakernel"]["losses"], result["trajectory"]["inductor"]["losses"]))
            mega_passed = all(result["trajectory"].get(name, {}).get("passed") and result["replay_correctness"].get(name, {}).get("passed") for name in methods if name.startswith("mega"))
            result["status"] = "ok" if mega_passed else "error"
        except Exception as exc:
            import traceback
            result.pop("graphs", None)
            result.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return int(result["status"] == "error")


if __name__ == "__main__":
    raise SystemExit(main())
