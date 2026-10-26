# Megakernel

Experimental compiler that turns a static, functional PyTorch training step into one persistent cooperative CUDA kernel. The step can include the model forward pass, loss, gradients, and SGD parameter updates.

The current backend schedules operators in topological order, uses grid barriers for dependencies, inlines single-consumer pointwise operations, and reuses temporary storage after its last reader. BF16 and FP16 matrix multiplications use WMMA tensor cores. FP32 uses a tiled SIMT implementation for accuracy.

This is an initial compiler research prototype. Performance results belong in `results/` and are not implied by reducing launch count.

## Install

Linux, Python 3.10+, PyTorch 2.6+, an NVIDIA SM80+ GPU, a compatible CUDA development toolkit with `nvcc`, and a C++ compiler are required for generated kernel execution. CPU capture and IR reference checks also run on macOS.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m examples.train
```

Both A100 and H100 are supported by the generic SM80+ code path. Hopper-specific TMA/WGMMA scheduling is not implemented.

## Compile a training step

```python
import torch
import megakernel

def train_step(weights, x, target):
    def loss(w):
        return ((torch.relu(x @ w) - target) ** 2).mean()
    gradients, value = torch.func.grad_and_value(loss)(weights)
    return value, gradients, weights - 1e-3 * gradients

compiled = megakernel.compile(train_step, (weights, x, target))
loss, gradients, weights = compiled(weights, x, target)
```

`examples/train.py` uses `torch.nn.Linear` modules through `torch.func.functional_call`. Parameters may be nested dictionaries or tuples. Updated parameters are returned explicitly; inputs are not mutated. The result does not register a new autograd function. Different shapes, strides, devices, or dtypes require another compilation.

Inspect the compiler without a GPU:

```python
program, output_tree = megakernel.capture(train_step, cpu_examples)
from megakernel.codegen import generate
cuda_source = generate(program).source
```

## Benchmark

```bash
python -m benchmarks.train_mlp --dtype bfloat16 \
  --batch 32 --input 128 --hidden 256 --output 64 \
  --json results/a100-small-bf16.json
```

The harness checks every loss, parameter gradient, and updated parameter against eager PyTorch, then compares a repeated SGD trajectory. It reports fixed-input step latency for eager, eager with CUDA Graphs, the megakernel, the megakernel with CUDA Graphs, and `torch.compile(mode="max-autotune")`, which requests Inductor's CUDA Graph optimization. Timings exclude compilation, use CUDA events, and include median/p10/p90 over repeated samples. Graph timing keeps inputs fixed; the separate trajectory check advances training state.

Use identical shapes and precision when comparing results. `--skip-inductor` is a debugging option and does not establish an advantage over compiled PyTorch.

## Supported subset

- FP32, BF16, and FP16 strided tensors with static shapes.
- Matrix multiplication and `addmm`, broadcast arithmetic, ReLU and its backward, scalar powers, sums and means, dtype casts, fills, and supported storage views.
- Pure tensor functions, including explicit differentiation through `torch.func.grad_and_value`.

Unsupported operations produce an error. There is no silent eager fallback. Attention, dropout/RNG, convolutions, dynamic shapes, arbitrary input mutation, distributed collectives, and asynchronous task-queue scheduling are not implemented.

Temporary values currently use FP32 workspace storage with explicit rounding at low-precision tensor boundaries. Fusion reduces materialization but does not keep an entire training model on-chip. Large GEMMs can be much faster in cuBLAS/Inductor than in these initial tile implementations.

## Design and evidence

See `docs/design.md` for the execution model and `docs/infrastructure.md` for the GPU access record. CPU tests validate full training IR semantics, views, reductions, diagnostics, and arena lifetimes. GPU tests compile and execute generated CUDA, compare gradients and parameter updates across precisions, and exercise CUDA Graph replay.
