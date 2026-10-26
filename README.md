# Megakernel

Megakernel compiles a small, explicit PyTorch training step into one cooperative CUDA kernel. The generated kernel can include the forward pass, loss, gradients, and SGD update, so a replayed step has no per-operator Python launch overhead.

The current backend schedules operators in topological order, uses grid barriers for dependencies, inlines single-consumer pointwise operations, and reuses temporary storage after its last reader. BF16 and FP16 matrix multiplications use WMMA tensor cores. FP32 uses a tiled SIMT implementation for accuracy.

This is a scoped research release for static functional training on NVIDIA SM80 GPUs. On one A100, the [held-out training report](results/a100-optimization/README.md) measured 15.28 versus 22.85 μs per step for a tiny online MLP and 20.32 versus 24.24 μs for a second online MLP, comparing the in-place megakernel with the fastest measured Inductor variant. The two larger tested MLPs were slower than Inductor. It is not a general PyTorch compiler and does not claim universal speedups.

## Install

Linux, Python 3.10+, PyTorch 2.8+, an NVIDIA SM80+ GPU, a CUDA toolkit with `nvcc`, and a C++ compiler are required for generated kernel execution. CPU capture and IR checks run without CUDA.

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m examples.online_training
```

The backend targets SM80+, including A100 and H100. Release evidence is A100-only; H100 has not been tested and Hopper TMA/WGMMA paths are not implemented.

## Compile a training step

```python
import torch
import megakernel

def train_step(weights, x, target):
    def loss(w):
        return ((torch.relu(x @ w) - target) ** 2).mean()
    gradients, value = torch.func.grad_and_value(loss)(weights)
    return value, gradients, weights - 1e-3 * gradients

compiled = megakernel.compile_training(train_step, (weights, x, target))
loss, gradients, weights = compiled(weights, x, target)
```

`compile_training` verifies that gradients match state structure and derives the update mapping, including nested state trees. The supplied state must be detached from autograd because the training function computes its own gradients. `compiled.capture(...)` creates a CUDA Graph after warmup, restores the supplied state after warmup and capture, and returns a reusable `CapturedProgram`; replay outputs borrow captured buffers with fixed shapes and addresses. See the [100-update example](examples/online_training.py).

Parameters may be nested dictionaries or tuples. Inputs and operations must stay on one device; resolve lazy negative or conjugate input views before compiling. Updated parameters are returned explicitly. The compiler proves direct last-read state commits where safe and stages all other updates transactionally; input mutation and arbitrary aliasing remain unsupported. Different shapes, strides, devices, or dtypes require another compilation.

Inspect the compiler without a GPU:

```python
program, output_tree = megakernel.capture(train_step, cpu_examples)
from megakernel.codegen import generate
cuda_source = generate(program).source
```

## Benchmark

```bash
bash scripts/run_train_loop_suite.sh results/a100-optimization/heldout
```

The [training-loop harness](benchmarks/train_loop.py) uses FP32 master weights, BF16 compute, FP32 loss, and FP32 SGD updates. It checks every loss, gradient, and updated parameter for 100 steps and separately checks a captured 20-step replay. Timed graphs include state feedback over four resident batches. They compare pure and in-place megakernels, captured eager PyTorch, and default and eager-rounding Inductor variants. CUDA event timings exclude compilation, capture, reset, and warmup, and retain raw samples. Numerical failures remain visible for all baselines. The [methodology](docs/a100-methodology.md) explains the held-out seeds and fixed tolerance policy; the [result](results/a100-optimization/README.md) includes both wins and regressions. The older [native-BF16 fixed-step harness](benchmarks/train_mlp.py) remains available for the historical report.

## Supported subset

- FP32, BF16, and FP16 strided tensors with static shapes.
- Matrix multiplication and `addmm`, broadcast arithmetic, ReLU and its backward, scalar powers, sums and means, dtype casts, fills, and supported storage views.
- Pure tensor functions, including explicit differentiation through `torch.func.grad_and_value`.

Unsupported operations produce an error. There is no silent eager fallback. Attention, dropout/RNG, convolutions, dynamic shapes, arbitrary input mutation, distributed collectives, and asynchronous task-queue scheduling are not implemented.

Temporary values use FP32 workspace storage with explicit rounding at low-precision boundaries. Fusion reduces materialization but does not keep an entire model on-chip. The tested larger GEMMs run faster in Inductor; the report keeps those regressions visible.

## Design and evidence

See [compiler design](docs/design.md), [A100 evidence](results/a100-optimization/README.md), and [GPU access record](docs/infrastructure.md). CPU tests validate IR semantics, views, reductions, diagnostics, and arena lifetimes. The complete A100 suite passed 113 tests that compile and execute generated CUDA, compare training outputs across precisions, and exercise CUDA Graph replay.
