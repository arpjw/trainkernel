# TrainKernel

TrainKernel compiles a supported static PyTorch training step into one
cooperative CUDA megakernel. The generated kernel can include the forward pass,
loss, gradients, and SGD update, so a captured step has no per-operator Python
launch overhead.

The current backend schedules operators in topological order, uses grid barriers for dependencies, inlines single-consumer pointwise operations, and reuses temporary storage after its last reader. BF16 and FP16 matrix multiplications use WMMA tensor cores. FP32 uses a tiled SIMT implementation for accuracy.

This is a scoped v0.1 research release for static functional training on NVIDIA
SM80 GPUs. It is not a general PyTorch compiler and does not claim universal
speedups.

| A100 workload | TrainKernel | Fastest measured Inductor | TrainKernel / Inductor |
| --- | ---: | ---: | ---: |
| tiny online MLP | 15.28 μs | 22.85 μs | **0.669, 1.49x faster** |
| online MLP | 20.32 μs | 24.24 μs | **0.838, 1.19x faster** |
| small MLP | 66.04 μs | 22.92 μs | 2.894, slower |
| medium MLP | 255.32 μs | 42.19 μs | 6.052, slower |

These are complete forward, backward, and update steps captured as CUDA Graphs.
The [held-out report](results/a100-optimization/README.md) publishes all seeds,
raw timings, correctness checks, baseline failures, and the two regressions.

## Install

Linux, Python 3.10+, PyTorch 2.8+, an NVIDIA SM80+ GPU, a CUDA toolkit with `nvcc`, and a C++ compiler are required for generated kernel execution. CPU capture and IR checks run without CUDA.

```bash
python -m pip install trainkernel
```

For development:

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
import trainkernel

def train_step(weights, x, target):
    def loss(w):
        return ((torch.relu(x @ w) - target) ** 2).mean()
    gradients, value = torch.func.grad_and_value(loss)(weights)
    return value, gradients, weights - 1e-3 * gradients

compiled = trainkernel.compile_training(train_step, (weights, x, target))
loss, gradients, weights = compiled(weights, x, target)
```

`compile_training` returns a typed `CompiledTrainingStep`. It verifies that
gradients match state structure and derives the update mapping, including nested
state trees. The supplied state must be detached from autograd because the
training function computes its own gradients. `compiled.capture(...)` creates a
CUDA Graph after warmup, restores the supplied state after warmup and capture,
and returns a reusable `CapturedProgram`; replay outputs borrow captured buffers
with fixed shapes and addresses. See the [100-update example](examples/online_training.py).

Parameters may be nested dictionaries or tuples. Inputs and operations must stay on one device; resolve lazy negative or conjugate input views before compiling. Updated parameters are returned explicitly. The compiler proves direct last-read state commits where safe and stages all other updates transactionally; input mutation and arbitrary aliasing remain unsupported. Different shapes, strides, devices, or dtypes require another compilation.

Inspect the compiler without a GPU:

```python
program, output_tree = trainkernel.capture(train_step, cpu_examples)
from trainkernel.codegen import generate
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

## Related work and roadmap

Most adjacent public megakernel systems currently focus on inference:
[HazyResearch Megakernels](https://github.com/HazyResearch/Megakernels) targets
low-latency Llama inference, [MegaBake](https://github.com/megabake/megabake)
captures PyTorch inference workloads for SM90, [AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel)
generates Llama-family inference kernels, and [Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219)
targets multi-GPU model inference. TrainKernel instead explores compiling the
stateful forward, loss, backward, and optimizer update of a small training step
on A100 into one kernel. It does not claim priority for megakernel compilation
in general.

The next engineering priorities are a measured shape-aware dispatcher that can
delegate GEMM-heavy steps to Inductor, an experimental `torch.compile` backend
after AOTAutograd, AdamW, and additional real online-learning workloads. These
are roadmap items, not v0.1 capabilities.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
