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
The [held-out report](https://github.com/arpjw/trainkernel/blob/main/results/a100-optimization/README.md) publishes all seeds,
raw timings, correctness checks, baseline failures, and the two regressions.

The H100 validation keeps the same comparison policy and adds an
architecture-specific launch calibration:

| H100 workload | TrainKernel | Fastest measured Inductor | TrainKernel / Inductor |
| --- | ---: | ---: | ---: |
| tiny online MLP | 12.80 μs | 15.58 μs | **0.824, 1.21x faster** |
| online MLP | 13.41 μs | 21.76 μs | **0.616, 1.62x faster** |
| small MLP | 53.86 μs | 18.84 μs | 2.859, slower |
| medium MLP | 191.09 μs | 30.58 μs | 6.257, slower |

The [H100 report](https://github.com/arpjw/trainkernel/blob/main/results/h100-optimization/README.md)
retains all 12 held-out runs, the classification-head validation, and rejected
WGMMA experiments. An opt-in SM90a WGMMA backend is correct across tiles,
tails, training trajectories, and graph replay, but it is slower on the tested
small and medium workloads and is not the selected path.

An independent release run adds a one-hot least-squares classification head.
Across three seeds, TrainKernel's in-place path measured 13.07 us versus
18.63 us for the fastest measured Inductor configuration at the median, or
1.43x faster. The [release evidence](https://github.com/arpjw/trainkernel/blob/main/results/release-validation/README.md)
retains all raw samples and correctness failures. Its cold-cache setup took
53.62 seconds and required an estimated 7.82 million updates to amortize, so
this is a steady-state small-workload result rather than a claim about
short-lived jobs.

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

The default backend targets SM80+, with measured A100 and H100 evidence. The
experimental BF16 WGMMA path requires an H100-class compute capability 9.0 GPU,
CUDA 12.8 in the validated environment, and explicit `hopper_wgmma=True`.
Hopper TMA, warp specialization, and thread-block clusters are not implemented.

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
with fixed shapes and addresses. See the [100-update example](https://github.com/arpjw/trainkernel/blob/main/examples/online_training.py).

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

For a clean release-candidate check, including the useful classification-head
workload and package build:

```bash
bash scripts/run_release_validation.sh results/release-validation
```

The [training-loop harness](https://github.com/arpjw/trainkernel/blob/main/benchmarks/train_loop.py) uses FP32 master weights, BF16 compute, FP32 loss, and FP32 SGD updates. It checks every loss, gradient, and updated parameter for 100 steps and separately checks a captured 20-step replay. Timed graphs include state feedback over four resident batches. They compare pure and in-place megakernels, captured eager PyTorch, and default and eager-rounding Inductor variants. CUDA event timings exclude compilation, capture, reset, and warmup, and retain raw samples. Numerical failures remain visible for all baselines. The [methodology](https://github.com/arpjw/trainkernel/blob/main/docs/a100-methodology.md) explains the held-out seeds and fixed tolerance policy; the [result](https://github.com/arpjw/trainkernel/blob/main/results/a100-optimization/README.md) includes both wins and regressions. The older [native-BF16 fixed-step harness](https://github.com/arpjw/trainkernel/blob/main/benchmarks/train_mlp.py) remains available for the historical report.

## Supported subset

- FP32, BF16, and FP16 strided tensors with static shapes.
- Matrix multiplication and `addmm`, broadcast arithmetic, ReLU and its backward, scalar powers, sums and means, dtype casts, fills, and supported storage views.
- Pure tensor functions, including explicit differentiation through `torch.func.grad_and_value`.

Unsupported operations produce an error. There is no silent eager fallback. Attention, dropout/RNG, convolutions, dynamic shapes, arbitrary input mutation, distributed collectives, and asynchronous task-queue scheduling are not implemented.

Temporary values use FP32 workspace storage with explicit rounding at low-precision boundaries. Fusion reduces materialization but does not keep an entire model on-chip. The tested larger GEMMs run faster in Inductor; the report keeps those regressions visible.

## Design and evidence

See [compiler design](https://github.com/arpjw/trainkernel/blob/main/docs/design.md), [A100 optimization evidence](https://github.com/arpjw/trainkernel/blob/main/results/a100-optimization/README.md), [H100 evidence](https://github.com/arpjw/trainkernel/blob/main/results/h100-optimization/README.md), [independent release validation](https://github.com/arpjw/trainkernel/blob/main/results/release-validation/README.md), and [GPU access record](https://github.com/arpjw/trainkernel/blob/main/docs/infrastructure.md). CPU tests validate IR semantics, views, reductions, diagnostics, and arena lifetimes. GPU suites compile and execute generated CUDA, compare training outputs across precisions, and exercise CUDA Graph replay.

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

Apache License 2.0. See [LICENSE](https://github.com/arpjw/trainkernel/blob/main/LICENSE) and [NOTICE](https://github.com/arpjw/trainkernel/blob/main/NOTICE).
