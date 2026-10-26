# TrainKernel 0.1.0

TrainKernel 0.1.0 is a research release that compiles a supported static
PyTorch training step into one cooperative CUDA kernel on NVIDIA SM80 GPUs.
The supported step can include the forward pass, loss, backward pass, and SGD
update.

## What works

- Typed compile, training, and captured-program APIs.
- FP32, BF16, and FP16 static tensor programs in the documented operator
  subset.
- Stateful CUDA Graph replay with checked direct or transactional updates.
- One generated device kernel for each validated complete training step.
- Apache 2.0 licensing, Python packaging, CPU CI, and reproducible A100
  evidence.

## A100 result

On the held-out suite, TrainKernel beat the fastest measured Inductor
configuration on two small online MLPs and lost on two larger MLPs. An
independent three-seed one-hot least-squares classification head measured
13.07 us for TrainKernel versus 18.63 us for the fastest measured Inductor
configuration at the across-seed median, or 1.43x faster.

The release keeps the important cost visible: a fresh TrainKernel compile and
capture took 53.62 seconds on that classification workload and required an
estimated 7.82 million updates to amortize. This is a scoped steady-state
result, not a claim of general PyTorch acceleration.

## Validation

- 116 GPU tests passed on an independently provisioned A100 SXM4 80 GB.
- 100 stateful example updates matched eager PyTorch.
- 65 CPU tests passed, with 51 CUDA-only tests skipped on the CPU runner.
- The wheel and source distribution passed `twine check`.

See the [release evidence](https://github.com/arpjw/trainkernel/blob/main/results/release-validation/README.md),
[optimization report](https://github.com/arpjw/trainkernel/blob/main/results/a100-optimization/README.md),
and [documented limitations](https://github.com/arpjw/trainkernel#supported-subset).
