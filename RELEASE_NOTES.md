# TrainKernel 0.2.0

TrainKernel 0.2.0 extends the research compiler's measured scope from A100 to
H100. It compiles a supported static PyTorch training step into one cooperative
CUDA kernel. The supported step can include the forward pass, loss, backward
pass, and SGD update.

## What works

- Typed compile, training, and captured-program APIs.
- FP32, BF16, and FP16 static tensor programs in the documented operator
  subset.
- Stateful CUDA Graph replay with checked direct or transactional updates.
- One generated device kernel for each validated complete training step.
- Apache 2.0 licensing, Python packaging, CPU CI, and reproducible A100 and
  H100 evidence.

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

## H100 result

On the held-out H100 suite, TrainKernel measured 12.80 us versus 15.58 us for
the fastest measured Inductor variant on tiny-online, and 13.41 us versus
21.76 us on online. It remained slower on small and medium at 53.86 us versus
18.84 us and 191.09 us versus 30.58 us. The H100-specific launch calibration
improved medium by about 8.6% from the unchanged-backend baseline.

The independent H100 classification head measured 12.77 us for TrainKernel
versus 15.65 us for the fastest measured Inductor configuration at the
across-seed median, or 1.22x faster.

## Experimental SM90a backend

This release adds an opt-in BF16 WGMMA path for compute capability 9.0. It is
correct across exact tiles, irregular tails, multi-K accumulation, 100-step
training, and CUDA Graph replay. It is not the default because the tested
WGMMA schedules were slower than the selected WMMA path on the small and
medium training workloads. The negative grouped-K and register-source results
are published with the calibration evidence.

See the [H100 report](https://github.com/arpjw/trainkernel/blob/main/results/h100-optimization/README.md)
and [H100 methodology](https://github.com/arpjw/trainkernel/blob/main/docs/h100-methodology.md).
The [A100 release evidence](https://github.com/arpjw/trainkernel/blob/main/results/release-validation/README.md),
[A100 optimization report](https://github.com/arpjw/trainkernel/blob/main/results/a100-optimization/README.md),
and [documented limitations](https://github.com/arpjw/trainkernel#supported-subset)
remain part of the release record.
