# H100 baseline methodology

This document defines the first Hopper validation pass. It is a compatibility
and measurement protocol, not a claim that the A100 backend is optimized for
H100.

## Baseline

The baseline runs the unchanged TrainKernel compiler on one NVIDIA H100 80 GB
SXM GPU. It records Python, PyTorch, CUDA, driver, device capability, device
count, and the exact git revision before running any benchmark. The full GPU
test suite and the 100-update online-training example must pass before timing
results are considered valid.

The four-shape matrix is the same as the A100 report: tiny-online
(B4/I32/H64/O16), online (B1/I128/H128/O16), small (B32/I128/H256/O64), and
medium (B128/I512/H1024/O256), with seeds 101, 211, and 307. It uses the same
30 paired CUDA-event samples, 20-step captured graphs, correctness gates, and
Inductor variants. A separate three-seed classification-head run uses one-hot
teacher labels, mean squared error, BF16 matrix compute, FP32 master weights,
and SGD.

Compilation, capture, warmup, reset, and CUDA Graph setup remain outside timed
CUDA-event intervals. Setup amortization is reported separately and includes
compilation plus graph capture. Cold and warm compiler-cache states must not be
combined into a single deployment-cost claim.

## Hopper optimization pass

Launch and tile candidates are selected only on seed 20261025. The selected
H100 SXM matrix keeps the existing tiny and online configurations, uses 132
blocks with shared skew 16 for small, and uses 264 blocks at two blocks per SM
with shared skew 16 for medium. Held-out seeds remain 101, 211, and 307.

The opt-in `hopper_wgmma=True` path targets SM90a and BF16. It must pass exact
tiles, irregular tails, multi-K accumulation, training trajectories, and CUDA
Graph replay before timing is considered. Shared-source, grouped-K, and
register-source-A schedules are compared only on the calibration seed. A
Hopper-specific implementation is not selected merely because it is correct.
The chosen public configuration must be the fastest validated candidate for
the useful workload.

Stage instrumentation changes generated code and is diagnostic only. It may
locate bottlenecks but cannot support latency claims. Held-out claims use the
uninstrumented paired CUDA-event harness.

## Interpretation

Running the WMMA/cooperative backend on H100 establishes portability and a
before-measurement. The experimental WGMMA path establishes correct SM90a
execution, not a speedup. Tensor Memory Accelerator, warp specialization, and
thread-block-cluster features remain unimplemented. Any speed comparison must
retain the strongest measured Inductor result, even if a baseline fails the
declared numerical tolerance; passing references are reported separately.

The baseline is a gate for implementation work. A Hopper backend should first
preserve the A100 correctness suite, then add architecture-specific kernels
behind explicit capability dispatch. A positive release claim requires
correctness across 100-step trajectories and replay, held-out seeds, raw
timings, setup accounting, and unchanged A100 evidence.
