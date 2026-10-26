# A100 release validation

This directory records the independent release-candidate run for TrainKernel
0.1.0 at git revision `2ab10aac1e91bb69d6b86d3d76d3ada4d57a115f`.
The run used an NVIDIA A100 SXM4 80 GB in a second Secure Cloud data center,
separate from the machine used for the optimization report. It used PyTorch
2.8.0+cu128, CUDA toolkit 12.8.93, and driver 595.91.07.

The complete GPU suite passed **116 tests**. The example then ran 100 stateful
updates that matched eager PyTorch, with each captured replay executing one
generated CUDA training kernel. The wheel and source distribution both passed
`twine check`.

## Classification-head workload

The release workload is a small resident two-layer classification head trained
with FP32 master weights, BF16 matrix compute, one-hot teacher labels, mean
squared error, and SGD. It is not a cross-entropy benchmark or a claim about
end-to-end production classifier throughput.

Each method executes a complete forward, backward, and update step. Timings are
medians of 30 paired CUDA-event samples after warmup. The fastest-all Inductor
column retains the quickest measured Inductor result even when that result
missed the predeclared 100-step numerical tolerance.

| Seed | TrainKernel in-place | Fastest-all Inductor | Ratio | Fastest passing reference |
| ---: | ---: | ---: | ---: | --- |
| 101 | 13.068 us | 18.629 us | 0.701 | strict in-place Inductor, 19.391 us |
| 211 | 15.376 us | 22.146 us | 0.694 | captured eager, 92.510 us |
| 307 | 12.546 us | 17.970 us | 0.698 | strict in-place Inductor, 18.700 us |
| **median** | **13.068 us** | **18.629 us** | **0.698** | **19.391 us** |

TrainKernel was 1.43x faster than the fastest measured Inductor configuration
at the across-seed median. TrainKernel and captured eager passed the full
100-step trajectory and 20-step replay checks on every seed. Strict Inductor
passed on seeds 101 and 307; no measured Inductor variant passed the declared
tolerance on seed 211. The speed comparison still includes the faster failing
variants so that numerical filtering cannot make TrainKernel's result easier.

## Setup and amortization

Setup is compilation plus CUDA Graph capture for both TrainKernel and the
fastest correctness-passing reference. The first seed used a cold TrainKernel
extension cache: setup took 53.624 seconds versus 4.164 seconds for strict
Inductor, giving an estimated break-even of about 7.82 million updates. Seeds
211 and 307 reused compiled artifacts and measured 0.301 and 0.254 seconds of
TrainKernel setup. Their estimates were 1,515 and zero updates, respectively.

These values are cache-state-sensitive accounting estimates, not deployment
guarantees. The cold-cache result is the conservative number for a fresh
machine and is kept prominent even though steady-state execution is faster.

## Files

- `full-gpu-tests.txt`: complete test-suite summary.
- `online-training.txt`: stateful 100-update example summary.
- `classification-head/seed*.json`: configuration, raw timing samples,
  bootstrap intervals, correctness records, source hashes, resource use, and
  setup accounting for every seed.
- `classification-head/seed*.stdout`: the benchmark's original console record.
- `environment.txt`: software, hardware, package version, and exact git SHA.
- `distribution-check.txt`: wheel and source-distribution metadata checks.

The temporary validation pod had no persistent volume. Evidence was copied
before teardown, the pod was terminated, and a follow-up provider read returned
not found.
