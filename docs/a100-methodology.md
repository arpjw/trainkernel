# A100 training benchmark methodology

This document defines the final comparison protocol for the static MLP
training compiler. It is a measurement specification, not a performance
claim.

## Workloads and data

The suite reports all four predeclared shapes: tiny-online (B4/I32/H64/O16),
online (B1/I128/H128/O16), small (B32/I128/H256/O64), and medium
(B128/I512/H1024/O256). Each run creates four fixed teacher-generated batches
from a recorded seed. Final validation uses independent seeds 101, 211, and
307. The exploratory tuning seed 20261025 and preliminary seeds 17, 29, and
43 are excluded from final evidence.

Parameters and optimizer state are FP32. Inputs, forward activations, and
backward matrix operations use BF16 casts. The loss is computed in FP32. Every
parameter gradient and every updated parameter is checked against eager
PyTorch for 100 cyclic training steps. This is a teacher-target sanity check,
not a convergence study or a claim about production training quality.

## Compared implementations

Every workload records seven methods: the pure megakernel, the in-place
megakernel with checked direct or staged state outputs, CUDA Graph replay of
eager PyTorch, pure Inductor, in-place Inductor, and eager-rounding strict
variants of both Inductor forms. The strongest all-Inductor result and the
strongest baseline that passes numerical and replay correctness are reported
separately. Baseline accuracy failures remain visible in the JSON and do not
get silently converted into speed wins. The Inductor mode is recorded per run;
the online shape uses standard mode because max-autotune compilation failed on
the calibration run.

## Timing

Each method is captured as a CUDA Graph containing 20 cyclic training steps.
The graph includes the parameter updates and state feedback. Compilation,
capture, state reset, and warmup are outside CUDA event intervals. For each
sample, all methods start from the same parameter state; methods are
interleaved with alternating order to reduce drift from short-term device
effects. The final run requests 30 paired samples per method and records raw
samples, medians, p10, and p90.

The in-run bootstrap resamples same-index timing samples for each method with
deterministic seed 0 and 2,000 draws, then takes the ratio of resampled
medians. It reports ratios where values below 1.0 mean the numerator is
faster. Confidence intervals describe variation in this run; they do not
account for workload, driver, clock, or machine selection.

The JSON also reports compilation amortization against the fastest
correctness-passing reference. `break_even_steps` estimates the number of
steady-state updates needed to recover compilation time from the per-step
saving; it is null when the megakernel is slower or no passing reference is
available. This is a single-run accounting estimate, not a deployment cost
model.

## Limits

Evidence comes from one A100 80 GB pod with unlocked clocks. No hardware
performance counters were collected, so the measurements do not establish
occupancy, instruction throughput, memory bandwidth, or energy efficiency.
They also do not establish H100 behavior, long-run convergence, data-loader
throughput, distributed scaling, or optimizer coverage beyond the explicit SGD
step. All shapes and seeds are retained, including regressions and baseline
accuracy failures.
