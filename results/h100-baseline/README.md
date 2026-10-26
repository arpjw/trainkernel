# H100 unchanged-backend baseline

Measured 2026-10-26 on one NVIDIA H100 SXM 80 GB at git revision
`2a1188bc9a7cf36ce434f616d663e55e42c48274`. This run used the unchanged
A100-selected WMMA backend. It establishes H100 compatibility and a fixed
before-measurement for Hopper work. It does not claim Hopper-specific
optimization.

Each latency is the median of three seed-specific medians for a complete
forward, backward, and SGD update. Ratios are TrainKernel divided by the
fastest measured Inductor variant, so less than 1 is faster.

| Workload | B/I/H/O | TrainKernel | Fastest all Inductor | Ratio | Seed ratio range |
| --- | --- | ---: | ---: | ---: | ---: |
| tiny-online | 4/32/64/16 | 12.74 us | 15.49 us | **0.827** | 0.822-0.828 |
| online | 1/128/128/16 | 13.29 us | 21.56 us | **0.616** | 0.616-0.617 |
| small | 32/128/256/64 | 54.85 us | 18.80 us | 2.921 | 2.915-2.938 |
| medium | 128/512/1024/256 | 209.03 us | 30.35 us | 6.892 | 6.863-6.931 |

All 12 runs passed the 100-step TrainKernel trajectory and 20-step CUDA Graph
replay checks. The fastest-all comparison retains Inductor variants that miss
the fixed numerical gate. Each JSON also identifies the fastest passing
reference. The full GPU suite passed 118 tests before the timing run, and the
100-update online example passed.

Raw results are under [heldout](heldout/). The original console logs,
[environment](environment.txt), [full GPU test output](full-gpu-tests.txt),
[online example](online-training.txt), and [distribution check](distribution-check.txt)
are retained beside them.

First-observed setup took about 31 seconds per shape. Subsequent observed setup
medians were 0.18 to 0.23 seconds. These describe run order, not guaranteed
cold and warm cache states. Setup is excluded from CUDA-event latency.
