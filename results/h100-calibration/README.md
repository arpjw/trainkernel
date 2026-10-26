# H100 calibration evidence

These files are exploratory measurements on seed 20261025. They select launch
geometry and reject slower Hopper experiments. They are not held-out evidence.

## Selected launch candidates

| Workload | Selected options | Calibration latency |
| --- | --- | ---: |
| tiny-online | 1 block, 512 threads | 12.77 us |
| online | 1 block, 1024 threads, split-K GEMV | 13.34 us |
| small | 132 blocks, 128 threads, shared skew 16 | 56.53 us |
| medium | 264 blocks, 2 blocks per SM, shared skew 16 | 190.88 us |

The unchanged A100-selected medium launch measured 208.43 us on the same
calibration seed. The selected 190.88 us candidate improved it by about 8.4%
before held-out validation. See [tuning](tuning/) for every candidate, raw
sample, correctness check, resource record, and source hash.

## Rejected WGMMA schedules

| Schedule | Small | Medium | Result |
| --- | ---: | ---: | --- |
| shared-source, one K slice per group | 93.59 us | 344.73 us | correct, slower |
| shared-source, four K slices per group | 92.87 us | 355.29 us | correct, slower |
| register-source A | 104.85 us | 389.76 us | correct, slower |

The grouped-K JSON files and register-source JSON files are under
[rejected](rejected/). The isolated register-source microprobe improved a
single 64x16x64 kernel from 4.08 us to 3.84 us, but the useful training
workloads regressed. It was therefore not selected.

## Stage diagnostics

The [profiles](profiles/) directory records instrumented per-stage timings for
small and medium WMMA and WGMMA kernels. Instrumentation changes generated code,
so these profiles are diagnostic only and do not support latency claims. They
showed the WGMMA matrix stages, not the surrounding pointwise work, as the
dominant regression.
