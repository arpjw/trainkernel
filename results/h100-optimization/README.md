# H100 training megakernel result

Measured 2026-10-26 on one NVIDIA H100 SXM 80 GB. The H100 pass validates the
existing training megakernel, calibrates launch geometry on a separate seed,
and adds a correct opt-in SM90a WGMMA research backend. The selected public
result continues to use WMMA because every tested WGMMA schedule was slower on
the useful small and medium training workloads.

## Held-out latency

Each number is the median of three seed-specific medians, in microseconds per
complete forward, backward, and SGD update. The ratio is TrainKernel divided by
the fastest measured Inductor variant, so less than 1 is faster. B/I/H/O means
batch, input width, hidden width, and output width.

| Workload | B/I/H/O | TrainKernel | Fastest all Inductor | Ratio | Ratio range | Raw seeds |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| tiny-online | 4/32/64/16 | 12.80 us | 15.58 us | **0.824** | 0.815-0.825 | [101](heldout/tiny-online-seed101.json), [211](heldout/tiny-online-seed211.json), [307](heldout/tiny-online-seed307.json) |
| online | 1/128/128/16 | 13.41 us | 21.76 us | **0.616** | 0.616-0.616 | [101](heldout/online-seed101.json), [211](heldout/online-seed211.json), [307](heldout/online-seed307.json) |
| small | 32/128/256/64 | 53.86 us | 18.84 us | 2.859 | 2.848-2.904 | [101](heldout/small-seed101.json), [211](heldout/small-seed211.json), [307](heldout/small-seed307.json) |
| medium | 128/512/1024/256 | 191.09 us | 30.58 us | 6.257 | 6.174-6.332 | [101](heldout/medium-seed101.json), [211](heldout/medium-seed211.json), [307](heldout/medium-seed307.json) |

TrainKernel is 1.21x faster on tiny-online and 1.62x faster on online. It is
2.86x and 6.26x slower on small and medium. The H100 launch calibration reduced
the medium latency from the unchanged-backend baseline of 209.03 us to
191.09 us, about 8.6%, and reduced small from 54.85 us to 53.86 us. Tiny and
online remained within run-to-run variation.

The fastest-all column includes all measured Inductor variants, including a
variant that misses the declared numerical tolerance. Each raw JSON separately
records the fastest passing reference. TrainKernel and captured eager passed
the 100-step trajectory and 20-step replay gates on every seed.

## Classification-head validation

The independent workload uses a one-hot least-squares classification head,
FP32 master weights, BF16 matrix compute, and SGD.

| Seed | TrainKernel in-place | Fastest all Inductor | Ratio | Fastest passing reference |
| ---: | ---: | ---: | ---: | ---: |
| 101 | 12.818 us | 15.673 us | 0.818 | 15.795 us |
| 211 | 12.757 us | 15.582 us | 0.819 | 61.712 us, captured eager |
| 307 | 12.770 us | 15.646 us | 0.816 | 15.754 us |
| **median** | **12.770 us** | **15.646 us** | **0.818** | **15.795 us** |

This is a 1.22x across-seed median win against the fastest measured Inductor
configuration. Raw results are under [classification-head](classification-head/).

## Hopper experiments

The calibration seed was fixed at 20261025 and never appears in the held-out
tables. On that seed, the best WMMA launch measured 56.53 us for small and
190.88 us for medium. The initial shared-source WGMMA schedule measured
93.59 us and 344.73 us. Grouping four K slices measured 92.87 us and
355.29 us. Register-source A measured 104.85 us and 389.76 us despite a 6%
win in the isolated 64x16x64 microprobe. These negative results are preserved
under [the calibration directory](../h100-calibration/).

The WGMMA path is opt-in through `hopper_wgmma=True`, requires compute
capability 9.0 and an SM90a toolchain target, and is experimental. It passed
exact tiles, irregular tails, multi-K accumulation, 100-step training, and
CUDA Graph replay. It is not the selected fast path. TMA, warp specialization,
thread-block clusters, and distributed training are not implemented.

## GPU validation

- The [complete H100 suite](full-gpu-tests.txt) passed **133 tests** in
  26 minutes 28 seconds. It compiled and ran generated CUDA across FP32, BF16,
  and FP16 paths, irregular tensor-core tails, state commits, reductions, and
  CUDA Graph replay.
- The [100-update example](online-training.txt) matched eager loss, gradients,
  and parameters, with one generated training kernel per replay.
- The [standalone WGMMA probe](wgmma-probe.txt) matched PyTorch within
  `9.54e-7` maximum absolute error for both shared-source and register-source A.
  Its 4.08 us versus 3.84 us diagnostic timing is not an end-to-end claim.

## Method and provenance

Each method is captured as a 20-step stateful CUDA Graph over four fixed,
GPU-resident batches. Compilation, capture, reset, warmup, and setup are
excluded from CUDA-event timing. Every run retains 30 paired timing samples,
100-step numerical diagnostics, replay checks, kernel resources, and source
hashes. The held-out files share TrainKernel SM90 source SHA-256
`9736572cb085315ce845260e65df9564a69bc0d0bfb129bf2540d860a7bd7c3a`.

First-observed setup took 29.84 to 34.12 seconds by shape. Subsequent observed
setup medians were 0.21 to 0.24 seconds. Those labels preserve run order and do
not guarantee cold and warm cache state.

The benchmarked functional compiler sources match 0.2.0. Package version
metadata was bumped after GPU timing, so the raw files retain the earlier
`trainkernel/__init__.py` hash. [Environment and source provenance](environment.txt)
records that boundary explicitly. The 0.2.0 wheel and source distribution both
passed [the package metadata check](distribution-check.txt).

Reproduce the selected held-out matrix with:

```bash
bash scripts/run_h100_heldout_suite.sh results/h100-optimization/heldout
python -m scripts.summarize_h100 results/h100-optimization/heldout
```

See [the H100 methodology](../../docs/h100-methodology.md) for the fixed
comparison and interpretation rules.

The SM90a implementation follows NVIDIA's [PTX WGMMA instruction and fragment
layouts](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions)
and the [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/).
