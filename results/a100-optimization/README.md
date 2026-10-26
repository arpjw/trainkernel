# A100 training megakernel result

Measured 2026-10-26 on one A100 SXM4 80 GB. This release supports a static, explicit PyTorch training subset: FP32 master parameters, BF16 forward and backward matrix compute, FP32 loss and SGD updates. The compiler generated one cooperative CUDA kernel for each complete tested step. It beat the fastest measured Inductor configuration on two small online MLPs and lost decisively on two larger MLPs. This is a scoped research result, not a general PyTorch speedup claim.

## Held-out latency

Each number is the median of three seed-specific medians, in microseconds per forward, backward, and update step. The ratio is megakernel / fastest measured Inductor, so less than 1 is faster. The final column gives the range of seed-specific ratios. B/I/H/O denotes batch, input width, hidden width, and output width.

| Workload | B/I/H/O | Mega in-place | Fastest Inductor | Eager CUDA Graph | Ratio | Ratio range | Raw seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| tiny-online | 4/32/64/16 | 15.28 | 22.85 | 92.05 | **0.669** | 0.663–0.669 | [101](heldout/tiny-online-seed101.json), [211](heldout/tiny-online-seed211.json), [307](heldout/tiny-online-seed307.json) |
| online | 1/128/128/16 | 20.32 | 24.24 | 83.68 | **0.838** | 0.838–0.844 | [101](heldout/online-seed101.json), [211](heldout/online-seed211.json), [307](heldout/online-seed307.json) |
| small | 32/128/256/64 | 66.04 | 22.92 | 92.47 | **2.894** | 2.881–2.903 | [101](heldout/small-seed101.json), [211](heldout/small-seed211.json), [307](heldout/small-seed307.json) |
| medium | 128/512/1024/256 | 255.32 | 42.19 | 121.05 | **6.052** | 5.961–6.090 | [101](heldout/medium-seed101.json), [211](heldout/medium-seed211.json), [307](heldout/medium-seed307.json) |

The online win is 1.19× against the fastest measured Inductor configuration; the tiny-online win is 1.49×. Large matrix products expose the current cooperative kernel's limited GEMM throughput. No automatic fallback to cuBLAS or Inductor is claimed. The [historical native-BF16 report](../a100/README.md) used a different precision and state-feedback policy, so its latencies are not a before/after comparison for this table.

## Correctness and comparison policy

All 12 held-out runs passed 100 cyclic steps against eager PyTorch for **every loss, every parameter gradient, and every FP32 updated parameter**. Both pure and in-place megakernels passed an independent 20-step CUDA Graph state-feedback replay. The captured eager reference also passed every gate. All twelve teacher-target trajectories had lower final-cycle loss than initial-cycle loss; this is a sanity check, not a convergence claim. Numerical tolerances were fixed before the held-out run: FP32 loss and updates `rtol=2e-4, atol=2e-5`; BF16-compute gradients retained in FP32 `rtol=0.04, atol=0.004`. Actual maximum errors and first failure steps are in each JSON.

The fastest-Inductor column includes all four measured variants, even when a variant missed the declared 100-step numerical gate. This avoids making the speed comparison easier by excluding a fast baseline. The eager-rounding Inductor configuration passed on both online seeds 211 and 307; the megakernel was also faster than that correctness-passing Inductor variant there, at ratios 0.839 and 0.838. No Inductor variant passed the tight 100-step gate on all tiny-online seeds; captured eager PyTorch provides the passing reference. Eager-rounding Inductor passed all three small and all three medium seeds, where the megakernel regressed. These are observed tolerance results, not a claim that Inductor is unusable or mathematically incorrect.

Configurations and Inductor modes were selected only on seed 20261025. The final script ran seeds 101, 211, and 307 for **all four predeclared shapes**. Max-autotune Inductor was used on tiny-online, small, and medium. Standard Inductor was used on online after the max-autotune Triton build failed on the calibration run; the [failure JSON](calibration/compare-online-v4.json) is retained. Both modes were compared for the other shapes on the calibration seed. The [calibration directory](calibration/) retains exploratory candidates from earlier source versions and is not held-out evidence.

Each method was captured as a CUDA Graph of 20 stateful steps over four fixed, GPU-resident batches. Pure methods copy updates into the state buffers inside the graph; in-place methods update those buffers in the compiled step. Compilation, capture, reset, and warmup are excluded from CUDA event timing. Each run records 30 interleaved, alternating-order samples per method, plus raw per-step samples, p10/p90, compilation time, source hashes, kernel resources, numerical diagnostics, and a same-index 2,000-draw bootstrap ratio. Those bootstrap intervals describe variation within a run. The table's seed ranges show another source of variation; one GPU with unlocked clocks does not establish general speedups across machines or drivers.

## Device validation and provenance

- [Full A100 test log](evidence/full-gpu-tests-v5.txt): **113 passed**, including FP32/BF16/FP16 training, irregular WMMA tails, split-K row products, reductions, direct and staged commits, side epilogues, and CUDA Graph replay.
- [Memory check on mixed multi-block commits](evidence/memcheck-direct-v5.txt), [synchronization check](evidence/synccheck-direct-v5.txt), and [WMMA tail memory check](evidence/memcheck-wmma-v5.txt): **zero sanitizer errors** in the representative cases checked.
- [Standalone training example](evidence/online-example-v5.txt): 100 stateful graph replays matched eager loss, gradients, and parameters.
- [Profiler record](evidence/profile-tiny-v5.json): one megakernel device kernel and twelve stateful Inductor device kernels in a separate one-step diagnostic run. CUDA Graph launch events were excluded from the device kernel count. The [generated CUDA](evidence/profile-tiny-v5.cu) has SHA-256 `8d0c118d9b829cfce0205dc0670d20300bddcceb8ba4d3088fbe3aa96b32dcb8`, matching the held-out tiny-online in-place kernel.
- [Environment](environment.json): PyTorch 2.8.0+cu128, CUDA toolkit 12.8.93, driver 595.91.07, A100 SXM4 80 GB, and exact hardware identifier. Clocks could not be fixed by this pod's permissions; [clock attempt](evidence/clock-policy.json). Hardware counters were unavailable due to [ERR_NVGPUCTRPERM](evidence/ncu-tiny.csv). No occupancy, throughput, or energy claim is inferred from the diagnostic profiler.

The [suite log](evidence/heldout-suite-v5.txt) and `python -m scripts.summarize_a100 results/a100-optimization/heldout` validate twelve complete runs with one shared compiler and benchmark source hash set. The hashes are embedded in every raw JSON; for example, `megakernel/codegen.py` is `1e453913fd820b91c3f883628dc3535e4bf1b0e11d6f84ea74532827df4eb68f` and `benchmarks/train_loop.py` is `b7c597a717bd653fd2267c10749133770ff8b72024e12dea11809d74b0ddd3b0`.

## Reproduce

On an A100 with the recorded Runpod image, CUDA toolkit, and PyTorch version:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0 TORCHINDUCTOR_COMPILE_THREADS=2
python -m pytest -q
python -m examples.online_training
bash scripts/run_train_loop_suite.sh results/a100-optimization/heldout
```

The benchmark covers a small two-layer ReLU MLP with SGD, static shapes, resident synthetic batches, and explicit `torch.func.grad_and_value` differentiation. It does not include data loading, host-to-device transfers, distributed work, AdamW, attention, dropout, dynamic shapes, or arbitrary PyTorch autograd. H100 and Hopper-specific execution remain untested.
