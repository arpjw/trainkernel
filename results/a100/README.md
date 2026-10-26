# A100 validation and benchmarks

The compiler generates a correct single-kernel training step for the tested static MLPs. **No meaningful speed advantage over the strongest Inductor CUDA Graph baseline was established.** The smallest case is near parity; larger cases remain slower. These results support continued compiler research, not a general training speedup claim.

Measured on 2026-10-25 PDT using one NVIDIA A100 SXM4 80 GB, PyTorch 2.8.0+cu128, CUDA toolkit 12.8.93, and driver 595.91.07. The exact image, source revision, file hashes, hardware, and measurement settings are in [environment.json](environment.json).

## Measured step latency

Medians in microseconds. B/I/H/O denotes batch, input width, hidden width, and output width. Each case computes the forward pass, MSE loss, all parameter gradients, and SGD-updated parameters. All values are BF16, including parameters, gradients, loss, and updates.

| Case | B/I/H/O | Eager CUDA Graph | Megakernel CUDA Graph | Inductor CUDA Graph | Inductor with eager rounding, CUDA Graph |
| --- | --- | ---: | ---: | ---: | ---: |
| [micro](micro-bf16.json) | 1/8/8/8 | 53.64 | 10.71 | 10.84 | 10.50 |
| [tiny](tiny-bf16.json) | 4/32/64/16 | 50.73 | 28.94 | 17.14 | 16.80 |
| [small](small-bf16.json) | 32/128/256/64 | 64.34 | 71.24 | 18.84* | 18.57 |
| [medium](medium-bf16.json) | 128/512/1024/256 | 81.91 | 447.49 | 36.14 | 34.64 |

The Inductor graph baselines directly capture optimized `max-autotune-no-cudagraphs` kernels, removing Python wrapper overhead. The last column additionally enables `emulate_precision_casts`, preserving eager-style low-precision rounding. Default `torch.compile(mode="max-autotune")` and uncaptured eager/megakernel timings are also included in every raw JSON. The Python runtime wrapper still has substantial overhead, so captured latency is not the latency of a normal uncaptured Python call.

An asterisk marks a default Inductor result that missed the benchmark's eager-comparison tolerance. The eager-rounding Inductor baseline passed all four cases. PyTorch documents that fusion can omit intermediate low-precision casts; the observed difference is consistent with that policy. See [the PyTorch 2.8 precision setting](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/config.py#L602). Baseline accuracy reports remain in the JSON; benchmark `status: ok` means the megakernel passed and measurements completed, rather than asserting every baseline uses identical rounding.

## Correctness and launch evidence

- [tests.txt](tests.txt): **41 passed on A100**, including FP32/BF16/FP16 training trajectories, irregular matrix tiles, views, casts, graph replay, `nn.Linear` training, and zero-beta `addmm` behavior.
- Every recorded training case checks all nine outputs and a five-step SGD trajectory against eager PyTorch. Written tolerances are included in each JSON. This is not a long-run convergence study.
- [profile.json](profile.json): the small MLP step produces exactly **one GPU kernel event**. The profiled Inductor invocation launches one CUDA Graph containing nine GPU kernel events. Profiling was separate from timed measurements.
- [profile.cu](profile.cu): the generated CUDA for the profiled step, with its SHA256 in the profiler record.

The scheduler materializes 16 operations, inlines 11 operations, and uses nine synchronization boundaries. One-block programs use block barriers. Compatible outputs are written directly into their nine output buffers. Temporary arena reservations are conservative and still include directly placed roots.

## Measurement limits

Timings use CUDA events, 20 samples of 10 step executions, and 10 timing warmup executions after compilation/capture setup. The raw JSON contains p10/p90 as well as medians. Inputs remain fixed during timing; the separate trajectory check advances parameter state. Data loading and copying updated state back into a persistent training-loop buffer are outside these timings.

These are measurements from one pod with unlocked GPU clocks. They do not establish performance across models, shapes, longer training runs, or other architectures. H100 execution has not been tested. The code currently uses basic synchronous WMMA tiles and static synchronization stages; improving GEMM tiling, epilogue fusion, and tile-level dependencies is further work.

## Reproduce

Use the recorded Runpod PyTorch development image, then from the repository root:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=2
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m benchmarks.train_mlp --dtype bfloat16 --batch 1 --input 8 --hidden 8 --output 8 --json results/a100/micro-bf16.json
python -m benchmarks.train_mlp --dtype bfloat16 --batch 4 --input 32 --hidden 64 --output 16 --json results/a100/tiny-bf16.json
python -m benchmarks.train_mlp --dtype bfloat16 --batch 32 --input 128 --hidden 256 --output 64 --json results/a100/small-bf16.json
python -m benchmarks.train_mlp --dtype bfloat16 --batch 128 --input 512 --hidden 1024 --output 256 --json results/a100/medium-bf16.json
python -m scripts.profile_step --json results/a100/profile.json
```
