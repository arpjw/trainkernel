# Benchmark harnesses

`train_mlp.py` measures one fixed-input training step and is useful for
operator-level latency comparisons. `train_loop.py` measures a stateful loop:
it cycles through seeded batches and feeds updated parameters into the next step.
Pure methods copy updates inside each captured CUDA graph; in-place methods
commit directly or use their own staged copy. The
reported latency is per step across a multi-step replay. Compilation and graph
capture are reported outside the timed region.

The loop benchmark compares the megakernel with raw `torch.compile` Inductor
kernels and eager PyTorch, all captured in equivalent CUDA graphs. It checks
every loss, gradient, and updated parameter against eager PyTorch for the
complete trajectory. Default and eager-rounding Inductor configurations are
reported separately, including numerical failures; the harness never relaxes
tolerances to manufacture a speedup.

The predeclared workload set is: `tiny-online` (B4/I32/H64/O16), `online`
(B1/I128/H128/O16), `small` (B32/I128/H256/O64), and `medium`
(B128/I512/H1024/O256). Results should be collected for every workload rather
than selecting only the fastest case.

The same harness can model an online personalization or classification head by
using one-hot teacher targets. This keeps the compiler contract unchanged while
making the target semantics explicit:

```bash
python -m benchmarks.train_loop --task classification-head \
  --batch 32 --input 128 --hidden 256 --output 16 \
  --max-blocks 108 --samples 30 --json results/a100/classification-head.json
```

The workload reports compile time, steady-state latency, and the estimated
break-even update count for recovering compilation time against the fastest
correctness-passing reference. It remains a synthetic SGD sanity check, not a
claim about a production recommender or classifier.

Example on a CUDA host:

```bash
python -m benchmarks.train_loop \
  --batch 128 --input 512 --hidden 1024 --output 256 \
  --steps 20 --json results/a100/train-loop-medium.json
```

Run the full predeclared suite across three held-out seeds with
`scripts/run_train_loop_suite.sh results/a100/train-loop`. The JSON labels the
raw Inductor baseline as `inductor` and its eager-rounding variant as
`inductor_strict`; baseline numerical mismatches remain visible without
invalidating the megakernel correctness gate.

This harness uses FP32 master weights with BF16 compute and checks losses,
gradients, and updated master weights. It is an SGD benchmark. It does not
claim AdamW behavior or data-loader throughput; the 100-step trajectory is a
sanity check for loss behavior, not a convergence study.
