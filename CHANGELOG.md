# Changelog

All notable changes are documented here. TrainKernel follows semantic
versioning while its public API is experimental.

## 0.1.0 - 2026-10-26

- Compiles a supported static functional PyTorch training step into one
  cooperative CUDA kernel.
- Supports FP32, BF16, and FP16 tensors on NVIDIA SM80 or newer GPUs.
- Provides typed `CompiledFunction`, `CompiledTrainingStep`, and
  `CapturedProgram` interfaces.
- Checks loss, every gradient, and every updated parameter across stateful
  training trajectories.
- Publishes held-out A100 comparisons against eager PyTorch and Inductor,
  including workloads where TrainKernel is slower.
- Includes a one-hot least-squares classification-head workload and end-to-end
  setup break-even reporting.
