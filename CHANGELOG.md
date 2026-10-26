# Changelog

All notable changes are documented here. TrainKernel follows semantic
versioning while its public API is experimental.

## 0.2.0 - 2026-10-26

- Validates the complete compiler and held-out training harness on one NVIDIA
  H100 SXM 80 GB.
- Adds a bounded H100 launch tuner and a separate held-out H100 suite.
- Improves the medium H100 workload from 209.03 us to 191.09 us while keeping
  tiny, online, and small results visible.
- Adds an opt-in SM90a BF16 WGMMA backend with exact-tile, irregular-tail,
  training-trajectory, and CUDA Graph replay validation.
- Preserves shared-source, grouped-K, and register-source WGMMA experiments,
  including the result that every measured WGMMA schedule was slower than the
  selected WMMA path on useful small and medium training workloads.
- Separates first-observed and subsequent setup measurements without assuming
  cold or warm compiler-cache state.

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
