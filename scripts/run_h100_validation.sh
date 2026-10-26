#!/usr/bin/env bash
set -euo pipefail

# Baseline H100 validation. This deliberately reuses the A100 comparison
# protocol before any Hopper-specific code is enabled.
out_dir=${1:-results/h100-baseline}
mkdir -p "$out_dir/heldout" "$out_dir/classification-head"

python - <<'PY' > "$out_dir/environment.txt"
import platform
import subprocess

import torch
import trainkernel

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for H100 validation")
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("trainkernel:", trainkernel.__version__)
print("device:", torch.cuda.get_device_name())
print("capability:", torch.cuda.get_device_capability())
print("device_count:", torch.cuda.device_count())
print("git:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >> "$out_dir/environment.txt"
nvcc_bin=${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}
"$nvcc_bin" --version >> "$out_dir/environment.txt"

python -m pytest -q | tee "$out_dir/full-gpu-tests.txt"
python -m examples.online_training | tee "$out_dir/online-training.txt"

# The four-shape matrix and seed policy are shared with the A100 held-out run.
bash scripts/run_train_loop_suite.sh "$out_dir/heldout" | tee "$out_dir/heldout-suite.txt"

for seed in 101 211 307; do
  python -m benchmarks.train_loop \
    --task classification-head \
    --batch 4 --input 32 --hidden 64 --output 16 \
    --max-blocks 1 --threads-per-block 512 \
    --inductor-mode max-autotune-no-cudagraphs \
    --seed "$seed" --samples 30 \
    --json "$out_dir/classification-head/seed${seed}.json" \
    > "$out_dir/classification-head/seed${seed}.stdout" 2>&1
done

python -m build --outdir "$out_dir/distributions"
python -m twine check "$out_dir"/distributions/* | tee "$out_dir/distribution-check.txt"
