#!/usr/bin/env bash
set -euo pipefail

out_dir=${1:-results/h100-optimization/heldout}
mkdir -p "$out_dir"

# Configurations were selected only on calibration seed 20261025. Tiny and
# online retain the established one-block launch. Small and medium use the
# best WMMA launch from the bounded H100 SXM calibration set. The experimental
# WGMMA path is deliberately excluded because it lost on the calibration run.
for seed in 101 211 307; do
  echo "Running tiny-online seed $seed"
  python -m benchmarks.train_loop --batch 4 --input 32 --hidden 64 --output 16 \
    --max-blocks 1 --threads-per-block 512 --inductor-mode max-autotune-no-cudagraphs \
    --seed "$seed" --samples 30 --json "$out_dir/tiny-online-seed${seed}.json" \
    > "$out_dir/tiny-online-seed${seed}.stdout" 2>&1
  echo "Running online seed $seed"
  python -m benchmarks.train_loop --batch 1 --input 128 --hidden 128 --output 16 \
    --max-blocks 1 --threads-per-block 1024 --split-k-gemv --inductor-mode default \
    --seed "$seed" --samples 30 --json "$out_dir/online-seed${seed}.json" \
    > "$out_dir/online-seed${seed}.stdout" 2>&1
  echo "Running small seed $seed"
  python -m benchmarks.train_loop --batch 32 --input 128 --hidden 256 --output 64 \
    --max-blocks 132 --shared-skew 16 --inductor-mode max-autotune-no-cudagraphs \
    --seed "$seed" --samples 30 --json "$out_dir/small-seed${seed}.json" \
    > "$out_dir/small-seed${seed}.stdout" 2>&1
  echo "Running medium seed $seed"
  python -m benchmarks.train_loop --batch 128 --input 512 --hidden 1024 --output 256 \
    --max-blocks 264 --blocks-per-sm 2 --shared-skew 16 --inductor-mode max-autotune-no-cudagraphs \
    --seed "$seed" --samples 30 --json "$out_dir/medium-seed${seed}.json" \
    > "$out_dir/medium-seed${seed}.stdout" 2>&1
done
python -m scripts.summarize_h100 "$out_dir"
