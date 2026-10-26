"""Validate the predeclared A100 result matrix and print a compact table."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


CASES = ("tiny-online", "online", "small", "medium")
SHAPES = ((4, 32, 64, 16), (1, 128, 128, 16), (32, 128, 256, 64), (128, 512, 1024, 256))
SEEDS = (101, 211, 307)
METHODS = (
    "megakernel", "mega_inplace", "eager_graph", "inductor",
    "inductor_strict", "inductor_inplace", "inductor_inplace_strict",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = []
    source = None
    for case, shape in zip(CASES, SHAPES):
        measurements = []
        for seed in SEEDS:
            path = args.directory / f"{case}-seed{seed}.json"
            result = json.loads(path.read_text())
            config = result["config"]
            if result["status"] != "ok" or config["seed"] != seed or tuple(config[k] for k in ("batch", "input", "hidden", "output")) != shape:
                raise ValueError(f"invalid run: {path}")
            if source is None:
                source = result["source_sha256"]
            elif result["source_sha256"] != source:
                raise ValueError(f"source hashes differ: {path}")
            if set(result["timings"]) != set(METHODS):
                raise ValueError(f"missing or extra methods: {path}")
            for method in ("megakernel", "mega_inplace", "eager_graph"):
                trajectory = result["trajectory"][method]
                if not trajectory["passed"] or not trajectory["all_finite"] or len(trajectory["losses"]) < 100 or not result["replay_correctness"][method]["passed"]:
                    raise ValueError(f"correctness gate failed for {method}: {path}")
            if result["timings"]["mega_inplace"]["samples"] != 30:
                raise ValueError(f"expected 30 samples: {path}")
            mega = result["timings"]["mega_inplace"]["median_ms"]
            ind = result["timings"][result["fastest_all_inductor"]]["median_ms"]
            eager = result["timings"]["eager_graph"]["median_ms"]
            measurements.append((mega, ind, eager, mega / ind))
        values = lambda index: statistics.median(row[index] for row in measurements)
        rows.append((case, *(values(i) * 1000 for i in range(3)), values(3),
                     min(row[3] for row in measurements), max(row[3] for row in measurements)))
    print("| Case | Mega in-place (μs) | Fastest Inductor (μs) | Eager graph (μs) | Mega/Inductor | Ratio range across seeds |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for case, mega, ind, eager, ratio, low, high in rows:
        print(f"| {case} | {mega:.2f} | {ind:.2f} | {eager:.2f} | {ratio:.3f} | {low:.3f}–{high:.3f} |")
    print(f"Validated {len(CASES) * len(SEEDS)} complete runs with one source hash set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
