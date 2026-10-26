"""Validate and summarize the held-out H100 training matrix.

The matrix is deliberately kept separate from calibration results.  Failed
Inductor numerical gates remain in the fastest measured Inductor column; the
passing-reference column is reported independently.
"""
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
REQUIRED_CORRECTNESS = ("megakernel", "mega_inplace", "eager_graph")


def _setup(result: dict, name: str, reference: str) -> tuple[float, float, float | None]:
    """Return method setup, reference setup, and an honest break-even count."""
    amortization = result.get("amortization", {}).get(name)
    if amortization:
        setup = float(amortization["setup_seconds"])
        ref_setup = float(amortization["reference_setup_seconds"])
    else:
        compilation = result["compilation_seconds"]
        capture = result["capture_seconds"]
        setup = float(compilation[name]) + float(capture[name])
        ref_setup = float(compilation[reference]) + float(capture[reference])
    savings_ms = float(result["timings"][reference]["median_ms"]) - float(result["timings"][name]["median_ms"])
    incremental = setup - ref_setup
    # A negative incremental setup means the candidate was already amortized
    # relative to the observed reference run. Report zero, never a negative
    # deployment claim. A non-positive steady-state saving has no break-even.
    break_even = max(0.0, incremental * 1000.0 / savings_ms) if savings_ms > 0 else None
    return setup, ref_setup, break_even


def _validate(path: Path, shape: tuple[int, int, int, int], seed: int, source: object) -> tuple[dict, object]:
    result = json.loads(path.read_text())
    config = result.get("config", {})
    if (result.get("status") != "ok" or config.get("seed") != seed
            or config.get("task") != "regression"
            or tuple(config.get(k) for k in ("batch", "input", "hidden", "output")) != shape):
        raise ValueError(f"invalid run: {path}")
    if source is not None and result.get("source_sha256") != source:
        raise ValueError(f"source hashes differ: {path}")
    if set(result.get("timings", {})) != set(METHODS):
        raise ValueError(f"missing or extra methods: {path}")
    for method in REQUIRED_CORRECTNESS:
        trajectory = result["trajectory"].get(method, {})
        replay = result["replay_correctness"].get(method, {})
        if not trajectory.get("passed") or not trajectory.get("all_finite") or len(trajectory.get("losses", [])) < 100 or not replay.get("passed"):
            raise ValueError(f"correctness gate failed for {method}: {path}")
    if result["timings"]["mega_inplace"].get("samples") != 30:
        raise ValueError(f"expected 30 samples: {path}")
    fastest_all = result.get("fastest_all_inductor")
    if fastest_all not in METHODS or not fastest_all.startswith("inductor"):
        raise ValueError(f"invalid fastest_all_inductor: {path}")
    measured_fastest_all = min(
        (name for name in METHODS if name.startswith("inductor")),
        key=lambda name: result["timings"][name]["median_ms"],
    )
    if fastest_all != measured_fastest_all:
        raise ValueError(f"fastest_all_inductor is not the measured minimum: {path}")
    fastest_passing = result.get("fastest_passing_reference")
    if fastest_passing is not None:
        if fastest_passing not in METHODS or fastest_passing.startswith("mega"):
            raise ValueError(f"invalid fastest_passing_reference: {path}")
        if not result["trajectory"].get(fastest_passing, {}).get("passed") or not result["replay_correctness"].get(fastest_passing, {}).get("passed"):
            raise ValueError(f"fastest_passing_reference failed correctness: {path}")
    passing_references = [
        name for name in METHODS
        if not name.startswith("mega")
        and result["trajectory"].get(name, {}).get("passed")
        and result["replay_correctness"].get(name, {}).get("passed")
    ]
    measured_fastest_passing = min(
        passing_references, key=lambda name: result["timings"][name]["median_ms"]
    ) if passing_references else None
    if fastest_passing != measured_fastest_passing:
        raise ValueError(f"fastest_passing_reference is not the measured passing minimum: {path}")
    return result, result.get("source_sha256", source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    rows = []
    source = None
    for case, shape in zip(CASES, SHAPES):
        measurements = []
        for seed in SEEDS:
            result, source = _validate(args.directory / f"{case}-seed{seed}.json", shape, seed, source)
            mega = float(result["timings"]["mega_inplace"]["median_ms"])
            all_inductor = float(result["timings"][result["fastest_all_inductor"]]["median_ms"])
            passing_name = result.get("fastest_passing_reference")
            passing = float(result["timings"][passing_name]["median_ms"]) if passing_name else None
            setup, ref_setup, break_even = _setup(result, "mega_inplace", passing_name or result["fastest_all_inductor"])
            measurements.append({"mega": mega, "all": all_inductor, "passing": passing, "setup": setup, "ref_setup": ref_setup, "break_even": break_even})
        median = lambda key: statistics.median(m[key] for m in measurements if m[key] is not None)
        ratios_all = [m["mega"] / m["all"] for m in measurements]
        ratios_passing = [m["mega"] / m["passing"] for m in measurements if m["passing"] is not None]
        rows.append((case, median("mega") * 1000, median("all") * 1000, median("passing") * 1000 if ratios_passing else None,
                     median("setup"), median("ref_setup"), median("break_even") if all(m["break_even"] is not None for m in measurements) else None,
                     statistics.median(ratios_all), min(ratios_all), max(ratios_all),
                     statistics.median(ratios_passing) if ratios_passing else None,
                     min(ratios_passing) if ratios_passing else None, max(ratios_passing) if ratios_passing else None))
    print("| Case | Mega in-place (us) | Fastest all Inductor (us) | Fastest passing reference (us) | Mega/all-Inductor | All ratio range | Mega/passing reference | Passing ratio range | Median setup (s) | Median break-even steps |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        case, mega, all_inductor, passing, setup, _, break_even, ratio, low, high, ratio_p, low_p, high_p = row
        fmt = lambda value: "n/a" if value is None else f"{value:.3f}"
        print(f"| {case} | {mega:.2f} | {all_inductor:.2f} | {fmt(passing)} | {ratio:.3f} | {low:.3f}-{high:.3f} | {fmt(ratio_p)} | {fmt(low_p)}-{fmt(high_p)} | {setup:.2f} | {fmt(break_even)} |")
    print(f"Validated {len(CASES) * len(SEEDS)} complete H100 runs with one source hash set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
