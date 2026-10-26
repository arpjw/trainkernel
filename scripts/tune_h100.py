"""Calibration tuner for H100 launch and tile choices.

This is deliberately exploratory: it uses one fixed calibration seed and must
never be presented as held-out evidence.  Results are written after every
trial, so a preempted GPU run can be resumed with the same ``--json`` path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch

from benchmarks.train_loop import _capture, _close, _paired_time, _trajectory, make_batches, step_master
from trainkernel import compile as mk_compile

CASES = {
    "tiny-online": (4, 32, 64, 16),
    "online": (1, 128, 128, 16),
    "small": (32, 128, 256, 64),
    "medium": (128, 512, 1024, 256),
}

# H100 has 132 SMs.  Keep the set small enough to be useful as a calibration
# pass while explicitly testing one and two resident blocks per SM.
CONFIGS = {
    "tiny-online": [
        {"max_blocks": 1, "threads_per_block": 128},
        {"max_blocks": 1, "threads_per_block": 512},
        {"max_blocks": 1, "threads_per_block": 1024},
        {"max_blocks": 4, "threads_per_block": 512},
    ],
    "online": [
        {"max_blocks": 1, "threads_per_block": 128},
        {"max_blocks": 1, "threads_per_block": 512},
        {"max_blocks": 1, "threads_per_block": 1024, "split_k_gemv": True},
        {"max_blocks": 4, "threads_per_block": 512, "split_k_gemv": True},
        {"max_blocks": 16, "threads_per_block": 1024, "split_k_gemv": True},
    ],
    "small": [
        # Retain the selected A100 launch as a control. H100's larger SM count
        # does not by itself prove that a wider cooperative grid is faster.
        {"max_blocks": 108, "threads_per_block": 128, "shared_skew": 16},
        {"max_blocks": 132, "threads_per_block": 128},
        {"max_blocks": 132, "threads_per_block": 128, "shared_skew": 16},
        {"max_blocks": 132, "threads_per_block": 128, "gemm_k_tile": 16},
        {"max_blocks": 132, "threads_per_block": 128, "gemm_tile_n": 64},
        {"max_blocks": 264, "blocks_per_sm": 2, "threads_per_block": 128, "shared_skew": 16},
        {"max_blocks": 264, "blocks_per_sm": 2, "threads_per_block": 128, "gemm_tile_n": 64},
        {"max_blocks": 132, "threads_per_block": 128, "hopper_wgmma": True},
        {"max_blocks": 264, "blocks_per_sm": 2, "threads_per_block": 128, "hopper_wgmma": True},
    ],
    "medium": [
        {"max_blocks": 216, "blocks_per_sm": 2, "threads_per_block": 128, "shared_skew": 16},
        {"max_blocks": 132, "threads_per_block": 128},
        {"max_blocks": 132, "threads_per_block": 128, "shared_skew": 16},
        {"max_blocks": 132, "threads_per_block": 128, "gemm_k_tile": 16},
        {"max_blocks": 132, "threads_per_block": 128, "gemm_tile_n": 64},
        {"max_blocks": 264, "blocks_per_sm": 2, "threads_per_block": 128, "shared_skew": 16},
        {"max_blocks": 264, "blocks_per_sm": 2, "threads_per_block": 128, "gemm_tile_n": 64},
        {"max_blocks": 132, "threads_per_block": 128, "hopper_wgmma": True},
        {"max_blocks": 264, "blocks_per_sm": 2, "threads_per_block": 128, "hopper_wgmma": True},
    ],
}


def _sources(root: Path) -> dict[str, str]:
    paths = sorted((root / "trainkernel").glob("*.py"))
    paths.extend((root / "benchmarks" / "train_loop.py", Path(__file__).resolve()))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--only", type=int)
    args = parser.parse_args(argv)
    configs = CONFIGS[args.case]
    if args.only is not None and not 0 <= args.only < len(configs):
        parser.error(f"--only must be between 0 and {len(configs) - 1}")
    if not torch.cuda.is_available():
        parser.error("an H100 CUDA device is required")
    capability = torch.cuda.get_device_capability()
    if capability != (9, 0):
        parser.error(f"expected H100 compute capability 9.0, found {capability}")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if properties.multi_processor_count != 132:
        parser.error(
            "this fixed candidate set requires an H100 SXM with 132 SMs; "
            f"found {properties.name} with {properties.multi_processor_count} SMs"
        )
    hardware = {
        "name": properties.name,
        "capability": list(capability),
        "sm_count": properties.multi_processor_count,
    }
    root = Path(__file__).resolve().parents[1]
    source_sha256 = _sources(root)
    result = {
        "purpose": "exploratory H100 calibration, not held-out evidence",
        "gpu_assumption": "H100 SXM with 132 SMs",
        "hardware": hardware,
        "case": args.case, "shape": CASES[args.case], "seed": 20261025,
        "source_sha256": source_sha256, "trials": [],
    }
    if args.json.exists():
        prior = json.loads(args.json.read_text())
        if prior.get("case") != args.case or prior.get("source_sha256") != source_sha256 or prior.get("hardware") != hardware:
            parser.error("existing result has a different case, source hash, or GPU; choose a new --json path")
        result = prior
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    params, batches = make_batches(*CASES[args.case], torch.bfloat16, "cuda", seed=20261025)
    params = tuple(v.float() for v in params)
    fn = lambda pp, x, y: step_master(pp, x, y, 1e-3)
    for index, options in enumerate(configs):
        if args.only is not None and index != args.only:
            continue
        if any(t.get("index") == index for t in result["trials"]):
            continue
        trial = {"index": index, "compiler_options": options}
        try:
            start = time.perf_counter()
            mega = mk_compile(fn, (params, *batches[0]), inplace_outputs={5: 0, 6: 1, 7: 2, 8: 3}, **options)
            trial["compile_seconds"] = time.perf_counter() - start
            trial["resources"] = mega.runner.resource_usage()
            trial["cuda_sha256"] = hashlib.sha256(mega.source.encode()).hexdigest()
            losses, _ = _trajectory(mega, params, batches, 100)
            trial["trajectory"] = {"passed": True, "steps": 100, "initial_loss": losses[0], "final_loss": losses[-1]}
            state = tuple(v.clone() for v in params)
            graph = _capture(mega, state, batches, 20)
            for dst, src in zip(state, params):
                dst.copy_(src)
            graph.replay()
            _, expected = _trajectory(fn, params, batches, 20)
            _close(state, expected, torch.float32)
            trial["replay_correctness"] = {"passed": True}
            trial["timing"] = _paired_time([(graph, state)], params, 20, 30, 10)[0]
            trial["status"] = "ok"
        except Exception as exc:  # preserve failures for calibration review
            trial.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        result["trials"].append(trial)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(trial), flush=True)
    passing = [t for t in result["trials"] if t.get("status") == "ok"]
    if passing:
        result["best_index"] = min(passing, key=lambda t: t["timing"]["median_ms"])["index"]
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    requested = [t for t in result["trials"] if args.only is None or t.get("index") == args.only]
    return int(not requested or any(t.get("status") != "ok" for t in requested))


if __name__ == "__main__":
    raise SystemExit(main())
