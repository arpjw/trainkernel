import json

from scripts.summarize_h100 import CASES, SEEDS, SHAPES, main


def _write_matrix(tmp_path):
    source = {"trainkernel/runtime.py": "abc"}
    for case, shape in zip(CASES, SHAPES):
        for seed in SEEDS:
            timings = {name: {"median_ms": value, "samples": 30} for name, value in {
                "megakernel": 0.011, "mega_inplace": 0.010, "eager_graph": 0.020,
                "inductor": 0.013, "inductor_strict": 0.014, "inductor_inplace": 0.012,
                "inductor_inplace_strict": 0.015,
            }.items()}
            methods = {name: {"passed": True, "all_finite": True, "losses": [1.0] * 100} for name in ("megakernel", "mega_inplace", "eager_graph", "inductor_inplace")}
            result = {
                "status": "ok", "config": {"seed": seed, "batch": shape[0], "input": shape[1], "hidden": shape[2], "output": shape[3], "task": "regression"},
                "source_sha256": source, "timings": timings, "trajectory": methods,
                "replay_correctness": {name: {"passed": True} for name in methods},
                "fastest_all_inductor": "inductor_inplace", "fastest_passing_reference": "inductor_inplace",
                "compilation_seconds": {name: 1.0 for name in timings}, "capture_seconds": {name: 1.0 for name in timings},
            }
            (tmp_path / f"{case}-seed{seed}.json").write_text(json.dumps(result))


def test_h100_summary_validates_and_reports_both_reference_columns(tmp_path, capsys):
    _write_matrix(tmp_path)
    assert main([str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Fastest all Inductor" in output
    assert "Fastest passing reference" in output
    assert "break-even" in output
    assert "not cold/warm cache guarantees" in output
    assert "Validated 12 complete H100 runs" in output


def test_h100_summary_rejects_failed_passing_reference(tmp_path):
    _write_matrix(tmp_path)
    path = tmp_path / "tiny-online-seed101.json"
    result = json.loads(path.read_text())
    result["trajectory"]["inductor_inplace"]["passed"] = False
    path.write_text(json.dumps(result))
    try:
        main([str(tmp_path)])
    except ValueError as exc:
        assert "fastest_passing_reference failed" in str(exc)
    else:
        raise AssertionError("failed passing reference should be rejected")
