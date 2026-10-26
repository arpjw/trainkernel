from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_h100_heldout_suite.sh"


def test_h100_heldout_matrix_is_fixed_and_excludes_rejected_wgmma():
    source = SCRIPT.read_text()
    assert "for seed in 101 211 307" in source
    assert source.count("python -m benchmarks.train_loop") == 4
    assert "--max-blocks 132 --shared-skew 16" in source
    assert "--max-blocks 264 --blocks-per-sm 2 --shared-skew 16" in source
    assert "--hopper-wgmma" not in source
    assert "seed 20261025" in source
