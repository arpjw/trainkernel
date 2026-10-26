import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
HELDOUT = ROOT / "results" / "h100-optimization" / "heldout"
CLASSIFICATION = ROOT / "results" / "h100-optimization" / "classification-head"
CASES = ("tiny-online", "online", "small", "medium")
SEEDS = (101, 211, 307)


def _load(path: Path) -> dict:
    assert path.is_file(), f"missing release evidence: {path}"
    return json.loads(path.read_text())


@pytest.mark.skipif(not HELDOUT.exists(), reason="repository release evidence is not packaged in the sdist")
def test_h100_heldout_evidence_is_complete_and_source_consistent():
    source_hashes = []
    for case in CASES:
        for seed in SEEDS:
            result = _load(HELDOUT / f"{case}-seed{seed}.json")
            assert result["status"] == "ok"
            assert result["config"]["seed"] == seed
            assert result["config"]["hopper_wgmma"] is False
            assert result["trajectory"]["mega_inplace"]["passed"] is True
            assert result["replay_correctness"]["mega_inplace"]["passed"] is True
            source_hashes.append(result["source_sha256"])
    assert all(hashes == source_hashes[0] for hashes in source_hashes)


@pytest.mark.skipif(not CLASSIFICATION.exists(), reason="repository release evidence is not packaged in the sdist")
def test_h100_classification_evidence_is_complete():
    for seed in SEEDS:
        result = _load(CLASSIFICATION / f"seed{seed}.json")
        assert result["status"] == "ok"
        assert result["task"] == "classification-head"
        assert result["trajectory"]["mega_inplace"]["passed"] is True
        assert result["replay_correctness"]["mega_inplace"]["passed"] is True
        assert result["speed_ratio_vs_fastest_all_inductor"]["mega_inplace"] < 1.0
