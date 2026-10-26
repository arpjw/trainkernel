from scripts.tune_h100 import CASES, CONFIGS, main


def test_h100_cases_cover_existing_suite_and_sm_count():
    assert tuple(CASES) == ("tiny-online", "online", "small", "medium")
    assert CASES["medium"] == (128, 512, 1024, 256)
    for case in ("small", "medium"):
        assert any(c.get("max_blocks") == 132 for c in CONFIGS[case])
        assert any(c.get("max_blocks") == 264 and c.get("blocks_per_sm") == 2 for c in CONFIGS[case])
        assert all(c.get("threads_per_block") == 128 for c in CONFIGS[case])
    assert any(c.get("max_blocks") == 108 for c in CONFIGS["small"])
    assert any(c.get("max_blocks") == 216 for c in CONFIGS["medium"])


def test_h100_tuner_rejects_bad_only(tmp_path):
    try:
        main(["--case", "tiny-online", "--json", str(tmp_path / "out.json"), "--only", "99"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("invalid --only should be rejected")
