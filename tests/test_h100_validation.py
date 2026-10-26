from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_h100_validation.sh"


def test_h100_validation_script_is_executable_and_complete():
    assert SCRIPT.stat().st_mode & 0o111
    source = SCRIPT.read_text()
    required_fragments = (
        'torch.cuda.get_device_capability()',
        '--query-gpu=name,uuid,driver_version,memory.total',
        'python -m pytest -q',
        'python -m examples.online_training',
        'scripts/run_train_loop_suite.sh',
        'python -m scripts.summarize_h100',
        '--task classification-head',
        'python -m build',
        'python -m twine check',
    )
    for fragment in required_fragments:
        assert fragment in source


def test_h100_validation_uses_independent_output_names():
    source = SCRIPT.read_text()
    assert 'results/h100-baseline' in source
    assert 'full-gpu-tests.txt' in source
    assert 'heldout-suite.txt' in source
    assert 'heldout-h100-summary.txt' in source
    assert 'environment.txt' in source
