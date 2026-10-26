from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_sm90_wgmma.py"


def test_sm90_probe_is_isolated_and_explicitly_sm90a():
    source = SCRIPT.read_text()
    assert "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16" in source
    assert "fence.proxy.async.shared::cta" in source
    assert '"%32, %33, 0, 1, 1, 0, 0;' in source
    assert "-gencode=arch=compute_90a,code=sm_90a" in source
    assert 'TORCH_CUDA_ARCH_LIST"] = "9.0a"' in source
    assert "this probe requires H100 compute capability 9.0" in source
    assert "not a performance benchmark" in source
    assert "def run(seed: int = 0" in source


def test_sm90_probe_cli_parses_without_cuda_execution():
    source = SCRIPT.read_text()
    assert "--tolerance" in source
    assert "--seed" in source
