import pytest
import torch

from trainkernel import capture, compile as mk_compile
from trainkernel.codegen import generate
from trainkernel.sm90 import emit_wgmma


def _read_a(r, k): return f"A[{r} * 17 + {k}]"
def _read_b(k, c): return f"B[{k} * 19 + {c}]"
def _store(i, v): return f"C[{i}] = {v};"


def test_wgmma_emission_metadata_and_protocol():
    out = emit_wgmma(128, 32, 96, _read_a, _read_b, _store)
    assert out.tile_m == 64 and out.tile_n == 64 and out.tile_k == 16
    assert out.shared_a_elements == out.shared_b_elements == 1024
    assert out.source.count("wgmma.mma_async") == 2
    assert "{%0,%1,%2,%3" in out.source
    assert "sm90_d[" not in out.source
    assert "fence.proxy.async.shared::cta" in out.source
    assert ", 0, 1, 1, 0, 0;" in out.source
    assert ", 1, 1, 1, 0, 0;" in out.source


def test_wgmma_emission_masks_tails_and_calls_callbacks():
    source = emit_wgmma(65, 17, 67, _read_a, _read_b, _store, lambda i, v: f"({v}) + bias[{i}]").source
    assert "sm90_ar < 65LL && sm90_ak < 17LL" in source
    assert "sm90_bn < 67LL && sm90_bk < 17LL" in source
    assert "sm90_r0 + 8 < 65LL && sm90_c0 + 1 < 67LL" in source
    assert "bias[sm90_i]" in source


def test_wgmma_rejects_nonpositive_dimensions():
    try:
        emit_wgmma(0, 16, 64, _read_a, _read_b, _store)
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive dimensions must be rejected")
    with pytest.raises(ValueError, match="BF16"):
        emit_wgmma(64, 16, 64, _read_a, _read_b, _store, value_type="__half")


def test_hopper_codegen_is_explicit_and_preserves_default_path():
    a = torch.randn(64, 16, dtype=torch.bfloat16)
    b = torch.randn(16, 64, dtype=torch.bfloat16)
    program, _ = capture(lambda a, b: a @ b, (a, b))
    baseline = generate(program)
    hopper = generate(program, hopper_wgmma=True)
    assert "wgmma.mma_async" not in baseline.source
    assert "wgmma.mma_async" in hopper.source
    assert hopper.tensor_core_operations == 1
    assert hopper.max_work_blocks == 1
    with pytest.raises(ValueError, match="boolean"):
        generate(program, hopper_wgmma=1)
    with pytest.raises(ValueError, match="128 threads"):
        generate(program, hopper_wgmma=True, threads_per_block=256)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires an SM90 Hopper GPU",
)
@pytest.mark.parametrize("shape", [(64, 16, 64), (65, 17, 67)])
def test_hopper_wgmma_cuda_tiles_and_tails(shape):
    m, k, n = shape
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    compiled = mk_compile(lambda a, b: a @ b, (a, b), hopper_wgmma=True)
    torch.testing.assert_close(compiled(a, b), a @ b, rtol=0.04, atol=0.004)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires an SM90 Hopper GPU",
)
def test_hopper_wgmma_training_trajectory_and_graph_replay():
    from benchmarks.train_loop import _capture, _close, _trajectory, make_batches, step_master

    raw, batches = make_batches(8, 32, 64, 16, torch.bfloat16, "cuda", seed=20261025)
    params = tuple(value.float() for value in raw)
    fn = lambda state, x, target: step_master(state, x, target, 1e-3)
    compiled = mk_compile(
        fn,
        (params, *batches[0]),
        hopper_wgmma=True,
        max_blocks=1,
        inplace_outputs={5: 0, 6: 1, 7: 2, 8: 3},
    )
    _trajectory(compiled, params, batches, 20)
    state = tuple(value.clone() for value in params)
    graph = _capture(compiled, state, batches, 10)
    for destination, source in zip(state, params):
        destination.copy_(source)
    graph.replay()
    _, expected = _trajectory(fn, params, batches, 10)
    _close(state, expected, torch.float32)
