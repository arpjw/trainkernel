"""Adversarial tests for observable GEMM side epilogues."""
import pytest
import torch

from trainkernel import capture, compile
from trainkernel.codegen import generate, _Generator


def test_shared_observable_root_and_update_are_discovered():
    a = torch.randn(4, 17)
    b = torch.randn(17, 4)
    lr = torch.randn(1)

    def fn(a, b, lr):
        root = a @ b
        return root, root * lr

    program, _ = capture(fn, (a, b, lr))
    scheduler = _Generator(program, blocks_per_sm=1, fuse_pointwise=True)
    generated = scheduler.generate()
    assert scheduler.side_epilogues
    for chain in scheduler.side_epilogues.values():
        assert all(op.output.buffer in scheduler.output_bindings for op in chain)
    assert generated.materialized_operations == 1


def test_branched_observable_outputs_share_one_gemm():
    a = torch.randn(4, 17)
    b = torch.randn(17, 4)
    def fn(a, b):
        root = a @ b
        return root, root + 1, root * 2
    program, _ = capture(fn, (a, b))
    scheduler = _Generator(program, blocks_per_sm=1, fuse_pointwise=True)
    generated = scheduler.generate()
    assert scheduler.side_epilogues
    assert generated.materialized_operations == 1


def test_square_transpose_descendant_is_not_fused():
    a = torch.randn(4, 8)
    b = torch.randn(8, 4)
    program, _ = capture(lambda a, b: (a @ b, (a @ b).t()), (a, b))
    generated = generate(program)
    assert generated.materialized_operations >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA SM80+ required")
@pytest.mark.parametrize("dtype,k", [(torch.float32, 4), (torch.bfloat16, 4), (torch.float16, 4),
                                     (torch.float32, 17), (torch.bfloat16, 17), (torch.float16, 17)])
def test_side_epilogue_cuda_parity_and_graph_replay(dtype, k):
    device = torch.device("cuda")
    a = torch.randn(4, k, device=device, dtype=dtype)
    b = torch.randn(k, 4, device=device, dtype=dtype)
    lr = torch.tensor([0.01], device=device, dtype=dtype)

    def fn(a, b, lr):
        root = a @ b
        return root, root * lr, torch.relu(root + 1)

    compiled = compile(fn, (a, b, lr))
    expected = fn(a, b, lr)
    actual = compiled(a, b, lr)
    tol = {"rtol": 0.04, "atol": 0.004} if dtype == torch.bfloat16 else {"rtol": 0.008, "atol": 0.0005} if dtype == torch.float16 else {"rtol": 2e-4, "atol": 2e-5}
    torch.testing.assert_close(actual, expected, **tol)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(a, b, lr)
    graph.replay()
    torch.testing.assert_close(captured, expected, **tol)
