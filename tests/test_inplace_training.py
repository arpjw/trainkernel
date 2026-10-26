"""Opt-in transactional output-to-input state commits."""
import pytest
import torch

from megakernel import compile as mk_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM80+")


def _step(p, x):
    w, b = p
    def loss(q):
        z = x @ q[0] + q[1]
        return (z * z).mean()
    grads, value = torch.func.grad_and_value(loss)(p)
    updated = tuple(q - 1e-3 * g for q, g in zip(p, grads))
    return value, grads[0], grads[1], updated[0], updated[1]


def test_inplace_state_commit_matches_pure_step_and_preserves_outputs():
    device = torch.device("cuda")
    p = (torch.randn(8, 8, device=device), torch.randn(8, device=device))
    x = torch.randn(4, 8, device=device)
    fn = mk_compile(_step, (p, x), inplace_outputs={3: 0, 4: 1})
    expected_state = tuple(q.clone() for q in p)
    for _ in range(3):
        expected = _step(expected_state, x)
        actual = fn(p, x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(p[0], expected[3])
        torch.testing.assert_close(p[1], expected[4])
        expected_state = (expected[3], expected[4])


def test_inplace_state_commit_supports_cuda_graph_feedback():
    device = torch.device("cuda")
    p = (torch.randn(4, 4, device=device), torch.randn(4, device=device))
    x = torch.randn(2, 4, device=device)
    compiled = mk_compile(_step, (p, x), inplace_outputs={3: 0, 4: 1})
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(p, x)
    for _ in range(10):
        graph.replay()
    assert torch.isfinite(captured[0]).all()
    assert torch.isfinite(p[0]).all() and torch.isfinite(p[1]).all()


def test_inplace_mapping_rejects_duplicates_and_bad_slots():
    x = torch.randn(2, 4, device="cuda")
    p = (torch.randn(4, 4, device="cuda"), torch.randn(4, device="cuda"))
    with pytest.raises(ValueError, match="duplicate|mapping|output"):
        mk_compile(_step, (p, x), inplace_outputs={3: 0, 4: 0})
    with pytest.raises(ValueError, match="index|output|input"):
        mk_compile(_step, (p, x), inplace_outputs={99: 0})


def test_cross_update_commit_is_transactional():
    device = torch.device("cuda")
    x = torch.randn(4, 4, device=device)
    y = torch.randn(4, 4, device=device)

    def swap(a, b):
        return a + 0, b + 0

    compiled = mk_compile(swap, (x, y), inplace_outputs={0: 1, 1: 0})
    original_x, original_y = x.clone(), y.clone()
    compiled(x, y)
    torch.testing.assert_close(x, original_y)
    torch.testing.assert_close(y, original_x)
