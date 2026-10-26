import pytest
import torch

from benchmarks.train_loop import _capture, _close_training, _trajectory, make_batches, step, step_master
from megakernel import compile as compile_megakernel


def test_stateful_sgd_trajectory_is_deterministic_cpu():
    params_a, batches_a = make_batches(3, 5, 7, 2, torch.float32, torch.device("cpu"), 3)
    params_b, batches_b = make_batches(3, 5, 7, 2, torch.float32, torch.device("cpu"), 3)
    for a, b in zip(params_a, params_b):
        torch.testing.assert_close(a, b)
    for (xa, ya), (xb, yb) in zip(batches_a, batches_b):
        torch.testing.assert_close(xa, xb)
        torch.testing.assert_close(ya, yb)
    state = params_a
    losses = []
    for i in range(5):
        loss, _, state = step(state, *batches_a[i % 3], 1e-3)
        losses.append(float(loss))
    assert all(torch.isfinite(torch.tensor(losses)))
    assert len(losses) == 5


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_make_batches_preserves_dtype(dtype):
    params, batches = make_batches(2, 3, 4, 2, dtype, torch.device("cpu"), 2)
    assert all(p.dtype == dtype for p in params)
    assert all(x.dtype == dtype and y.dtype == dtype for x, y in batches)


def test_master_trajectory_accepts_bf16_batches():
    params, batches = make_batches(2, 3, 4, 2, torch.bfloat16, torch.device("cpu"), 2)
    master = tuple(p.float() for p in params)
    losses, state = _trajectory(lambda p, x, y: step_master(p, x, y, 1e-3), master, batches, 3)
    assert len(losses) == 3
    assert all(torch.isfinite(torch.tensor(losses)))
    assert all(p.dtype == torch.float32 for p in state)


def test_false_training_output_is_rejected():
    params, batches = make_batches(2, 3, 4, 2, torch.bfloat16, torch.device("cpu"), 1)
    master = tuple(p.float() for p in params)
    expected = step_master(master, *batches[0], 1e-3)
    bad = (expected[0], expected[1], tuple(p + 1 for p in expected[2]))
    with pytest.raises(AssertionError):
        _close_training(bad, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_captured_feedback_replay_matches_eager():
    params, batches = make_batches(2, 3, 4, 2, torch.bfloat16, torch.device("cuda"), 2)
    master = tuple(p.float() for p in params)
    method = compile_megakernel(lambda p, x, y: step_master(p, x, y, 1e-3), (master, *batches[0]), max_blocks=1)
    state = tuple(p.clone() for p in master)
    graph = _capture(method, state, batches, 3)
    expected = _trajectory(lambda p, x, y: step_master(p, x, y, 1e-3), master, batches, 3)[1]
    for dst, src in zip(state, master):
        dst.copy_(src)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(state, expected, rtol=0.04, atol=0.004)
