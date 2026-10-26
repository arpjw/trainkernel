import torch
from trainkernel import capture

def test_capture_mlp_forward_backwardish_ops():
    def fn(x, w, b):
        return torch.relu(x @ w + b).mean()
    x = torch.randn(4, 3)
    w = torch.randn(3, 5)
    b = torch.randn(5)
    program, spec = capture(fn, (x, w, b))
    assert [i.shape for i in program.inputs] == [(4, 3), (3, 5), (5,)]
    assert any(op.op == "mm" for op in program.operations)
    assert any(op.op == "relu" for op in program.operations)
    assert program.outputs[0].shape == ()

def test_capture_rejects_non_tensor_inputs():
    try:
        capture(lambda x, n: x + n, (torch.ones(2), 1))
    except TypeError as e:
        assert "tensors only" in str(e)
    else:
        raise AssertionError("expected TypeError")


def test_capture_rejects_cross_device_copy():
    import pytest
    with pytest.raises(ValueError, match="cross-device"):
        capture(lambda x: x.to("meta"), (torch.ones(3),))


def test_capture_rejects_lazy_negative_input():
    import pytest
    with pytest.raises(ValueError, match="lazy negative"):
        capture(lambda x: x + 1, (torch._neg_view(torch.ones(3)),))
