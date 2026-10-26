"""Adversarial checks for GEMM epilogue fusion.

The structural cases run on CPU fake tensors; execution cases require an SM80+
GPU and are skipped on developer machines without CUDA.
"""
import pytest
import torch

from megakernel import capture, compile
from megakernel.codegen import generate


def _program(fn, inputs):
    program, _ = capture(fn, inputs)
    return program


def test_observable_mm_is_not_fused():
    a, b = torch.randn(4, 8), torch.randn(8, 4)
    program = _program(lambda a, b: (a @ b, torch.relu(a @ b)), (a, b))
    generated = generate(program)
    # The first GEMM is returned directly and cannot be replaced by its
    # pointwise consumer.
    assert generated.materialized_operations >= 2


def test_transpose_input_and_broadcast_chain_remain_safe():
    a, b, bias = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 4)

    def fn(a, b, bias):
        # The GEMM result is the second operand of sub. The transposed input is
        # a view and must not be treated as a direct contiguous root.
        return bias - (a @ b.t())

    program = _program(fn, (a, b, bias))
    generated = generate(program)
    assert generated.materialized_operations >= 1


def test_fused_chain_preserves_original_rounding_boundary():
    a, b, bias = (torch.randn(4, 8, dtype=torch.bfloat16),
                  torch.randn(8, 4, dtype=torch.bfloat16),
                  torch.randn(4, dtype=torch.bfloat16))

    def fn(a, b, bias):
        value = a @ b
        return torch.relu(value + bias)

    program = _program(fn, (a, b, bias))
    generated = generate(program)
    # The source must contain an explicit BF16 conversion for the GEMM result
    # before the fused add/relu expression.
    assert "__float2bfloat16_rn" in generated.source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA SM80+ required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_gradient_matches_eager(dtype):
    device = torch.device("cuda")
    a = torch.randn(4, 8, device=device, dtype=dtype)
    b = torch.randn(8, 4, device=device, dtype=dtype)
    target = torch.randn(4, 4, device=device, dtype=dtype)

    def step(a, b, target):
        def loss(x, y):
            return ((torch.relu(x @ y) - target) ** 2).mean()
        grads, value = torch.func.grad_and_value(loss, argnums=(0, 1))(a, b)
        return value, grads

    compiled = compile(step, (a, b, target))
    expected = step(a, b, target)
    actual = compiled(a, b, target)
    tol = {"rtol": 0.04, "atol": 0.004} if dtype == torch.bfloat16 else {"rtol": 2e-4, "atol": 2e-5}
    torch.testing.assert_close(actual, expected, **tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA SM80+ required")
def test_addmm_beta_zero_nan_bias_is_finite():
    device = torch.device("cuda")
    a = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
    b = torch.randn(8, 4, device=device, dtype=torch.bfloat16)
    bias = torch.full((4, 4), float("nan"), device=device, dtype=torch.bfloat16)

    def fn(a, b, bias):
        return torch.addmm(bias, a, b, beta=0.0)

    compiled = compile(fn, (a, b, bias))
    actual = compiled(a, b, bias)
    expected = fn(a, b, bias)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.004)
    assert torch.isfinite(actual).all()
