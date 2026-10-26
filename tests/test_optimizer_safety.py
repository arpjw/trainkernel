"""Safety and variant coverage for optimizer knobs."""
import pytest
import torch

from megakernel import capture, compile as mk_compile
from megakernel.codegen import generate


def _program(fn, inputs):
    program, _ = capture(fn, inputs)
    return program


def test_variant_flags_and_inplace_mapping_validate_before_codegen():
    x = torch.randn(4, 8)
    program = _program(lambda x: x + 1, (x,))
    for kwargs in ({"gemm_k_tile": 7}, {"gemm_tile_n": 7}, {"shared_skew": 3}, {"blocks_per_sm": 0}, {"max_blocks": 0}):
        with pytest.raises(ValueError):
            generate(program, **kwargs)
    with pytest.raises(ValueError, match="index"):
        generate(program, inplace_outputs={4: 0})
    duplicate_program = _program(lambda x: (x + 1, x + 2), (x,))
    with pytest.raises(ValueError, match="distinct|target"):
        generate(duplicate_program, inplace_outputs={0: 0, 1: 0})

    a = torch.randn(2, 3)
    p = _program(lambda a: a.sum(), (a,))
    with pytest.raises(ValueError, match="shape"):
        generate(p, inplace_outputs={0: 0})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM80+")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_large_reduction_two_pass(dtype, reduction):
    device = torch.device("cuda")
    if reduction == "mean":
        x = torch.randn(32769, device=device, dtype=dtype)
        fn = lambda x: x.mean()
    else:
        x = torch.randn(5003, 3, device=device, dtype=dtype)
        fn = lambda x: x.sum(dim=0)
    compiled = mk_compile(fn, (x,), max_blocks=108)
    actual = compiled(x)
    expected = fn(x)
    tol = {"rtol": 0.04, "atol": 0.004} if dtype == torch.bfloat16 else {"rtol": 0.008, "atol": 0.0005} if dtype == torch.float16 else {"rtol": 2e-4, "atol": 2e-5}
    torch.testing.assert_close(actual, expected, **tol)
    assert compiled.runner.generated.grid_barriers >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM80+")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("k_tile", [16, 32])
def test_irregular_wmma_variants(dtype, k_tile):
    device = torch.device("cuda")
    a = torch.randn(17, 33, device=device, dtype=dtype)
    b = torch.randn(33, 65, device=device, dtype=dtype)
    fn = lambda a, b: a @ b
    compiled = mk_compile(fn, (a, b), gemm_k_tile=k_tile, shared_skew=16)
    torch.testing.assert_close(compiled(a, b), fn(a, b), rtol=0.04, atol=0.004)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM80+")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_irregular_wmma_tile64(dtype):
    device = torch.device("cuda")
    a = torch.randn(17, 33, device=device, dtype=dtype)
    b = torch.randn(33, 65, device=device, dtype=dtype)
    fn = lambda a, b: a @ b
    compiled = mk_compile(fn, (a, b), gemm_tile_n=64, gemm_k_tile=16)
    torch.testing.assert_close(compiled(a, b), fn(a, b), rtol=0.04, atol=0.004)
