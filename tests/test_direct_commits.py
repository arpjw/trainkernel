"""Structural and CUDA-gated coverage for direct input commits."""
import pytest
import torch

from trainkernel import capture, compile as mk_compile
from trainkernel.codegen import generate


def _program(fn, inputs):
    program, _ = capture(fn, inputs)
    return program


def test_standard_pointwise_update_gets_direct_commit():
    x = torch.randn(512)
    program = _program(lambda x: x + 1, (x,))
    generated = generate(program, inplace_outputs={0: 0})
    assert generated.direct_state_commits == 1


def test_original_input_return_cannot_direct_commit():
    x = torch.randn(16)
    program = _program(lambda x: x.detach(), (x,))
    with pytest.raises(ValueError, match="computed values|input views"):
        generate(program, inplace_outputs={0: 0})
    program = _program(lambda x: (x, x + 1), (x,))
    assert generate(program, inplace_outputs={1: 0}).direct_state_commits == 0


def test_transposed_update_is_staged_or_rejected():
    x = torch.randn(8, 8)
    program = _program(lambda x: x.t(), (x,))
    with pytest.raises(ValueError, match="inplace"):
        generate(program, inplace_outputs={0: 0})
    program = _program(lambda x: x.t() + 1, (x,))
    assert generate(program, inplace_outputs={0: 0}).direct_state_commits == 0


def test_cross_input_swap_stays_transactional():
    x, y = torch.randn(32), torch.randn(32)
    program = _program(lambda x, y: (y + 0, x + 0), (x, y))
    generated = generate(program, inplace_outputs={0: 0, 1: 1})
    assert generated.direct_state_commits < 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM80+")
def test_multiblock_direct_commit_updates_state_and_version():
    x = torch.randn(8192, device="cuda")
    compiled = mk_compile(lambda x: x + 1, (x,), inplace_outputs={0: 0}, threads_per_block=512)
    version = x._version
    expected = x.clone() + 1
    result = compiled(x)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(x, expected)
    assert x._version > version


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM80+")
def test_requires_grad_input_rejected_for_direct_commit():
    x = torch.randn(64, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="detached|autograd"):
        mk_compile(lambda x: x + 1, (x,), inplace_outputs={0: 0})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM80+")
def test_mixed_commit_multiblock_swap_and_original_snapshot():
    x,y=torch.randn(8192,device='cuda'),torch.randn(8192,device='cuda')
    old_x,old_y=x.clone(),y.clone()
    fn=lambda x,y:(y+0,x+0)
    compiled=mk_compile(fn,(x,y),inplace_outputs={0:0,1:1})
    assert compiled.runner.generated.direct_state_commits == 1
    compiled(x,y)
    torch.testing.assert_close(x,old_y)
    torch.testing.assert_close(y,old_x)
    snapshot=mk_compile(lambda x:(x,x+1),(x,),inplace_outputs={1:0})
    before=x.clone()
    original,updated=snapshot(x)
    torch.testing.assert_close(original,before)
    torch.testing.assert_close(updated,before+1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM80+")
def test_cross_element_update_retains_staged_commit():
    x=torch.randn(33,33,device='cuda')
    expected=x.t()+1
    compiled=mk_compile(lambda x:x.t()+1,(x,),inplace_outputs={0:0})
    assert compiled.runner.generated.direct_state_commits == 0
    torch.testing.assert_close(compiled(x),expected)
