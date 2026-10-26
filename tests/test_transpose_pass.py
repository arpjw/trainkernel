import pytest
import torch
from megakernel import capture
from megakernel.ir import validate
from megakernel.passes import canonicalize_transposed_mm
from megakernel.reference import run


@pytest.mark.parametrize('observed',[False,True])
def test_transpose_rewrite_preserves_views_and_observable_roots(observed):
    a,b=torch.randn(3,5),torch.randn(5,7)
    def fn(a,b):
        x=a@b
        return (x,x.t()+1) if observed else x.t()+1
    program,_=capture(fn,(a,b))
    rewritten=canonicalize_transposed_mm(program)
    validate(rewritten)
    torch.testing.assert_close(run(rewritten,(a,b)),run(program,(a,b)))
    assert (rewritten==program)==observed
    assert canonicalize_transposed_mm(rewritten)==rewritten


def test_dependent_transposed_matmuls_rewrite_together():
    a,b,c=torch.randn(3,5),torch.randn(5,7),torch.randn(3,4)
    def fn(a,b,c):
        x=(a@b).t()
        y=(x@c).t()
        return y+1
    program,_=capture(fn,(a,b,c))
    rewritten=canonicalize_transposed_mm(program)
    validate(rewritten)
    torch.testing.assert_close(run(rewritten,(a,b,c)),run(program,(a,b,c)))
    assert canonicalize_transposed_mm(rewritten)==rewritten
