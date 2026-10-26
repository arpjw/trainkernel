"""Malformed IR must fail before producing an unsafe CUDA memory access."""
import pytest
from megakernel.ir import TensorSpec,Operation,Program
from megakernel.codegen import generate


def spec(name,shape,strides=None,buffer=None,offset=0):
    if strides is None:
        import math
        strides=tuple(math.prod(shape[i+1:]) for i in range(len(shape)))
    return TensorSpec(name,shape,'float32',strides,buffer or name,offset)


def test_views_and_ssa_are_bounded():
    x=spec('x',(2,3))
    with pytest.raises(ValueError,match='allocation'):
        generate(Program((x,),(),(spec('bad',(2,3),(300,1),'x'),)))
    future=spec('future',(2,3));out=spec('out',(2,3))
    with pytest.raises(ValueError,match='future|undefined'):
        generate(Program((x,),(Operation('bad','add',(x,future),out),),(out,)))
    with pytest.raises(ValueError,match='new roots'):
        generate(Program((x,),(Operation('bad','add',(x,1),x),),(x,)))


def test_operator_shapes_cannot_overrun_roots():
    a,b=spec('a',(2,3)),spec('b',(3,4))
    out=spec('out',(2,2))
    with pytest.raises(ValueError,match='matrix output shape'):
        generate(Program((a,b),(Operation('bad','mm',(a,b),out),),(out,)))
    b=spec('b',(2,2));out=spec('out',(2,3))
    with pytest.raises(ValueError,match='broadcast'):
        generate(Program((a,b),(Operation('bad','add',(a,b),out),),(out,)))
