"""Pure IR rewrites that preserve the public tensor output contract."""
from __future__ import annotations
from dataclasses import replace
from .ir import Operation, Program, TensorSpec


def canonicalize_transposed_mm(program: Program) -> Program:
    """Compute `(A @ B).T` as `B.T @ A.T` when every use is that view.

    PyTorch backward graphs often return transposed weight gradients. Choosing
    their final physical orientation avoids a separate transpose/cast pass.
    Observable untransposed results and other views keep their original layout.
    """
    changes: dict[str, tuple[int, int]] = {}
    for op in program.operations:
        if op.op != 'mm' or len(op.output.shape) != 2:
            continue
        a,b=op.inputs[:2]
        if not isinstance(a,TensorSpec) or not isinstance(b,TensorSpec) or len(a.shape)!=2 or len(b.shape)!=2:
            continue
        m,n=op.output.shape
        if min(m,n)<=1 or a.shape!=(m,b.shape[0]) or b.shape[1]!=n:
            continue
        uses=[v for user in program.operations for v in user.inputs if isinstance(v,TensorSpec) and v.buffer==op.output.buffer]
        uses.extend(v for v in program.outputs if v.buffer==op.output.buffer)
        if uses and all(v.shape==(n,m) and v.strides==(1,n) and v.offset==0 for v in uses):
            changes[op.output.buffer]=(m,n)
    if not changes:
        return program
    # Flipping a consumer also flips its operand view. Select downstream
    # candidates first, so adjacent rewrites do not undo each other's layout.
    selected: dict[str, tuple[int, int]] = {}
    for op in reversed(program.operations):
        buffer = op.output.buffer
        if buffer not in changes:
            continue
        if any(user.output.buffer in selected for user in program.operations for value in user.inputs if isinstance(value, TensorSpec) and value.buffer == buffer):
            continue
        selected[buffer] = changes[buffer]
    changes = selected

    def view(value: TensorSpec | int | float) -> TensorSpec | int | float:
        if not isinstance(value,TensorSpec) or value.buffer not in changes:
            return value
        m,n=changes[value.buffer]
        if value.shape==(n,m) and value.strides==(1,n) and value.offset==0:
            return replace(value,strides=(m,1))
        # Transposing an operand of another rewritten GEMM creates this view.
        if value.shape==(m,n) and value.strides==(n,1) and value.offset==0:
            return replace(value,strides=(1,m))
        raise ValueError('megakernel: unexpected view in transposed GEMM rewrite')

    operations=[]
    for op in program.operations:
        if op.output.buffer in changes:
            a,b=op.inputs[:2]
            assert isinstance(a,TensorSpec) and isinstance(b,TensorSpec)
            operands=(replace(b,shape=b.shape[::-1],strides=b.strides[::-1]),replace(a,shape=a.shape[::-1],strides=a.strides[::-1]))
            m,n=changes[op.output.buffer]
            output=replace(op.output,shape=(n,m),strides=(m,1))
            operations.append(replace(op,inputs=tuple(view(x) for x in operands),output=output))
        else:
            operations.append(replace(op,inputs=tuple(view(x) for x in op.inputs)))
    outputs=tuple(view(x) for x in program.outputs)
    assert all(isinstance(x,TensorSpec) for x in outputs)
    return Program(program.inputs,tuple(operations),outputs)
