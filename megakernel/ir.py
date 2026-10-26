"""Small, serialisable intermediate representation for megakernel programs."""
from dataclasses import dataclass, field
import math

@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    strides: tuple[int, ...]
    buffer: str
    offset: int = 0

@dataclass(frozen=True)
class Operation:
    name: str
    op: str
    inputs: tuple[TensorSpec | int | float, ...]
    output: TensorSpec
    attrs: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class Program:
    inputs: tuple[TensorSpec, ...]
    operations: tuple[Operation, ...]
    outputs: tuple[TensorSpec, ...]


def validate(program: Program) -> None:
    """Check SSA ordering and view bounds before generating memory accesses."""
    roots: dict[str, tuple[TensorSpec, int]] = {}

    def metadata(spec: TensorSpec) -> None:
        if len(spec.shape) != len(spec.strides) or any(type(d) is not int or d < 0 for d in spec.shape) or any(type(s) is not int or s < 0 for s in spec.strides) or type(spec.offset) is not int or spec.offset < 0:
            raise ValueError("megakernel: invalid tensor shape, stride, or offset")

    def read(spec: TensorSpec) -> None:
        metadata(spec)
        if spec.buffer not in roots:
            raise ValueError(f"megakernel: undefined or future buffer {spec.buffer}")
        root, extent = roots[spec.buffer]
        if root.dtype != spec.dtype:
            raise ValueError("megakernel: views must preserve the root dtype")
        if math.prod(spec.shape) and spec.offset + sum((d - 1) * s for d, s in zip(spec.shape, spec.strides)) >= extent:
            raise ValueError("megakernel: tensor view exceeds its root allocation")

    for spec in program.inputs:
        metadata(spec)
        if spec.buffer in roots or spec.offset:
            raise ValueError("megakernel: input roots must be distinct and have zero offset")
        extent = 0 if not math.prod(spec.shape) else 1 + sum((d - 1) * s for d, s in zip(spec.shape, spec.strides))
        roots[spec.buffer] = (spec, extent)
    for op in program.operations:
        for operand in op.inputs:
            if isinstance(operand, TensorSpec):
                read(operand)
        spec = op.output
        metadata(spec)
        if spec.buffer in roots or spec.offset:
            raise ValueError("megakernel: operation outputs must define new roots at offset zero")
        roots[spec.buffer] = (spec, math.prod(spec.shape))
    for spec in program.outputs:
        read(spec)
