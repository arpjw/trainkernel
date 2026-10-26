"""Small, serialisable intermediate representation for megakernel programs."""
from dataclasses import dataclass, field

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
