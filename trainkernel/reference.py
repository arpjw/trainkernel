"""CPU/PyTorch IR interpreter for validation; never a runtime fallback."""
from __future__ import annotations

from collections.abc import Sequence
import torch
from .ir import Program, TensorSpec


def run(program: Program, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != len(program.inputs):
        raise ValueError("reference: incorrect input count")
    env = {spec.buffer: tensor for spec, tensor in zip(program.inputs, inputs)}

    def read(spec: TensorSpec) -> torch.Tensor:
        root = env[spec.buffer]
        return torch.as_strided(root, spec.shape, spec.strides, root.storage_offset() + spec.offset)

    def value(operand: TensorSpec | int | float) -> torch.Tensor | int | float:
        return read(operand) if isinstance(operand, TensorSpec) else operand

    for op in program.operations:
        args = [value(x) for x in op.inputs]
        kind = op.op
        if kind == "mm":
            result = args[0] @ args[1]
        elif kind == "addmm":
            result = torch.addmm(args[0], args[1], args[2], alpha=op.attrs.get("alpha", 1), beta=op.attrs.get("beta", 1))
        elif kind == "add":
            result = torch.add(args[0], args[1], alpha=op.attrs.get("alpha", 1))
        elif kind == "sub":
            result = torch.sub(args[0], args[1], alpha=op.attrs.get("alpha", 1))
        elif kind == "mul":
            result = args[0] * args[1]
        elif kind == "div":
            result = args[0] / args[1]
        elif kind == "neg":
            result = -args[0]
        elif kind == "relu":
            result = torch.relu(args[0])
        elif kind == "pow":
            result = args[0] ** args[1]
        elif kind in {"sum", "mean"}:
            result = getattr(args[0], kind)(dim=op.attrs["dims"], keepdim=op.attrs["keepdim"], dtype=getattr(torch, op.output.dtype))
        elif kind == "threshold_backward":
            result = torch.where(args[1] <= op.attrs["threshold"], 0, args[0])
        elif kind == "full":
            result = torch.full(op.output.shape, op.attrs["value"], dtype=getattr(torch, op.output.dtype), device=inputs[0].device)
        elif kind == "cast":
            result = args[0].to(dtype=getattr(torch, op.output.dtype))
        else:
            raise ValueError(f"reference: unsupported operation {kind}")
        env[op.output.buffer] = result.to(dtype=getattr(torch, op.output.dtype)).contiguous()
    return tuple(read(spec) for spec in program.outputs)
