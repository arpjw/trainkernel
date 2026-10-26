"""Static FX capture for pure tensor functions, including functional training."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from .codegen import contiguous_strides
from .ir import Operation, Program, TensorSpec

_DTYPES = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}
_VIEWS = {"aten.alias.default", "aten.detach.default", "aten.t.default", "aten.transpose.int", "aten.permute.default", "aten.view.default", "aten._unsafe_view.default", "aten.reshape.default", "aten.expand.default", "aten.squeeze.default", "aten.squeeze.dim", "aten.unsqueeze.default"}
_BINARY = {"add", "sub", "mul", "div", "pow"}


def _metadata(node: torch.fx.Node) -> tuple[tuple[int, ...], str]:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"megakernel: missing tensor metadata for {node.name}")
    if value.dtype not in _DTYPES:
        raise ValueError(f"megakernel: unsupported dtype {value.dtype} at {node.name}")
    if any(not isinstance(dim, int) for dim in value.shape):
        raise ValueError("megakernel: dynamic shapes are unsupported")
    return tuple(value.shape), _DTYPES[value.dtype]


def _view(node: torch.fx.Node, root: TensorSpec, shape: tuple[int, ...], dtype: str) -> TensorSpec:
    target = str(node.target)
    strides = root.strides
    if target in {"aten.t.default", "aten.transpose.int", "aten.permute.default"}:
        axes = list(range(len(root.shape)))
        if target == "aten.t.default":
            axes = axes[::-1]
        elif target == "aten.transpose.int":
            a, b = int(node.args[1]), int(node.args[2])
            axes[a], axes[b] = axes[b], axes[a]
        else:
            axes = [int(a) for a in node.args[1]]
        strides = tuple(root.strides[a] for a in axes)
    elif target == "aten.expand.default":
        padded_shape = (1,) * (len(shape) - len(root.shape)) + root.shape
        padded_stride = (0,) * (len(shape) - len(root.shape)) + root.strides
        strides = tuple(0 if old == 1 and new != 1 else stride for old, new, stride in zip(padded_shape, shape, padded_stride))
    elif target not in {"aten.alias.default", "aten.detach.default"}:
        # Derive the view from our chosen physical layout, which can differ from
        # FX metadata after we materialize an operation into contiguous storage.
        try:
            meta = torch.empty_strided(root.shape, root.strides, device="meta")
            strides = tuple(meta.view(shape).stride())
        except RuntimeError as exc:
            raise ValueError(f"megakernel: {target} requires a contiguous copy; use .contiguous() before reshaping") from exc
    return TensorSpec(node.name, shape, dtype, strides, root.buffer, root.offset)


def capture(fn: Callable[..., Any], example_inputs: Sequence[Any]) -> tuple[Program, TreeSpec]:
    """Trace a pure function. Inputs may contain nested tensor pytrees.

    Scalars should be captured Python constants. Tensor constants, data-dependent
    control flow, input mutations, and unsupported ATen operations are rejected.
    Fake capture avoids running training updates against the user's real tensors.
    """
    flat_inputs, input_spec = tree_flatten(tuple(example_inputs))
    if not flat_inputs or any(not isinstance(x, torch.Tensor) for x in flat_inputs):
        raise TypeError("megakernel: example_inputs must contain tensors only and cannot be empty")
    if any(x.dtype not in _DTYPES for x in flat_inputs):
        raise ValueError("megakernel: unsupported dtype; expected float32, float16, or bfloat16")
    if any(x.layout != torch.strided for x in flat_inputs):
        raise ValueError("megakernel: only strided tensors are supported")
    from torch.func import functionalize
    from torch.fx.experimental.proxy_tensor import make_fx

    def flattened(*args: torch.Tensor) -> Any:
        return fn(*tree_unflatten(list(args), input_spec))

    try:
        gm = make_fx(functionalize(flattened), tracing_mode="fake", _allow_non_fake_inputs=True)(*flat_inputs)
    except Exception as exc:
        raise RuntimeError(f"megakernel: FX capture failed: {exc}") from exc

    specs: dict[torch.fx.Node, TensorSpec] = {}
    inputs: list[TensorSpec] = []
    operations: list[Operation] = []
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        if node.op == "placeholder":
            shape, dtype = _metadata(node)
            tensor = flat_inputs[len(inputs)]
            spec = TensorSpec(node.name, shape, dtype, tuple(tensor.stride()), node.name)
            specs[node] = spec
            inputs.append(spec)
            continue
        if node.op != "call_function":
            raise ValueError(f"megakernel: unsupported FX node {node.op} ({node.target}); pass tensor constants as explicit inputs")
        target = str(node.target)
        if target.startswith("aten.copy_"):
            raise ValueError("megakernel: input mutation is unsupported; return updated tensors instead")
        shape, dtype = _metadata(node)
        tensor_args = [specs[arg] for arg in node.args if isinstance(arg, torch.fx.Node)]
        if target in _VIEWS:
            specs[node] = _view(node, tensor_args[0], shape, dtype)
            continue
        short = target.split(".")[1] if target.startswith("aten.") else target
        attrs: dict[str, object] = {}
        operands: tuple[TensorSpec | int | float, ...]

        def value(arg: Any) -> TensorSpec | int | float:
            if isinstance(arg, torch.fx.Node):
                return specs[arg]
            if isinstance(arg, (int, float)):
                return arg
            raise ValueError(f"megakernel: unsupported argument in {target}: {type(arg).__name__}")

        if short in _BINARY:
            if short == "div" and (len(node.args) > 2 or node.kwargs.get("rounding_mode") is not None):
                raise ValueError("megakernel: rounded division is unsupported")
            operands = tuple(value(arg) for arg in node.args[:2])
            if short == "pow" and not isinstance(operands[1], (int, float)):
                raise ValueError("megakernel: pow supports a scalar exponent only")
            if short in {"add", "sub"}:
                attrs["alpha"] = node.kwargs.get("alpha", node.args[2] if len(node.args) > 2 else 1)
            op = short
        elif short == "addmm":
            operands = tuple(tensor_args)
            attrs.update(alpha=node.kwargs.get("alpha", 1), beta=node.kwargs.get("beta", 1))
            op = short
        elif short in {"neg", "relu", "mm"}:
            operands = tuple(tensor_args)
            op = short
        elif short in {"sum", "mean"}:
            root = tensor_args[0]
            dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
            if dims is None or dims == [] or dims == ():
                dims = tuple(range(len(root.shape)))
            elif isinstance(dims, int):
                dims = (dims,)
            dims = tuple(sorted({int(d) % len(root.shape) for d in dims})) if root.shape else ()
            attrs.update(dims=dims, keepdim=bool(node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)))
            operands = (root,)
            op = short
        elif short == "threshold_backward":
            operands = tuple(tensor_args[:2])
            attrs["threshold"] = node.args[2]
            op = short
        elif short in {"ones_like", "zeros_like", "full_like"}:
            fill = 1 if short == "ones_like" else 0 if short == "zeros_like" else node.args[1]
            operands = (value(fill),)
            attrs["value"] = fill
            op = "full"
        elif short in {"_to_copy", "clone"}:
            operands = (tensor_args[0],)
            op = "cast"
        else:
            raise ValueError(f"megakernel: unsupported ATen op {target} at {node.name}")
        output = TensorSpec(node.name, shape, dtype, contiguous_strides(shape), node.name)
        specs[node] = output
        operations.append(Operation(node.name, op, operands, output, attrs))
    out_node = next(node for node in gm.graph.nodes if node.op == "output")
    flat_outputs, output_spec = tree_flatten(out_node.args[0])
    if any(not isinstance(node, torch.fx.Node) or node not in specs for node in flat_outputs):
        raise ValueError("megakernel: outputs must contain tensors only")
    return Program(tuple(inputs), tuple(operations), tuple(specs[node] for node in flat_outputs)), output_spec


def compile(fn: Callable[..., Any], example_inputs: Sequence[Any], **options: Any) -> Callable[..., Any]:
    """Compile a static pure function; explicit training graphs return gradients."""
    program, output_spec = capture(fn, example_inputs)
    _, input_spec = tree_flatten(tuple(example_inputs))
    from .runtime import compile_program
    runner = compile_program(program, example_inputs, **options)

    def wrapped(*args: Any) -> Any:
        flat, actual_spec = tree_flatten(args)
        if actual_spec != input_spec:
            raise ValueError("megakernel: runtime input pytree differs from the compiled signature")
        return tree_unflatten(list(runner(*flat)), output_spec)

    wrapped.program = program
    wrapped.runner = runner
    wrapped.source = runner.source
    return wrapped
