"""CUDA extension compilation and a checked, static execution boundary."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from .codegen import GeneratedKernel, generate
from .ir import Program


@dataclass
class ExecutionBuffers:
    """Caller-owned storage. Reuse only when previous work on it is ordered."""

    outputs: tuple[torch.Tensor, ...]
    workspace: torch.Tensor


class CompiledProgram:
    """One specialized kernel. Calls allocate independent buffers for stream safety.

    The result has no autograd registration: compile a functional training step
    containing explicit differentiation to compute gradients and parameter updates.
    """

    def __init__(self, program: Program, generated: GeneratedKernel, module: Any, device: torch.device):
        self.program = program
        self.generated = generated
        self.source = generated.source
        self.module = module
        self.device = device

    def _validate_inputs(self, inputs: tuple[torch.Tensor, ...]) -> None:
        if len(inputs) != len(self.program.inputs):
            raise ValueError(f"megakernel: expected {len(self.program.inputs)} inputs, got {len(inputs)}")
        for i, (tensor, spec) in enumerate(zip(inputs, self.program.inputs)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"megakernel: input {i} must be a tensor")
            if tensor.device != self.device or tuple(tensor.shape) != spec.shape or tuple(tensor.stride()) != spec.strides or tensor.dtype != getattr(torch, spec.dtype):
                raise ValueError(f"megakernel: input {i} does not match compiled device, shape, stride, or dtype")

    def allocate(self) -> ExecutionBuffers:
        return ExecutionBuffers(
            tuple(torch.empty(spec.shape, dtype=getattr(torch, spec.dtype), device=self.device) for spec in self.program.outputs),
            torch.empty(self.generated.workspace_elements, dtype=torch.float32, device=self.device),
        )

    def run_into(self, *inputs: torch.Tensor, buffers: ExecutionBuffers) -> tuple[torch.Tensor, ...]:
        self._validate_inputs(inputs)
        if len(buffers.outputs) != len(self.program.outputs):
            raise ValueError("megakernel: incorrect output buffer count")
        if buffers.workspace.device != self.device or buffers.workspace.dtype != torch.float32 or not buffers.workspace.is_contiguous() or buffers.workspace.numel() < self.generated.workspace_elements:
            raise ValueError("megakernel: workspace does not match compiled requirements")
        # Caller buffers must be disjoint from each other and the inputs. A
        # conservative storage-level check prevents writes clobbering dependencies.
        output_storages = []
        for out, spec in zip(buffers.outputs, self.program.outputs):
            if out.device != self.device or out.dtype != getattr(torch, spec.dtype) or tuple(out.shape) != spec.shape or not out.is_contiguous():
                raise ValueError("megakernel: output buffer does not match compiled requirements")
            if out.numel():
                output_storages.append(out.untyped_storage().data_ptr())
        if buffers.workspace.numel():
            output_storages.append(buffers.workspace.untyped_storage().data_ptr())
        input_storages = {tensor.untyped_storage().data_ptr() for tensor in inputs if tensor.numel()}
        if len(set(output_storages)) != len(output_storages) or input_storages.intersection(output_storages):
            raise ValueError("megakernel: execution buffers must not alias inputs or each other")
        self.module.launch(list(inputs), list(buffers.outputs), buffers.workspace)
        return buffers.outputs

    def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.run_into(*inputs, buffers=self.allocate())


def compile_program(program: Program, example_inputs: tuple[Any, ...] | list[Any], *, blocks_per_sm: int = 1, fuse_pointwise: bool = True, verbose: bool = False) -> CompiledProgram:
    """Compile for the current input device, requiring Linux, CUDA and SM80+."""
    flat, _ = tree_flatten(example_inputs)
    if not flat or any(not isinstance(x, torch.Tensor) for x in flat):
        raise TypeError("megakernel: nonempty tensor inputs required")
    if not torch.cuda.is_available() or any(x.device.type != "cuda" for x in flat):
        raise RuntimeError("megakernel: CUDA SM80+ GPU required; CPU IR inspection is available through capture()")
    device = flat[0].device
    if any(x.device != device for x in flat):
        raise ValueError("megakernel: all inputs must use the same CUDA device")
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 0):
        raise RuntimeError("megakernel: NVIDIA compute capability 8.0 or newer required")
    generated = generate(program, blocks_per_sm=blocks_per_sm, fuse_pointwise=fuse_pointwise)
    from torch.utils.cpp_extension import CUDA_HOME, load_inline
    if CUDA_HOME is None:
        raise RuntimeError("megakernel: CUDA toolkit with nvcc is required to compile generated kernels")
    name = "megakernel_" + hashlib.sha256((generated.source + str(capability) + torch.__version__).encode()).hexdigest()[:20]
    # Restrict compilation to the actual target rather than relying on a global
    # TORCH_CUDA_ARCH_LIST, which may target a different GPU or unsupported SM.
    arch = f"-gencode=arch=compute_{capability[0]}{capability[1]},code=sm_{capability[0]}{capability[1]}"
    with torch.cuda.device(device):
        module = load_inline(
            name=name,
            cpp_sources="void launch(std::vector<torch::Tensor>, std::vector<torch::Tensor>, torch::Tensor);",
            cuda_sources=generated.source,
            functions=["launch"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--fmad=false", arch],
            extra_cflags=["-O3"],
            with_cuda=True,
            verbose=verbose,
        )
    result = CompiledProgram(program, generated, module, device)
    result._validate_inputs(tuple(flat))
    return result
