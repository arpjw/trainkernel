"""CUDA extension compilation and a checked, static execution boundary."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
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

    def __init__(self, program: Program, generated: GeneratedKernel, module: Any, device: torch.device, inplace_outputs: dict[int, int] | None = None):
        self.program = program
        self.generated = generated
        self.source = generated.source
        self.module = module
        self.device = device
        self.inplace_outputs = dict(inplace_outputs or {})

    def resource_usage(self) -> dict[str, int]:
        """Compiler resource attributes; these do not require GPU counters."""
        with torch.cuda.device(self.device):
            values = self.module.resource_usage()
        return dict(zip(("registers_per_thread", "local_bytes_per_thread", "shared_bytes_per_block", "resident_blocks_per_sm"), values))

    def _validate_inputs(self, inputs: tuple[torch.Tensor, ...]) -> None:
        if len(inputs) != len(self.program.inputs):
            raise ValueError(f"trainkernel: expected {len(self.program.inputs)} inputs, got {len(inputs)}")
        for i, (tensor, spec) in enumerate(zip(inputs, self.program.inputs)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"trainkernel: input {i} must be a tensor")
            if tensor.is_neg() or tensor.is_conj():
                raise ValueError("trainkernel: resolve lazy negative or conjugate input views before execution")
            if tensor.device != self.device or tuple(tensor.shape) != spec.shape or tuple(tensor.stride()) != spec.strides or tensor.dtype != getattr(torch, spec.dtype):
                raise ValueError(f"trainkernel: input {i} does not match compiled device, shape, stride, or dtype")
        if any(inputs[i].requires_grad for i in self.inplace_outputs.values()):
            raise ValueError("trainkernel: inplace targets must be detached from autograd")

    def allocate(self, *inputs: torch.Tensor) -> ExecutionBuffers:
        if self.inplace_outputs:
            self._validate_inputs(inputs)
        return ExecutionBuffers(
            tuple(inputs[self.inplace_outputs[i]] if i in self.inplace_outputs else torch.empty(spec.shape, dtype=getattr(torch, spec.dtype), device=self.device) for i, spec in enumerate(self.program.outputs)),
            torch.empty(self.generated.workspace_elements, dtype=torch.float32, device=self.device),
        )

    def run_into(self, *inputs: torch.Tensor, buffers: ExecutionBuffers) -> tuple[torch.Tensor, ...]:
        self._validate_inputs(inputs)
        if len(buffers.outputs) != len(self.program.outputs):
            raise ValueError("trainkernel: incorrect output buffer count")
        if buffers.workspace.device != self.device or buffers.workspace.dtype != torch.float32 or not buffers.workspace.is_contiguous() or buffers.workspace.numel() < self.generated.workspace_elements:
            raise ValueError("trainkernel: workspace does not match compiled requirements")
        # Caller buffers must be disjoint from each other and the inputs. A
        # conservative storage-level check prevents writes clobbering dependencies.
        output_storages = []
        allowed_storages = set()
        for slot, (out, spec) in enumerate(zip(buffers.outputs, self.program.outputs)):
            if out.device != self.device or out.dtype != getattr(torch, spec.dtype) or tuple(out.shape) != spec.shape or not out.is_contiguous():
                raise ValueError("trainkernel: output buffer does not match compiled requirements")
            if out.numel():
                output_storages.append(out.untyped_storage().data_ptr())
            if slot in self.inplace_outputs:
                target_index = self.inplace_outputs[slot]
                target = inputs[target_index]
                if out.data_ptr() != target.data_ptr() or tuple(out.stride()) != tuple(target.stride()):
                    raise ValueError("trainkernel: inplace output must bind its declared input")
                if target.numel():
                    storage = target.untyped_storage().data_ptr()
                    if any(i != target_index and value.numel() and value.untyped_storage().data_ptr() == storage for i, value in enumerate(inputs)):
                        raise ValueError("trainkernel: inplace target must not alias another input")
                    allowed_storages.add(storage)
        if buffers.workspace.numel():
            output_storages.append(buffers.workspace.untyped_storage().data_ptr())
        input_storages = {tensor.untyped_storage().data_ptr() for tensor in inputs if tensor.numel()}
        if len(set(output_storages)) != len(output_storages) or input_storages.intersection(output_storages) - allowed_storages:
            raise ValueError("trainkernel: execution buffers must not alias inputs or each other")
        self.module.launch(list(inputs), list(buffers.outputs), buffers.workspace)
        if self.inplace_outputs:
            torch.autograd.graph.increment_version([inputs[i] for i in self.inplace_outputs.values()])
        return buffers.outputs

    def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.run_into(*inputs, buffers=self.allocate(*inputs))


def compile_program(program: Program, example_inputs: tuple[Any, ...] | list[Any], *, blocks_per_sm: int = 1, max_blocks: int | None = None, fuse_pointwise: bool = True, fuse_epilogues: bool = True, gemm_k_tile: int = 32, gemm_tile_n: int = 32, shared_skew: int = 0, split_k_gemv: bool = False, threads_per_block: int = 128, inline_input_casts: bool = True, hopper_wgmma: bool = False, inplace_outputs: dict[int, int] | None = None, verbose: bool = False) -> CompiledProgram:
    """Compile for the current input device, requiring Linux, CUDA and SM80+."""
    flat, _ = tree_flatten(example_inputs)
    if not flat or any(not isinstance(x, torch.Tensor) for x in flat):
        raise TypeError("trainkernel: nonempty tensor inputs required")
    if len(flat) != len(program.inputs):
        raise ValueError("trainkernel: input count differs from the program signature")
    if not torch.cuda.is_available() or any(x.device.type != "cuda" for x in flat):
        raise RuntimeError("trainkernel: CUDA SM80+ GPU required; CPU IR inspection is available through capture()")
    device = flat[0].device
    if any(x.device != device for x in flat):
        raise ValueError("trainkernel: all inputs must use the same CUDA device")
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 0):
        raise RuntimeError("trainkernel: NVIDIA compute capability 8.0 or newer required")
    if type(hopper_wgmma) is not bool:
        raise ValueError("trainkernel: hopper_wgmma must be a boolean")
    if hopper_wgmma and capability != (9, 0):
        raise RuntimeError("trainkernel: hopper_wgmma requires an SM90a Hopper GPU")
    generated = generate(program, blocks_per_sm=blocks_per_sm, max_blocks=max_blocks, fuse_pointwise=fuse_pointwise, fuse_epilogues=fuse_epilogues, gemm_k_tile=gemm_k_tile, gemm_tile_n=gemm_tile_n, shared_skew=shared_skew, split_k_gemv=split_k_gemv, threads_per_block=threads_per_block, inline_input_casts=inline_input_casts, hopper_wgmma=hopper_wgmma, inplace_outputs=inplace_outputs)
    if any(flat[i].requires_grad for i in (inplace_outputs or {}).values()):
        raise ValueError("trainkernel: inplace targets must be detached from autograd")
    from torch.utils.cpp_extension import CUDA_HOME, load_inline
    if CUDA_HOME is None:
        raise RuntimeError("trainkernel: CUDA toolkit with nvcc is required to compile generated kernels")
    # Restrict compilation to the actual target rather than relying on a global
    # TORCH_CUDA_ARCH_LIST, which may target a different GPU or unsupported SM.
    target = "90a" if hopper_wgmma else f"{capability[0]}{capability[1]}"
    cache_identity = (generated.source, capability, target, hopper_wgmma, torch.__version__)
    name = "trainkernel_" + hashlib.sha256(repr(cache_identity).encode()).hexdigest()[:20]
    arch = f"-gencode=arch=compute_{target},code=sm_{target}"
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if hopper_wgmma:
        # PyTorch otherwise adds a generic sm_90 pass which cannot assemble
        # architecture-conditional WGMMA instructions.
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        with torch.cuda.device(device):
            module = load_inline(
                name=name,
                cpp_sources="void launch(std::vector<torch::Tensor>, std::vector<torch::Tensor>, torch::Tensor); std::vector<int64_t> resource_usage();",
                cuda_sources=generated.source,
                functions=["launch", "resource_usage"],
                extra_cuda_cflags=["-O3", "-std=c++17", "--fmad=false", arch],
                extra_cflags=["-O3"],
                with_cuda=True,
                verbose=verbose,
            )
    finally:
        if hopper_wgmma:
            if previous_arch_list is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list
    result = CompiledProgram(program, generated, module, device, inplace_outputs)
    result._validate_inputs(tuple(flat))
    return result
