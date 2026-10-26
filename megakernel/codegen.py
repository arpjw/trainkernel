"""Static CUDA code generation for a cooperative persistent training kernel.

The initial scheduler is deliberately conservative: operators execute in topological
order with a grid barrier. Single-use pointwise values are evaluated inline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from .ir import Operation, Program, TensorSpec

_DTYPES = {"float32": "float", "float16": "__half", "bfloat16": "__nv_bfloat16"}
_POINTWISE = {"add", "sub", "mul", "div", "neg", "relu", "pow", "threshold_backward", "full", "cast"}


def numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))


def _literal(value: int | float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("megakernel: non-finite scalar constants are unsupported")
    return f"{value:.17e}f"


def _round(expr: str, dtype: str) -> str:
    if dtype == "float32":
        return f"({expr})"
    if dtype == "bfloat16":
        return f"__bfloat162float(__float2bfloat16_rn({expr}))"
    if dtype == "float16":
        return f"__half2float(__float2half_rn({expr}))"
    raise ValueError(f"megakernel: unsupported CUDA dtype {dtype}")


@dataclass(frozen=True)
class GeneratedKernel:
    source: str
    workspace_elements: int
    materialized_operations: int
    inlined_operations: int
    tensor_core_operations: int
    blocks_per_sm: int
    max_work_blocks: int
    grid_barriers: int
    direct_outputs: int


class _Generator:
    def __init__(self, program: Program, *, blocks_per_sm: int, fuse_pointwise: bool):
        self.program = program
        self.blocks_per_sm = blocks_per_sm
        self.producers = {op.output.buffer: op for op in program.operations}
        self.input_indices = {spec.buffer: i for i, spec in enumerate(program.inputs)}
        counts: Counter[str] = Counter()
        for op in program.operations:
            counts.update(x.buffer for x in op.inputs if isinstance(x, TensorSpec))
        output_buffers = {x.buffer for x in program.outputs}
        # Do not duplicate a pointwise expression across multiple consumers or outputs.
        self.inline = {
            op.output.buffer for op in program.operations
            if fuse_pointwise and op.op in _POINTWISE
            and counts[op.output.buffer] == 1 and op.output.buffer not in output_buffers
        }
        self.output_bindings: dict[str, tuple[int, str]] = {}
        for index, spec in enumerate(program.outputs):
            producer = self.producers.get(spec.buffer)
            if producer is not None and producer.output.dtype == spec.dtype and spec.offset == 0 and numel(spec.shape) == numel(producer.output.shape) and all(dim == 1 or a == b for dim, a, b in zip(spec.shape, spec.strides, contiguous_strides(spec.shape))):
                self.output_bindings.setdefault(spec.buffer, (index, spec.dtype))
        self.materialized = [op for op in program.operations if op.output.buffer not in self.inline]
        self.offsets, self.workspace_elements = self._plan_workspace()
        self.tensor_core_count = 0

    def _plan_workspace(self) -> tuple[dict[str, int], int]:
        # A small best-fit arena reuses storage after the last materialized reader.
        def roots(spec: TensorSpec) -> set[str]:
            if spec.buffer not in self.inline:
                return {spec.buffer}
            return set().union(*(roots(x) for x in self.producers[spec.buffer].inputs if isinstance(x, TensorSpec)))
        last: dict[str, int] = {}
        for i, op in enumerate(self.materialized):
            for x in op.inputs:
                if isinstance(x, TensorSpec):
                    for root in roots(x):
                        last[root] = i
        for spec in self.program.outputs:
            for root in roots(spec):
                last[root] = len(self.materialized)
        offsets: dict[str, int] = {}
        active: list[tuple[str, int, int]] = []
        free: list[tuple[int, int]] = []
        total = 0
        for i, op in enumerate(self.materialized):
            remaining = []
            for root, start, size in active:
                if last.get(root, -1) < i:
                    free.append((start, size))
                else:
                    remaining.append((root, start, size))
            active = remaining
            size = max(64, ((numel(op.output.shape) + 63) // 64) * 64)
            candidates = [(n, j, start) for j, (start, n) in enumerate(free) if n >= size]
            if candidates:
                n, j, start = min(candidates)
                free.pop(j)
                if n > size:
                    free.append((start + size, n - size))
            else:
                start = total
                total += size
            offsets[op.output.buffer] = start
            active.append((op.output.buffer, start, size))
        return offsets, max(total, 1)

    @staticmethod
    def coords(shape: tuple[int, ...], index: str) -> tuple[str, ...]:
        return tuple(f"((({index}) / {max(1, math.prod(shape[i + 1:]))}LL) % {max(1, dim)}LL)" for i, dim in enumerate(shape))

    def read(self, spec: TensorSpec, coords: tuple[str, ...]) -> str:
        index = " + ".join([f"{spec.offset}LL"] + [f"(({c}) * {s}LL)" for c, s in zip(coords, spec.strides) if s])
        if spec.buffer in self.inline:
            return _round(self.point_expr(self.producers[spec.buffer], index), spec.dtype)
        if spec.buffer in self.input_indices:
            i = self.input_indices[spec.buffer]
            raw = f"static_cast<const {_DTYPES[spec.dtype]}*>(args.inputs[{i}])[{index}]"
            if spec.dtype == "bfloat16":
                return f"__bfloat162float({raw})"
            if spec.dtype == "float16":
                return f"__half2float({raw})"
            return raw
        if spec.buffer in self.output_bindings:
            slot, dtype = self.output_bindings[spec.buffer]
            raw = f"static_cast<const {_DTYPES[dtype]}*>(args.outputs[{slot}])[{index}]"
            return f"__bfloat162float({raw})" if dtype == "bfloat16" else f"__half2float({raw})" if dtype == "float16" else raw
        return f"workspace[{self.offsets[spec.buffer]}LL + ({index})]"

    def store(self, spec: TensorSpec, index: str, value: str) -> str:
        if spec.buffer in self.output_bindings:
            slot, dtype = self.output_bindings[spec.buffer]
            cast = f"__float2bfloat16_rn({value})" if dtype == "bfloat16" else f"__float2half_rn({value})" if dtype == "float16" else value
            return f"static_cast<{_DTYPES[dtype]}*>(args.outputs[{slot}])[{index}] = {cast};"
        return f"workspace[{self.offsets[spec.buffer]}LL + ({index})] = {_round(value, spec.dtype)};"

    def operand(self, operand: TensorSpec | int | float, output: TensorSpec, index: str) -> str:
        if not isinstance(operand, TensorSpec):
            return _literal(operand)
        coords = self.coords(output.shape, index)
        if len(operand.shape) > len(coords):
            raise ValueError("megakernel: invalid broadcast rank")
        trailing = coords[len(coords) - len(operand.shape):] if operand.shape else ()
        return self.read(operand, tuple("0" if dim == 1 else c for dim, c in zip(operand.shape, trailing)))

    def point_expr(self, op: Operation, index: str) -> str:
        values = [self.operand(x, op.output, index) for x in op.inputs]
        if op.op in {"add", "sub"}:
            sign = "+" if op.op == "add" else "-"
            return f"({values[0]} {sign} ({_literal(op.attrs.get('alpha', 1))} * {values[1]}))"
        if op.op in {"mul", "div"}:
            sign = "*" if op.op == "mul" else "/"
            return f"({values[0]} {sign} {values[1]})"
        if op.op == "neg":
            return f"(-({values[0]}))"
        if op.op == "relu":
            return f"mk_relu({values[0]})"
        if op.op == "pow":
            exponent = op.inputs[1] if len(op.inputs) > 1 else op.attrs.get("exponent", 2)
            if not isinstance(exponent, (float, int)):
                raise ValueError("megakernel: pow exponent must be a scalar")
            if exponent == 2:
                return f"mk_square({values[0]})"
            return f"powf({values[0]}, {_literal(exponent)})"
        if op.op == "threshold_backward":
            return f"(({values[1]} <= {_literal(op.attrs.get('threshold', 0))}) ? 0.0f : {values[0]})"
        if op.op == "full":
            return _literal(op.attrs.get("value", op.inputs[0] if op.inputs else 0))
        if op.op == "cast":
            return values[0]
        raise ValueError(f"megakernel: unsupported pointwise operation {op.op}")

    def emit_pointwise(self, op: Operation) -> str:
        expr = self.point_expr(op, "i")
        return f"""for (int64_t i = tid; i < {numel(op.output.shape)}LL; i += nt) {{
            {self.store(op.output, 'i', expr)}
        }}"""

    def emit_reduction(self, op: Operation) -> str:
        source = op.inputs[0]
        if not isinstance(source, TensorSpec):
            raise ValueError("megakernel: reduction requires a tensor")
        dims = tuple(int(d) % len(source.shape) for d in op.attrs.get("dims", tuple(range(len(source.shape))))) if source.shape else ()
        dims = tuple(sorted(set(dims)))
        kept = [d for d in range(len(source.shape)) if d not in dims]
        kept_shape = tuple(source.shape[d] for d in kept)
        reduced_shape = tuple(source.shape[d] for d in dims)
        out_coords = self.coords(kept_shape, "row")
        red_coords = self.coords(reduced_shape, "r")
        coordinates = ["0"] * len(source.shape)
        for d, coord in zip(kept, out_coords):
            coordinates[d] = coord
        for d, coord in zip(dims, red_coords):
            coordinates[d] = coord
        value = self.read(source, tuple(coordinates))
        size = numel(reduced_shape)
        if size <= 128 and numel(op.output.shape) >= 8:
            finish = "acc" if op.op == "sum" else f"(acc / {size}.0f)" if size else "NAN"
            return f"""for (int64_t row = tid; row < {numel(op.output.shape)}LL; row += nt) {{
                float acc = 0.0f;
                for (int64_t r = 0; r < {size}LL; ++r) acc += {value};
                {self.store(op.output, 'row', finish)}
            }}"""
        finish = "shared_reduce[0]" if op.op == "sum" else f"(shared_reduce[0] / {size}.0f)" if size else "NAN"
        return f"""for (int64_t row = blockIdx.x; row < {numel(op.output.shape)}LL; row += gridDim.x) {{
            float acc = 0.0f;
            for (int64_t r = threadIdx.x; r < {size}LL; r += blockDim.x) acc += {value};
            shared_reduce[threadIdx.x] = acc;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
                if (threadIdx.x < s) shared_reduce[threadIdx.x] += shared_reduce[threadIdx.x + s];
                __syncthreads();
            }}
            if (threadIdx.x == 0) {self.store(op.output, 'row', finish)}
            __syncthreads();
        }}"""

    def emit_mm(self, op: Operation) -> str:
        a, b = op.inputs[1:3] if op.op == "addmm" else op.inputs[:2]
        if not isinstance(a, TensorSpec) or not isinstance(b, TensorSpec) or len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("megakernel: mm requires compatible rank-two tensors")
        m, k = a.shape
        _, n = b.shape
        def epilogue(acc: str, index: str) -> str:
            if op.op != "addmm":
                return acc
            alpha, beta = op.attrs.get("alpha", 1), op.attrs.get("beta", 1)
            if alpha == 0 and beta == 0:
                return "0.0f"
            if beta == 0:
                return f"({_literal(alpha)} * ({acc}))"
            bias = self.operand(op.inputs[0], op.output, index)
            if alpha == 0:
                return f"({_literal(beta)} * ({bias}))"
            return f"({_literal(alpha)} * ({acc}) + {_literal(beta)} * ({bias}))"
        if a.dtype == b.dtype and a.dtype in {"bfloat16", "float16"}:
            self.tensor_core_count += 1
            cast = "__float2bfloat16_rn" if a.dtype == "bfloat16" else "__float2half_rn"
            typ = _DTYPES[a.dtype]
            tile_size = 16 if m <= 16 and n <= 16 else 32
            warp_columns = tile_size // 16
            active_warps = warp_columns * warp_columns
            av = self.read(a, ("ar", "ak"))
            bv = self.read(b, ("bk", "bc"))
            return f"""{{
            auto* sa = reinterpret_cast<{typ}*>(shared_a);
            auto* sb = reinterpret_cast<{typ}*>(shared_b);
            const int warp = threadIdx.x / 32;
            for (int64_t tile = blockIdx.x; tile < {((m + tile_size - 1)//tile_size)*((n + tile_size - 1)//tile_size)}LL; tile += gridDim.x) {{
                const int64_t row = (tile / {(n+tile_size-1)//tile_size or 1}LL) * {tile_size};
                const int64_t col = (tile % {(n+tile_size-1)//tile_size or 1}LL) * {tile_size};
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
                nvcuda::wmma::fill_fragment(acc, 0.0f);
                for (int64_t kk = 0; kk < {k}LL; kk += 16) {{
                    for (int p = threadIdx.x; p < {tile_size * 16}; p += blockDim.x) {{
                        const int64_t ar = row + p / 16, ak = kk + p % 16;
                        const int64_t bk = kk + p / {tile_size}, bc = col + p % {tile_size};
                        sa[p] = {cast}((ar < {m}LL && ak < {k}LL) ? {av} : 0.0f);
                        sb[p] = {cast}((bk < {k}LL && bc < {n}LL) ? {bv} : 0.0f);
                    }}
                    __syncthreads();
                    if (warp < {active_warps}) {{
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, {typ}, nvcuda::wmma::row_major> af;
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, {typ}, nvcuda::wmma::row_major> bf;
                    nvcuda::wmma::load_matrix_sync(af, sa + (warp / {warp_columns}) * 16 * 16, 16);
                    nvcuda::wmma::load_matrix_sync(bf, sb + (warp % {warp_columns}) * 16, {tile_size});
                    nvcuda::wmma::mma_sync(acc, af, bf, acc);
                    }}
                    __syncthreads();
                }}
                if (warp < {active_warps}) nvcuda::wmma::store_matrix_sync(shared_c + (warp / {warp_columns}) * 16 * {tile_size} + (warp % {warp_columns}) * 16, acc, {tile_size}, nvcuda::wmma::mem_row_major);
                __syncthreads();
                for (int p = threadIdx.x; p < {tile_size * tile_size}; p += blockDim.x) {{
                    int64_t rr = row + p / {tile_size}, cc = col + p % {tile_size};
                    if (rr < {m}LL && cc < {n}LL) {self.store(op.output, f'rr * {n}LL + cc', epilogue('shared_c[p]', f'rr * {n}LL + cc'))}
                }}
                __syncthreads();
            }}
            }}"""
        # Accurate FP32 fallback; kept separate from low-precision tensor-core paths.
        av = self.read(a, ("ar", "ak"))
        bv = self.read(b, ("bk", "bc"))
        return f"""for (int64_t tile = blockIdx.x; tile < {((m+15)//16)*((n+7)//8)}LL; tile += gridDim.x) {{
            int64_t row = (tile / {(n+7)//8 or 1}LL) * 16;
            int64_t col = (tile % {(n+7)//8 or 1}LL) * 8;
            const int rr = threadIdx.x / 8, cc = threadIdx.x % 8;
            float acc = 0.0f;
            for (int64_t kk = 0; kk < {k}LL; kk += 16) {{
                for (int p = threadIdx.x; p < 256; p += blockDim.x) {{
                    int64_t ar = row + p / 16, ak = kk + p % 16;
                    shared_float_a[p] = (ar < {m}LL && ak < {k}LL) ? {av} : 0.0f;
                }}
                for (int p = threadIdx.x; p < 128; p += blockDim.x) {{
                    int64_t bk = kk + p / 8, bc = col + p % 8;
                    shared_float_b[p] = (bk < {k}LL && bc < {n}LL) ? {bv} : 0.0f;
                }}
                __syncthreads();
                for (int j = 0; j < 16; ++j) acc = fmaf(shared_float_a[rr * 16 + j], shared_float_b[j * 8 + cc], acc);
                __syncthreads();
            }}
            if (row + rr < {m}LL && col + cc < {n}LL) {self.store(op.output, f'(row + rr) * {n}LL + col + cc', epilogue('acc', f'(row + rr) * {n}LL + col + cc'))}
            __syncthreads();
        }}"""

    def generate(self) -> GeneratedKernel:
        for spec in (*self.program.inputs, *(op.output for op in self.program.operations), *self.program.outputs):
            if spec.dtype not in _DTYPES:
                raise ValueError(f"megakernel: unsupported CUDA dtype {spec.dtype}")
            if any(d < 0 for d in spec.shape) or len(spec.shape) != len(spec.strides) or any(s < 0 for s in spec.strides):
                raise ValueError("megakernel: invalid static shape or stride")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", spec.buffer):
                raise ValueError("megakernel: invalid buffer identifier")
        statements = []
        stage_reads: list[tuple[int, int]] = []
        stage_writes: list[tuple[int, int]] = []
        barriers = 0

        def read_regions(spec: TensorSpec) -> list[tuple[int, int]]:
            if spec.buffer in self.inline:
                return [region for value in self.producers[spec.buffer].inputs if isinstance(value, TensorSpec) for region in read_regions(value)]
            if spec.buffer in self.input_indices:
                return []
            start = self.offsets[spec.buffer]
            return [(start, start + numel(self.producers[spec.buffer].output.shape))]

        def overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
            return left[0] < right[1] and right[0] < left[1]

        for op in self.materialized:
            if op.output.strides != contiguous_strides(op.output.shape):
                # Size-one dimensions may have arbitrary equivalent strides.
                if any(n > 1 and a != b for n, a, b in zip(op.output.shape, op.output.strides, contiguous_strides(op.output.shape))):
                    raise ValueError(f"megakernel: materialized output must be contiguous: {op.name}")
            if op.op in {"mm", "addmm"}:
                code = self.emit_mm(op)
            elif op.op in {"sum", "mean"}:
                code = self.emit_reduction(op)
            elif op.op in _POINTWISE:
                code = self.emit_pointwise(op)
            else:
                raise ValueError(f"megakernel: unsupported CUDA operation {op.op}")
            reads = [region for value in op.inputs if isinstance(value, TensorSpec) for region in read_regions(value)]
            start = self.offsets[op.output.buffer]
            write = (start, start + numel(op.output.shape))
            # Remove barriers only when both dependencies and arena reuse permit
            # overlap. Include write-after-read hazards on recycled allocations.
            hazard = any(overlap(old, new) for old in stage_writes for new in (*reads, write)) or any(overlap(old, write) for old in stage_reads)
            if hazard:
                statements.append("mk_grid_sync(grid);")
                barriers += 1
                stage_reads.clear()
                stage_writes.clear()
            statements.append(f"// {op.name}: {op.op}\n{{\n{code}\n}}")
            stage_reads.extend(reads)
            stage_writes.append(write)
        if stage_writes:
            statements.append("mk_grid_sync(grid);")
            barriers += 1
        for i, spec in enumerate(self.program.outputs):
            if spec.buffer in self.output_bindings and self.output_bindings[spec.buffer][0] == i:
                continue
            value = self.read(spec, self.coords(spec.shape, "i"))
            stored = f"__float2bfloat16_rn({value})" if spec.dtype == "bfloat16" else f"__float2half_rn({value})" if spec.dtype == "float16" else value
            statements.append(f"for (int64_t i = tid; i < {numel(spec.shape)}LL; i += nt) static_cast<{_DTYPES[spec.dtype]}*>(args.outputs[{i}])[i] = {stored};")
        work_blocks = [1]
        for op in self.materialized:
            if op.op in {"mm", "addmm"}:
                a, b = op.inputs[1:3] if op.op == "addmm" else op.inputs[:2]
                tm, tn = ((16, 16) if a.shape[0] <= 16 and b.shape[1] <= 16 else (32, 32)) if a.dtype in {"float16", "bfloat16"} else (16, 8)
                work_blocks.append(((a.shape[0] + tm - 1) // tm) * ((b.shape[1] + tn - 1) // tn))
            elif op.op in {"sum", "mean"}:
                source = op.inputs[0]
                dims = tuple(int(d) % len(source.shape) for d in op.attrs.get("dims", tuple(range(len(source.shape))))) if source.shape else ()
                reduction_size = math.prod(source.shape[d] for d in set(dims))
                rows = numel(op.output.shape)
                work_blocks.append((rows + 127) // 128 if reduction_size <= 128 and rows >= 8 else rows)
            else:
                work_blocks.append((numel(op.output.shape) + 127) // 128)
        work_blocks.extend((numel(output.shape) + 127) // 128 for output in self.program.outputs)
        max_work_blocks = max(work_blocks)
        source = _CUDA_PREAMBLE.replace("@INPUT_COUNT@", str(max(1, len(self.program.inputs)))).replace("@OUTPUT_COUNT@", str(max(1, len(self.program.outputs))))
        source += "\n".join(statements) + "\n}\n" + _CUDA_HOST.replace("@INPUT_COUNT@", str(len(self.program.inputs))).replace("@OUTPUT_COUNT@", str(len(self.program.outputs))).replace("@BLOCKS_PER_SM@", str(self.blocks_per_sm)).replace("@MAX_WORK_BLOCKS@", str(max_work_blocks))
        return GeneratedKernel(source, self.workspace_elements, len(self.materialized), len(self.inline), self.tensor_core_count, self.blocks_per_sm, max_work_blocks, barriers, len(self.output_bindings))


def generate(program: Program, *, blocks_per_sm: int = 1, fuse_pointwise: bool = True) -> GeneratedKernel:
    """Generate CUDA source without requiring a CUDA device or CUDA toolkit."""
    if blocks_per_sm < 1 or blocks_per_sm > 8:
        raise ValueError("megakernel: blocks_per_sm must be between 1 and 8")
    return _Generator(program, blocks_per_sm=blocks_per_sm, fuse_pointwise=fuse_pointwise).generate()


_CUDA_PREAMBLE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <cmath>
#include <algorithm>
#include <cstdint>
namespace cg = cooperative_groups;
__device__ __forceinline__ void mk_grid_sync(cg::grid_group& grid) {
    if (gridDim.x == 1) __syncthreads();
    else grid.sync();
}
struct Arguments { const void* inputs[@INPUT_COUNT@]; void* outputs[@OUTPUT_COUNT@]; };
__device__ __forceinline__ float mk_relu(float x) { return x < 0.0f ? 0.0f : x; }
__device__ __forceinline__ float mk_square(float x) { return x * x; }
__global__ void megakernel_entry(Arguments args, float* workspace) {
    auto grid = cg::this_grid();
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t nt = int64_t(gridDim.x) * blockDim.x;
    __shared__ __align__(32) unsigned short shared_a[512];
    __shared__ __align__(32) unsigned short shared_b[512];
    __shared__ __align__(32) float shared_c[1024];
    __shared__ float shared_reduce[128];
    __shared__ float shared_float_a[256];
    __shared__ float shared_float_b[128];
'''

_CUDA_HOST = r'''
void launch(std::vector<torch::Tensor> inputs, std::vector<torch::Tensor> outputs, torch::Tensor workspace) {
    TORCH_CHECK(inputs.size() == @INPUT_COUNT@ && outputs.size() == @OUTPUT_COUNT@, "incorrect tensor count");
    TORCH_CHECK(workspace.is_cuda(), "CUDA workspace required");
    c10::cuda::CUDAGuard guard(workspace.device());
    Arguments args{};
    for (size_t i = 0; i < inputs.size(); ++i) {
        TORCH_CHECK(inputs[i].is_cuda() && inputs[i].device() == workspace.device(), "input device mismatch");
        args.inputs[i] = inputs[i].data_ptr();
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        TORCH_CHECK(outputs[i].is_cuda() && outputs[i].device() == workspace.device(), "output device mismatch");
        args.outputs[i] = outputs[i].data_ptr();
    }
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, workspace.get_device()));
    TORCH_CHECK(prop.cooperativeLaunch, "device does not support cooperative launch");
    int active = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, megakernel_entry, 128, 0));
    TORCH_CHECK(active > 0, "kernel has no resident blocks");
    int blocks = std::min(prop.multiProcessorCount * std::min(active, @BLOCKS_PER_SM@), @MAX_WORK_BLOCKS@);
    float* arena = workspace.data_ptr<float>();
    void* kernel_args[] = {&args, &arena};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(megakernel_entry), dim3(blocks), dim3(128), kernel_args, 0, at::cuda::getCurrentCUDAStream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
