"""Static CUDA code generation for a cooperative persistent training kernel.

The initial scheduler is deliberately conservative: operators execute in topological
order with a grid barrier. Single-use pointwise values are evaluated inline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from .ir import Operation, Program, TensorSpec, validate

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
    direct_state_commits: int = 0


class _Generator:
    def __init__(self, program: Program, *, blocks_per_sm: int, fuse_pointwise: bool,
                 max_blocks: int | None = None, fuse_epilogues: bool = True,
                 gemm_k_tile: int = 32, gemm_tile_n: int = 32, shared_skew: int = 0, split_k_gemv: bool = False, threads_per_block: int = 128, inline_input_casts: bool = True, inplace_outputs: dict[int, int] | None = None):
        self.program = program
        self.blocks_per_sm = blocks_per_sm
        self.max_blocks = max_blocks
        self.gemm_k_tile = gemm_k_tile
        self.gemm_tile_n = gemm_tile_n
        self.shared_skew = shared_skew
        self.split_k_gemv = split_k_gemv
        self.threads_per_block = threads_per_block
        self.inplace_outputs = dict(inplace_outputs or {})
        aliased_buffers = {program.outputs[i].buffer for i in self.inplace_outputs}
        self.producers = {op.output.buffer: op for op in program.operations}
        self.input_indices = {spec.buffer: i for i, spec in enumerate(program.inputs)}
        counts: Counter[str] = Counter()
        for op in program.operations:
            counts.update(x.buffer for x in op.inputs if isinstance(x, TensorSpec))
        output_buffers = {x.buffer for x in program.outputs}
        self.epilogues: dict[str, tuple[tuple[Operation, ...], str, Operation]] = {}
        self.side_epilogues: dict[str, tuple[Operation, ...]] = {}
        side_outputs: dict[str, TensorSpec] = {}
        replacements: dict[str, Operation] = {}
        fused_skip: set[str] = set()
        if fuse_epilogues:
            users: dict[str, list[Operation]] = {}
            for candidate in program.operations:
                for value in candidate.inputs:
                    if isinstance(value, TensorSpec):
                        users.setdefault(value.buffer, []).append(candidate)
            op_index = {op.output.buffer: i for i, op in enumerate(program.operations)}
            for mm_index, mm in enumerate(program.operations):
                if mm.op not in {"mm", "addmm"} or mm.output.buffer in output_buffers:
                    continue
                chain: list[Operation] = []
                current = mm.output.buffer
                while len(users.get(current, [])) == 1:
                    nxt = users[current][0]
                    if nxt.op not in _POINTWISE or nxt.output.buffer in output_buffers:
                        break
                    # The substituted GEMM root must be consumed with the
                    # same logical layout. A square transpose/view can have
                    # the same shape, so shape-only checks are insufficient.
                    current_spec = self.producers.get(current, None)
                    if current_spec is None:
                        break
                    matching = [value for value in nxt.inputs
                                if isinstance(value, TensorSpec) and value.buffer == current]
                    if not matching or any(value.shape != current_spec.output.shape
                                           or value.strides != contiguous_strides(value.shape)
                                           or value.offset != 0 for value in matching):
                        break
                    chain.append(nxt)
                    fused_skip.add(current)
                    current = nxt.output.buffer
                    if len(users.get(current, [])) != 1:
                        break
                if chain:
                    final = chain[-1]
                    # Fusion is safe only for direct, contiguous tensor roots
                    # and shape-preserving pointwise chains. External values
                    # must already exist when the replacement executes.
                    safe = all(op.output.shape == mm.output.shape and op.output.offset == 0
                               and op.output.strides == contiguous_strides(op.output.shape)
                               for op in chain)
                    external: list[TensorSpec] = []
                    chain_buffers = {mm.output.buffer} | {op.output.buffer for op in chain}
                    for cop in chain:
                        for value in cop.inputs:
                            if isinstance(value, TensorSpec) and value.buffer not in chain_buffers:
                                if value.offset != 0 or value.strides != contiguous_strides(value.shape):
                                    safe = False
                                producer_index = op_index.get(value.buffer)
                                if producer_index is not None and producer_index >= mm_index:
                                    safe = False
                                if value.buffer not in {x.buffer for x in external}:
                                    external.append(value)
                    if not safe:
                        fused_skip.difference_update({mm.output.buffer, *[op.output.buffer for op in chain]})
                        continue
                    fused_inputs = tuple(mm.inputs) + tuple(external)
                    fused = Operation(mm.name + "_epilogue", mm.op, fused_inputs, final.output, mm.attrs)
                    self.epilogues[fused.output.buffer] = (tuple(chain), mm.output.buffer, mm)
                    replacements[mm.output.buffer] = fused
                    fused_skip.add(final.output.buffer)
                    self.producers[fused.output.buffer] = fused
            # A second, conservative case keeps the GEMM root and writes
            # returned pointwise descendants from the same output tile. Only
            # externally visible descendants have storage outside this tile.
            for mm_index, mm in enumerate(program.operations):
                if mm.op not in {"mm", "addmm"} or mm.output.buffer in replacements or mm.output.buffer in fused_skip:
                    continue
                known = {mm.output.buffer}
                chain = []
                external: dict[str, TensorSpec] = {}
                for nxt in program.operations[mm_index + 1:]:
                    tensors = [x for x in nxt.inputs if isinstance(x, TensorSpec)]
                    if nxt.op not in _POINTWISE or not any(x.buffer in known for x in tensors):
                        continue
                    if nxt.output.shape != mm.output.shape or nxt.output.buffer in fused_skip or nxt.output.buffer in aliased_buffers or nxt.output.offset or nxt.output.strides != contiguous_strides(nxt.output.shape):
                        continue
                    if any(x.buffer in known and (x.shape != mm.output.shape or x.strides != contiguous_strides(x.shape) or x.offset) for x in tensors):
                        continue
                    others = [x for x in tensors if x.buffer not in known]
                    if any(op_index.get(x.buffer, -1) >= mm_index for x in others):
                        continue
                    chain.append(nxt)
                    known.add(nxt.output.buffer)
                    external.update((x.buffer, x) for x in others)
                derived = known - {mm.output.buffer}
                returned = derived & output_buffers
                # Discard branches that do not lead to a stored return value.
                needed = set(returned)
                for candidate in reversed(chain):
                    if candidate.output.buffer in needed:
                        needed.update(x.buffer for x in candidate.inputs if isinstance(x, TensorSpec) and x.buffer in derived)
                chain = [op for op in chain if op.output.buffer in needed]
                derived &= needed
                if not returned:
                    continue
                if any(user.output.buffer not in derived for buf in derived - returned for user in users.get(buf, [])):
                    continue
                if any(out.offset or out.shape != self.producers[out.buffer].output.shape or out.strides != contiguous_strides(out.shape) for out in program.outputs if out.buffer in returned):
                    continue
                replacement = Operation(mm.name, mm.op, mm.inputs + tuple(external.values()), mm.output, mm.attrs)
                replacements[mm.output.buffer] = replacement
                self.producers[mm.output.buffer] = replacement
                self.side_epilogues[mm.output.buffer] = tuple(chain)
                fused_skip.update(derived)
                side_outputs.update((op.output.buffer, op.output) for op in chain if op.output.buffer in returned)
        self.fused_replacements = replacements
        # Do not duplicate a pointwise expression across multiple consumers or outputs.
        self.inline = {
            op.output.buffer for op in program.operations
            if fuse_pointwise and op.op in _POINTWISE
            and (counts[op.output.buffer] == 1 or (inline_input_casts and op.op == "cast" and isinstance(op.inputs[0], TensorSpec) and op.inputs[0].buffer in self.input_indices)) and op.output.buffer not in output_buffers
            and op.output.buffer not in self.epilogues
        }
        # Direct state writes are allowed only for a final elementwise use of
        # an input. Other mappings retain the staged transactional commit.
        schedule = []
        for candidate in program.operations:
            if candidate.output.buffer in replacements:
                schedule.append(replacements[candidate.output.buffer])
            elif candidate.output.buffer not in fused_skip and candidate.output.buffer not in self.inline:
                schedule.append(candidate)
        def raw_inputs(spec: TensorSpec) -> set[str]:
            if spec.buffer in self.input_indices:
                return {spec.buffer}
            if spec.buffer in self.inline:
                return set().union(*(raw_inputs(v) for v in self.producers[spec.buffer].inputs if isinstance(v, TensorSpec)))
            return set()
        def identity_access(spec: TensorSpec, target: str, shape: tuple[int, ...]) -> bool:
            if target not in raw_inputs(spec):
                return True
            if spec.shape != shape or spec.offset or spec.strides != contiguous_strides(shape):
                return False
            if spec.buffer == target:
                return True
            return all(identity_access(v, target, shape) for v in self.producers[spec.buffer].inputs if isinstance(v, TensorSpec))
        self.direct_aliases: dict[int, int] = {}
        output_counts = Counter(x.buffer for x in program.outputs)
        for slot, input_index in self.inplace_outputs.items():
            output = program.outputs[slot]
            target = program.inputs[input_index].buffer
            owner = next((i for i, op in enumerate(schedule) if op.output.buffer == output.buffer), None)
            if owner is None or schedule[owner].op not in _POINTWISE or output_counts[output.buffer] != 1:
                continue
            if output.offset or output.strides != contiguous_strides(output.shape) or any(out.buffer == target for out in program.outputs):
                continue
            if any(target in raw_inputs(v) for op in schedule[owner+1:] for v in op.inputs if isinstance(v, TensorSpec)):
                continue
            if not all(identity_access(v, target, output.shape) for v in schedule[owner].inputs if isinstance(v, TensorSpec)):
                continue
            self.direct_aliases[slot] = input_index
        self.input_regions: dict[str, tuple[int, int]] = {}
        cursor = -1
        for spec in program.inputs:
            size = max(1, numel(spec.shape))
            self.input_regions[spec.buffer] = (cursor - size, cursor)
            cursor -= size + 1
        self.alias_output_regions = {program.outputs[slot].buffer: self.input_regions[program.inputs[index].buffer] for slot, index in self.direct_aliases.items()}
        self.output_bindings: dict[str, tuple[int, str]] = {}
        for index, spec in enumerate(program.outputs):
            if index in self.inplace_outputs and index not in self.direct_aliases:
                continue
            producer = self.producers.get(spec.buffer)
            if producer is not None and producer.output.dtype == spec.dtype and spec.offset == 0 and numel(spec.shape) == numel(producer.output.shape) and all(dim == 1 or a == b for dim, a, b in zip(spec.shape, spec.strides, contiguous_strides(spec.shape))):
                self.output_bindings.setdefault(spec.buffer, (index, spec.dtype))
        if not side_outputs.keys() <= self.output_bindings.keys():
            raise ValueError("megakernel: side epilogues require direct output bindings")
        self.native_regions: dict[str, tuple[int, int]] = {}
        for buffer in self.output_bindings:
            if buffer in self.alias_output_regions:
                self.native_regions[buffer] = self.alias_output_regions[buffer]
            else:
                size = max(1, numel(self.producers[buffer].output.shape))
                self.native_regions[buffer] = (cursor - size, cursor)
                cursor -= size + 1
        self.discarded_mm_roots = {
            root for root, chain in self.side_epilogues.items()
            if root not in output_buffers and all(user.output.buffer in {op.output.buffer for op in chain}
                for user in program.operations if any(isinstance(x, TensorSpec) and x.buffer == root for x in user.inputs))
        }
        self.discarded_regions: dict[str, tuple[int, int]] = {}
        for root in sorted(self.discarded_mm_roots):
            size = max(1, numel(self.producers[root].output.shape))
            self.discarded_regions[root] = (cursor - size, cursor)
            cursor -= size + 1
        self.materialized = []
        for op in program.operations:
            if op.output.buffer in self.fused_replacements:
                self.materialized.append(self.fused_replacements[op.output.buffer])
            elif op.output.buffer in fused_skip:
                continue
            elif op.output.buffer not in self.inline:
                self.materialized.append(op)
        self.offsets, self.workspace_elements = self._plan_workspace()
        # Native allocations use disjoint virtual regions for dependencies and
        # never consume arena space. Declared aliases share their input region.
        for buffer, spec in side_outputs.items():
            self.offsets[buffer] = self.native_regions[buffer][0]
        self.reduction_scratch = self.workspace_elements
        if any(op.op in {"sum", "mean"} and isinstance(op.inputs[0], TensorSpec) and numel(op.inputs[0].shape) > 4096 and numel(op.output.shape) <= 4 for op in self.materialized):
            self.workspace_elements += 1024
        self.internal_barriers = 0
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
            if op.output.buffer in self.native_regions:
                offsets[op.output.buffer] = self.native_regions[op.output.buffer][0]
                continue
            if op.output.buffer in self.discarded_regions:
                offsets[op.output.buffer] = self.discarded_regions[op.output.buffer][0]
                continue
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
        if operand.buffer in getattr(self, "_tile_values", {}):
            return self._tile_values[operand.buffer]
        if hasattr(self, "_epilogue_operand") and self._epilogue_operand is not None and operand.buffer == self._epilogue_operand[0]:
            return self._epilogue_operand[1]
        coords = self.coords(output.shape, index)
        if len(operand.shape) > len(coords):
            raise ValueError("megakernel: invalid broadcast rank")
        trailing = coords[len(coords) - len(operand.shape):] if operand.shape else ()
        if any(a != 1 and a != b for a, b in zip(operand.shape, output.shape[-len(operand.shape):] if operand.shape else ())):
            raise ValueError("megakernel: incompatible broadcast dimensions")
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

    def epilogue_expr(self, buffer: str, value: str, index: str) -> str:
        record = self.epilogues.get(buffer)
        if record is None:
            return value
        chain, buffer, original = record
        value = _round(value, original.output.dtype)
        for op in chain:
            self._epilogue_operand = (buffer, value)
            value = _round(self.point_expr(op, index), op.output.dtype)
            self._epilogue_operand = None
            buffer = op.output.buffer
        return value

    def emit_pointwise(self, op: Operation) -> str:
        expr = self.point_expr(op, "i")
        return f"""for (int64_t i = tid; i < {numel(op.output.shape)}LL; i += nt) {{
            {self.store(op.output, 'i', expr)}
        }}"""

    def store_mm(self, op: Operation, index: str, value: str) -> str:
        value = self.epilogue_expr(op.output.buffer, value, index)
        chain = self.side_epilogues.get(op.output.buffer, ())
        if not chain:
            return self.store(op.output, index, value)
        lines = ["{", f"float tile_root = {_round(value, op.output.dtype)};"]
        if op.output.buffer not in self.discarded_mm_roots:
            lines.append(self.store(op.output, index, "tile_root"))
        self._tile_values = {op.output.buffer: "tile_root"}
        for i, descendant in enumerate(chain):
            name = f"tile_value_{i}"
            lines.append(f"float {name} = {_round(self.point_expr(descendant, index), descendant.output.dtype)};")
            self._tile_values[descendant.output.buffer] = name
            if descendant.output.buffer in self.output_bindings:
                lines.append(self.store(descendant.output, index, name))
        self._tile_values = {}
        return "\n".join([*lines, "}"])

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
        if numel(op.output.shape) != numel(kept_shape):
            raise ValueError("megakernel: reduction output shape does not match its axes")
        if size > 4096 and numel(op.output.shape) <= 4:
            parts = min(256, (size + 511) // 512)
            rows = numel(op.output.shape)
            finish = "acc" if op.op == "sum" else f"(acc / {size}.0f)"
            self.internal_barriers += 2
            # The fixed scratch region is reused only after both grid barriers.
            # Each partial has its own slot, independent of cooperative grid size.
            return f"""{{
                for (int64_t part = blockIdx.x; part < {rows * parts}LL; part += gridDim.x) {{
                    const int64_t row = part / {parts};
                    const int64_t begin = (part % {parts}) * 512;
                    float acc = 0.0f;
                    for (int64_t r = begin + threadIdx.x; r < min(begin + 512LL, {size}LL); r += blockDim.x) acc += {value};
                    acc = mk_block_sum(acc, shared_reduce);
                    if (threadIdx.x == 0) workspace[{self.reduction_scratch}LL + part] = acc;
                    __syncthreads();
                }}
                mk_grid_sync(grid);
                for (int64_t row = blockIdx.x; row < {rows}LL; row += gridDim.x) {{
                    float acc = 0.0f;
                    for (int r = threadIdx.x; r < {parts}; r += blockDim.x) acc += workspace[{self.reduction_scratch}LL + row * {parts} + r];
                    acc = mk_block_sum(acc, shared_reduce);
                    if (threadIdx.x == 0) {self.store(op.output, 'row', finish)}
                    __syncthreads();
                }}
                mk_grid_sync(grid);
            }}"""
        if size <= 128 and numel(op.output.shape) >= 8:
            finish = "acc" if op.op == "sum" else f"(acc / {size}.0f)" if size else "NAN"
            return f"""for (int64_t row = tid; row < {numel(op.output.shape)}LL; row += nt) {{
                float acc = 0.0f;
                for (int64_t r = 0; r < {size}LL; ++r) acc += {value};
                {self.store(op.output, 'row', finish)}
            }}"""
        finish = "acc" if op.op == "sum" else f"(acc / {size}.0f)" if size else "NAN"
        return f"""for (int64_t row = blockIdx.x; row < {numel(op.output.shape)}LL; row += gridDim.x) {{
            float acc = 0.0f;
            for (int64_t r = threadIdx.x; r < {size}LL; r += blockDim.x) acc += {value};
            acc = mk_block_sum(acc, shared_reduce);
            if (threadIdx.x == 0) {self.store(op.output, 'row', finish)}
            __syncthreads();
        }}"""

    def emit_mm(self, op: Operation) -> str:
        a, b = op.inputs[1:3] if op.op == "addmm" else op.inputs[:2]
        if not isinstance(a, TensorSpec) or not isinstance(b, TensorSpec) or len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("megakernel: mm requires compatible rank-two tensors")
        m, k = a.shape
        _, n = b.shape
        if op.output.shape != (m, n):
            raise ValueError("megakernel: matrix output shape does not match its inputs")
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
        if self.split_k_gemv and m == 1 and k >= 64:
            from .gemv import emit_gemv
            return emit_gemv(m, k, n, lambda r, q: self.read(a, (r, q)), lambda q, c: self.read(b, (q, c)),
                             lambda index, value: self.store_mm(op, index, value), lambda index, value: epilogue(value, index), threads=self.threads_per_block).source
        if k <= 8 or m <= 4:
            # For online and micro-batch gradients, WMMA's 32-wide K staging
            # wastes work and synchronization. Each thread owns one output
            # element and performs the exact FP32 accumulation directly.
            value = "acc"
            index = f"rr * {n}LL + cc"
            result = epilogue(value, index)
            return f"""for (int64_t i = tid; i < {m * n}LL; i += nt) {{
                const int64_t rr = i / {n}LL, cc = i % {n}LL;
                float acc = 0.0f;
                #pragma unroll 4
                for (int kk = 0; kk < {k}; ++kk) acc = fmaf({self.read(a, ('rr', 'kk'))}, {self.read(b, ('kk', 'cc'))}, acc);
                {self.store_mm(op, 'i', result)}
            }}"""
        if a.dtype == b.dtype and a.dtype in {"bfloat16", "float16"}:
            self.tensor_core_count += 1
            cast = "__float2bfloat16_rn" if a.dtype == "bfloat16" else "__float2half_rn"
            typ = _DTYPES[a.dtype]
            if self.gemm_tile_n == 64 and m >= 16 and n >= 64:
                from .wmma import emit_wmma
                return emit_wmma(m, k, n, lambda r, q: self.read(a, (r, q)), lambda q, c: self.read(b, (q, c)),
                                 lambda index, value: self.store_mm(op, index, value),
                                 lambda index, value: epilogue(value, index), k_tile=self.gemm_k_tile, value_type=typ, cast=cast).source
            tile_size = 16 if m <= 16 and n <= 16 else 32
            a_pitch = self.gemm_k_tile + self.shared_skew
            b_pitch = tile_size + self.shared_skew
            c_pitch = tile_size + self.shared_skew // 2
            warp_columns = tile_size // 16
            active_warps = warp_columns * warp_columns
            transpose_a = a.strides == (1, a.shape[0])
            a_row = f"row + p % {tile_size}" if transpose_a else f"row + p / {self.gemm_k_tile}"
            a_k = f"kk + p / {tile_size}" if transpose_a else f"kk + p % {self.gemm_k_tile}"
            a_store = f"sa[(p % {tile_size}) * {a_pitch} + p / {tile_size}]" if transpose_a else f"sa[(p / {self.gemm_k_tile}) * {a_pitch} + p % {self.gemm_k_tile}]"
            av = self.read(a, ("ar", "ak"))
            # A transposed B view has unit stride in K.  Assign adjacent
            # loader lanes to adjacent K elements and scatter into the
            # row-major WMMA tile; this keeps the global read coalesced while
            # preserving the layout expected by load_matrix_sync.
            transpose_b = b.strides == (1, b.shape[0])
            if transpose_b:
                bv = self.read(b, (f"kk + p % {self.gemm_k_tile}", f"col + p / {self.gemm_k_tile}"))
                b_store = f"sb[(p % {self.gemm_k_tile}) * {b_pitch} + p / {self.gemm_k_tile}]"
                b_valid = f"(kk + p % {self.gemm_k_tile} < {k}LL && col + p / {self.gemm_k_tile} < {n}LL)"
            else:
                bv = self.read(b, ("bk", "bc"))
                b_store = f"sb[(p / {tile_size}) * {b_pitch} + p % {tile_size}]"
                b_valid = f"(bk < {k}LL && bc < {n}LL)"
            return f"""{{
            auto* sa = reinterpret_cast<{typ}*>(shared_a);
            auto* sb = reinterpret_cast<{typ}*>(shared_b);
            const int warp = threadIdx.x / 32;
            for (int64_t tile = blockIdx.x; tile < {((m + tile_size - 1)//tile_size)*((n + tile_size - 1)//tile_size)}LL; tile += gridDim.x) {{
                const int64_t row = (tile / {(n+tile_size-1)//tile_size or 1}LL) * {tile_size};
                const int64_t col = (tile % {(n+tile_size-1)//tile_size or 1}LL) * {tile_size};
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
                nvcuda::wmma::fill_fragment(acc, 0.0f);
                // Stage a 32-wide K tile and consume it with two WMMA
                // instructions before the next barrier. This removes one
                // shared-memory synchronization per K tile on SM80.
                for (int64_t kk = 0; kk < {k}LL; kk += {self.gemm_k_tile}) {{
                    for (int p = threadIdx.x; p < {tile_size * self.gemm_k_tile}; p += blockDim.x) {{
                        const int64_t ar = {a_row}, ak = {a_k};
                        const int64_t bk = kk + p / {tile_size}, bc = col + p % {tile_size};
                        {a_store} = {cast}((ar < {m}LL && ak < {k}LL) ? {av} : 0.0f);
                        {b_store} = {cast}({b_valid} ? {bv} : 0.0f);
                    }}
                    __syncthreads();
                    if (warp < {active_warps}) {{
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, {typ}, nvcuda::wmma::row_major> af;
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, {typ}, nvcuda::wmma::row_major> bf;
                    nvcuda::wmma::load_matrix_sync(af, sa + (warp / {warp_columns}) * 16 * {a_pitch}, {a_pitch});
                    nvcuda::wmma::load_matrix_sync(bf, sb + (warp % {warp_columns}) * 16, {b_pitch});
                    nvcuda::wmma::mma_sync(acc, af, bf, acc);
                    {'' if self.gemm_k_tile == 16 else f'nvcuda::wmma::load_matrix_sync(af, sa + (warp / {warp_columns}) * 16 * {a_pitch} + 16, {a_pitch});'}
                    {'' if self.gemm_k_tile == 16 else f'nvcuda::wmma::load_matrix_sync(bf, sb + (warp % {warp_columns}) * 16 + {b_pitch} * 16, {b_pitch});'}
                    {'' if self.gemm_k_tile == 16 else 'nvcuda::wmma::mma_sync(acc, af, bf, acc);'}
                    }}
                    __syncthreads();
                }}
                if (warp < {active_warps}) nvcuda::wmma::store_matrix_sync(shared_c + (warp / {warp_columns}) * 16 * {c_pitch} + (warp % {warp_columns}) * 16, acc, {c_pitch}, nvcuda::wmma::mem_row_major);
                __syncthreads();
                for (int p = threadIdx.x; p < {tile_size * tile_size}; p += blockDim.x) {{
                    int64_t rr = row + p / {tile_size}, cc = col + p % {tile_size};
                    if (rr < {m}LL && cc < {n}LL) {self.store_mm(op, f'rr * {n}LL + cc', epilogue(f'shared_c[(p / {tile_size}) * {c_pitch} + p % {tile_size}]', f'rr * {n}LL + cc'))}
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
            if (row + rr < {m}LL && col + cc < {n}LL) {self.store_mm(op, f'(row + rr) * {n}LL + col + cc', epilogue('acc', f'(row + rr) * {n}LL + col + cc'))}
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
                return [self.input_regions[spec.buffer]] if self.direct_aliases else []
            if spec.buffer in self.native_regions:
                return [self.native_regions[spec.buffer]]
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
            write = self.native_regions.get(op.output.buffer, (start, start + numel(op.output.shape)))
            writes = ([] if op.output.buffer in self.discarded_mm_roots else [write]) + [(self.offsets[x.output.buffer], self.offsets[x.output.buffer] + numel(x.output.shape)) for x in self.side_epilogues.get(op.output.buffer, ()) if x.output.buffer in self.output_bindings]
            # Remove barriers only when both dependencies and arena reuse permit
            # overlap. Include write-after-read hazards on recycled allocations.
            hazard = any(overlap(old, new) for old in stage_writes for new in (*reads, *writes)) or any(overlap(old, new) for old in stage_reads for new in writes)
            if hazard:
                statements.append("mk_grid_sync(grid);")
                barriers += 1
                stage_reads.clear()
                stage_writes.clear()
            statements.append(f"// {op.name}: {op.op}\n{{\n{code}\n}}")
            stage_reads.extend(reads)
            stage_writes.extend(writes)
        needs_output_copy = any(self.output_bindings.get(spec.buffer, (-1, ""))[0] != i for i, spec in enumerate(self.program.outputs))
        if stage_writes and needs_output_copy:
            statements.append("mk_grid_sync(grid);")
            barriers += 1
        copied_outputs = False
        for i, spec in enumerate(self.program.outputs):
            if i in self.inplace_outputs:
                continue
            if spec.buffer in self.output_bindings and self.output_bindings[spec.buffer][0] == i:
                continue
            value = self.read(spec, self.coords(spec.shape, "i"))
            stored = f"__float2bfloat16_rn({value})" if spec.dtype == "bfloat16" else f"__float2half_rn({value})" if spec.dtype == "float16" else value
            statements.append(f"for (int64_t i = tid; i < {numel(spec.shape)}LL; i += nt) static_cast<{_DTYPES[spec.dtype]}*>(args.outputs[{i}])[i] = {stored};")
            copied_outputs = True
        if self.inplace_outputs:
            if copied_outputs:
                statements.append("mk_grid_sync(grid);")
                barriers += 1
            statements.append("// Transactional input commit: all input reads have completed.")
            for slot in self.inplace_outputs:
                if slot in self.direct_aliases:
                    continue
                spec = self.program.outputs[slot]
                value = self.read(spec, self.coords(spec.shape, "i"))
                stored = f"__float2bfloat16_rn({value})" if spec.dtype == "bfloat16" else f"__float2half_rn({value})" if spec.dtype == "float16" else value
                statements.append(f"for (int64_t i = tid; i < {numel(spec.shape)}LL; i += nt) static_cast<{_DTYPES[spec.dtype]}*>(args.outputs[{slot}])[i] = {stored};")
        work_blocks = [1]
        for op in self.materialized:
            if op.op in {"mm", "addmm"}:
                a, b = op.inputs[1:3] if op.op == "addmm" else op.inputs[:2]
                if self.split_k_gemv and a.shape[0] == 1 and a.shape[1] >= 64:
                    work_blocks.append((b.shape[1] + 31) // 32)
                elif a.shape[1] <= 8 or a.shape[0] <= 4:
                    work_blocks.append((numel(op.output.shape) + self.threads_per_block - 1) // self.threads_per_block)
                else:
                    tm, tn = ((32, 64) if self.gemm_tile_n == 64 and a.shape[0] >= 16 and b.shape[1] >= 64 else (16, 16) if a.shape[0] <= 16 and b.shape[1] <= 16 else (32, 32)) if a.dtype in {"float16", "bfloat16"} else (16, 8)
                    work_blocks.append(((a.shape[0] + tm - 1) // tm) * ((b.shape[1] + tn - 1) // tn))
            elif op.op in {"sum", "mean"}:
                source = op.inputs[0]
                dims = tuple(int(d) % len(source.shape) for d in op.attrs.get("dims", tuple(range(len(source.shape))))) if source.shape else ()
                reduction_size = math.prod(source.shape[d] for d in set(dims))
                rows = numel(op.output.shape)
                work_blocks.append(rows * min(256, (reduction_size + 511) // 512) if reduction_size > 4096 and rows <= 4 else (rows + self.threads_per_block - 1) // self.threads_per_block if reduction_size <= 128 and rows >= 8 else rows)
            else:
                work_blocks.append((numel(op.output.shape) + self.threads_per_block - 1) // self.threads_per_block)
        work_blocks.extend((numel(output.shape) + self.threads_per_block - 1) // self.threads_per_block for output in self.program.outputs)
        max_work_blocks = max(work_blocks)
        source = _CUDA_PREAMBLE.replace("@INPUT_COUNT@", str(max(1, len(self.program.inputs)))).replace("@OUTPUT_COUNT@", str(max(1, len(self.program.outputs))))
        source = source.replace("@SHARED_A@", str(max(1024, 32 * (self.gemm_k_tile + self.shared_skew)))).replace("@SHARED_B@", str(max(1024, self.gemm_k_tile * (32 + self.shared_skew), self.gemm_k_tile * 64 if self.gemm_tile_n == 64 else 0))).replace("@SHARED_C@", str(max(1024, 32 * (32 + self.shared_skew // 2), 2048 if self.gemm_tile_n == 64 else 0)))
        source += "\n".join(statements) + "\n}\n" + _CUDA_HOST.replace("@INPUT_COUNT@", str(len(self.program.inputs))).replace("@OUTPUT_COUNT@", str(len(self.program.outputs))).replace("@BLOCKS_PER_SM@", str(self.blocks_per_sm)).replace("@MAX_WORK_BLOCKS@", str(max_work_blocks)).replace("@MAX_BLOCKS@", str(-1 if self.max_blocks is None else self.max_blocks)).replace("@THREADS@", str(self.threads_per_block))
        return GeneratedKernel(source, self.workspace_elements, len(self.materialized), len(self.inline), self.tensor_core_count, self.blocks_per_sm, max_work_blocks, barriers + self.internal_barriers, len(self.output_bindings), len(self.direct_aliases))


def generate(program: Program, *, blocks_per_sm: int = 1, fuse_pointwise: bool = True,
             max_blocks: int | None = None, fuse_epilogues: bool = True,
             gemm_k_tile: int = 32, gemm_tile_n: int = 32, shared_skew: int = 0, split_k_gemv: bool = False, threads_per_block: int = 128, inline_input_casts: bool = True, inplace_outputs: dict[int, int] | None = None) -> GeneratedKernel:
    """Generate CUDA source without requiring a CUDA device or CUDA toolkit."""
    validate(program)
    from .passes import canonicalize_transposed_mm
    program = canonicalize_transposed_mm(program)
    validate(program)
    if type(blocks_per_sm) is not int or blocks_per_sm < 1 or blocks_per_sm > 8:
        raise ValueError("megakernel: blocks_per_sm must be between 1 and 8")
    if max_blocks is not None and (type(max_blocks) is not int or max_blocks < 1):
        raise ValueError("megakernel: max_blocks must be positive when provided")
    if type(gemm_k_tile) is not int or gemm_k_tile not in {16, 32}:
        raise ValueError("megakernel: gemm_k_tile must be 16 or 32")
    if type(gemm_tile_n) is not int or gemm_tile_n not in {32, 64}:
        raise ValueError("megakernel: gemm_tile_n must be 32 or 64")
    if type(shared_skew) is not int or shared_skew not in {0, 16}:
        raise ValueError("megakernel: shared_skew must be 0 or 16")
    if type(threads_per_block) is not int or threads_per_block not in {128, 256, 512, 1024}:
        raise ValueError("megakernel: threads_per_block must be 128, 256, 512, or 1024")
    if threads_per_block != 128:
        for op in program.operations:
            if op.op in {"mm", "addmm"}:
                a = op.inputs[1] if op.op == "addmm" else op.inputs[0]
                if a.shape[0] > 4 and a.shape[1] > 8:
                    raise ValueError("megakernel: larger blocks require direct SIMT/GEMV matrix paths")
    mapping = dict(inplace_outputs or {})
    if any(type(i) is not int or type(j) is not int or not 0 <= i < len(program.outputs) or not 0 <= j < len(program.inputs) for i, j in mapping.items()):
        raise ValueError("megakernel: invalid inplace output/input index")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("megakernel: inplace outputs must target distinct inputs")
    produced = {op.output.buffer for op in program.operations}
    for slot, target in mapping.items():
        output, input_spec = program.outputs[slot], program.inputs[target]
        if output.shape != input_spec.shape or output.dtype != input_spec.dtype or input_spec.strides != contiguous_strides(input_spec.shape):
            raise ValueError("megakernel: inplace output must match a contiguous input shape and dtype")
        if output.buffer not in produced:
            raise ValueError("megakernel: inplace outputs must be computed values, not input views")
    return _Generator(program, blocks_per_sm=blocks_per_sm, fuse_pointwise=fuse_pointwise,
                      max_blocks=max_blocks, fuse_epilogues=fuse_epilogues,
                      gemm_k_tile=gemm_k_tile, gemm_tile_n=gemm_tile_n, shared_skew=shared_skew, split_k_gemv=split_k_gemv, threads_per_block=threads_per_block, inline_input_casts=inline_input_casts, inplace_outputs=mapping).generate()


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
__device__ __forceinline__ float mk_block_sum(float value, float* scratch) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    value = threadIdx.x < blockDim.x / 32 ? scratch[lane] : 0.0f;
    if (warp == 0) for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}
__global__ void megakernel_entry(Arguments args, float* workspace) {
    auto grid = cg::this_grid();
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t nt = int64_t(gridDim.x) * blockDim.x;
    __shared__ __align__(32) unsigned short shared_a[@SHARED_A@];
    __shared__ __align__(32) unsigned short shared_b[@SHARED_B@];
    __shared__ __align__(32) float shared_c[@SHARED_C@];
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
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, megakernel_entry, @THREADS@, 0));
    TORCH_CHECK(active > 0, "kernel has no resident blocks");
    int blocks = std::min(prop.multiProcessorCount * std::min(active, @BLOCKS_PER_SM@),
                          @MAX_BLOCKS@ < 0 ? @MAX_WORK_BLOCKS@ : std::min(@MAX_WORK_BLOCKS@, @MAX_BLOCKS@));
    float* arena = workspace.data_ptr<float>();
    void* kernel_args[] = {&args, &arena};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(megakernel_entry), dim3(blocks), dim3(@THREADS@), kernel_args, 0, at::cuda::getCurrentCUDAStream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int64_t> resource_usage() {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, megakernel_entry));
    int active = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, megakernel_entry, @THREADS@, 0));
    return {attributes.numRegs, static_cast<int64_t>(attributes.localSizeBytes), static_cast<int64_t>(attributes.sharedSizeBytes), active};
}
'''
