"""Experimental SM90a WGMMA source emission for BF16 GEMMs.

This emitter is intentionally independent of the production scheduler.  It
uses the validated 64x16 @ 16x64 BF16 tile and keeps all Hopper-specific PTX
behind a small, testable boundary.
"""

from dataclasses import dataclass
from typing import Callable


Read = Callable[[str, str], str]
Store = Callable[[str, str], str]
Epilogue = Callable[[str, str], str]


@dataclass(frozen=True)
class SM90Emission:
    source: str
    tile_m: int = 64
    tile_n: int = 64
    tile_k: int = 16
    warps: int = 4
    shared_a_elements: int = 1024
    shared_b_elements: int = 1024
    shared_c_elements: int = 0

    @property
    def shared_bytes(self) -> int:
        return (self.shared_a_elements + self.shared_b_elements) * 2


def emit_wgmma(
    m: int,
    k: int,
    n: int,
    read_a: Read,
    read_b: Read,
    store: Store,
    epilogue: Epilogue | None = None,
    *,
    value_type: str = "__nv_bfloat16",
    cast: str = "__float2bfloat16_rn",
    k_group_size: int = 1,
    register_source_a: bool = False,
) -> SM90Emission:
    """Emit a tiled BF16 SM90a WGMMA GEMM.

    A and B are read in logical row-major ``A[m,k]`` and ``B[k,n]`` order.
    Each 64x16/64x16 operand tile is packed into the K-major interleaved
    no-swizzle layout required by WGMMA.  Tail elements are zero-filled.
    """
    if min(m, k, n) <= 0:
        raise ValueError("WGMMA dimensions must be positive")
    if value_type != "__nv_bfloat16" or cast != "__float2bfloat16_rn":
        raise ValueError("WGMMA currently supports BF16 operands only")
    if type(k_group_size) is not int or k_group_size not in (1, 2, 4):
        raise ValueError("WGMMA k_group_size must be 1, 2, or 4")
    if type(register_source_a) is not bool:
        raise ValueError("WGMMA register_source_a must be a boolean")
    ep = epilogue or (lambda _index, value: value)
    mt, nt = (m + 63) // 64, (n + 63) // 64
    chunks = (k + 15) // 16
    groups = (chunks + k_group_size - 1) // k_group_size
    regs = ",".join(f"%{i}" for i in range(32))
    constraints = ", ".join(f'"+f"(d{i})' for i in range(32))
    first_constraints = ", ".join(f'"=f"(d{i})' for i in range(32))
    descriptors = (
        "sm90_da, sm90_db"
        if k_group_size == 1
        else ", ".join(name for stage in range(k_group_size) for name in (f"sm90_da{stage}", f"sm90_db{stage}"))
    )
    lines = [
        "{",
        f"  auto* sm90_a = reinterpret_cast<{value_type}*>(shared_a);",
        f"  auto* sm90_b = reinterpret_cast<{value_type}*>(shared_b);",
        f"  float d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15;",
        f"  float d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31;",
        f"  uint64_t {descriptors};",
        *(["  const int sm90_lane = threadIdx.x & 31, sm90_warp = threadIdx.x >> 5;"] if register_source_a else []),
        f"  for (int64_t sm90_tile = blockIdx.x; sm90_tile < {mt * nt}LL; sm90_tile += gridDim.x) {{",
        f"    const int64_t sm90_row = (sm90_tile / {nt}LL) * 64LL;",
        f"    const int64_t sm90_col = (sm90_tile % {nt}LL) * 64LL;",
    ]
    for group in range(groups):
        first_chunk = group * k_group_size
        group_chunks = min(k_group_size, chunks - first_chunk)
        # Register fragments are redeclared for each unrolled K group. Keep
        # those declarations scoped without perturbing the shared-source path.
        if register_source_a:
            lines.append("    {")
        if register_source_a:
            lines += [
                f"    for (int p = threadIdx.x; p < {1024 * group_chunks}; p += blockDim.x) {{",
                f"      const int sm90_stage = p / 1024, sm90_local = p % 1024;",
                "      const int sm90_r = sm90_local / 16, sm90_q = sm90_local % 16;",
                f"      const int64_t sm90_kk = {first_chunk}LL * 16LL + int64_t(sm90_stage) * 16LL;",
                "      const int64_t sm90_bn = sm90_col + sm90_r, sm90_bk = sm90_kk + sm90_q;",
                "      const int sm90_bp = sm90_stage * 1024 + (sm90_r % 8) * 8 + (sm90_r / 8) * 64 + (sm90_q % 8) + (sm90_q / 8) * 512;",
                f"      sm90_b[sm90_bp] = {cast}((sm90_bn < {n}LL && sm90_bk < {k}LL) ? {read_b('sm90_bk', 'sm90_bn')} : 0.0f);",
                "    }",
                "    __syncthreads();",
                '    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");',
            ]
        elif group_chunks == 1:
            kk = first_chunk * 16
            lines += [
                "    for (int p = threadIdx.x; p < 1024; p += blockDim.x) {",
                "      const int sm90_r = p / 16, sm90_q = p % 16;",
                f"      const int64_t sm90_ar = sm90_row + sm90_r, sm90_ak = {kk}LL + sm90_q;",
                "      const int sm90_ap = (sm90_r % 8) * 8 + (sm90_r / 8) * 64 + (sm90_q % 8) + (sm90_q / 8) * 512;",
                f"      sm90_a[sm90_ap] = {cast}((sm90_ar < {m}LL && sm90_ak < {k}LL) ? {read_a('sm90_ar', 'sm90_ak')} : 0.0f);",
                f"      const int64_t sm90_bn = sm90_col + sm90_r, sm90_bk = {kk}LL + sm90_q;",
                "      const int sm90_bp = (sm90_r % 8) * 8 + (sm90_r / 8) * 64 + (sm90_q % 8) + (sm90_q / 8) * 512;",
                f"      sm90_b[sm90_bp] = {cast}((sm90_bn < {n}LL && sm90_bk < {k}LL) ? {read_b('sm90_bk', 'sm90_bn')} : 0.0f);",
                "    }",
                "    __syncthreads();",
                '    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");',
            ]
        else:
            lines += [
                f"    for (int p = threadIdx.x; p < {1024 * group_chunks}; p += blockDim.x) {{",
                f"      const int sm90_stage = p / 1024, sm90_local = p % 1024;",
                "      const int sm90_r = sm90_local / 16, sm90_q = sm90_local % 16;",
                f"      const int64_t sm90_kk = {first_chunk}LL * 16LL + int64_t(sm90_stage) * 16LL;",
                "      const int64_t sm90_ar = sm90_row + sm90_r, sm90_ak = sm90_kk + sm90_q;",
                "      const int sm90_ap = sm90_stage * 1024 + (sm90_r % 8) * 8 + (sm90_r / 8) * 64 + (sm90_q % 8) + (sm90_q / 8) * 512;",
                f"      sm90_a[sm90_ap] = {cast}((sm90_ar < {m}LL && sm90_ak < {k}LL) ? {read_a('sm90_ar', 'sm90_ak')} : 0.0f);",
                "      const int64_t sm90_bn = sm90_col + sm90_r, sm90_bk = sm90_kk + sm90_q;",
                "      const int sm90_bp = sm90_stage * 1024 + (sm90_r % 8) * 8 + (sm90_r / 8) * 64 + (sm90_q % 8) + (sm90_q / 8) * 512;",
                f"      sm90_b[sm90_bp] = {cast}((sm90_bn < {n}LL && sm90_bk < {k}LL) ? {read_b('sm90_bk', 'sm90_bn')} : 0.0f);",
                "    }",
                "    __syncthreads();",
                '    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");',
            ]
        operand_groups = []
        for stage in range(group_chunks):
            da, db = (("sm90_da", "sm90_db") if k_group_size == 1 else (f"sm90_da{stage}", f"sm90_db{stage}"))
            if not register_source_a:
                lines += [
                    f'    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"({da}) : "l"(sm90_a + {stage} * 1024));',
                    f'    {da} = (({da} & 0x3ffffULL) >> 4) | (uint64_t(1024 >> 4) << 16) | (uint64_t(128 >> 4) << 32);',
                ]
            lines += [
                f'    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"({db}) : "l"(sm90_b + {stage} * 1024));',
                f'    {db} = (({db} & 0x3ffffULL) >> 4) | (uint64_t(1024 >> 4) << 16) | (uint64_t(128 >> 4) << 32);',
            ]
            if register_source_a:
                prefix = f"sm90_ra{stage}"
                row = "(sm90_row + int64_t(sm90_warp * 16 + sm90_lane / 4))"
                col = f"({(first_chunk + stage) * 16}LL + int64_t((sm90_lane & 3) * 2))"
                def aval(row_expr: str, col_expr: str) -> str:
                    return f"{cast}((({row_expr}) < {m}LL && ({col_expr}) < {k}LL) ? {read_a(row_expr, col_expr)} : 0.0f)"
                lines += [
                    f"    union {{ __nv_bfloat162 v; uint32_t u; }} {prefix}0, {prefix}1, {prefix}2, {prefix}3;",
                    f"    {prefix}0.v = __halves2bfloat162({aval(row, col)}, {aval(row, f'({col}) + 1LL')});",
                    f"    {prefix}1.v = __halves2bfloat162({aval(f'({row}) + 8LL', col)}, {aval(f'({row}) + 8LL', f'({col}) + 1LL')});",
                    f"    {prefix}2.v = __halves2bfloat162({aval(row, f'({col}) + 8LL')}, {aval(row, f'({col}) + 9LL')});",
                    f"    {prefix}3.v = __halves2bfloat162({aval(f'({row}) + 8LL', f'({col}) + 8LL')}, {aval(f'({row}) + 8LL', f'({col}) + 9LL')});",
                ]
                operand_groups.append([f'"r"({prefix}{i}.u)' for i in range(4)] + [f'"l"({db})'])
            else:
                operand_groups.append([f'"l"({da})', f'"l"({db})'])
        lines.append('    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");')
        outputs = first_constraints if first_chunk == 0 else constraints
        asm_lines = []
        operand_constraints = []
        for stage in range(group_chunks):
            scale = 0 if first_chunk + stage == 0 else 1
            operand_constraints.extend(operand_groups[stage])
            if register_source_a:
                base = 32 + stage * 5
                a_regs = ",".join(f"%{base + i}" for i in range(4))
                asm_lines.append(f'      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {{{regs}}}, {{{a_regs}}}, %{base + 4}, {scale}, 1, 1, 0;\\n\\t"')
            else:
                base = 32 + stage * 2
                asm_lines.append(f'      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {{{regs}}}, %{base}, %{base + 1}, {scale}, 1, 1, 0, 0;\\n\\t"')
        input_constraints = ", ".join(operand_constraints)
        lines += [
            '    asm volatile(',
            *asm_lines,
            '      "wgmma.commit_group.sync.aligned;\\n\\t"',
            '      "wgmma.wait_group.sync.aligned 0;"',
            f'      : {outputs}',
            f'      : {input_constraints});',
            '    __syncthreads();',
        ]
        if register_source_a:
            lines.append('    }')
    lines += ([] if register_source_a else [
        "    const int sm90_lane = threadIdx.x & 31, sm90_warp = threadIdx.x >> 5;",
    ]) + [
        "    const int sm90_base_row = sm90_warp * 16 + sm90_lane / 4, sm90_base_col = (sm90_lane & 3) * 2;",
    ]
    for group in range(8):
        column = group * 8
        group_lines = [
            f"    const int64_t sm90_c{group} = sm90_col + sm90_base_col + {column};",
            "    const int64_t sm90_r0 = sm90_row + sm90_base_row;" if group == 0 else "",
            f"    if (sm90_r0 < {m}LL && sm90_c{group} < {n}LL) {{ const int64_t sm90_i = sm90_r0 * {n}LL + sm90_c{group}; {store('sm90_i', ep('sm90_i', f'd{4 * group}'))} }}",
            f"    if (sm90_r0 < {m}LL && sm90_c{group} + 1 < {n}LL) {{ const int64_t sm90_i = sm90_r0 * {n}LL + sm90_c{group} + 1; {store('sm90_i', ep('sm90_i', f'd{4 * group + 1}'))} }}",
            f"    if (sm90_r0 + 8 < {m}LL && sm90_c{group} < {n}LL) {{ const int64_t sm90_i = (sm90_r0 + 8) * {n}LL + sm90_c{group}; {store('sm90_i', ep('sm90_i', f'd{4 * group + 2}'))} }}",
            f"    if (sm90_r0 + 8 < {m}LL && sm90_c{group} + 1 < {n}LL) {{ const int64_t sm90_i = (sm90_r0 + 8) * {n}LL + sm90_c{group} + 1; {store('sm90_i', ep('sm90_i', f'd{4 * group + 3}'))} }}",
        ]
        lines.extend(line for line in group_lines if line)
    lines += [
        "    __syncthreads();",
        "  }",
        "}",
    ]
    return SM90Emission("\n".join(lines), shared_a_elements=0 if register_source_a else 1024 * k_group_size, shared_b_elements=1024 * k_group_size)
