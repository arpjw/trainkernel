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
    ep = epilogue or (lambda _index, value: value)
    mt, nt = (m + 63) // 64, (n + 63) // 64
    chunks = (k + 15) // 16
    regs = ",".join(f"%{i}" for i in range(32))
    constraints = ", ".join(f'"+f"(d{i})' for i in range(32))
    first_constraints = ", ".join(f'"=f"(d{i})' for i in range(32))
    lines = [
        "{",
        f"  auto* sm90_a = reinterpret_cast<{value_type}*>(shared_a);",
        f"  auto* sm90_b = reinterpret_cast<{value_type}*>(shared_b);",
        f"  float d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15;",
        f"  float d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31;",
        "  uint64_t sm90_da, sm90_db;",
        f"  for (int64_t sm90_tile = blockIdx.x; sm90_tile < {mt * nt}LL; sm90_tile += gridDim.x) {{",
        f"    const int64_t sm90_row = (sm90_tile / {nt}LL) * 64LL;",
        f"    const int64_t sm90_col = (sm90_tile % {nt}LL) * 64LL;",
    ]
    for chunk in range(chunks):
        kk = chunk * 16
        lines += [
            f"    for (int p = threadIdx.x; p < 1024; p += blockDim.x) {{",
            "      const int sm90_r = p / 16, sm90_q = p % 16;",
            f"      const int64_t sm90_ar = sm90_row + sm90_r, sm90_ak = {kk}LL + sm90_q;",
            "      const int sm90_ap = (sm90_r % 8) * 8 + (sm90_r / 8) * 64 + (sm90_q % 8) + (sm90_q / 8) * 512;",
            f"      sm90_a[sm90_ap] = {cast}((sm90_ar < {m}LL && sm90_ak < {k}LL) ? {read_a('sm90_ar', 'sm90_ak')} : 0.0f);",
            "      const int64_t sm90_bn = sm90_col + sm90_r, sm90_bk = " + str(kk) + "LL + sm90_q;",
            "      const int sm90_bp = (sm90_r % 8) * 8 + (sm90_r / 8) * 64 + (sm90_q % 8) + (sm90_q / 8) * 512;",
            f"      sm90_b[sm90_bp] = {cast}((sm90_bn < {n}LL && sm90_bk < {k}LL) ? {read_b('sm90_bk', 'sm90_bn')} : 0.0f);",
            "    }",
            "    __syncthreads();",
            '    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");',
            '    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(sm90_da) : "l"(sm90_a));',
            '    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(sm90_db) : "l"(sm90_b));',
            '    sm90_da = ((sm90_da & 0x3ffffULL) >> 4) | (uint64_t(1024 >> 4) << 16) | (uint64_t(128 >> 4) << 32);',
            '    sm90_db = ((sm90_db & 0x3ffffULL) >> 4) | (uint64_t(1024 >> 4) << 16) | (uint64_t(128 >> 4) << 32);',
            '    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");',
        ]
        outputs = first_constraints if chunk == 0 else constraints
        lines += [
            '    asm volatile(',
            f'      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {{{regs}}}, %32, %33, {0 if chunk == 0 else 1}, 1, 1, 0, 0;\\n\\t"',
            '      "wgmma.commit_group.sync.aligned;\\n\\t"',
            '      "wgmma.wait_group.sync.aligned 0;"',
            f'      : {outputs}',
            '      : "l"(sm90_da), "l"(sm90_db));',
            '    __syncthreads();',
        ]
    lines += [
        "    const int sm90_lane = threadIdx.x & 31, sm90_warp = threadIdx.x >> 5;",
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
    return SM90Emission("\n".join(lines))
