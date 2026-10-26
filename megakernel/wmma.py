"""Optional 32x64 WMMA GEMM source emitter.

This module deliberately has no dependency on the scheduler.  The callbacks
are the small boundary needed to retain scheduler-specific addressing and
epilogue fusion while keeping this experimental tile independently testable.
"""

from dataclasses import dataclass
from typing import Callable


Read = Callable[[str, str], str]
Store = Callable[[str, str], str]
Epilogue = Callable[[str, str], str]


@dataclass(frozen=True)
class WMMAEmission:
    source: str
    tile_m: int = 32
    tile_n: int = 64
    warps: int = 4
    shared_a_elements: int = 0
    shared_b_elements: int = 0
    shared_c_elements: int = 2048

    # A/B are FP16/BF16 (2 bytes each); C is FP32 (4 bytes).  Keeping these
    # separate avoids treating the mixed-width allocation as one element type.
    @property
    def shared_a_bytes(self) -> int:
        return self.shared_a_elements * 2

    @property
    def shared_b_bytes(self) -> int:
        return self.shared_b_elements * 2

    @property
    def shared_c_bytes(self) -> int:
        return self.shared_c_elements * 4

    @property
    def shared_bytes(self) -> int:
        return self.shared_a_bytes + self.shared_b_bytes + self.shared_c_bytes

    @property
    def shared_elements(self) -> int:
        return self.shared_a_elements + self.shared_b_elements + self.shared_c_elements


def emit_wmma(
    m: int,
    k: int,
    n: int,
    read_a: Read,
    read_b: Read,
    store: Store,
    epilogue: Epilogue | None = None,
    *,
    k_tile: int = 32,
    value_type: str = "__nv_bfloat16",
    cast: str = "__float2bfloat16_rn",
) -> WMMAEmission:
    """Emit one 32x64 low-precision GEMM work tile.

    ``read_a(row, k)`` and ``read_b(k, col)`` return device expressions.
    ``store(index, value)`` must return a complete CUDA statement.  The
    optional epilogue receives the logical output index and accumulator
    expression, and may return a replacement expression.
    """
    if min(m, k, n) < 0 or k_tile not in (16, 32):
        raise ValueError("WMMA dimensions must be non-negative and k_tile must be 16 or 32")
    ep = epilogue or (lambda _index, value: value)
    a_tile = 32 * k_tile
    b_tile = k_tile * 64
    chunks = (k + k_tile - 1) // k_tile
    # Each warp owns one 16-row A slab and two 16-column output slabs.
    lines = [
        "{",
        f"  auto* wmma_a = reinterpret_cast<{value_type}*>(shared_a);",
        f"  auto* wmma_b = reinterpret_cast<{value_type}*>(shared_b);",
        "  const int wmma_warp = threadIdx.x >> 5;",
        f"  for (int64_t wtile = blockIdx.x; wtile < {((m + 31) // 32) * ((n + 63) // 64)}LL; wtile += gridDim.x) {{",
        f"    const int64_t wrow = (wtile / {max(1, (n + 63) // 64)}LL) * 32LL;",
        f"    const int64_t wcol = (wtile % {max(1, (n + 63) // 64)}LL) * 64LL;",
        "    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> acc0, acc1;",
        "    nvcuda::wmma::fill_fragment(acc0, 0.0f);",
        "    nvcuda::wmma::fill_fragment(acc1, 0.0f);",
        f"    for (int64_t wkk = 0; wkk < {k}LL; wkk += {k_tile}) {{",
        f"      for (int p = threadIdx.x; p < {max(a_tile, b_tile)}; p += blockDim.x) {{",
        f"        if (p < {a_tile}) {{ const int64_t ar = wrow + p / {k_tile}, ak = wkk + p % {k_tile}; wmma_a[p] = {cast}((ar < {m}LL && ak < {k}LL) ? {read_a('ar', 'ak')} : 0.0f); }}",
        f"        if (p < {b_tile}) {{ const int64_t bk = wkk + p / 64, bc = wcol + p % 64; wmma_b[p] = {cast}((bk < {k}LL && bc < {n}LL) ? {read_b('bk', 'bc')} : 0.0f); }}",
        "      }",
        "      __syncthreads();",
        "      if (wmma_warp < 4) {",
        f"        const int arow = (wmma_warp >> 1) * 16;",
        "        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16," + value_type + ",nvcuda::wmma::row_major> af;",
        "        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16," + value_type + ",nvcuda::wmma::row_major> bf0, bf1;",
        f"        nvcuda::wmma::load_matrix_sync(af, wmma_a + arow * {k_tile}, {k_tile});",
        "        nvcuda::wmma::load_matrix_sync(bf0, wmma_b + (wmma_warp & 1) * 16, 64);",
        "        nvcuda::wmma::load_matrix_sync(bf1, wmma_b + (wmma_warp & 1) * 16 + 32, 64);",
        "        nvcuda::wmma::mma_sync(acc0, af, bf0, acc0);",
        "        nvcuda::wmma::mma_sync(acc1, af, bf1, acc1);",
    ]
    if k_tile == 32:
        lines += [
            "        nvcuda::wmma::load_matrix_sync(af, wmma_a + arow * 32 + 16, 32);",
            "        nvcuda::wmma::load_matrix_sync(bf0, wmma_b + 16 * 64 + (wmma_warp & 1) * 16, 64);",
            "        nvcuda::wmma::load_matrix_sync(bf1, wmma_b + 16 * 64 + (wmma_warp & 1) * 16 + 32, 64);",
            "        nvcuda::wmma::mma_sync(acc0, af, bf0, acc0);",
            "        nvcuda::wmma::mma_sync(acc1, af, bf1, acc1);",
        ]
    lines += [
        "      }",
        "      __syncthreads();",
        "    }",
        "    if (wmma_warp < 4) {",
        "      nvcuda::wmma::store_matrix_sync(shared_c + (wmma_warp >> 1) * 16 * 64 + (wmma_warp & 1) * 16, acc0, 64, nvcuda::wmma::mem_row_major);",
        "      nvcuda::wmma::store_matrix_sync(shared_c + (wmma_warp >> 1) * 16 * 64 + (wmma_warp & 1) * 16 + 32, acc1, 64, nvcuda::wmma::mem_row_major);",
        "    }",
        "    __syncthreads();",
        "    for (int p = threadIdx.x; p < 2048; p += blockDim.x) {",
        "      const int64_t rr = wrow + p / 64, cc = wcol + p % 64;",
        f"      if (rr < {m}LL && cc < {n}LL) {{ const int64_t oi = rr * {n}LL + cc; {store('oi', ep('oi', 'shared_c[p]'))} }}",
        "    }",
        "    __syncthreads();",
        "  }",
        "}",
    ]
    return WMMAEmission("\n".join(lines), shared_a_elements=a_tile, shared_b_elements=b_tile)

