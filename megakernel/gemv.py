"""Optional cooperative GEMV source emitter."""

from dataclasses import dataclass
from typing import Callable

Read = Callable[[str, str], str]
Store = Callable[[str, str], str]
Epilogue = Callable[[str, str], str]


@dataclass(frozen=True)
class GEMVEmission:
    source: str
    tile_n: int = 32
    warps: int = 4
    threads: int = 128
    shared_c_elements: int = 128


def emit_gemv(
    m: int, k: int, n: int, read_a: Read, read_b: Read, store: Store,
    epilogue: Epilogue | None = None,
    *, threads: int = 128,
) -> GEMVEmission:
    """Emit a 128-thread, four-warp GEMV with a 32-column output tile."""
    if min(m, k, n) < 0:
        raise ValueError("GEMV dimensions must be non-negative")
    if threads not in {128, 256, 512, 1024}:
        raise ValueError("unsupported GEMV block size")
    warps = threads // 32
    ep = epilogue or (lambda _index, value: value)
    lines = [
        "{",
        "  for (int64_t tile = blockIdx.x; tile < "
        f"{m * ((n + 31) // 32)}LL; tile += gridDim.x) {{",
        f"    const int64_t row = tile / {max(1, (n + 31) // 32)}LL;",
        f"    const int64_t col0 = (tile % {max(1, (n + 31) // 32)}LL) * 32LL;",
        "    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;",
        "    const int64_t col = col0 + lane;",
        "    float acc = 0.0f;",
        f"    for (int64_t kk = warp; kk < {k}LL; kk += {warps}) {{",
        f"      if (col < {n}LL) acc = fmaf({read_a('row', 'kk')}, {read_b('kk', 'col')}, acc);",
        "    }",
        "    shared_c[warp * 32 + lane] = acc;",
        "    __syncthreads();",
        "    if (warp == 0 && col < " + f"{n}LL) {{",
        f"      float sum = 0.0f; for (int w = 0; w < {warps}; ++w) sum += shared_c[w * 32 + lane];",
        "      const int64_t oi = row * " + f"{n}LL + col;",
        f"      {store('oi', ep('oi', 'sum'))}",
        "    }",
        "    __syncthreads();",
        "  }",
        "}",
    ]
    return GEMVEmission("\n".join(lines), warps=warps, threads=threads, shared_c_elements=threads)

