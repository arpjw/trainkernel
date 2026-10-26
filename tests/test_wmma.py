from trainkernel.wmma import emit_wmma
import pytest
import torch


def _read_a(row, col):
    return f"A[{row} * 97 + {col}]"


def _read_b(row, col):
    return f"B[{row} * 83 + {col}]"


def _store(index, value):
    return f"C[{index}] = {value};"


def test_wmma_32x64_metadata_and_dual_accumulators():
    out = emit_wmma(37, 49, 70, _read_a, _read_b, _store)
    assert (out.tile_m, out.tile_n, out.warps) == (32, 64, 4)
    assert out.shared_a_elements == 32 * 32
    assert out.shared_b_elements == 32 * 64
    assert out.shared_c_elements == 32 * 64
    assert out.shared_elements == 5120
    assert (out.shared_a_bytes, out.shared_b_bytes, out.shared_c_bytes, out.shared_bytes) == (2048, 4096, 8192, 14336)
    assert out.source.count("mma_sync") == 4
    assert "acc0" in out.source and "acc1" in out.source
    assert "wmma_b + (wmma_warp & 1) * 16, 64" in out.source
    assert "wmma_b + (wmma_warp & 1) * 16 + 32, 64" in out.source
    assert "wmma_b + 16 * 64 + (wmma_warp & 1) * 16" in out.source
    assert "rr < 37LL && cc < 70LL" in out.source


def test_wmma_masks_a_and_b_tails_and_applies_epilogue():
    out = emit_wmma(
        3,
        17,
        65,
        _read_a,
        _read_b,
        _store,
        lambda index, value: f"({value} + bias[{index}])",
        k_tile=16,
    )
    assert "ar < 3LL && ak < 17LL" in out.source
    assert "bk < 17LL && bc < 65LL" in out.source
    assert "shared_c[p] + bias[oi]" in out.source
    assert out.source.count("mma_sync") == 2
    assert "wmma_b + (wmma_warp & 1) * 16, 64" in out.source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k_tile", [16, 32])
def test_wmma_cuda_tail_matmul(dtype, k_tile, tmp_path):
    """Compile and execute the isolated emitter, including non-tile tails."""
    typ = "half" if dtype is torch.float16 else "__nv_bfloat16"
    cast = "__float2half_rn" if dtype is torch.float16 else "__float2bfloat16_rn"
    read_a = (lambda r, c: f"__half2float(A[{r} * 47 + {c}])") if dtype is torch.float16 else (lambda r, c: f"__bfloat162float(A[{r} * 47 + {c}])")
    read_b = (lambda r, c: f"__half2float(B[{r} * 79 + {c}])") if dtype is torch.float16 else (lambda r, c: f"__bfloat162float(B[{r} * 79 + {c}])")
    emission = emit_wmma(33, 47, 79, read_a, read_b, lambda i, v: f"C[{i}] = {v};", k_tile=k_tile, value_type=typ, cast=cast)
    source = f'''#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
__global__ void mm_kernel(const {typ}* A, const {typ}* B, float* C) {{
  __shared__ __align__(32) unsigned char shared_a[{32 * k_tile * 2}];
  __shared__ __align__(32) unsigned char shared_b[{64 * k_tile * 2}];
  __shared__ float shared_c[2048];
{emission.source}
}}
void run(torch::Tensor A, torch::Tensor B, torch::Tensor C) {{ mm_kernel<<<2,128>>>(reinterpret_cast<const {typ}*>(A.data_ptr()), reinterpret_cast<const {typ}*>(B.data_ptr()), C.data_ptr<float>()); C10_CUDA_KERNEL_LAUNCH_CHECK(); }}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{ m.def("run", &run); }}
'''
    from torch.utils.cpp_extension import load_inline
    name = f"wmma_tail_{'f16' if dtype is torch.float16 else 'bf16'}_{k_tile}"
    mod = load_inline(name, cpp_sources="", cuda_sources=source, functions=None, extra_cuda_cflags=["-O2"], build_directory=str(tmp_path), verbose=False)
    a = torch.randn((33, 47), device="cuda", dtype=dtype)
    b = torch.randn((47, 79), device="cuda", dtype=dtype)
    out = torch.empty((33, 79), device="cuda", dtype=torch.float32)
    mod.run(a, b, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, a.float() @ b.float(), rtol=2e-4, atol=1e-4)

