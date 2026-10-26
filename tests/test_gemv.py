from megakernel.gemv import emit_gemv
import pytest
import torch


def test_gemv_geometry_and_tail_mask():
    out = emit_gemv(1, 127, 65, lambda r, k: f"A[{r}*127+{k}]", lambda k, c: f"B[{k}*65+{c}]", lambda i, v: f"C[{i}]={v};")
    assert (out.tile_n, out.warps, out.threads) == (32, 4, 128)
    assert out.shared_c_elements == 128
    assert "kk += 4" in out.source
    assert "col < 65LL" in out.source
    assert "__syncthreads();" in out.source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gemv_cuda(dtype, tmp_path):
    typ = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "__nv_bfloat16"}[dtype]
    cv = {torch.float32: "A[row * 127 + kk]", torch.float16: "__half2float(A[row * 127 + kk])", torch.bfloat16: "__bfloat162float(A[row * 127 + kk])"}[dtype]
    bv = {torch.float32: "B[kk * 65 + col]", torch.float16: "__half2float(B[kk * 65 + col])", torch.bfloat16: "__bfloat162float(B[kk * 65 + col])"}[dtype]
    e = emit_gemv(1, 127, 65, lambda r, k: cv, lambda k, c: bv, lambda i, v: f"C[{i}] = {v};")
    src = f'''#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAException.h>
__global__ void g(const {typ}* A, const {typ}* B, float* C) {{ __shared__ __align__(32) float shared_c[128]; {e.source} }}
void run(torch::Tensor a, torch::Tensor b, torch::Tensor c) {{ g<<<2,128>>>(reinterpret_cast<const {typ}*>(a.data_ptr()), reinterpret_cast<const {typ}*>(b.data_ptr()), c.data_ptr<float>()); C10_CUDA_KERNEL_LAUNCH_CHECK(); }}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{ m.def("run",&run); }}'''
    from torch.utils.cpp_extension import load_inline
    mod = load_inline(f"gemv_{str(dtype).split('.')[-1]}", cpp_sources="", cuda_sources=src, functions=None, extra_cuda_cflags=["-O2"], build_directory=str(tmp_path), verbose=False)
    a = torch.randn(1, 127, device="cuda", dtype=dtype); b = torch.randn(127, 65, device="cuda", dtype=dtype); c = torch.empty(1, 65, device="cuda")
    mod.run(a, b, c); torch.cuda.synchronize()
    torch.testing.assert_close(c, a.float() @ b.float(), rtol=2e-4, atol=1e-4)

