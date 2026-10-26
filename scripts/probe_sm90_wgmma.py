"""Standalone, experimental Hopper WGMMA correctness probe.

This is intentionally separate from TrainKernel's production codegen.  It
compiles one 128-thread SM90a kernel for a 64x16 BF16 by 16x64 BF16 GEMM,
decodes the documented warpgroup accumulator layout, and compares the result
with PyTorch.  It is a correctness probe, not a performance benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from textwrap import dedent

import torch
from torch.utils.cpp_extension import CUDA_HOME, load_inline


CUDA_SOURCE = dedent(
    r'''
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDAGuard.h>
    #include <c10/cuda/CUDAException.h>
    #include <cuda.h>
    #include <cuda_bf16.h>
    #include <cuda_runtime.h>
    #include <cstdint>

    __device__ __forceinline__ uint64_t matrix_desc(const void* ptr, uint32_t lbo, uint32_t sbo) {
        // PTX matrix descriptors use 16-byte units for start, leading, and
        // stride offsets.  This probe intentionally uses no swizzle.
        uint64_t shared_address;
        asm("cvta.to.shared.u64 %0, %1;" : "=l"(shared_address) : "l"(ptr));
        const uint64_t start = (shared_address & 0x3ffffULL) >> 4;
        return start | (uint64_t(lbo >> 4) << 16) | (uint64_t(sbo >> 4) << 32);
    }

    __global__ void probe_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, float* out) {
        __shared__ __align__(128) __nv_bfloat16 sa[64 * 16];
        __shared__ __align__(128) __nv_bfloat16 sb[16 * 64];
        // A K-major, no-swizzle WGMMA layout is interleaved even though the
        // logical A matrix is row-major. Pack each 8-row by 8-K group using
        // the PTX canonical layout strides ((8,8),(8,2)):((8,64),(1,512)).
        for (int i = int(threadIdx.x); i < 64 * 16; i += int(blockDim.x)) {
            const int m = i / 16, k = i % 16;
            const int physical = (m % 8) * 8 + (m / 8) * 64 + (k % 8) + (k / 8) * 512;
            sa[physical] = a[i];
        }
        // WGMMA reads B column-major. Treat each logical output column as a
        // row of the same K-major canonical layout while staging.
        for (int i = int(threadIdx.x); i < 16 * 64; i += int(blockDim.x)) {
            const int k = i / 64, n = i % 64;
            const int physical = (n % 8) * 8 + (n / 8) * 64 + (k % 8) + (k / 8) * 512;
            sb[physical] = b[i];
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

        // LBO advances to the second 8-element K group after all 64 rows;
        // SBO advances to the next group of eight rows.
        const uint64_t da = matrix_desc(sa, 1024, 128);
        const uint64_t db = matrix_desc(sb, 1024, 128);
        float d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15;
        float d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31;
        asm volatile(
            "wgmma.fence.sync.aligned;\n\t"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
            "%32, %33, 0, 1, 1, 0, 0;\n\t"
            "wgmma.commit_group.sync.aligned;\n\t"
            "wgmma.wait_group.sync.aligned 0;\n\t"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3),
              "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7),
              "=f"(d8), "=f"(d9), "=f"(d10), "=f"(d11),
              "=f"(d12), "=f"(d13), "=f"(d14), "=f"(d15),
              "=f"(d16), "=f"(d17), "=f"(d18), "=f"(d19),
              "=f"(d20), "=f"(d21), "=f"(d22), "=f"(d23),
              "=f"(d24), "=f"(d25), "=f"(d26), "=f"(d27),
              "=f"(d28), "=f"(d29), "=f"(d30), "=f"(d31)
            : "l"(da), "l"(db));

        const int lane = int(threadIdx.x) & 31;
        const int warp = int(threadIdx.x) >> 5;
        const int row = warp * 16 + lane / 4;
        const int col = (lane & 3) * 2;
        const float d[32] = {d0, d1, d2, d3, d4, d5, d6, d7,
                             d8, d9, d10, d11, d12, d13, d14, d15,
                             d16, d17, d18, d19, d20, d21, d22, d23,
                             d24, d25, d26, d27, d28, d29, d30, d31};
        for (int g = 0; g < 8; ++g) {
            const int c = col + 8 * g;
            out[row * 64 + c] = d[4 * g + 0];
            out[row * 64 + c + 1] = d[4 * g + 1];
            out[(row + 8) * 64 + c] = d[4 * g + 2];
            out[(row + 8) * 64 + c + 1] = d[4 * g + 3];
        }
    }

    void launch(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
        TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(), "CUDA tensors required");
        TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kFloat, "unexpected dtypes");
        TORCH_CHECK(a.sizes() == torch::IntArrayRef({64, 16}) && b.sizes() == torch::IntArrayRef({16, 64}) && out.sizes() == torch::IntArrayRef({64, 64}), "unexpected shapes");
        TORCH_CHECK(a.device() == b.device() && a.device() == out.device(), "device mismatch");
        c10::cuda::CUDAGuard guard(a.device());
        probe_kernel<<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
            out.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    '''
)


def run(seed: int = 0, tolerance: float = 1e-2) -> dict[str, object]:
    if CUDA_HOME is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA toolkit and a CUDA device are required")
    capability = torch.cuda.get_device_capability()
    if capability != (9, 0):
        raise RuntimeError(f"this probe requires H100 compute capability 9.0, found {capability}")
    torch.manual_seed(seed)
    a = torch.randn((64, 16), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((16, 64), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((64, 64), device="cuda", dtype=torch.float32)
    name = "trainkernel_sm90a_probe_" + hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()[:16]
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        module = load_inline(
            name=name,
            cpp_sources="void launch(torch::Tensor, torch::Tensor, torch::Tensor);",
            cuda_sources=CUDA_SOURCE,
            functions=["launch"],
            extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"],
            with_cuda=True,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list
    module.launch(a, b, out)
    torch.cuda.synchronize()
    expected = a.float() @ b.float()
    max_error = float((out - expected).abs().max())
    passed = max_error <= tolerance
    return {"passed": passed, "max_abs_error": max_error, "tolerance": tolerance, "shape": [64, 16, 64], "capability": list(capability)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    args = parser.parse_args()
    result = run(args.seed, args.tolerance)
    print(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
