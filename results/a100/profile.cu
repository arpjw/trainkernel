
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
struct Arguments { const void* inputs[6]; void* outputs[9]; };
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
// mm: mm
{
{
            auto* sa = reinterpret_cast<__nv_bfloat16*>(shared_a);
            auto* sb = reinterpret_cast<__nv_bfloat16*>(shared_b);
            const int warp = threadIdx.x / 32;
            for (int64_t tile = blockIdx.x; tile < 8LL; tile += gridDim.x) {
                const int64_t row = (tile / 8LL) * 32;
                const int64_t col = (tile % 8LL) * 32;
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
                nvcuda::wmma::fill_fragment(acc, 0.0f);
                for (int64_t kk = 0; kk < 128LL; kk += 16) {
                    for (int p = threadIdx.x; p < 512; p += blockDim.x) {
                        const int64_t ar = row + p / 16, ak = kk + p % 16;
                        const int64_t bk = kk + p / 32, bc = col + p % 32;
                        sa[p] = __float2bfloat16_rn((ar < 32LL && ak < 128LL) ? __bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[4])[0LL + ((ar) * 128LL) + ((ak) * 1LL)]) : 0.0f);
                        sb[p] = __float2bfloat16_rn((bk < 128LL && bc < 256LL) ? __bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[0])[0LL + ((bk) * 256LL) + ((bc) * 1LL)]) : 0.0f);
                    }
                    __syncthreads();
                    if (warp < 4) {
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> af;
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> bf;
                    nvcuda::wmma::load_matrix_sync(af, sa + (warp / 2) * 16 * 16, 16);
                    nvcuda::wmma::load_matrix_sync(bf, sb + (warp % 2) * 16, 32);
                    nvcuda::wmma::mma_sync(acc, af, bf, acc);
                    }
                    __syncthreads();
                }
                if (warp < 4) nvcuda::wmma::store_matrix_sync(shared_c + (warp / 2) * 16 * 32 + (warp % 2) * 16, acc, 32, nvcuda::wmma::mem_row_major);
                __syncthreads();
                for (int p = threadIdx.x; p < 1024; p += blockDim.x) {
                    int64_t rr = row + p / 32, cc = col + p % 32;
                    if (rr < 32LL && cc < 256LL) workspace[0LL + (rr * 256LL + cc)] = __bfloat162float(__float2bfloat16_rn(shared_c[p]));
                }
                __syncthreads();
            }
            }
}
mk_grid_sync(grid);
// relu: relu
{
for (int64_t i = tid; i < 8192LL; i += nt) {
            workspace[8192LL + (i)] = __bfloat162float(__float2bfloat16_rn(mk_relu(__bfloat162float(__float2bfloat16_rn((workspace[0LL + (0LL + (((((0LL + (((((i) / 256LL) % 32LL)) * 256LL) + (((((i) / 1LL) % 256LL)) * 1LL)) / 256LL) % 32LL)) * 256LL) + (((((0LL + (((((i) / 256LL) % 32LL)) * 256LL) + (((((i) / 1LL) % 256LL)) * 1LL)) / 1LL) % 256LL)) * 1LL))] + (1.00000000000000000e+00f * __bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[1])[0LL + (((((0LL + (((((i) / 256LL) % 32LL)) * 256LL) + (((((i) / 1LL) % 256LL)) * 1LL)) / 1LL) % 256LL)) * 1LL)]))))))));
        }
}
mk_grid_sync(grid);
// mm_1: mm
{
{
            auto* sa = reinterpret_cast<__nv_bfloat16*>(shared_a);
            auto* sb = reinterpret_cast<__nv_bfloat16*>(shared_b);
            const int warp = threadIdx.x / 32;
            for (int64_t tile = blockIdx.x; tile < 2LL; tile += gridDim.x) {
                const int64_t row = (tile / 2LL) * 32;
                const int64_t col = (tile % 2LL) * 32;
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
                nvcuda::wmma::fill_fragment(acc, 0.0f);
                for (int64_t kk = 0; kk < 256LL; kk += 16) {
                    for (int p = threadIdx.x; p < 512; p += blockDim.x) {
                        const int64_t ar = row + p / 16, ak = kk + p % 16;
                        const int64_t bk = kk + p / 32, bc = col + p % 32;
                        sa[p] = __float2bfloat16_rn((ar < 32LL && ak < 256LL) ? workspace[8192LL + (0LL + ((ar) * 256LL) + ((ak) * 1LL))] : 0.0f);
                        sb[p] = __float2bfloat16_rn((bk < 256LL && bc < 64LL) ? __bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[2])[0LL + ((bk) * 64LL) + ((bc) * 1LL)]) : 0.0f);
                    }
                    __syncthreads();
                    if (warp < 4) {
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> af;
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> bf;
                    nvcuda::wmma::load_matrix_sync(af, sa + (warp / 2) * 16 * 16, 16);
                    nvcuda::wmma::load_matrix_sync(bf, sb + (warp % 2) * 16, 32);
                    nvcuda::wmma::mma_sync(acc, af, bf, acc);
                    }
                    __syncthreads();
                }
                if (warp < 4) nvcuda::wmma::store_matrix_sync(shared_c + (warp / 2) * 16 * 32 + (warp % 2) * 16, acc, 32, nvcuda::wmma::mem_row_major);
                __syncthreads();
                for (int p = threadIdx.x; p < 1024; p += blockDim.x) {
                    int64_t rr = row + p / 32, cc = col + p % 32;
                    if (rr < 32LL && cc < 64LL) workspace[0LL + (rr * 64LL + cc)] = __bfloat162float(__float2bfloat16_rn(shared_c[p]));
                }
                __syncthreads();
            }
            }
}
mk_grid_sync(grid);
// sub: sub
{
for (int64_t i = tid; i < 2048LL; i += nt) {
            workspace[2048LL + (i)] = __bfloat162float(__float2bfloat16_rn((__bfloat162float(__float2bfloat16_rn((workspace[0LL + (0LL + (((((0LL + (((((i) / 64LL) % 32LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)) / 64LL) % 32LL)) * 64LL) + (((((0LL + (((((i) / 64LL) % 32LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)) / 1LL) % 64LL)) * 1LL))] + (1.00000000000000000e+00f * __bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[3])[0LL + (((((0LL + (((((i) / 64LL) % 32LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)) / 1LL) % 64LL)) * 1LL)]))))) - (1.00000000000000000e+00f * __bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[5])[0LL + (((((i) / 64LL) % 32LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)])))));
        }
}
mk_grid_sync(grid);
// mean: mean
{
for (int64_t row = blockIdx.x; row < 1LL; row += gridDim.x) {
            float acc = 0.0f;
            for (int64_t r = threadIdx.x; r < 2048LL; r += blockDim.x) acc += __bfloat162float(__float2bfloat16_rn(mk_square(workspace[2048LL + (0LL + (((((0LL + (((((r) / 64LL) % 32LL)) * 64LL) + (((((r) / 1LL) % 64LL)) * 1LL)) / 64LL) % 32LL)) * 64LL) + (((((0LL + (((((r) / 64LL) % 32LL)) * 64LL) + (((((r) / 1LL) % 64LL)) * 1LL)) / 1LL) % 64LL)) * 1LL))])));
            shared_reduce[threadIdx.x] = acc;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) shared_reduce[threadIdx.x] += shared_reduce[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) static_cast<__nv_bfloat16*>(args.outputs[0])[row] = __float2bfloat16_rn((shared_reduce[0] / 2048.0f));
            __syncthreads();
        }
}
// mul_1: mul
{
for (int64_t i = tid; i < 2048LL; i += nt) {
            workspace[4096LL + (i)] = __bfloat162float(__float2bfloat16_rn((__bfloat162float(__float2bfloat16_rn((__bfloat162float(__float2bfloat16_rn(1.00000000000000000e+00f)) / 2.04800000000000000e+03f))) * __bfloat162float(__float2bfloat16_rn((__bfloat162float(__float2bfloat16_rn(powf(workspace[2048LL + (0LL + (((((0LL + (((((0LL + (((((i) / 64LL) % 32LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)) / 64LL) % 32LL)) * 64LL) + (((((0LL + (((((i) / 64LL) % 32LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)) / 1LL) % 64LL)) * 1LL)) / 64LL) % 32LL)) * 64LL) + (((((0LL + (((((0LL + (((((i) / 64LL) % 32LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)) / 64LL) % 32LL)) * 64LL) + (((((0LL + (((((i) / 64LL) % 32LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)) / 1LL) % 64LL)) * 1LL)) / 1LL) % 64LL)) * 1LL))], 1.00000000000000000e+00f))) * 2.00000000000000000e+00f))))));
        }
}
mk_grid_sync(grid);
// sum_1: sum
{
for (int64_t row = tid; row < 64LL; row += nt) {
                float acc = 0.0f;
                for (int64_t r = 0; r < 32LL; ++r) acc += workspace[4096LL + (0LL + (((((r) / 1LL) % 32LL)) * 64LL) + (((((row) / 1LL) % 64LL)) * 1LL))];
                static_cast<__nv_bfloat16*>(args.outputs[4])[row] = __float2bfloat16_rn(acc);
            }
}
// mm_2: mm
{
{
            auto* sa = reinterpret_cast<__nv_bfloat16*>(shared_a);
            auto* sb = reinterpret_cast<__nv_bfloat16*>(shared_b);
            const int warp = threadIdx.x / 32;
            for (int64_t tile = blockIdx.x; tile < 16LL; tile += gridDim.x) {
                const int64_t row = (tile / 2LL) * 32;
                const int64_t col = (tile % 2LL) * 32;
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
                nvcuda::wmma::fill_fragment(acc, 0.0f);
                for (int64_t kk = 0; kk < 32LL; kk += 16) {
                    for (int p = threadIdx.x; p < 512; p += blockDim.x) {
                        const int64_t ar = row + p / 16, ak = kk + p % 16;
                        const int64_t bk = kk + p / 32, bc = col + p % 32;
                        sa[p] = __float2bfloat16_rn((ar < 256LL && ak < 32LL) ? workspace[8192LL + (0LL + ((ar) * 1LL) + ((ak) * 256LL))] : 0.0f);
                        sb[p] = __float2bfloat16_rn((bk < 32LL && bc < 64LL) ? workspace[4096LL + (0LL + ((bk) * 64LL) + ((bc) * 1LL))] : 0.0f);
                    }
                    __syncthreads();
                    if (warp < 4) {
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> af;
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> bf;
                    nvcuda::wmma::load_matrix_sync(af, sa + (warp / 2) * 16 * 16, 16);
                    nvcuda::wmma::load_matrix_sync(bf, sb + (warp % 2) * 16, 32);
                    nvcuda::wmma::mma_sync(acc, af, bf, acc);
                    }
                    __syncthreads();
                }
                if (warp < 4) nvcuda::wmma::store_matrix_sync(shared_c + (warp / 2) * 16 * 32 + (warp % 2) * 16, acc, 32, nvcuda::wmma::mem_row_major);
                __syncthreads();
                for (int p = threadIdx.x; p < 1024; p += blockDim.x) {
                    int64_t rr = row + p / 32, cc = col + p % 32;
                    if (rr < 256LL && cc < 64LL) static_cast<__nv_bfloat16*>(args.outputs[3])[rr * 64LL + cc] = __float2bfloat16_rn(shared_c[p]);
                }
                __syncthreads();
            }
            }
}
// mm_3: mm
{
{
            auto* sa = reinterpret_cast<__nv_bfloat16*>(shared_a);
            auto* sb = reinterpret_cast<__nv_bfloat16*>(shared_b);
            const int warp = threadIdx.x / 32;
            for (int64_t tile = blockIdx.x; tile < 8LL; tile += gridDim.x) {
                const int64_t row = (tile / 8LL) * 32;
                const int64_t col = (tile % 8LL) * 32;
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
                nvcuda::wmma::fill_fragment(acc, 0.0f);
                for (int64_t kk = 0; kk < 64LL; kk += 16) {
                    for (int p = threadIdx.x; p < 512; p += blockDim.x) {
                        const int64_t ar = row + p / 16, ak = kk + p % 16;
                        const int64_t bk = kk + p / 32, bc = col + p % 32;
                        sa[p] = __float2bfloat16_rn((ar < 32LL && ak < 64LL) ? workspace[4096LL + (0LL + ((ar) * 64LL) + ((ak) * 1LL))] : 0.0f);
                        sb[p] = __float2bfloat16_rn((bk < 64LL && bc < 256LL) ? __bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[2])[0LL + ((bk) * 1LL) + ((bc) * 64LL)]) : 0.0f);
                    }
                    __syncthreads();
                    if (warp < 4) {
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> af;
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> bf;
                    nvcuda::wmma::load_matrix_sync(af, sa + (warp / 2) * 16 * 16, 16);
                    nvcuda::wmma::load_matrix_sync(bf, sb + (warp % 2) * 16, 32);
                    nvcuda::wmma::mma_sync(acc, af, bf, acc);
                    }
                    __syncthreads();
                }
                if (warp < 4) nvcuda::wmma::store_matrix_sync(shared_c + (warp / 2) * 16 * 32 + (warp % 2) * 16, acc, 32, nvcuda::wmma::mem_row_major);
                __syncthreads();
                for (int p = threadIdx.x; p < 1024; p += blockDim.x) {
                    int64_t rr = row + p / 32, cc = col + p % 32;
                    if (rr < 32LL && cc < 256LL) workspace[32768LL + (rr * 256LL + cc)] = __bfloat162float(__float2bfloat16_rn(shared_c[p]));
                }
                __syncthreads();
            }
            }
}
mk_grid_sync(grid);
// threshold_backward: threshold_backward
{
for (int64_t i = tid; i < 8192LL; i += nt) {
            workspace[40960LL + (i)] = __bfloat162float(__float2bfloat16_rn(((workspace[8192LL + (0LL + (((((i) / 256LL) % 32LL)) * 256LL) + (((((i) / 1LL) % 256LL)) * 1LL))] <= 0.00000000000000000e+00f) ? 0.0f : workspace[32768LL + (0LL + (((((i) / 256LL) % 32LL)) * 256LL) + (((((i) / 1LL) % 256LL)) * 1LL))])));
        }
}
mk_grid_sync(grid);
// sum_2: sum
{
for (int64_t row = tid; row < 256LL; row += nt) {
                float acc = 0.0f;
                for (int64_t r = 0; r < 32LL; ++r) acc += workspace[40960LL + (0LL + (((((r) / 1LL) % 32LL)) * 256LL) + (((((row) / 1LL) % 256LL)) * 1LL))];
                static_cast<__nv_bfloat16*>(args.outputs[2])[row] = __float2bfloat16_rn(acc);
            }
}
// mm_4: mm
{
{
            auto* sa = reinterpret_cast<__nv_bfloat16*>(shared_a);
            auto* sb = reinterpret_cast<__nv_bfloat16*>(shared_b);
            const int warp = threadIdx.x / 32;
            for (int64_t tile = blockIdx.x; tile < 32LL; tile += gridDim.x) {
                const int64_t row = (tile / 8LL) * 32;
                const int64_t col = (tile % 8LL) * 32;
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
                nvcuda::wmma::fill_fragment(acc, 0.0f);
                for (int64_t kk = 0; kk < 32LL; kk += 16) {
                    for (int p = threadIdx.x; p < 512; p += blockDim.x) {
                        const int64_t ar = row + p / 16, ak = kk + p % 16;
                        const int64_t bk = kk + p / 32, bc = col + p % 32;
                        sa[p] = __float2bfloat16_rn((ar < 128LL && ak < 32LL) ? __bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[4])[0LL + ((ar) * 1LL) + ((ak) * 128LL)]) : 0.0f);
                        sb[p] = __float2bfloat16_rn((bk < 32LL && bc < 256LL) ? workspace[40960LL + (0LL + ((bk) * 256LL) + ((bc) * 1LL))] : 0.0f);
                    }
                    __syncthreads();
                    if (warp < 4) {
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> af;
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> bf;
                    nvcuda::wmma::load_matrix_sync(af, sa + (warp / 2) * 16 * 16, 16);
                    nvcuda::wmma::load_matrix_sync(bf, sb + (warp % 2) * 16, 32);
                    nvcuda::wmma::mma_sync(acc, af, bf, acc);
                    }
                    __syncthreads();
                }
                if (warp < 4) nvcuda::wmma::store_matrix_sync(shared_c + (warp / 2) * 16 * 32 + (warp % 2) * 16, acc, 32, nvcuda::wmma::mem_row_major);
                __syncthreads();
                for (int p = threadIdx.x; p < 1024; p += blockDim.x) {
                    int64_t rr = row + p / 32, cc = col + p % 32;
                    if (rr < 128LL && cc < 256LL) static_cast<__nv_bfloat16*>(args.outputs[1])[rr * 256LL + cc] = __float2bfloat16_rn(shared_c[p]);
                }
                __syncthreads();
            }
            }
}
mk_grid_sync(grid);
// sub_1: sub
{
for (int64_t i = tid; i < 32768LL; i += nt) {
            static_cast<__nv_bfloat16*>(args.outputs[5])[i] = __float2bfloat16_rn((__bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[0])[0LL + (((((i) / 256LL) % 128LL)) * 256LL) + (((((i) / 1LL) % 256LL)) * 1LL)]) - (1.00000000000000000e+00f * __bfloat162float(__float2bfloat16_rn((__bfloat162float(static_cast<const __nv_bfloat16*>(args.outputs[1])[0LL + (((((0LL + (((((i) / 256LL) % 128LL)) * 256LL) + (((((i) / 1LL) % 256LL)) * 1LL)) / 256LL) % 128LL)) * 256LL) + (((((0LL + (((((i) / 256LL) % 128LL)) * 256LL) + (((((i) / 1LL) % 256LL)) * 1LL)) / 1LL) % 256LL)) * 1LL)]) * 1.00000000000000002e-03f))))));
        }
}
// sub_2: sub
{
for (int64_t i = tid; i < 256LL; i += nt) {
            static_cast<__nv_bfloat16*>(args.outputs[6])[i] = __float2bfloat16_rn((__bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[1])[0LL + (((((i) / 1LL) % 256LL)) * 1LL)]) - (1.00000000000000000e+00f * __bfloat162float(__float2bfloat16_rn((__bfloat162float(static_cast<const __nv_bfloat16*>(args.outputs[2])[0LL + (((((0LL + (((((i) / 1LL) % 256LL)) * 1LL)) / 1LL) % 256LL)) * 1LL)]) * 1.00000000000000002e-03f))))));
        }
}
// sub_3: sub
{
for (int64_t i = tid; i < 16384LL; i += nt) {
            static_cast<__nv_bfloat16*>(args.outputs[7])[i] = __float2bfloat16_rn((__bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[2])[0LL + (((((i) / 64LL) % 256LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)]) - (1.00000000000000000e+00f * __bfloat162float(__float2bfloat16_rn((__bfloat162float(static_cast<const __nv_bfloat16*>(args.outputs[3])[0LL + (((((0LL + (((((i) / 64LL) % 256LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)) / 64LL) % 256LL)) * 64LL) + (((((0LL + (((((i) / 64LL) % 256LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)) / 1LL) % 64LL)) * 1LL)]) * 1.00000000000000002e-03f))))));
        }
}
// sub_4: sub
{
for (int64_t i = tid; i < 64LL; i += nt) {
            static_cast<__nv_bfloat16*>(args.outputs[8])[i] = __float2bfloat16_rn((__bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[3])[0LL + (((((i) / 1LL) % 64LL)) * 1LL)]) - (1.00000000000000000e+00f * __bfloat162float(__float2bfloat16_rn((__bfloat162float(static_cast<const __nv_bfloat16*>(args.outputs[4])[0LL + (((((0LL + (((((i) / 1LL) % 64LL)) * 1LL)) / 1LL) % 64LL)) * 1LL)]) * 1.00000000000000002e-03f))))));
        }
}
mk_grid_sync(grid);
}

void launch(std::vector<torch::Tensor> inputs, std::vector<torch::Tensor> outputs, torch::Tensor workspace) {
    TORCH_CHECK(inputs.size() == 6 && outputs.size() == 9, "incorrect tensor count");
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
    int blocks = std::min(prop.multiProcessorCount * std::min(active, 1), 256);
    float* arena = workspace.data_ptr<float>();
    void* kernel_args[] = {&args, &arena};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(megakernel_entry), dim3(blocks), dim3(128), kernel_args, 0, at::cuda::getCurrentCUDAStream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
