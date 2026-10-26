
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
    __shared__ __align__(32) unsigned short shared_a[1024];
    __shared__ __align__(32) unsigned short shared_b[1024];
    __shared__ __align__(32) float shared_c[1024];
    __shared__ float shared_reduce[128];
    __shared__ float shared_float_a[256];
    __shared__ float shared_float_b[128];
// mm_epilogue: mm
{
for (int64_t i = tid; i < 256LL; i += nt) {
                const int64_t rr = i / 64LL, cc = i % 64LL;
                float acc = 0.0f;
                #pragma unroll 4
                for (int kk = 0; kk < 32; ++kk) acc = fmaf(__bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[4])[0LL + ((rr) * 32LL) + ((kk) * 1LL)]), __bfloat162float(__float2bfloat16_rn(static_cast<const float*>(args.inputs[0])[0LL + (((((0LL + ((kk) * 64LL) + ((cc) * 1LL)) / 64LL) % 32LL)) * 64LL) + (((((0LL + ((kk) * 64LL) + ((cc) * 1LL)) / 1LL) % 64LL)) * 1LL)])), acc);
                workspace[0LL + (i)] = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(mk_relu(__bfloat162float(__float2bfloat16_rn((__bfloat162float(__float2bfloat16_rn(acc)) + (1.00000000000000000e+00f * __bfloat162float(__float2bfloat16_rn(static_cast<const float*>(args.inputs[1])[0LL + (((((0LL + (((((i) / 1LL) % 64LL)) * 1LL)) / 1LL) % 64LL)) * 1LL)])))))))))));
            }
}
mk_grid_sync(grid);
// mm_1: mm
{
for (int64_t i = tid; i < 64LL; i += nt) {
                const int64_t rr = i / 16LL, cc = i % 16LL;
                float acc = 0.0f;
                #pragma unroll 4
                for (int kk = 0; kk < 64; ++kk) acc = fmaf(workspace[0LL + (0LL + ((rr) * 64LL) + ((kk) * 1LL))], __bfloat162float(__float2bfloat16_rn(static_cast<const float*>(args.inputs[2])[0LL + (((((0LL + ((kk) * 16LL) + ((cc) * 1LL)) / 16LL) % 64LL)) * 16LL) + (((((0LL + ((kk) * 16LL) + ((cc) * 1LL)) / 1LL) % 16LL)) * 1LL)])), acc);
                workspace[256LL + (i)] = __bfloat162float(__float2bfloat16_rn(acc));
            }
}
mk_grid_sync(grid);
// sub: sub
{
for (int64_t i = tid; i < 64LL; i += nt) {
            workspace[320LL + (i)] = (((__bfloat162float(__float2bfloat16_rn((workspace[256LL + (0LL + (((((0LL + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL))] + (1.00000000000000000e+00f * __bfloat162float(__float2bfloat16_rn(static_cast<const float*>(args.inputs[3])[0LL + (((((0LL + (((((0LL + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)]))))))) - (1.00000000000000000e+00f * (__bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[5])[0LL + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)])))));
        }
}
mk_grid_sync(grid);
// mean: mean
{
for (int64_t row = blockIdx.x; row < 1LL; row += gridDim.x) {
            float acc = 0.0f;
            for (int64_t r = threadIdx.x; r < 64LL; r += blockDim.x) acc += (mk_square(workspace[320LL + (0LL + (((((0LL + (((((r) / 16LL) % 4LL)) * 16LL) + (((((r) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((r) / 16LL) % 4LL)) * 16LL) + (((((r) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL))]));
            acc = mk_block_sum(acc, shared_reduce);
            if (threadIdx.x == 0) static_cast<float*>(args.outputs[0])[row] = (acc / 64.0f);
            __syncthreads();
        }
}
// _to_copy_6: cast
{
for (int64_t i = tid; i < 64LL; i += nt) {
            workspace[256LL + (i)] = __bfloat162float(__float2bfloat16_rn((((((1.00000000000000000e+00f) / 6.40000000000000000e+01f)) * (((powf(workspace[320LL + (0LL + (((((0LL + (((((0LL + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((0LL + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 16LL) % 4LL)) * 16LL) + (((((0LL + (((((i) / 16LL) % 4LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL))], 1.00000000000000000e+00f)) * 2.00000000000000000e+00f))))));
        }
}
mk_grid_sync(grid);
// sum_1: sum
{
for (int64_t row = tid; row < 16LL; row += nt) {
                float acc = 0.0f;
                for (int64_t r = 0; r < 4LL; ++r) acc += workspace[256LL + (0LL + (((((r) / 1LL) % 4LL)) * 16LL) + (((((row) / 1LL) % 16LL)) * 1LL))];
                workspace[320LL + (row)] = __bfloat162float(__float2bfloat16_rn(acc));
            }
}
// mm_2: mm
{
for (int64_t i = tid; i < 1024LL; i += nt) {
                const int64_t rr = i / 16LL, cc = i % 16LL;
                float acc = 0.0f;
                #pragma unroll 4
                for (int kk = 0; kk < 4; ++kk) acc = fmaf(workspace[0LL + (0LL + ((rr) * 1LL) + ((kk) * 64LL))], workspace[256LL + (0LL + ((kk) * 16LL) + ((cc) * 1LL))], acc);
                {
float tile_root = __bfloat162float(__float2bfloat16_rn(acc));
float tile_value_0 = (tile_root);
static_cast<float*>(args.outputs[3])[i] = tile_value_0;
}
            }
}
// mm_3_epilogue: mm
{
for (int64_t i = tid; i < 256LL; i += nt) {
                const int64_t rr = i / 64LL, cc = i % 64LL;
                float acc = 0.0f;
                #pragma unroll 4
                for (int kk = 0; kk < 16; ++kk) acc = fmaf(workspace[256LL + (0LL + ((rr) * 16LL) + ((kk) * 1LL))], __bfloat162float(__float2bfloat16_rn(static_cast<const float*>(args.inputs[2])[0LL + (((((0LL + ((kk) * 1LL) + ((cc) * 16LL)) / 16LL) % 64LL)) * 16LL) + (((((0LL + ((kk) * 1LL) + ((cc) * 16LL)) / 1LL) % 16LL)) * 1LL)])), acc);
                workspace[384LL + (i)] = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(((workspace[0LL + (0LL + (((((i) / 64LL) % 4LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL))] <= 0.00000000000000000e+00f) ? 0.0f : __bfloat162float(__float2bfloat16_rn(acc)))))));
            }
}
mk_grid_sync(grid);
// sum_2: sum
{
for (int64_t row = tid; row < 64LL; row += nt) {
                float acc = 0.0f;
                for (int64_t r = 0; r < 4LL; ++r) acc += workspace[384LL + (0LL + (((((r) / 1LL) % 4LL)) * 64LL) + (((((row) / 1LL) % 64LL)) * 1LL))];
                workspace[256LL + (row)] = __bfloat162float(__float2bfloat16_rn(acc));
            }
}
// mm_4: mm
{
for (int64_t i = tid; i < 2048LL; i += nt) {
                const int64_t rr = i / 64LL, cc = i % 64LL;
                float acc = 0.0f;
                #pragma unroll 4
                for (int kk = 0; kk < 4; ++kk) acc = fmaf(__bfloat162float(static_cast<const __nv_bfloat16*>(args.inputs[4])[0LL + ((rr) * 1LL) + ((kk) * 32LL)]), workspace[384LL + (0LL + ((kk) * 64LL) + ((cc) * 1LL))], acc);
                {
float tile_root = __bfloat162float(__float2bfloat16_rn(acc));
float tile_value_0 = (tile_root);
static_cast<float*>(args.outputs[1])[i] = tile_value_0;
}
            }
}
// _to_copy_7: cast
{
for (int64_t i = tid; i < 16LL; i += nt) {
            static_cast<float*>(args.outputs[4])[i] = workspace[320LL + (0LL + (((((i) / 1LL) % 16LL)) * 1LL))];
        }
}
mk_grid_sync(grid);
// _to_copy_9: cast
{
for (int64_t i = tid; i < 64LL; i += nt) {
            static_cast<float*>(args.outputs[2])[i] = workspace[256LL + (0LL + (((((i) / 1LL) % 64LL)) * 1LL))];
        }
}
// sub_1: sub
{
for (int64_t i = tid; i < 2048LL; i += nt) {
            static_cast<float*>(args.outputs[5])[i] = (static_cast<const float*>(args.inputs[0])[0LL + (((((i) / 64LL) % 32LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)] - (1.00000000000000000e+00f * ((static_cast<const float*>(args.outputs[1])[0LL + (((((0LL + (((((i) / 64LL) % 32LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)) / 64LL) % 32LL)) * 64LL) + (((((0LL + (((((i) / 64LL) % 32LL)) * 64LL) + (((((i) / 1LL) % 64LL)) * 1LL)) / 1LL) % 64LL)) * 1LL)] * 1.00000000000000002e-03f))));
        }
}
mk_grid_sync(grid);
// sub_2: sub
{
for (int64_t i = tid; i < 64LL; i += nt) {
            static_cast<float*>(args.outputs[6])[i] = (static_cast<const float*>(args.inputs[1])[0LL + (((((i) / 1LL) % 64LL)) * 1LL)] - (1.00000000000000000e+00f * ((static_cast<const float*>(args.outputs[2])[0LL + (((((0LL + (((((i) / 1LL) % 64LL)) * 1LL)) / 1LL) % 64LL)) * 1LL)] * 1.00000000000000002e-03f))));
        }
}
// sub_3: sub
{
for (int64_t i = tid; i < 1024LL; i += nt) {
            static_cast<float*>(args.outputs[7])[i] = (static_cast<const float*>(args.inputs[2])[0LL + (((((i) / 16LL) % 64LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)] - (1.00000000000000000e+00f * ((static_cast<const float*>(args.outputs[3])[0LL + (((((0LL + (((((i) / 16LL) % 64LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 16LL) % 64LL)) * 16LL) + (((((0LL + (((((i) / 16LL) % 64LL)) * 16LL) + (((((i) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)] * 1.00000000000000002e-03f))));
        }
}
// sub_4: sub
{
for (int64_t i = tid; i < 16LL; i += nt) {
            static_cast<float*>(args.outputs[8])[i] = (static_cast<const float*>(args.inputs[3])[0LL + (((((i) / 1LL) % 16LL)) * 1LL)] - (1.00000000000000000e+00f * ((static_cast<const float*>(args.outputs[4])[0LL + (((((0LL + (((((i) / 1LL) % 16LL)) * 1LL)) / 1LL) % 16LL)) * 1LL)] * 1.00000000000000002e-03f))));
        }
}
// Transactional input commit: all input reads have completed.
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
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, megakernel_entry, 512, 0));
    TORCH_CHECK(active > 0, "kernel has no resident blocks");
    int blocks = std::min(prop.multiProcessorCount * std::min(active, 1),
                          1 < 0 ? 4 : std::min(4, 1));
    float* arena = workspace.data_ptr<float>();
    void* kernel_args[] = {&args, &arena};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(megakernel_entry), dim3(blocks), dim3(512), kernel_args, 0, at::cuda::getCurrentCUDAStream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int64_t> resource_usage() {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, megakernel_entry));
    int active = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, megakernel_entry, 512, 0));
    return {attributes.numRegs, static_cast<int64_t>(attributes.localSizeBytes), static_cast<int64_t>(attributes.sharedSizeBytes), active};
}
