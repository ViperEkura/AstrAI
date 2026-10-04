#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>

namespace astrai::muon_ns::syrk {

// A 64x64 triangular CTA, with four warps computing one 32x32 quadrant
// each. Inspired by the tile dispatch in StarrickLiu/fused-muon (Apache-2.0),
// but implemented using CUDA WMMA and shared memory without CuTe/CUTLASS.
__global__ void syrk64_kernel(const __nv_bfloat16* input,
                              __nv_bfloat16* output,
                              int dim,
                              int reduction,
                              int64_t stride0,
                              int64_t stride1) {
    using namespace nvcuda;
    constexpr int kTile = 64;
    constexpr int kMma = 16;
    constexpr int kTileElements = kTile * kTile;
    constexpr int kPanelStride = kTile + 8;
    constexpr int kPanelElements = kTile * kPanelStride;
    constexpr int kOutputStride = kTile + 8;

    int tile_row = static_cast<int>((sqrtf(8.0f * blockIdx.x + 1.0f) - 1.0f) * 0.5f);
    while (tile_row * (tile_row + 1) / 2 > blockIdx.x) {
        --tile_row;
    }
    while ((tile_row + 1) * (tile_row + 2) / 2 <= blockIdx.x) {
        ++tile_row;
    }
    const int tile_col = blockIdx.x - tile_row * (tile_row + 1) / 2;
    const int row = tile_row * kTile;
    const int col = tile_col * kTile;

    __shared__ __align__(16) __nv_bfloat16 panel_a[kPanelElements];
    __shared__ __align__(16) __nv_bfloat16 panel_b[kPanelElements];
    __shared__ float accumulator[kTileElements];
    __shared__ __nv_bfloat16 rounded[kTile * kOutputStride];

    const int warp = threadIdx.x / warpSize;
    const int warp_row = (warp / 2) * 32;
    const int warp_col = (warp % 2) * 32;
    wmma::fragment<wmma::matrix_a, kMma, kMma, kMma, __nv_bfloat16, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, kMma, kMma, kMma, __nv_bfloat16, wmma::col_major> b0, b1;
    wmma::fragment<wmma::accumulator, kMma, kMma, kMma, float> p00, p01, p10, p11;
    wmma::fill_fragment(p00, 0.0f);
    wmma::fill_fragment(p01, 0.0f);
    wmma::fill_fragment(p10, 0.0f);
    wmma::fill_fragment(p11, 0.0f);

    for (int k_base = 0; k_base < reduction; k_base += kTile) {
        // Each cp.async transfers eight BF16 values along the contiguous K axis.
        constexpr int kSegments = kTile * (kTile / 8);
        for (int segment = threadIdx.x; segment < kSegments; segment += blockDim.x) {
            const int outer = segment / (kTile / 8);
            const int inner = segment % (kTile / 8) * 8;
            const int panel_index = outer * kPanelStride + inner;
            const auto* source_a = input + (row + outer) * stride0 + (k_base + inner) * stride1;
            const auto* source_b = input + (col + outer) * stride0 + (k_base + inner) * stride1;
            const uint32_t target_a = __cvta_generic_to_shared(panel_a + panel_index);
            const uint32_t target_b = __cvta_generic_to_shared(panel_b + panel_index);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(target_a),
                         "l"(source_a));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(target_b),
                         "l"(source_b));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        for (int k = 0; k < kTile; k += kMma) {
            const int a_offset = warp_row * kPanelStride + k;
            const int b_offset = warp_col * kPanelStride + k;
            wmma::load_matrix_sync(a0, panel_a + a_offset, kPanelStride);
            wmma::load_matrix_sync(a1, panel_a + a_offset + kMma * kPanelStride, kPanelStride);
            wmma::load_matrix_sync(b0, panel_b + b_offset, kPanelStride);
            wmma::load_matrix_sync(b1, panel_b + b_offset + kMma * kPanelStride, kPanelStride);
            wmma::mma_sync(p00, a0, b0, p00);
            wmma::mma_sync(p01, a0, b1, p01);
            wmma::mma_sync(p10, a1, b0, p10);
            wmma::mma_sync(p11, a1, b1, p11);
        }
        __syncthreads();
    }

    float* warp_output = accumulator + warp_row * kTile + warp_col;
    wmma::store_matrix_sync(warp_output, p00, kTile, wmma::mem_row_major);
    wmma::store_matrix_sync(warp_output + kMma, p01, kTile, wmma::mem_row_major);
    wmma::store_matrix_sync(warp_output + kMma * kTile, p10, kTile, wmma::mem_row_major);
    wmma::store_matrix_sync(warp_output + kMma * kTile + kMma, p11, kTile, wmma::mem_row_major);
    __syncthreads();

    for (int index = threadIdx.x; index < kTileElements; index += blockDim.x) {
        const int local_row = index / kTile;
        const int local_col = index % kTile;
        if (tile_row != tile_col || local_row >= local_col) {
            const __nv_bfloat16 result = __float2bfloat16_rn(accumulator[index]);
            rounded[local_row * kOutputStride + local_col] = result;
            if (tile_row == tile_col && local_row != local_col) {
                rounded[local_col * kOutputStride + local_row] = result;
            }
        }
    }
    __syncthreads();

    for (int index = threadIdx.x; index < kTileElements; index += blockDim.x) {
        const int local_row = index / kTile;
        const int local_col = index % kTile;
        output[(row + local_row) * dim + col + local_col] =
            rounded[local_row * kOutputStride + local_col];
        if (tile_row != tile_col) {
            output[(col + local_row) * dim + row + local_col] =
                rounded[local_col * kOutputStride + local_row];
        }
    }
}

inline void launch(const __nv_bfloat16* input,
                   __nv_bfloat16* output,
                   int dim,
                   int reduction,
                   int64_t stride0,
                   int64_t stride1,
                   cudaStream_t stream) {
    const int tiles = dim / 64;
    const dim3 grid(tiles * (tiles + 1) / 2);
    constexpr int kThreads = 128;
    syrk64_kernel<<<grid, kThreads, 0, stream>>>(input, output, dim, reduction, stride0, stride1);
}
} // namespace astrai::muon_ns::syrk
