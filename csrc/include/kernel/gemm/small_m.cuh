#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace astrai {
namespace gemm {

// One CTA reads an entire weight row and reuses it for up to four output rows.
template <int Rows>
__global__ void small_m_dot(const __nv_bfloat16* a,
                            const __nv_bfloat16* weight,
                            __nv_bfloat16* out,
                            int m,
                            int n,
                            int k) {
    const int col = blockIdx.x;
    const int row = blockIdx.y * Rows;
    const int tid = threadIdx.x;
    float acc[Rows]{};
    __shared__ float partial[Rows][4];
    for (int offset = tid * 8; offset < k; offset += 128 * 8) {
        const uint4 w = *reinterpret_cast<const uint4*>(weight + int64_t(col) * k + offset);
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (row + r < m) {
                const uint4 x = *reinterpret_cast<const uint4*>(a + (row + r) * k + offset);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[r] = fmaf(float(reinterpret_cast<const __nv_bfloat16*>(&x)[j]),
                                  float(reinterpret_cast<const __nv_bfloat16*>(&w)[j]), acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        for (int delta = 16; delta; delta >>= 1)
            acc[r] += __shfl_down_sync(0xffffffff, acc[r], delta);
        if ((tid & 31) == 0)
            partial[r][tid / 32] = acc[r];
    }
    __syncthreads();
    if (tid < Rows && row + tid < m)
        out[(row + tid) * n + col] = __float2bfloat16((partial[tid][0] + partial[tid][1]) +
                                                      (partial[tid][2] + partial[tid][3]));
}

} // namespace gemm
} // namespace astrai
