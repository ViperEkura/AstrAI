#pragma once
/*
 * Single-row (decode) GEMV: one warp owns one output row.
 *
 * The tile pipeline is built for m >= 16: every CTA tile pads M up to
 * kBlockM (64 or 128), so at decode's m == 1 it issues the padded MMA work
 * anyway and the kernel stays compute-bound on work no output consumes. The
 * decode route keeps the same math and skips the padding: lane l reads the
 * row's 16B weight chunk, the warp folds the partial dot through shuffles,
 * and lane 0 folds the epilogue factors into the one live row.
 *
 * Math contract (identical to epilogue/writer.cuh for row 0, column n):
 *   y[n] = ((sum_k a[k] * b[n, k]) * a_scale(0) * b_scale(n)) + bias[n]
 * a_scale/b_scale are the per-tensor scalars when their row/col count is 0.
 * Bias is the fused bf16 vector, added in fp32 before the single rounding.
 */
#include <cstdint>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <api/native/gemm.h>
#include <datatype/element.cuh>
#include <utils/define.cuh>

namespace astrai {
namespace gemm {

/* int8 weights enter the dot as their exact integer value. */
template <typename ElemB> DEVICE_FORCEINLINE float weight_to_float(ElemB x) {
    if constexpr (std::is_same_v<ElemB, int8_t>)
        return static_cast<float>(x);
    else
        return ElemTrait<ElemB>::to_float(x);
}

/*
 * PER = elements one lane consumes per round: one 16B weight chunk, so a
 * 2-byte weight reads 8 and a 1-byte weight reads 16. The activation side
 * reads the same element count in 8-element (16B) pieces.
 */
template <typename ElemA, typename ElemB, typename OutT, int kWarps>
__global__ void __launch_bounds__(kWarps * 32) gemv_kernel(GemmParams p) {
    static_assert(sizeof(ElemA) == 2, "activation operand must be 16-bit");
    static_assert(sizeof(ElemB) == 1 || sizeof(ElemB) == 2, "weight operand must be 8- or 16-bit");
    constexpr int PER = 16 / (int)sizeof(ElemB);
    constexpr int ROUND = 32 * PER;

    const int warp = blockIdx.x * kWarps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (warp >= p.n)
        return;

    const auto* __restrict__ a = static_cast<const ElemA*>(p.a_ptr);
    const auto* __restrict__ row = static_cast<const ElemB*>(p.b_ptr) + (int64_t)warp * p.b_ld;
    const int k = p.k;
    float acc = 0.0f;

    /*
     * 16B chunks need both bases and every row start aligned; otherwise the
     * scalar walk below still covers the row (short tail shapes).
     */
    const bool vectorized =
        ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(row)) & 15) == 0 &&
        p.b_ld % PER == 0;

    if (vectorized) {
        const int rounds = k / ROUND;
        int i = lane * PER;
        for (int r = 0; r < rounds; ++r, i += ROUND) {
            float av[PER];
#pragma unroll
            for (int c = 0; c < PER / 8; ++c)
                load8(a + i + c * 8, av + c * 8);
            float wv[PER];
            if constexpr (sizeof(ElemB) == 2) {
                load8(row + i, wv);
            } else {
                const uint4 raw = *reinterpret_cast<const uint4*>(row + i);
                const int8_t* bytes = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
                for (int t = 0; t < PER; ++t)
                    wv[t] = static_cast<float>(bytes[t]);
            }
#pragma unroll
            for (int t = 0; t < PER; ++t)
                acc += av[t] * wv[t];
        }
    }

    /* Tail (and the unaligned case): 32 lanes walk the remaining elements. */
    const int done = vectorized ? (k / ROUND) * ROUND : 0;
    for (int t = done + lane; t < k; t += 32)
        acc += ElemTrait<ElemA>::to_float(a[t]) * weight_to_float(row[t]);

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane != 0)
        return;
    /*
     * One live row, so the row factor is a_scale[0] and the per-tensor
     * factors fold in the same order as the tile epilogue.
     */
    float v = acc * (p.a_scale && p.a_scale_m == 0 ? *p.a_scale : 1.0f) *
              (p.b_scale && p.b_scale_n == 0 ? *p.b_scale : 1.0f);
    if (p.a_scale_m > 0)
        v *= p.a_scale[0];
    if (p.b_scale_n > 0)
        v *= p.b_scale[warp];
    if (p.bias_ptr != nullptr)
        v += ElemTrait<bf16>::to_float(static_cast<const bf16*>(p.bias_ptr)[warp]);
    static_cast<OutT*>(p.out_ptr)[warp] = ElemTrait<OutT>::from_float(v);
}

} // namespace gemm
} // namespace astrai
