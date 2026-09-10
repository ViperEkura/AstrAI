#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <type_traits>

#include "common/shape.cuh"

// GEMM-family pure POD/traits header — dtype-neutral: layout tags, element
// traits and the unified parameter POD shared by every element-type
// specialization.

namespace astrai {
namespace gemm {

// Operand storage tags (CUTLASS-style) relative to the canonical matrices
struct RowMajor {};
struct ColMajor {};

// Tile-geometry Shape: the shared static-shape vocabulary imported so the
// Shape<M, N, K> CTA recipes keep their spelling (policy.cuh).
using astrai::Shape;

// Element-type traits: per-dtype storage facts the smem layers price
// rings from (kBytes). The MMA K extent rides MmaShapeFor<MmaT>
// (common/mma.cuh) — it keys on the COMPUTE type, so a dequantized
// operand's storage K (32) is never conflated with the promoted cell's
// (16); dequant insertion factors ride gemm_mma_traits. Adding a dtype =
// one specialization here plus an MmaShapeFor<InT> cell.
template <typename T>
struct gemm_elem_traits;

template <>
struct gemm_elem_traits<__nv_fp8_e4m3> {
    static constexpr int kBytes = 1;
};

template <>
struct gemm_elem_traits<__nv_fp8_e5m2> {
    static constexpr int kBytes = 1;
};

template <>
struct gemm_elem_traits<__nv_bfloat16> {
    static constexpr int kBytes = 2;
};

template <>
struct gemm_elem_traits<int8_t> {
    static constexpr int kBytes = 1;
};

// MMA compute type per operand pair — the tensor-core input type both
// operands are brought to before mma.sync. Promotes to bf16 m16n8k16 when
// int8 rides exactly one side (W8A16, A8W16) or when exactly one side is
// fp8 and the other bf16; symmetric pairs keep their native mma
// (bf16 pass-through, fp8 mma, s8 mma with int32 accumulators).
template <typename ElemA, typename ElemB>
struct gemm_mma_traits {
    static constexpr bool kFp8A =
        std::is_same_v<ElemA, __nv_fp8_e4m3> || std::is_same_v<ElemA, __nv_fp8_e5m2>;
    static constexpr bool kFp8B =
        std::is_same_v<ElemB, __nv_fp8_e4m3> || std::is_same_v<ElemB, __nv_fp8_e5m2>;
    static constexpr bool kI8A = std::is_same_v<ElemA, int8_t>;
    static constexpr bool kI8B = std::is_same_v<ElemB, int8_t>;
    static constexpr bool kPromote =
        (kI8A != kI8B) ||
        ((kFp8A != kFp8B) && (std::is_same_v<ElemA, __nv_bfloat16> ||
                              std::is_same_v<ElemB, __nv_bfloat16>));
    using MmaT = std::conditional_t<kPromote, __nv_bfloat16, ElemA>;
    static constexpr bool kDequantA = !std::is_same_v<ElemA, MmaT>;
    static constexpr bool kDequantB = !std::is_same_v<ElemB, MmaT>;
};

// Unified GEMM parameter POD: one struct flows through the kernels; each
// kernel touches only the fields it needs.
struct GemmParams {
    // Operands + output; bias fuses into the epilogue (fp32 add before the
    // single bf16 rounding); null disables.
    const void* __restrict__ a_ptr = nullptr;
    const void* __restrict__ b_ptr = nullptr;
    const void* __restrict__ bias_ptr = nullptr;
    void* __restrict__ out_ptr = nullptr;

    // Per-operand dequant scales, folded multiplicatively into the epilogue:
    //   a_scale — length a_scale_m: 0 = device scalar over the whole output,
    //             m = per-row [m];
    //   b_scale — length b_scale_n: 0 = device scalar, n = per-output-
    //             channel [n] (weight-only quantization);
    // null/0 disable each side. Grouped-along-K scales cannot fold here.
    const float* __restrict__ a_scale = nullptr;
    const float* __restrict__ b_scale = nullptr;
    int a_scale_m = 0;
    int b_scale_n = 0;

    // Batched (bmm) geometry: grid.z steps these strides (0 broadcasts the
    // operand). Batch strides are extent products and can cross int32.
    int batch = 1;
    int m, n, k;
    // Row strides in elements; out_ld lets non-contiguous outputs cost
    // nothing. The output orientation rides the policy's LayoutOut tag.
    int a_ld, b_ld, out_ld;
    int64_t a_batch_stride = 0;
    int64_t b_batch_stride = 0;
    int64_t out_batch_stride = 0;

    // Raster order (runtime knob, picked from the problem's aspect):
    //   >0 — grouped raster: group of `raster` M-tile rows, M walked fastest
    //        (consecutive CTAs share one B column stripe);
    //   <0 — mirrored in N (consecutive CTAs share one A row stripe);
    //    0 — plain N-fastest raster.
    int raster = 8;
};

}  // namespace gemm
}  // namespace astrai
