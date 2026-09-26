#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <type_traits>

#include <utils/dtype.cuh>
#include <utils/shape.cuh>

// GEMM-family POD/traits header — dtype-neutral: layout tags, element traits
// and the unified parameter POD shared by every element-type specialization.
// The element types themselves (and their storage facts) come from the shared
// vocabulary in common/dtype.cuh, so a precision is named once for the whole
// kernel tree. Torch-free: these headers are what a host or device pass sees
// before any binding does.

namespace astrai {
namespace gemm {

// Operand storage tags (CUTLASS-style) relative to the canonical matrices
struct RowMajor {};
struct ColMajor {};

// Tile-geometry Shape: the shared static-shape vocabulary imported so the
// Shape<M, N, K> CTA recipes keep their spelling (policy.cuh).
using astrai::Shape;

// Element-type traits: the family-local spelling of the shared vocabulary's
// ElemTrait (common/dtype.cuh), which carries the per-dtype storage facts the
// smem layers price rings from (kBytes). The MMA K extent rides
// MmaShapeFor<MmaT> (common/mma.cuh) — it keys on the COMPUTE type, so a
// dequantized operand's storage K (32) is never conflated with the promoted
// cell's (16); dequant insertion factors ride gemm_mma_traits. An element type
// the vocabulary does not know is a compile error at the use site, never a
// silent fallback.
template <typename T>
using gemm_elem_traits = astrai::ElemTrait<T>;

// MMA compute type per operand pair — the tensor-core input type both
// operands are brought to before mma.sync. Promotes to bf16 m16n8k16 when
// int8 rides exactly one side (W8A16, A8W16, the only supported mixed
// pair); symmetric pairs keep their native mma (bf16 pass-through, fp8 mma,
// s8 mma with int32 accumulators).
template <typename ElemA, typename ElemB>
struct gemm_mma_traits {
    static constexpr bool kI8A = std::is_same_v<ElemA, int8_t>;
    static constexpr bool kI8B = std::is_same_v<ElemB, int8_t>;
    static constexpr bool kPromote = (kI8A != kI8B);
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
