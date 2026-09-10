#pragma once
// In-register dequantization functors: the mma.sync consumes MmaT fragments
// while quantized operands stage in their storage type, so the GEMM
// fragment load folds an exact expansion between the LDS and the mma —
// never a separate F2F round-trip pass.
//
// int8 -> bf16 is exact for the full [-128, 127] range, four LOP3-class
// instructions per element pair. The magnitude bits (0-6) and the sign bit
// (7) get separate LOP3s, because ORing the whole byte into a bf16 base
// breaks linearity (bit 7 spills into the exponent: bf16 carries only 7
// mantissa bits). Per byte u:
//   h = (u & 0x7F) | 0x4300          -> bf16 128 + u7   (exact: u7 <= 127)
//   s = (u & 0x80) | 0x4300          -> bf16 128 or 256 (the sign picks)
//   v = h - s = (128 + u7) - (128 + 128*b7) = u - 128*b7 = the int8 value
// Every intermediate lands on bf16-exact values (|v| <= 128), so no
// rounding occurs anywhere.
//
// fp8 -> bf16 (W-F8A16 weight-only) instead rides the hardware widen
// (cvt.rn.f16x2.e4m3x2 on sm_89+): an LOP3 exponent-rebias trick like
// int8's mis-maps e4m3 subnormals (exp field 0) and maps NaN to a finite
// value, while the hardware path is exact for normals, subnormals, +-0
// and NaN. The f16 -> bf16 step is an fp32 round-trip rounding to nearest
// — exact, because every fp8 value carries at most 4 significant bits and
// bf16 holds 8.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <type_traits>

namespace astrai {
namespace quant {

// (a & mask) | base as one SASS LOP3 (truth table 0xEA).
__device__ __forceinline__ unsigned lop3_and_or(unsigned a, unsigned mask,
                                                unsigned base) {
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;"
        : "=r"(r)
        : "r"(a), "r"(mask), "r"(base));
    return r;
}

// __hsub2 over bit-cast words (bf16x2 lanes).
__device__ __forceinline__ unsigned hsub2_words(unsigned a, unsigned b) {
    const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162 y = *reinterpret_cast<const __nv_bfloat162*>(&b);
    const __nv_bfloat162 d = __hsub2(x, y);
    return *reinterpret_cast<const unsigned*>(&d);
}

// Per-pair dequantization policy keyed on (storage, mma) element types.
// Unsupported pairs stay undefined — instantiating one is a compile error,
// never a silent fallback.
template <typename SrcT, typename MmaT>
struct DequantPair;

template <>
struct DequantPair<int8_t, __nv_bfloat16> {
    static constexpr unsigned kBase = 0x43004300u;  // bf16 128.0 per lane
    static constexpr unsigned kMagMask = 0x007f007fu;   // magnitude bits
    static constexpr unsigned kSignMask = 0x00800080u;  // sign bit

    // Two expand steps: magnitude word (128+u7) and sign word (128/256).
    static __device__ __forceinline__ unsigned expand(unsigned lanes,
                                                      unsigned sign) {
        return hsub2_words(lop3_and_or(lanes, kMagMask, kBase),
                           lop3_and_or(sign, kSignMask, kBase));
    }

    // One k-adjacent byte pair (u16: elements e0, e1) -> one bf16x2.
    // The PRMT spreads the pair to the two 16-bit lanes (bytes {0,2});
    // both LOP3s and the HSUB2 then run on the pair at once:
    // 4 SASS instructions. (A future humming-style offline byte interleave
    // could fold the spread into storage and drop the PRMT; the layout
    // derivation lives in docs/developer/cuda_kernels.md.)
    static __device__ __forceinline__ unsigned pair(unsigned short v) {
        const unsigned spread =
            __byte_perm((unsigned)v, 0, 0x4140u);  // (e0, 0, e1, 0)
        return expand(spread, spread);
    }
};

// fp8 widen via the hardware converter (exact incl. subnormals, +-0, NaN;
// see the header comment). One cvt + one f32 round-trip per pair; the two
// formats share the body — only the converter's interpretation constant
// differs. The per-format specializations below stay explicit, so an
// unsupported (SrcT, MmaT) pair still fails to compile.
template <__nv_fp8_interpretation_t Fmt>
struct Fp8WidenPair {
    static __device__ __forceinline__ unsigned pair(unsigned short v) {
        const __half2_raw h2 =
            __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, Fmt);
        const float2 f =
            __half22float2(*reinterpret_cast<const __half2*>(&h2));
        const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
        return *reinterpret_cast<const unsigned*>(&b);
    }
};

template <>
struct DequantPair<__nv_fp8_e4m3, __nv_bfloat16> : Fp8WidenPair<__NV_E4M3> {};
template <>
struct DequantPair<__nv_fp8_e5m2, __nv_bfloat16> : Fp8WidenPair<__NV_E5M2> {};

}  // namespace quant
}  // namespace astrai
