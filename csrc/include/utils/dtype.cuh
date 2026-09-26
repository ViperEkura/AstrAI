#pragma once
// Element types and their operations — the vocabulary every kernel family
// shares (attention, gemm, quantize, rotary). Torch-free and CUDA-only: the
// runtime spelling of "which precision" is torch's own `at::ScalarType` at
// every torch-facing surface, so this header owns no tag enum — it owns what
// torch does not have: the device-side element type and its operations.
//
//   ElemTrait<T>      element type -> {storage bytes, float conversions}
//
// The primary template is UNDEFINED and specializes per element type: an
// element type the vocabulary does not know is a compile error at the use
// site, never a silent fallback (the same discipline as MmaShapeFor in
// common/mma.cuh and gemm's storage traits, which alias this one).
//
// The pair-cell operations exist only for the 16-bit types — one 32-bit
// register holding two elements, the tensor-core operand and packed-store
// layout. 1-byte and 32-bit types pack differently and hit the helpers'
// static_assert instead of quietly producing the wrong layout.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <utils/define.cuh>

namespace astrai {

using bf16 = __nv_bfloat16;
using fp16 = __half;
using fp8_e4m3 = __nv_fp8_e4m3;
using fp8_e5m2 = __nv_fp8_e5m2;

// Element type -> storage facts and conversions.
template <typename T>
struct ElemTrait;

template <>
struct ElemTrait<bf16> {
    static constexpr int kBytes = 2;
    static constexpr int kPerCell = 2;  // elements per 32-bit cell

          static DEVICE_FORCEINLINE float to_float(bf16 x) {
        return __bfloat162float(x);
    }
          static DEVICE_FORCEINLINE bf16 from_float(float x) {
        return __float2bfloat16(x);
    }
    // Two elements into one 32-bit cell, element 0 in the low half — the mma
    // A/B operand order and the packed-store layout.
          static DEVICE_FORCEINLINE unsigned pack2(float lo, float hi) {
        __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<unsigned*>(&v);
    }
          static DEVICE_FORCEINLINE float2 unpack2(unsigned cell) {
        __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&cell);
        return __bfloat1622float2(v);
    }
};

template <>
struct ElemTrait<fp16> {
    static constexpr int kBytes = 2;
    static constexpr int kPerCell = 2;

          static DEVICE_FORCEINLINE float to_float(fp16 x) {
        return __half2float(x);
    }
          static DEVICE_FORCEINLINE fp16 from_float(float x) {
        return __float2half(x);
    }
          static DEVICE_FORCEINLINE unsigned pack2(float lo, float hi) {
        __half2 v = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<unsigned*>(&v);
    }
          static DEVICE_FORCEINLINE float2 unpack2(unsigned cell) {
        __half2 v = *reinterpret_cast<__half2*>(&cell);
        return __half22float2(v);
    }
};

// Storage facts only: the scaled fp8 conversions belong to quantize (which
// owns the scales and the delayed-scaling ring), gemm prices smem from kBytes.
template <> struct ElemTrait<fp8_e4m3> { static constexpr int kBytes = 1; };
template <> struct ElemTrait<fp8_e5m2> { static constexpr int kBytes = 1; };
template <> struct ElemTrait<int8_t> { static constexpr int kBytes = 1; };

template <>
struct ElemTrait<float> {
    static constexpr int kBytes = 4;

          static DEVICE_FORCEINLINE float to_float(float x) { return x; }
          static DEVICE_FORCEINLINE float from_float(float x) { return x; }
};

// ---------------------------------------------------------------------------
// Bulk helpers over 16-bit elements.
// ---------------------------------------------------------------------------

// Eight elements (one 16-byte chunk) as floats. src must be 16-byte aligned —
// every 16-bit row of a contiguous tile is.
template <typename T>
DEVICE_FORCEINLINE void load8(const T* src, float* out) {
    static_assert(ElemTrait<T>::kBytes == 2,
                  "load8 is a 16-bit-element helper (8 elements = 16 bytes)");
    uint4 raw = *reinterpret_cast<const uint4*>(src);
    float2 a = ElemTrait<T>::unpack2(raw.x), b = ElemTrait<T>::unpack2(raw.y);
    float2 c = ElemTrait<T>::unpack2(raw.z), d = ElemTrait<T>::unpack2(raw.w);
    out[0] = a.x; out[1] = a.y;
    out[2] = b.x; out[3] = b.y;
    out[4] = c.x; out[5] = c.y;
    out[6] = d.x; out[7] = d.y;
}

// Two elements as one packed 32-bit cell at dst (4-byte aligned, as every
// element pair at an even index of a contiguous row is).
template <typename T>
DEVICE_FORCEINLINE void store2(T* dst, float e0, float e1) {
    static_assert(ElemTrait<T>::kBytes == 2,
                  "store2 is a 16-bit-element helper (one cell = two elements)");
    *reinterpret_cast<unsigned*>(dst) = ElemTrait<T>::pack2(e0, e1);
}

}  // namespace astrai
