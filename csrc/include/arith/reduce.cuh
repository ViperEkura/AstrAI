// Shared warp/block reduction + atomic helpers — pure CUDA, no torch.
//
// Extracted from the attention and fp8 families so both share one
// implementation: warp_reduce (decode scalar kernel), group_reduce<G>
// (prefill scalar kernel), warp_reduce over maximum + atomic_max_float (fp8
// quantize amax). The combining op rides on a template parameter in the
// cutlass/functional.h functor shape (plus/maximum below), so one butterfly
// body serves every op.

#pragma once

#include <utils/define.cuh>

namespace astrai {

// Reduction combinators, one stateless functor per op (the
// cutlass/functional.h shape): compound assignment for plus, a ternary for
// maximum with an fmaxf specialization for float. maximum is the
// NaN-suppressing IEEE max (fmaxf drops a NaN operand); no max.NaN.f32
// variant because no caller wants NaN propagation.
template <typename T>
struct plus {
    DEVICE_FORCEINLINE T operator()(T lhs, const T& rhs) const {
        lhs += rhs;
        return lhs;
    }
};

template <typename T>
struct maximum {
    DEVICE_FORCEINLINE T operator()(const T& lhs, const T& rhs) const {
        return lhs < rhs ? rhs : lhs;
    }
};

template <>
struct maximum<float> {
    DEVICE_FORCEINLINE float operator()(const float& lhs, const float& rhs) const {
        return fmaxf(lhs, rhs);
    }
};

// Full-warp butterfly reduction over a binary op (32 lanes). The op rides
// on the template parameter list with a default, so the sum call sites read
// plain warp_reduce(v) and the max ones warp_reduce<maximum<float>>(v).
template <typename Op = plus<float>>
DEVICE_FORCEINLINE float warp_reduce(float v) {
    Op op;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = op(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
    return v;
}

// Sub-warp group reduction over G consecutive lanes (G a power of two).
// `mask` is the full participating-lane mask of the group (see the
// prefill scalar kernel's gmask computation).
template <int G, typename Op = plus<float>>
DEVICE_FORCEINLINE float group_reduce(float v, unsigned mask) {
    Op op;
#pragma unroll
    for (int o = G / 2; o > 0; o >>= 1)
        v = op(v, __shfl_xor_sync(mask, v, o));
    return v;
}

// Unsigned-bit-pattern atomicMax for non-negative floats; a null
// destination disables the update (kernels with optional amax slots).
DEVICE_FORCEINLINE void atomic_max_float(float* destination, float value) {
    if (destination)
        atomicMax(reinterpret_cast<unsigned*>(destination),
                  __float_as_uint(value));
}

}  // namespace astrai
