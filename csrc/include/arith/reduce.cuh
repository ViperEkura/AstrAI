// Shared warp/block reduction + atomic helpers — pure CUDA, no torch.
//
// Extracted from the attention and fp8 families so both share one
// implementation: warp_reduce_sum (decode scalar kernel), warp_reduce_max +
// atomic_max_float (fp8 quantize amax), group_reduce_sum<G> (prefill scalar
// kernel).

#pragma once

namespace astrai {

// Full-warp butterfly sum reduction (32 lanes).
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Full-warp butterfly max reduction (32 lanes).
__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

// Sub-warp group reduction over G consecutive lanes (G a power of two).
// `mask` is the full participating-lane mask of the group (see the
// prefill scalar kernel's gmask computation).
template <int G>
__device__ __forceinline__ float group_reduce_sum(float v, unsigned mask) {
#pragma unroll
    for (int o = G / 2; o > 0; o >>= 1)
        v += __shfl_xor_sync(mask, v, o);
    return v;
}

// Unsigned-bit-pattern atomicMax for non-negative floats; a null
// destination disables the update (kernels with optional amax slots).
__device__ __forceinline__ void atomic_max_float(float* destination,
                                                 float value) {
    if (destination)
        atomicMax(reinterpret_cast<unsigned*>(destination),
                  __float_as_uint(value));
}

}  // namespace astrai
