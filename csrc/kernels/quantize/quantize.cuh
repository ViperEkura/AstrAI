#pragma once
// FP8 quantize device code — pure CUDA, no torch: kernels take the
// QuantParams POD while the fp8 element type, input type and Dual
// orientation ride on template parameters, and the launcher is shared by
// the torch binding and the C tests. Per-dtype facts live in one traits
// specialization each (primary templates undefined — an unsupported
// dtype is a compile error, never a silent fallback), keyed on the fp8
// element type; the bindings name the raw __nv_* types from the output
// dtype directly — no format enum.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "common.h"
#include "common/launch.cuh"
#include "common/reduce.cuh"

namespace astrai {
namespace quant {
// Input element traits: the specialization supplies the scalar widen and,
// for 2-lane 16-bit inputs, names the native pair type; the shared base in
// detail below builds the 16B vector unpack and the pair load on top —
// adding a dtype is one thin specialization.
template <typename InT>
struct quant_in_traits;

namespace detail {

// Native 2-lane widen: the single per-dtype intrinsic fact.
__device__ __forceinline__ float2 widen(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}
__device__ __forceinline__ float2 widen(__half2 v) {
    return __half22float2(v);
}

// 2-lane 16-bit input body: 8 elements per 16B load, one native pair load
// per row. Everything but the widen above is dtype-independent.
template <typename InT, typename PairT>
struct pair_in_traits {
    static constexpr int kVecElems = 8;
    static __device__ __forceinline__ void load_vec(const uint4& raw,
                                                    float* f) {
        const PairT* p2 = reinterpret_cast<const PairT*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 p = widen(p2[j]);
            f[2 * j] = p.x;
            f[2 * j + 1] = p.y;
        }
    }
    static __device__ __forceinline__ void load_pair(const InT* p,
                                                     float* f) {
        const float2 v = widen(*reinterpret_cast<const PairT*>(p));
        f[0] = v.x;
        f[1] = v.y;
    }
};

}  // namespace detail

template <>
struct quant_in_traits<__nv_bfloat16>
    : detail::pair_in_traits<__nv_bfloat16, __nv_bfloat162> {
    static __device__ __forceinline__ float to_float(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
};

template <>
struct quant_in_traits<__half> : detail::pair_in_traits<__half, __half2> {
    static __device__ __forceinline__ float to_float(__half v) {
        return __half2float(v);
    }
};

template <>
struct quant_in_traits<float> {
    static constexpr int kVecElems = 4;
    static __device__ __forceinline__ float to_float(float v) { return v; }
    static __device__ __forceinline__ void load_vec(const uint4& raw,
                                                    float* f) {
        const float* w = reinterpret_cast<const float*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) f[j] = w[j];
    }
    static __device__ __forceinline__ void load_pair(const float* p,
                                                     float* f) {
        f[0] = p[0];
        f[1] = p[1];
    }
};

// FP8 convert traits, keyed on the fp8 element type (one specialization
// per format): round-nearest-even + satfinite, one
// float -> one byte and one pair -> one packed fp8x2 word. Primary
// template undefined; one specialization per format.
namespace detail {

// Shared pair body — only the converter's interpretation constant differs
// per format (mirrors dequant.cuh's Fp8WidenPair).
template <__nv_fp8_interpretation_t Fmt>
struct Fp8PackPair {
    static __device__ __forceinline__ unsigned pack(float a, float b) {
        return static_cast<unsigned>(__nv_cvt_float2_to_fp8x2(
            make_float2(a, b), __NV_SATFINITE, Fmt));
    }
};

}  // namespace detail

template <typename Fp8T>
struct fp8_cvt_traits;

template <>
struct fp8_cvt_traits<__nv_fp8_e4m3> : detail::Fp8PackPair<__NV_E4M3> {
    static __device__ __forceinline__ uint8_t cvt(float v) {
        return __nv_fp8_e4m3(v).__x;
    }
};

template <>
struct fp8_cvt_traits<__nv_fp8_e5m2> : detail::Fp8PackPair<__NV_E5M2> {
    static __device__ __forceinline__ uint8_t cvt(float v) {
        return __nv_fp8_e5m2(v).__x;
    }
};

// Block-wide amax reduce -> one atomic per block: warp-reduce, park one
// value per warp, thread 0 folds. kWarps must cover the block's warp count.
// With p.fold_ring, the last-finishing block additionally folds the final
// amax into the history window and publishes the next scale (atomicAdd
// ticket + fences), re-zeroing the amax slot and the counter for the next
// launch — the host-side delayed-scaling update chain disappears.
template <int kWarps>
__device__ __forceinline__ void publish_amax(const QuantParams& p,
                                             float v) {
    v = warp_reduce_max(v);
    __shared__ float slots[kWarps];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if ((tid & 31) == 0) slots[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
#pragma unroll
        for (int w = 1; w < kWarps; ++w) v = fmaxf(v, slots[w]);
        atomic_max_float(p.amax, v);
        if (!p.fold_ring) return;
        __threadfence();
        const unsigned int ticket = atomicAdd(p.done, 1u);
        __threadfence();
        // Total blocks over BOTH grid dimensions (the tiled kernel launches
        // a 2D grid; gridDim.x alone made the fold fire every gridDim.x-th
        // completion — recording round-local partial amaxes and re-folding
        // on tall tensors, silently clipping the published scale).
        const unsigned int total = gridDim.x * gridDim.y;
        if (ticket != total - 1u) return;
        p.hist[p.hist_idx] = *p.amax;
        float peak = p.hist[0];
        for (int i = 1; i < p.hist_len; ++i) peak = fmaxf(peak, p.hist[i]);
        *p.scale_out = fmaxf(peak / p.fp8_max / p.pow2_margin, 1e-12f);
        *p.amax = 0.0f;
        *p.done = 0u;
    }
}

// Elementwise quantize kernel (QuantLayout::RowMajor): vectorized 16B loads
// -> fp8 stores, fused amax over raw values.
template <typename Fp8T, typename InT>
__global__ void fp8_quantize_kernel(QuantParams p) {
    const float mult = *p.scale;
    const auto* x = static_cast<const InT*>(p.input_ptr);
    uint8_t* x8 = static_cast<uint8_t*>(p.output_ptr);
    float local_amax = 0.0f;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // One 16B load -> kVecElems bytes per step. Torch allocations are >=16B
    // aligned, so element 0 keeps the uint4 access natural; a misaligned
    // base (odd storage offset view) falls to the scalar tail via
    // total_vec = 0.
    constexpr int kVecElems = quant_in_traits<InT>::kVecElems;
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(x) |
          reinterpret_cast<uintptr_t>(x8)) &
         15) == 0;
    const int64_t total_vec = aligned ? p.total / kVecElems : 0;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec;
         i += stride) {
        float f[kVecElems];
        quant_in_traits<InT>::load_vec(xv[i], f);
        // One 32-bit word packs two fp8x2 pairs (4 elements).
        unsigned packed[kVecElems / 4];
#pragma unroll
        for (int j = 0; j < kVecElems / 4; ++j) {
            local_amax = fmaxf(
                local_amax,
                fmaxf(fmaxf(fabsf(f[4 * j]), fabsf(f[4 * j + 1])),
                      fmaxf(fabsf(f[4 * j + 2]), fabsf(f[4 * j + 3]))));
            const unsigned lo = fp8_cvt_traits<Fp8T>::pack(
                f[4 * j] * mult, f[4 * j + 1] * mult);
            const unsigned hi = fp8_cvt_traits<Fp8T>::pack(
                f[4 * j + 2] * mult, f[4 * j + 3] * mult);
            packed[j] = (lo & 0xffffu) | (hi << 16);
        }
        if constexpr (kVecElems == 8)
            reinterpret_cast<uint2*>(x8)[i] = make_uint2(packed[0], packed[1]);
        else
            reinterpret_cast<unsigned*>(x8)[i] = packed[0];
    }
    // Scalar tail (and full fallback for misaligned bases).
    for (int64_t i = total_vec * kVecElems + blockIdx.x * blockDim.x +
                      threadIdx.x;
         i < p.total; i += stride) {
        const float v = quant_in_traits<InT>::to_float(x[i]);
        local_amax = fmaxf(local_amax, fabsf(v));
        x8[i] = fp8_cvt_traits<Fp8T>::cvt(v * mult);
    }
    if (p.amax) publish_amax<8>(p, local_amax);
}

// Tiled transpose quantize (QuantLayout::Transposed/Dual): reads the
// [rows][cols] input once and writes the fp8 bytes transposed
// ([cols][rows], so the contract dim lands K-contiguous for NT GEMM
// operands) and, when Dual, the row-major copy too (compiled out
// otherwise). 64x32 tiles, one native pair load per row (a full 128B warp
// read); rows whose pair is unaligned or ragged (odd widths, misaligned
// bases) fall back to element loads in place. Staging goes through a byte
// tile whose pitch keeps the store stride coprime with the 32 banks.
// (+25-35% over the former 32x32 scalar kernel on sub-4M tensors; ~5%
// slower once DRAM-saturated — accepted for the single-kernel shape.)
template <typename Fp8T, typename InT, bool Dual>
__global__ void fp8_quantize_tiled_kernel(QuantParams p) {
    constexpr int kTileC = 64, kTileR = 32;
    // 34B pitch: staging stride is 17 words (coprime with the 32 banks) so
    // the pair-byte stores stay conflict-free, and the byte-wise consume
    // reads still span distinct words.
    __shared__ uint8_t tile[kTileC][kTileR + 2];
    const float mult = *p.scale;
    const auto* x = static_cast<const InT*>(p.input_ptr);
    const int r0 = blockIdx.y * kTileR;
    const int c0 = blockIdx.x * kTileC;
    const int r = r0 + threadIdx.y * 4;
    const int c = c0 + threadIdx.x * 2;  // cols even => the pair is in-bounds

    uint8_t q[4][2];
    float local_amax = 0.0f;
    // Vectorize the pair when both elements are in-bounds and the native
    // 2-element load is aligned; odd widths, misaligned bases and ragged
    // edges fall back to element loads row by row.
    constexpr int kPairAlign = 2 * (int)sizeof(InT);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        q[j][0] = 0;
        q[j][1] = 0;
        if (r + j < p.rows && c < p.cols) {
            const InT* a = x + (int64_t)(r + j) * p.cols + c;
            if (c + 1 < p.cols &&
                (reinterpret_cast<uintptr_t>(a) & (kPairAlign - 1)) == 0) {
                float f[2];
                quant_in_traits<InT>::load_pair(a, f);
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    local_amax = fmaxf(local_amax, fabsf(f[k]));
                    q[j][k] = fp8_cvt_traits<Fp8T>::cvt(f[k] * mult);
                }
            } else {
                const float v0 = quant_in_traits<InT>::to_float(a[0]);
                local_amax = fmaxf(local_amax, fabsf(v0));
                q[j][0] = fp8_cvt_traits<Fp8T>::cvt(v0 * mult);
                if (c + 1 < p.cols) {
                    const float v1 = quant_in_traits<InT>::to_float(a[1]);
                    local_amax = fmaxf(local_amax, fabsf(v1));
                    q[j][1] = fp8_cvt_traits<Fp8T>::cvt(v1 * mult);
                }
            }
        }
    }
    if (Dual) {
        uint8_t* out = static_cast<uint8_t*>(p.output_ptr);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (r + j < p.rows && c < p.cols) {
                uint8_t* o = out + (int64_t)(r + j) * p.cols + c;
                const int64_t off = (int64_t)(r + j) * p.cols + c;
                if (c + 1 < p.cols && (off & 1) == 0)
                    *reinterpret_cast<unsigned short*>(o) =
                        (unsigned short)(q[j][0] | (q[j][1] << 8));
                else {
                    o[0] = q[j][0];
                    if (c + 1 < p.cols) o[1] = q[j][1];
                }
            }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int k = 0; k < 2; ++k)
            tile[threadIdx.x * 2 + k][threadIdx.y * 4 + j] = q[j][k];
    __syncthreads();
    // Transposed scatter: output element (c, r) lives at c * rows + r;
    // threadIdx.x tracks r so each warp writes one contiguous run. tile is
    // [col][row]; warp y walks 8 columns, threads read down one column.
    uint8_t* out_t = static_cast<uint8_t*>(p.output_transposed_ptr);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int oc = c0 + threadIdx.y * 8 + i;
        if (oc < p.cols && r0 + threadIdx.x < p.rows)
            out_t[(int64_t)oc * p.rows + r0 + threadIdx.x] =
                tile[threadIdx.y * 8 + i][threadIdx.x];
    }
    if (p.amax) publish_amax<8>(p, local_amax);
}

// Unified quantize launcher: Tiled selects the transpose kernel
// (QuantLayout::Transposed/Dual) over the vectorized elementwise one.
// Kernels and traits are keyed on the fp8 element type (the launcher's
// template parameter — the bindings name the raw types); Dual picks the
// tiled instantiation with the row-major store (Transposed compiles it
// out). The transpose kernel vectorizes loads in-kernel and falls back to
// scalar loads at unaligned/ragged rows, so the host side picks only the
// grid.
template <typename Fp8T, typename InT, bool Tiled = false>
void launch_fp8_quantize(const QuantParams& p, cudaStream_t stream) {
    if constexpr (Tiled) {
        const dim3 grid((p.cols + 63) / 64, (p.rows + 31) / 32);
        if (grid.x == 0 || grid.y == 0) return;
        if (p.out_layout == QuantLayout::Dual)
            fp8_quantize_tiled_kernel<Fp8T, InT, true><<<grid, dim3(32, 8), 0, stream>>>(p);
        else
            fp8_quantize_tiled_kernel<Fp8T, InT, false><<<grid, dim3(32, 8), 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    } else {
        constexpr int kThreads = 256;
        constexpr int kVecElems = quant_in_traits<InT>::kVecElems;
        // Grid-stride loops: any grid >= 1 is correct; one block per 256
        // vectors plus the tail block covers tiny and misaligned tensors.
        const int64_t blocks = 1 + p.total / (kVecElems * kThreads);
        fp8_quantize_kernel<Fp8T, InT><<<blocks, kThreads, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
}

}  // namespace quant
}  // namespace astrai
