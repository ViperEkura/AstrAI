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
#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "common.h"
#include "common/launch.cuh"
#include "common/reduce.cuh"

namespace astrai {
namespace quant {
// Input element traits: the specialization supplies the scalar widen and,
// for 2-lane 16-bit inputs, names the native pair type; the shared base in
// detail below builds the pair load on top — adding a dtype is one thin
// specialization.
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

// 2-lane 16-bit input body: one native pair load per row. Everything but
// the widen above is dtype-independent.
template <typename InT, typename PairT>
struct pair_in_traits {
    using native_pair = PairT;
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
    using native_pair = float2;
    static __device__ __forceinline__ float to_float(float v) { return v; }
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
// per format.
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

// The kernel's warp count; the launcher's block is ``dim3(32, kQuantWarps)``
// and publish_amax parks one partial per warp, indexed by ``tid >> 5``.
inline constexpr int kQuantWarps = 8;

// Block-wide amax reduce -> one RMW per block: warp-reduce, park one value per
// warp, thread 0 folds. Blocks RMW their amax into amax_scratch[block id mod
// kFoldSlots] and the last-finishing block (atomicAdd ticket) folds the scratch
// into the history window, publishes the next scale and the round's amax, then
// re-zeroes for the next launch.
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
        const int slot =
            (blockIdx.y * gridDim.x + blockIdx.x) & (kFoldSlots - 1);
        atomic_max_float(p.amax_scratch + slot, v);
        __threadfence();
        const unsigned int ticket = atomicAdd(p.done, 1u);
        __threadfence();
        // Total blocks over BOTH grid dimensions (the tiled kernel launches
        // a 2D grid; gridDim.x alone made the fold fire every gridDim.x-th
        // completion — recording round-local partial amaxes and re-folding
        // on tall tensors, silently clipping the published scale).
        const unsigned int total = gridDim.x * gridDim.y;
        if (ticket != total - 1u) return;
        float peak = p.amax_scratch[0];
        for (int s = 1; s < kFoldSlots; ++s)
            peak = fmaxf(peak, p.amax_scratch[s]);
        p.hist[p.hist_idx] = peak;
        float win = p.hist[0];
        for (int i = 1; i < p.hist_len; ++i) win = fmaxf(win, p.hist[i]);
        const float next = fmaxf(win / p.fp8_max / p.pow2_margin, 1e-12f);
        *p.scale_out = next;
        // __frcp_rn is the correctly rounded reciprocal — bit-identical to
        // the ATen 1/x the host used to materialize (never fast-math: the
        // intrinsic pins the rounding mode).
        if (p.scale_recip_out) *p.scale_recip_out = __frcp_rn(next);
        if (p.amax) *p.amax = peak;
        for (int s = 0; s < kFoldSlots; ++s) p.amax_scratch[s] = 0.0f;
        *p.done = 0u;
    }
}

// The quantize kernel (merged 2026-09-19 from the former elementwise +
// tiled pair — ncu-verified tie in all three modes): one source serves
// RowMajor / Transposed / Dual. The orientation rides on which output
// pointers are live plus the output_ptr stride pair ((cols, 1) for the
// row-major placement); the transposed copy always takes the canonical
// contract c*rows + r derived from p.rows (not the pair's swap — that
// equals it only on square shapes, and the NT GEMM operand contract pins
// the layout anyway). The host folds leading dims so the kernel always
// sees one flat [rows][cols] view covering the whole buffer (1D and 3D
// inputs included). Work decomposition keeps warps on the unit-stride
// dimension on both sides: pair loads along input rows (128B per warp per
// row, four in flight on the full-tile fast path), row-major stores as 64B
// warp runs, transposed stores as 32B warp runs staged through the
// pitch-permuted shared tile (stride 17 words — coprime with the 32
// banks). Other stride pairs stay correct, just uncoalesced.
// Per-element dual conversion: the row-major orientation uses ``Fp8TA``, the
// transposed one ``Fp8TB``. A single format (the default) converts once and
// both orientations share the byte — only the hybrid training pair (E4M3
// forward / E5M2 backward, whose two GEMM sides must match) instantiates the
// second conversion, and ``if constexpr`` keeps the codegen of the
// single-format instantiations identical.
template <typename Fp8TA, typename Fp8TB>
struct dual_cvt {
    static constexpr bool kMixed = !std::is_same_v<Fp8TA, Fp8TB>;

    static __device__ __forceinline__ void store(uint8_t (*q)[2], uint8_t (*q2)[2],
                                                 int j, int k, float v) {
        q[j][k] = fp8_cvt_traits<Fp8TA>::cvt(v);
        if constexpr (kMixed) q2[j][k] = fp8_cvt_traits<Fp8TB>::cvt(v);
    }

    static __device__ __forceinline__ void zero(uint8_t (*q)[2], uint8_t (*q2)[2],
                                                int j) {
        q[j][0] = 0;
        q[j][1] = 0;
        if constexpr (kMixed) {
            q2[j][0] = 0;
            q2[j][1] = 0;
        }
    }

    static __device__ __forceinline__ uint8_t pick(const uint8_t (*q)[2],
                                                   const uint8_t (*q2)[2], int j,
                                                   int k) {
        if constexpr (kMixed) return q2[j][k];
        return q[j][k];
    }
};

template <typename Fp8T, typename InT, typename Fp8T2 = Fp8T>
__global__ void fp8_quantize_strided_kernel(QuantParams p) {
    constexpr int kTileC = 64, kTileR = 32;
    __shared__ uint8_t tile[kTileC][kTileR + 2];
    const float mult = *p.scale;
    const auto* x = static_cast<const InT*>(p.input_ptr);
    const int r0 = blockIdx.y * kTileR;
    const int c0 = blockIdx.x * kTileC;
    const int r = r0 + threadIdx.y * 4;
    const int c = c0 + threadIdx.x * 2;

    using Cvt = dual_cvt<Fp8T, Fp8T2>;
    uint8_t q[4][2];
    uint8_t q2[4][2];  // live only when the two orientations differ in format
    float local_amax = 0.0f;
    constexpr int kPairAlign = 2 * (int)sizeof(InT);
    using PairT = typename quant_in_traits<InT>::native_pair;
    const bool full_tile =
        r0 + kTileR <= p.rows && c0 + kTileC <= p.cols &&
        (reinterpret_cast<uintptr_t>(x) & (kPairAlign - 1)) == 0 &&
        ((p.cols & 1) == 0);
    if (full_tile) {
        const InT* a = x + (int64_t)r * p.cols + c;
        const PairT raws[4] = {
            *reinterpret_cast<const PairT*>(a),
            *reinterpret_cast<const PairT*>(a + (int64_t)p.cols),
            *reinterpret_cast<const PairT*>(a + 2 * (int64_t)p.cols),
            *reinterpret_cast<const PairT*>(a + 3 * (int64_t)p.cols)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float f[2];
            quant_in_traits<InT>::load_pair(
                reinterpret_cast<const InT*>(&raws[j]), f);
            local_amax = fmaxf(local_amax, fmaxf(fabsf(f[0]), fabsf(f[1])));
            Cvt::store(q, q2, j, 0, f[0] * mult);
            Cvt::store(q, q2, j, 1, f[1] * mult);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            Cvt::zero(q, q2, j);
            if (r + j < p.rows && c < p.cols) {
                const InT* a = x + (int64_t)(r + j) * p.cols + c;
                if (c + 1 < p.cols &&
                    (reinterpret_cast<uintptr_t>(a) & (kPairAlign - 1)) == 0) {
                    float f[2];
                    quant_in_traits<InT>::load_pair(a, f);
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        local_amax = fmaxf(local_amax, fabsf(f[k]));
                        Cvt::store(q, q2, j, k, f[k] * mult);
                    }
                } else {
                    const float v0 = quant_in_traits<InT>::to_float(a[0]);
                    local_amax = fmaxf(local_amax, fabsf(v0));
                    Cvt::store(q, q2, j, 0, v0 * mult);
                    if (c + 1 < p.cols) {
                        const float v1 = quant_in_traits<InT>::to_float(a[1]);
                        local_amax = fmaxf(local_amax, fabsf(v1));
                        Cvt::store(q, q2, j, 1, v1 * mult);
                    }
                }
            }
        }
    }
    // Primary output: placement from the stride pair (live for RowMajor
    // and Dual; null for pure-Transposed).
    uint8_t* out = static_cast<uint8_t*>(p.output_ptr);
    if (out != nullptr) {
        const int64_t s_r = p.out_row_stride, s_c = p.out_col_stride;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (r + j < p.rows && c < p.cols) {
                const int64_t off = (int64_t)(r + j) * s_r + (int64_t)c * s_c;
                uint8_t* o = out + off;
                if (c + 1 < p.cols && (off & 1) == 0)
                    *reinterpret_cast<unsigned short*>(o) =
                        (unsigned short)(q[j][0] | (q[j][1] << 8));
                else {
                    o[0] = q[j][0];
                    if (c + 1 < p.cols) o[1] = q[j][1];
                }
            }
    }
    // Transposed copy: canonical (1, rows) placement, staged through the
    // pitch-permuted tile so each warp writes one contiguous run.
    uint8_t* out_t = static_cast<uint8_t*>(p.output_transposed_ptr);
    if (out_t != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int k = 0; k < 2; ++k)
                tile[threadIdx.x * 2 + k][threadIdx.y * 4 + j] =
                    Cvt::pick(q, q2, j, k);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int oc = c0 + threadIdx.y * 8 + i;
            if (oc < p.cols && r0 + threadIdx.x < p.rows)
                out_t[(int64_t)oc * p.rows + r0 + threadIdx.x] =
                    tile[threadIdx.y * 8 + i][threadIdx.x];
        }
    }
    if (p.fold_ring) publish_amax<kQuantWarps>(p, local_amax);
}

// Quantize launcher: the mode (which pointers are live, the stride pair)
// is data, not template — one instantiation per (Fp8T, InT) for a single
// format, per (Fp8TA, Fp8TB, InT) when the two orientations differ. A
// degenerate axis still launches one block so the ring fold fires on empty
// tensors (the delayed-scaling publish must happen even when nothing
// quantizes).
template <typename Fp8T, typename InT, typename Fp8T2 = Fp8T>
void launch_fp8_quantize(const QuantParams& p, cudaStream_t stream) {
    const dim3 grid(std::max(1, (p.cols + 63) / 64),
                    std::max(1, (p.rows + 31) / 32));
    fp8_quantize_strided_kernel<Fp8T, InT, Fp8T2>
        <<<grid, dim3(32, kQuantWarps), 0, stream>>>(p);
    ASTRAI_LAUNCH_CHECK();
}

}  // namespace quant
}  // namespace astrai
