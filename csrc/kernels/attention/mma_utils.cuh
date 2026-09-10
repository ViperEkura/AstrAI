#pragma once
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common/pipeline.cuh"
#include "common/mma.cuh"
#include "softmax.cuh"

// Predicated cp.async (4-operand form) requires CUDA 11.2+.
// bf16 mma.sync requires sm_80+ (guarded at build time by ASTRAI_NO_MMA).
#if CUDART_VERSION < 11020
#error "AstrAI CUDA kernels require CUDA 11.2 or later (CUDART_VERSION >= 11020)."
#endif

namespace astrai {
namespace attention {

// ============================================================================
// KernelTraits — FlashAttention-v2 style compile-time configuration bundle.
//
// Bundles all dimension-dependent constants so device functions only need a
// single Traits template parameter rather than scattered <KD, NC8, KT2, ...>.
// ============================================================================
template <int HEAD_DIM_, int BC_, int WARPS_, int STAGES_>
struct KernelTraits {
    static constexpr int HEAD_DIM = HEAD_DIM_;
    static constexpr int BC      = BC_;       // K/V tile size along seq dim
    static constexpr int WARPS   = WARPS_;    // warps per block
    static constexpr int STAGES  = STAGES_;   // double-buffer stages (1 or 2)

    static constexpr int BR = 16;             // Q rows per warp (mma M=16)

    // Derived: mma tile counts from the shared mma_shape (m16n8k16 for bf16)
    static constexpr int KD  = HEAD_DIM / astrai::mma_shape<bf16>::k;  // Q/K k-slides
    static constexpr int NC8 = BC / 8;          // S n-tiles (N=8)
    static constexpr int KT2 = BC / astrai::mma_shape<bf16>::k;        // P k-tiles (K=16)
    static constexpr int DN8 = HEAD_DIM / 8;    // O n-tiles (N=8)

    static constexpr int LD = HEAD_DIM;         // smem leading dim

    // XOR swizzle chunk bits for ldmatrix bank-conflict avoidance.
    // mask = log2(LD/8) bits, clamped to stay within LD.
    static constexpr int SWIZ_MASK = (HEAD_DIM >= 64) ? 7 : (HEAD_DIM / 8 - 1);

    static constexpr int NUM_THREADS = WARPS * 32;
    static constexpr int VEC = 8;               // bf16 per cp.async unit (16 bytes)
    static constexpr int TOTAL = BC * HEAD_DIM; // total elements per tile
};

// ---- PTX wrappers ----
using bf16 = __nv_bfloat16;
// bf16 mma.sync lives in the shared astrai::mma_sync template (common/mma.cuh).

// read two adjacent bf16 from smem as one packed .b32 (elem0 low, elem1 high)
__device__ __forceinline__ unsigned ld2(const bf16* p) {
    return *reinterpret_cast<const unsigned*>(p);
}

// pack two floats into one bf16x2 as .b32
__device__ __forceinline__ unsigned pk2(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<unsigned*>(&v);
}

// pack two (non-contiguous) bf16 into one .b32
__device__ __forceinline__ unsigned pkb(bf16 a, bf16 b) {
    __nv_bfloat162 v;
    v.x = a;
    v.y = b;
    return *reinterpret_cast<unsigned*>(&v);
}

// ldmatrix lives in the shared template (common/mma.cuh):
// `astrai::ldmatrix_x2<bf16>` / `<bf16, /*Trans=*/true>` load the K/V
// fragments with the exact register layout mma expects.

// XOR swizzle for shared-memory column at 8-bf16 chunk granularity.
__device__ __forceinline__ int swiz_col(int d, int r, int mask = 7) {
    return ((d >> 3) ^ (r & mask)) << 3 | (d & 7);
}

// cp.async primitives live in the shared template (common/pipeline.cuh):
// `astrai::cp_async_16` (predicated), `astrai::cp_async_commit_group`,
// `astrai::cp_async_wait_group<N>` / `_wait_all` stage the K/V tiles.

// ---------------------------------------------------------------------------
// Q-load: load query rows directly from global memory into mma A-operand
// register layout. One call replaces ~15 duplicated lines in each MMA kernel.
// stride_row is p.q_h_stride for decode (q_len=1, G heads) or
//              p.q_l_stride for prefill (multi-q rows).
// ---------------------------------------------------------------------------
template <int KD>
__device__ inline void load_q_mma_frags(
    const bf16* __restrict__ q,
    int stride_row,
    int stride_d,
    int qra, int qrb,
    bool va, bool vb,
    int tid4,
    unsigned Qa[KD][4])
{
    #pragma unroll
    for (int kt = 0; kt < KD; kt++) {
        int c = kt * 16 + tid4 * 2;
        const unsigned* pau = reinterpret_cast<const unsigned*>(
            &q[qra * stride_row + c * stride_d]);
        const unsigned* pbu = reinterpret_cast<const unsigned*>(
            &q[qrb * stride_row + c * stride_d]);
        Qa[kt][0] = va ? pau[0] : 0u;
        Qa[kt][1] = vb ? pbu[0] : 0u;
        Qa[kt][2] = va ? pau[4] : 0u;
        Qa[kt][3] = vb ? pbu[4] : 0u;
    }
}

// ---------------------------------------------------------------------------
// K/V tile loader shared by the MMA kernels: stages one BC×HEAD_DIM tile
// into the double-buffered K/V rings via predicated cp.async with the XOR
// swizzle.  AddrFn maps (kc, d, valid) -> {k, v, valid} (KVAddr); the two
// kernels differ only in addressing (decode: KV::decode_addr with new-K/V
// persistence; prefill: resolve_token + kv_addr_from_token).
// ---------------------------------------------------------------------------
template <typename Traits, typename AddrFn>
__device__ inline void load_kv_tile(
    bf16* sK, bf16* sV,   // ring bases (STAGES * BC * LD each)
    int ti, int buf,      // tile index, ring slot
    int seq_len,
    const AddrFn& addr)
{
    int kv0 = ti * Traits::BC;
    bf16* dK = sK + buf * Traits::BC * Traits::LD;
    bf16* dV = sV + buf * Traits::BC * Traits::LD;
    #pragma unroll
    for (int i = threadIdx.x * Traits::VEC; i < Traits::TOTAL;
         i += Traits::NUM_THREADS * Traits::VEC) {
        int r = i / Traits::HEAD_DIM, d = i % Traits::HEAD_DIM;
        int kc = kv0 + r;
        bool valid = kc < seq_len;
        auto a = addr(kc, d, valid);
        int off = r * Traits::LD + swiz_col(d, r, Traits::SWIZ_MASK);
        astrai::cp_async_16(&dK[off], a.k, a.valid);
        astrai::cp_async_16(&dV[off], a.v, a.valid);
    }
    astrai::cp_async_commit_group();
}

// ---------------------------------------------------------------------------
// S = Q @ K^T  (Qa pre-loaded by the caller; `scale` applied post-mma in
// float to avoid bf16 precision loss).
// Traits provides KD, NC8, LD, and SWIZ_MASK.
// ---------------------------------------------------------------------------
template <typename Traits>
__device__ inline void mma_compute_scores(
    const unsigned Qa[Traits::KD][4],
    const bf16* __restrict__ sK,
    float scale,
    int lane,
    float Sacc[Traits::NC8][4])
{
    #pragma unroll
    for (int n8 = 0; n8 < Traits::NC8; n8++) {
        Sacc[n8][0] = Sacc[n8][1] = Sacc[n8][2] = Sacc[n8][3] = 0.0f;
        int krow_l = n8 * 8 + (lane & 7);
        int kcol_h = (lane & 8) ? 8 : 0;
        #pragma unroll
        for (int kt = 0; kt < Traits::KD; kt++) {
            unsigned b[2];
            astrai::ldmatrix_x2<bf16>(b, &sK[krow_l * Traits::LD
                + swiz_col(kt * 16 + kcol_h, krow_l, Traits::SWIZ_MASK)]);
            astrai::mma_sync<bf16>(Sacc[n8], Qa[kt], b, Sacc[n8]);
        }
        Sacc[n8][0] *= scale; Sacc[n8][1] *= scale;
        Sacc[n8][2] *= scale; Sacc[n8][3] *= scale;
    }
}

// ---------------------------------------------------------------------------
// Online softmax + Oacc rescale for one K/V tile.
//
// HasMask is a compile-time template bool: when false, the mask branch is
// entirely dead-code-eliminated from the inner unrolled loop.
// ---------------------------------------------------------------------------
template <typename Traits, bool HasMask>
__device__ inline void mma_softmax_tile(
    int kv0,
    int maxc0, int maxc1,
    int qrow0, int qrow1,
    int mask_b_stride, int mask_h_stride, int mask_l_stride,
    int mask_batch, int mask_head0, int mask_head1,
    const bool* __restrict__ mask,
    bool valid0, bool valid1,
    float Sacc[Traits::NC8][4],
    float Oacc[Traits::DN8][4],
    float& m0, float& m1,
    float& l0, float& l1,
    int lane)
{
    int tid4 = lane & 3;

    float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
    int mask_base0 = mask_batch * mask_b_stride + mask_head0 * mask_h_stride + qrow0 * mask_l_stride;
    int mask_base1 = mask_batch * mask_b_stride + mask_head1 * mask_h_stride + qrow1 * mask_l_stride;
    #pragma unroll
    for (int n8 = 0; n8 < Traits::NC8; n8++) {
        int cc = kv0 + n8 * 8 + 2 * tid4;
        int c1 = cc + 1;
        bool b0 = !valid0 || (cc >= maxc0) || (HasMask && !mask[mask_base0 + cc]);
        bool b1 = !valid0 || (c1 >= maxc0) || (HasMask && !mask[mask_base0 + c1]);
        bool b2 = !valid1 || (cc >= maxc1) || (HasMask && !mask[mask_base1 + cc]);
        bool b3 = !valid1 || (c1 >= maxc1) || (HasMask && !mask[mask_base1 + c1]);
        float s0 = b0 ? -FLT_MAX : Sacc[n8][0];
        float s1 = b1 ? -FLT_MAX : Sacc[n8][1];
        float s2 = b2 ? -FLT_MAX : Sacc[n8][2];
        float s3 = b3 ? -FLT_MAX : Sacc[n8][3];
        Sacc[n8][0] = s0; Sacc[n8][1] = s1;
        Sacc[n8][2] = s2; Sacc[n8][3] = s3;
        rmax0 = fmaxf(rmax0, fmaxf(s0, s1));
        rmax1 = fmaxf(rmax1, fmaxf(s2, s3));
    }
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

    float corr0, corr1, pn0, pn1;
    float nm0 = softmax_remax(m0, rmax0, corr0, pn0);
    float nm1 = softmax_remax(m1, rmax1, corr1, pn1);

    float rsum0 = 0.0f, rsum1 = 0.0f;
    #pragma unroll
    for (int n8 = 0; n8 < Traits::NC8; n8++) {
        float p0 = pn0 * __expf(Sacc[n8][0] - nm0);
        float p1 = pn0 * __expf(Sacc[n8][1] - nm0);
        float p2 = pn1 * __expf(Sacc[n8][2] - nm1);
        float p3 = pn1 * __expf(Sacc[n8][3] - nm1);
        Sacc[n8][0] = p0; Sacc[n8][1] = p1;
        Sacc[n8][2] = p2; Sacc[n8][3] = p3;
        rsum0 += p0 + p1;
        rsum1 += p2 + p3;
    }
    rsum0 += __shfl_xor_sync(0xFFFFFFFF, rsum0, 1);
    rsum0 += __shfl_xor_sync(0xFFFFFFFF, rsum0, 2);
    rsum1 += __shfl_xor_sync(0xFFFFFFFF, rsum1, 1);
    rsum1 += __shfl_xor_sync(0xFFFFFFFF, rsum1, 2);
    l0 = l0 * corr0 + rsum0;
    l1 = l1 * corr1 + rsum1;

    #pragma unroll
    for (int j = 0; j < Traits::DN8; j++) {
        Oacc[j][0] *= corr0; Oacc[j][1] *= corr0;
        Oacc[j][2] *= corr1; Oacc[j][3] *= corr1;
    }
}

// ---------------------------------------------------------------------------
// O += P @ V  (Sacc must contain P = attention weights after softmax).
// Traits provides DN8, KT2, LD, and SWIZ_MASK.
// ---------------------------------------------------------------------------
template <typename Traits>
__device__ inline void mma_pv_accumulate(
    float Sacc[][4],
    const bf16* __restrict__ sV,
    int lane,
    float Oacc[Traits::DN8][4])
{
    #pragma unroll
    for (int kt2 = 0; kt2 < Traits::KT2; kt2++) {
        unsigned Pa[4];
        Pa[0] = pk2(Sacc[kt2 * 2][0], Sacc[kt2 * 2][1]);
        Pa[1] = pk2(Sacc[kt2 * 2][2], Sacc[kt2 * 2][3]);
        Pa[2] = pk2(Sacc[kt2 * 2 + 1][0], Sacc[kt2 * 2 + 1][1]);
        Pa[3] = pk2(Sacc[kt2 * 2 + 1][2], Sacc[kt2 * 2 + 1][3]);
        int vrow_l = kt2 * 16 + (lane & 15);
        #pragma unroll
        for (int dn8 = 0; dn8 < Traits::DN8; dn8++) {
            unsigned b[2];
            astrai::ldmatrix_x2<bf16, true>(b, &sV[vrow_l * Traits::LD
                + swiz_col(dn8 * 8, vrow_l, Traits::SWIZ_MASK)]);
            astrai::mma_sync<bf16>(Oacc[dn8], Pa, b, Oacc[dn8]);
        }
    }
}

}  // namespace attention
}  // namespace astrai
