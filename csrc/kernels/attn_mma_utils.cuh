#pragma once
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

    // Derived: mma.sync.m16n8k16 tile counts
    static constexpr int KD  = HEAD_DIM / 16;  // Q/K k-slides
    static constexpr int NC8 = BC / 8;          // S n-tiles (N=8)
    static constexpr int KT2 = BC / 16;         // P k-tiles (K=16)
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

__device__ __forceinline__ void mma16816(float* d, const unsigned* a,
                                          const unsigned* b, const float* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

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

// ldmatrix: cooperatively load mma fragments from smem (one instruction per
// 16x16 / 16x8 tile) with the exact register layout mma expects.
__device__ __forceinline__ void ldmatrix_x4(unsigned* r, const bf16* p) {
    unsigned a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2(unsigned* r, const bf16* p) {
    unsigned a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2_trans(unsigned* r, const bf16* p) {
    unsigned a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
}

// XOR swizzle for shared-memory column at 8-bf16 chunk granularity.
__device__ __forceinline__ int swiz_col(int d, int r, int mask = 7) {
    return ((d >> 3) ^ (r & mask)) << 3 | (d & 7);
}

// cp.async: copy 16 bytes (8 bf16) from global to shared memory directly.
__device__ __forceinline__ void cp_async_16(bf16* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(smem_addr), "l"(gmem_ptr));
}

// Predicated cp.async: copy 16 bytes when `pred`, otherwise zero-fill.
// src_size=0 → no bytes read from src, so out-of-bounds src address is safe.
__device__ __forceinline__ void cp_async_16_pred(bf16* smem_ptr,
                                                  const void* gmem_ptr,
                                                  bool pred) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    int src_size = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_size));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

// ---------------------------------------------------------------------------
// Q-load: load query rows directly from global memory into mma A-operand
// register layout. One call replaces ~15 duplicated lines in each MMA kernel.
// stride_row is p.q_stride_h for decode (q_len=1, G heads) or
//              p.q_stride_l for prefill (multi-q rows).
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
// S = Q @ K^T  (Qa pre-loaded by the caller; scale applied post-mma in the
// caller to avoid bf16 precision loss).
// Traits provides KD, NC8, LD, and SWIZ_MASK.
// ---------------------------------------------------------------------------
template <typename Traits>
__device__ inline void mma_compute_scores(
    const unsigned Qa[Traits::KD][4],
    const bf16* __restrict__ sK,
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
            ldmatrix_x2(b, &sK[krow_l * Traits::LD
                + swiz_col(kt * 16 + kcol_h, krow_l, Traits::SWIZ_MASK)]);
            mma16816(Sacc[n8], Qa[kt], b, Sacc[n8]);
        }
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
    int mask_b_stride, int mask_q_stride,
    int mask_batch,
    const bool* __restrict__ mask,
    float Sacc[Traits::NC8][4],
    float Oacc[Traits::DN8][4],
    float& m0, float& m1,
    float& l0, float& l1,
    int lane)
{
    int tid4 = lane & 3;

    float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
    int mask_base0 = mask_batch * mask_b_stride + qrow0 * mask_q_stride;
    int mask_base1 = mask_batch * mask_b_stride + qrow1 * mask_q_stride;
    #pragma unroll
    for (int n8 = 0; n8 < Traits::NC8; n8++) {
        int cc = kv0 + n8 * 8 + 2 * tid4;
        int c1 = cc + 1;
        bool b0 = (cc >= maxc0) || (HasMask && !mask[mask_base0 + cc]);
        bool b1 = (c1 >= maxc0) || (HasMask && !mask[mask_base0 + c1]);
        bool b2 = (cc >= maxc1) || (HasMask && !mask[mask_base1 + cc]);
        bool b3 = (c1 >= maxc1) || (HasMask && !mask[mask_base1 + c1]);
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

    float nm0 = fmaxf(m0, rmax0), nm1 = fmaxf(m1, rmax1);
    float corr0 = __expf(m0 - nm0);
    float corr1 = __expf(m1 - nm1);
    float pn0 = (nm0 == -FLT_MAX) ? 0.0f : 1.0f;
    float pn1 = (nm1 == -FLT_MAX) ? 0.0f : 1.0f;

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
    m0 = nm0; m1 = nm1;

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
            ldmatrix_x2_trans(b, &sV[vrow_l * Traits::LD
                + swiz_col(dn8 * 8, vrow_l, Traits::SWIZ_MASK)]);
            mma16816(Oacc[dn8], Pa, b, Oacc[dn8]);
        }
    }
}
