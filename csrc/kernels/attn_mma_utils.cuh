#pragma once
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Shared MMA utilities for tensor-core GQA kernels.
// mma.sync.m16n8k16 PTX wrappers, ldmatrix helpers, and bf16 packing.

// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
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
// 16x16 / 16x8 tile) with the exact register layout mma expects — replaces the
// scalar per-thread fragment packing, cutting shared-load instructions and bank
// conflicts. Each lane supplies the shared address of one 8-wide row.
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
// Eliminates ldmatrix bank conflicts without LD padding: consecutive rows
// land in distinct bank groups. swiz_col(d, r, mask) = ((d>>3)^(r&mask))<<3 | (d&7).
// mask must cover log2(HEAD_DIM/8) chunk bits but stay within LD: use 7 for
// HEAD_DIM>=64 (8+ chunks), 3 for HEAD_DIM=32 (4 chunks). Default 7 keeps
// existing HEAD_DIM>=64 call sites working unchanged.
__device__ __forceinline__ int swiz_col(int d, int r, int mask = 7) {
    return ((d >> 3) ^ (r & mask)) << 3 | (d & 7);
}

// cp.async: copy 16 bytes (8 bf16) from global to shared memory directly,
// bypassing registers. Eliminates shared-store bank conflicts and cuts
// load-loop instruction count in half (1 cp.async vs 1 LDG + 1 STS).
// Requires sm_80+.
__device__ __forceinline__ void cp_async_16(bf16* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(smem_addr), "l"(gmem_ptr));
}

// Predicated cp.async: copy 16 bytes when `pred`, otherwise zero-fill the
// destination (src-size operand = 0 → no bytes read from src, so an
// out-of-bounds src address is never dereferenced). Lets full and partial
// tiles share one uniform async load path — no scalar fallback branch.
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

// Wait until at most N commit groups are still in flight. Used for
// double-buffered pipelining: wait_group<1> lets the next tile's cp.async
// continue while ensuring the current tile's data is ready.
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

// ---------------------------------------------------------------------------
// Shared MMA compute functions — used by both decode and prefill MMA kernels.
// Extracted because S=Q@K^T, online softmax, and P@V are structurally identical
// between the two kernels; only the per-row causal/mask bounds differ.
// ---------------------------------------------------------------------------

// S = Q @ K^T  (Qa pre-loaded by the caller; scale applied post-mma in the
// caller to avoid bf16 precision loss).
// LD and SWIZ_MASK are constexpr in the calling kernel — passing them as
// runtime ints lets the compiler fold them while keeping the signature clean.
template <int KD, int NC8>
__device__ inline void mma_compute_scores(
    const unsigned Qa[KD][4],
    const bf16* __restrict__ sK,
    int LD, 
    int SWIZ_MASK, 
    int lane,
    float Sacc[NC8][4])
{
    #pragma unroll
    for (int n8 = 0; n8 < NC8; n8++) {
        Sacc[n8][0] = Sacc[n8][1] = Sacc[n8][2] = Sacc[n8][3] = 0.0f;
        int krow_l = n8 * 8 + (lane & 7);
        int kcol_h = (lane & 8) ? 8 : 0;
        #pragma unroll
        for (int kt = 0; kt < KD; kt++) {
            unsigned b[2];
            ldmatrix_x2(b, &sK[krow_l * LD + swiz_col(kt * 16 + kcol_h, krow_l, SWIZ_MASK)]);
            mma16816(Sacc[n8], Qa[kt], b, Sacc[n8]);
        }
    }
}

// Online softmax + Oacc rescale for one K/V tile.
//   maxc0/maxc1: per-row KV column bounds (prefill: per-query-row causal limits;
//     decode: same value for both rows since q_len==1).
//   qrow0/qrow1: query row indices (for 3D mask indexing; decode passes 0).
//   mask_b_stride/mask_q_stride: mask layout (2D: mask_q_stride=0; 3D: =kv_len).
// Reads Sacc (Q@K^T scores), applies causal/mask, computes P = exp(S - nm),
// rescales Oacc by exp(m_old - nm), and updates m/l — all in place.
template <int NC8, int DN8>
__device__ inline void mma_softmax_tile(
    int kv0,
    int maxc0, 
    int maxc1,
    int qrow0, 
    int qrow1,
    int mask_b_stride, 
    int mask_q_stride,
    int mask_batch,
    const bool* __restrict__ mask,
    bool has_mask,
    float Sacc[NC8][4],
    float Oacc[DN8][4],
    float& m0, float& m1,
    float& l0, float& l1,
    int lane)
{
    int tid4 = lane & 3;

    // Mask out-of-bounds / masked columns: set -FLT_MAX so expf → 0 downstream
    // without per-element sentinel checks. Compute tile-local row maxima.
    float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
    int mask_base0 = mask_batch * mask_b_stride + qrow0 * mask_q_stride;
    int mask_base1 = mask_batch * mask_b_stride + qrow1 * mask_q_stride;
    #pragma unroll
    for (int n8 = 0; n8 < NC8; n8++) {
        int cc = kv0 + n8 * 8 + 2 * tid4;
        int c1 = cc + 1;
        bool b0 = (cc >= maxc0) || (has_mask && !mask[mask_base0 + cc]);
        bool b1 = (c1 >= maxc0) || (has_mask && !mask[mask_base0 + c1]);
        bool b2 = (cc >= maxc1) || (has_mask && !mask[mask_base1 + cc]);
        bool b3 = (c1 >= maxc1) || (has_mask && !mask[mask_base1 + c1]);
        float s0 = b0 ? -FLT_MAX : Sacc[n8][0];
        float s1 = b1 ? -FLT_MAX : Sacc[n8][1];
        float s2 = b2 ? -FLT_MAX : Sacc[n8][2];
        float s3 = b3 ? -FLT_MAX : Sacc[n8][3];
        Sacc[n8][0] = s0; Sacc[n8][1] = s1;
        Sacc[n8][2] = s2; Sacc[n8][3] = s3;
        rmax0 = fmaxf(rmax0, fmaxf(s0, s1));
        rmax1 = fmaxf(rmax1, fmaxf(s2, s3));
    }
    // Warp-reduce row maxima across the 4-lane thread group (xor 1, xor 2).
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

    // nm = max(running max m, tile-local max rmax) — updated running maximum.
    float nm0 = fmaxf(m0, rmax0), nm1 = fmaxf(m1, rmax1);
    // corr rescales Oacc and l by exp(m_old - nm). When all-masked (m == nm ==
    // -FLT_MAX), exp(0) = 1 — correct, no guard needed.
    float corr0 = __expf(m0 - nm0);
    float corr1 = __expf(m1 - nm1);
    // pn guards only the all-masked-row edge: if nm == -FLT_MAX, exp(S - nm)
    // gives 1 not 0 for masked entries. Two scalar masks replace 4*NC8
    // per-element comparisons.
    float pn0 = (nm0 == -FLT_MAX) ? 0.0f : 1.0f;
    float pn1 = (nm1 == -FLT_MAX) ? 0.0f : 1.0f;

    // P = exp(S - nm) for each element. Masked entries (Sacc = -FLT_MAX) give
    // exp(-inf) ≈ 0 naturally; pn zero-fills the all-masked-row edge.
    float rsum0 = 0.0f, rsum1 = 0.0f;
    #pragma unroll
    for (int n8 = 0; n8 < NC8; n8++) {
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
    for (int j = 0; j < DN8; j++) {
        Oacc[j][0] *= corr0; Oacc[j][1] *= corr0;
        Oacc[j][2] *= corr1; Oacc[j][3] *= corr1;
    }
}

// O += P @ V  (Sacc must contain P = attention weights after softmax).
template <int DN8, int KT2>
__device__ inline void mma_pv_accumulate(
    float Sacc[][4],
    const bf16* __restrict__ sV,
    int LD, int SWIZ_MASK, int lane,
    float Oacc[DN8][4])
{
    #pragma unroll
    for (int kt2 = 0; kt2 < KT2; kt2++) {
        unsigned Pa[4];
        Pa[0] = pk2(Sacc[kt2 * 2][0], Sacc[kt2 * 2][1]);
        Pa[1] = pk2(Sacc[kt2 * 2][2], Sacc[kt2 * 2][3]);
        Pa[2] = pk2(Sacc[kt2 * 2 + 1][0], Sacc[kt2 * 2 + 1][1]);
        Pa[3] = pk2(Sacc[kt2 * 2 + 1][2], Sacc[kt2 * 2 + 1][3]);
        int vrow_l = kt2 * 16 + (lane & 15);
        #pragma unroll
        for (int dn8 = 0; dn8 < DN8; dn8++) {
            unsigned b[2];
            ldmatrix_x2_trans(b, &sV[vrow_l * LD + swiz_col(dn8 * 8, vrow_l, SWIZ_MASK)]);
            mma16816(Oacc[dn8], Pa, b, Oacc[dn8]);
        }
    }
}
