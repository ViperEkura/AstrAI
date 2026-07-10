#pragma once
#include "gqa_common.cuh"
#include "gqa_mma_utils.cuh"

// Tensor-core decode via GQA head-packing with cp.async loads.
//
// Decode has q_len == 1, so S = q @ K^T is a GEMV per head — no tensor-core work
// on its own. But GQA gives us G = q_head / kv_head query heads that all share
// one kv_head. We pack those G heads into the M=16 rows of mma.sync.m16n8k16,
// turning G independent GEMVs into a single GEMM that reuses each loaded K/V tile
// across all G heads (K/V load is the decode bottleneck, so the reuse is the win,
// not the flops). Fragment layout is identical to the prefill mma kernel; the
// only differences are (1) the M rows come from different heads at position 0
// instead of different sequence positions of one head, and (2) causal masking is
// a single scalar bound shared by every row. One warp owns one (batch, kv_head);
// requires G <= 16.
//
// Optimizations:
//   - cp.async global→shared for K/V (bypasses registers, cuts instruction count)
//   - XOR swizzle (swiz_col): LD=HEAD_DIM, zero waste, no bank conflicts
//   - pre-scaled Q: Q scaled during load, softmax skips per-tile multiply
//   - single-buffer: keeps smem small for high occupancy

template <int HEAD_DIM, int BC>
__global__ void gqa_decode_attn_mma_kernel(GQAParams p) {
    constexpr int BR = 16;
    constexpr int KD = HEAD_DIM / 16;  // Q/K k-tiles
    constexpr int NC8 = BC / 8;        // S n-tiles (N=8 each)
    constexpr int KT2 = BC / 16;       // P k-tiles (K=16 each)
    constexpr int DN8 = HEAD_DIM / 8;  // O n-tiles (N=8 each)
    constexpr int LD = HEAD_DIM;       // XOR swizzle handles bank conflicts, zero waste
    constexpr int SWIZ_MASK = (HEAD_DIM >= 64) ? 7 : (HEAD_DIM / 8 - 1);

    const int lane = threadIdx.x;  // single warp
    const int gid = lane >> 2;     // 0..7 → rows gid, gid+8
    const int tid4 = lane & 3;

    const int kv_head = blockIdx.x;
    const int batch = blockIdx.y;
    const int G = p.q_head / p.kv_head;
    const int q_head0 = kv_head * G;

    extern __shared__ __align__(16) bf16 smem[];
    bf16* sK = smem;                  // [BC][LD]
    bf16* sV = sK + BC * LD;          // [BC][LD]
    bf16* sQ = sV + BC * LD;          // [BR][LD]

    // ---- stage Q into shared (pre-scaled, swizzled) ----
    bf16 scale_bf16 = __float2bfloat16(p.scale);
    for (int i = lane; i < BR * HEAD_DIM; i += 32) {
        int r = i / HEAD_DIM, d = i % HEAD_DIM;
        bf16 val = __float2bfloat16(0.0f);
        if (r < G) {
            int qh = q_head0 + r;
            val = p.q[(batch * p.q_head + qh) * HEAD_DIM + d];  // q_len == 1
        }
        sQ[r * LD + swiz_col(d, r, SWIZ_MASK)] = __hmul(val, scale_bf16);
    }
    __syncwarp();

    // Q resident A-fragments
    unsigned Qa[KD][4];
    int qrow_l = (lane & 7) + (lane & 8);
    int qcol_l = (lane & 16) ? 8 : 0;
#pragma unroll
    for (int kt = 0; kt < KD; kt++)
        ldmatrix_x4(Qa[kt], &sQ[qrow_l * LD + swiz_col(kt * 16 + qcol_l, qrow_l, SWIZ_MASK)]);

    float Oacc[DN8][4];
#pragma unroll
    for (int j = 0; j < DN8; j++)
        Oacc[j][0] = Oacc[j][1] = Oacc[j][2] = Oacc[j][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;

    const int kv_base = (batch * p.kv_head + kv_head) * p.kv_len * HEAD_DIM;
    const int mask_base = batch * p.kv_len;
    const int tiles = (p.kv_len + BC - 1) / BC;
    const int has_mask = p.use_mask && p.mask;

    for (int ti = 0; ti < tiles; ti++) {
        int kv0 = ti * BC;

        // ---- load K/V tile to shared (cp.async on full tiles) ----
        bool full_tile = (kv0 + BC <= p.kv_len);
        if (full_tile) {
            constexpr int VEC = 8;  // 8 bf16 = 16 bytes per cp.async
            int total = BC * HEAD_DIM;
#pragma unroll
            for (int i = lane * VEC; i < total; i += 32 * VEC) {
                int r = i / HEAD_DIM, d = i % HEAD_DIM;
                int kc = kv0 + r;
                cp_async_16(&sK[r * LD + swiz_col(d, r, SWIZ_MASK)],
                            &p.k[kv_base + kc * HEAD_DIM + d]);
                cp_async_16(&sV[r * LD + swiz_col(d, r, SWIZ_MASK)],
                            &p.v[kv_base + kc * HEAD_DIM + d]);
            }
            cp_async_commit();
            cp_async_wait_all();
        } else {
            for (int i = lane; i < BC * HEAD_DIM; i += 32) {
                int r = i / HEAD_DIM, d = i % HEAD_DIM;
                int kc = kv0 + r;
                bf16 z = __float2bfloat16(0.0f);
                sK[r * LD + swiz_col(d, r, SWIZ_MASK)] =
                    (kc < p.kv_len) ? p.k[kv_base + kc * HEAD_DIM + d] : z;
                sV[r * LD + swiz_col(d, r, SWIZ_MASK)] =
                    (kc < p.kv_len) ? p.v[kv_base + kc * HEAD_DIM + d] : z;
            }
        }
        __syncwarp();

        // S = Q @ K^T  (Q already pre-scaled, so Sacc includes scale)
        float Sacc[NC8][4];
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

        // ---- online softmax (Q pre-scaled → no per-tile scale multiply) ----
        float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
#pragma unroll
        for (int n8 = 0; n8 < NC8; n8++) {
            int cc = kv0 + n8 * 8 + 2 * tid4;
            bool bc0 = (cc >= p.kv_len) ||
                       (has_mask && !p.mask[mask_base + cc]);
            bool bc1 = (cc + 1 >= p.kv_len) ||
                       (has_mask && !p.mask[mask_base + cc + 1]);
            bool cz = p.is_causal;
            int off = p.causal_offset;
            bool bad0 = bc0 || (cz && cc > off);
            bool bad1 = bc1 || (cz && (cc + 1) > off);
            float s0 = bad0 ? -FLT_MAX : Sacc[n8][0];
            float s1 = bad1 ? -FLT_MAX : Sacc[n8][1];
            float s2 = bad0 ? -FLT_MAX : Sacc[n8][2];
            float s3 = bad1 ? -FLT_MAX : Sacc[n8][3];
            Sacc[n8][0] = s0; Sacc[n8][1] = s1; Sacc[n8][2] = s2; Sacc[n8][3] = s3;
            rmax0 = fmaxf(rmax0, fmaxf(s0, s1));
            rmax1 = fmaxf(rmax1, fmaxf(s2, s3));
        }
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

        float nm0 = fmaxf(m0, rmax0), nm1 = fmaxf(m1, rmax1);
        float corr0 = (nm0 == -FLT_MAX) ? 1.0f : __expf(m0 - nm0);
        float corr1 = (nm1 == -FLT_MAX) ? 1.0f : __expf(m1 - nm1);

        float rsum0 = 0.0f, rsum1 = 0.0f;
#pragma unroll
        for (int n8 = 0; n8 < NC8; n8++) {
            float p0 = (Sacc[n8][0] == -FLT_MAX) ? 0.0f : __expf(Sacc[n8][0] - nm0);
            float p1 = (Sacc[n8][1] == -FLT_MAX) ? 0.0f : __expf(Sacc[n8][1] - nm0);
            float p2 = (Sacc[n8][2] == -FLT_MAX) ? 0.0f : __expf(Sacc[n8][2] - nm1);
            float p3 = (Sacc[n8][3] == -FLT_MAX) ? 0.0f : __expf(Sacc[n8][3] - nm1);
            Sacc[n8][0] = p0; Sacc[n8][1] = p1; Sacc[n8][2] = p2; Sacc[n8][3] = p3;
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

        // O += P @ V
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
        __syncwarp();  // sK/sV reused next tile
    }

    // ---- write output ----
    float rl0 = (l0 > 1e-20f) ? (1.0f / l0) : 0.0f;
    float rl1 = (l1 > 1e-20f) ? (1.0f / l1) : 0.0f;
#pragma unroll
    for (int dn8 = 0; dn8 < DN8; dn8++) {
        int d = dn8 * 8 + 2 * tid4;
        int r0 = gid, r1 = gid + 8;
        if (r0 < G) {
            int o_off = (batch * p.q_head + q_head0 + r0) * HEAD_DIM + d;
            p.o[o_off] = __float2bfloat16(Oacc[dn8][0] * rl0);
            p.o[o_off + 1] = __float2bfloat16(Oacc[dn8][1] * rl0);
        }
        if (r1 < G) {
            int o_off = (batch * p.q_head + q_head0 + r1) * HEAD_DIM + d;
            p.o[o_off] = __float2bfloat16(Oacc[dn8][2] * rl1);
            p.o[o_off + 1] = __float2bfloat16(Oacc[dn8][3] * rl1);
        }
    }
}

// ---------------------------------------------------------------------------
// Split-K (FlashDecoding) decode: identical math to the kernel above, but the
// KV sequence is partitioned across gridDim.z blocks so that a decode with only
// batch*kv_head independent tasks can fill all SMs. Each (batch, kv_head, split)
// block computes an UN-normalised partial (Oacc, m, l) over its KV slice; the
// combine kernel below reduces across splits. Fixes the "grid too small"
// bottleneck (0.04 waves/SM → many blocks) for long-context, small-batch decode.
//
// Partial layout (float, contiguous):
//   o_part : [batch, q_head, num_splits, HEAD_DIM]
//   ml_part: [batch, q_head, num_splits, 2]  (m, l)
template <int HEAD_DIM, int BC>
__global__ void gqa_decode_attn_mma_splitk_kernel(GQAParams p,
                                                  float* __restrict__ o_part,
                                                  float* __restrict__ ml_part,
                                                  int num_splits) {
    constexpr int BR = 16;
    constexpr int KD = HEAD_DIM / 16;
    constexpr int NC8 = BC / 8;
    constexpr int KT2 = BC / 16;
    constexpr int DN8 = HEAD_DIM / 8;
    constexpr int LD = HEAD_DIM;
    constexpr int SWIZ_MASK = (HEAD_DIM >= 64) ? 7 : (HEAD_DIM / 8 - 1);

    const int lane = threadIdx.x;
    const int gid = lane >> 2;
    const int tid4 = lane & 3;

    const int kv_head = blockIdx.x;
    const int batch = blockIdx.y;
    const int split = blockIdx.z;
    const int G = p.q_head / p.kv_head;
    const int q_head0 = kv_head * G;

    extern __shared__ __align__(16) bf16 smem[];
    bf16* sK = smem;
    bf16* sV = sK + BC * LD;
    bf16* sQ = sV + BC * LD;

    bf16 scale_bf16 = __float2bfloat16(p.scale);
    for (int i = lane; i < BR * HEAD_DIM; i += 32) {
        int r = i / HEAD_DIM, d = i % HEAD_DIM;
        bf16 val = __float2bfloat16(0.0f);
        if (r < G) {
            int qh = q_head0 + r;
            val = p.q[(batch * p.q_head + qh) * HEAD_DIM + d];
        }
        sQ[r * LD + swiz_col(d, r, SWIZ_MASK)] = __hmul(val, scale_bf16);
    }
    __syncwarp();

    unsigned Qa[KD][4];
    int qrow_l = (lane & 7) + (lane & 8);
    int qcol_l = (lane & 16) ? 8 : 0;
#pragma unroll
    for (int kt = 0; kt < KD; kt++)
        ldmatrix_x4(Qa[kt], &sQ[qrow_l * LD + swiz_col(kt * 16 + qcol_l, qrow_l, SWIZ_MASK)]);

    float Oacc[DN8][4];
#pragma unroll
    for (int j = 0; j < DN8; j++)
        Oacc[j][0] = Oacc[j][1] = Oacc[j][2] = Oacc[j][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;

    const int kv_base = (batch * p.kv_head + kv_head) * p.kv_len * HEAD_DIM;
    const int mask_base = batch * p.kv_len;
    const int tiles_total = (p.kv_len + BC - 1) / BC;
    const int tiles_per_split = (tiles_total + num_splits - 1) / num_splits;
    const int ti_begin = split * tiles_per_split;
    const int ti_end = min(tiles_total, ti_begin + tiles_per_split);
    const int has_mask = p.use_mask && p.mask;

    for (int ti = ti_begin; ti < ti_end; ti++) {
        int kv0 = ti * BC;

        bool full_tile = (kv0 + BC <= p.kv_len);
        if (full_tile) {
            constexpr int VEC = 8;
            int total = BC * HEAD_DIM;
#pragma unroll
            for (int i = lane * VEC; i < total; i += 32 * VEC) {
                int r = i / HEAD_DIM, d = i % HEAD_DIM;
                int kc = kv0 + r;
                cp_async_16(&sK[r * LD + swiz_col(d, r, SWIZ_MASK)],
                            &p.k[kv_base + kc * HEAD_DIM + d]);
                cp_async_16(&sV[r * LD + swiz_col(d, r, SWIZ_MASK)],
                            &p.v[kv_base + kc * HEAD_DIM + d]);
            }
            cp_async_commit();
            cp_async_wait_all();
        } else {
            for (int i = lane; i < BC * HEAD_DIM; i += 32) {
                int r = i / HEAD_DIM, d = i % HEAD_DIM;
                int kc = kv0 + r;
                bf16 z = __float2bfloat16(0.0f);
                sK[r * LD + swiz_col(d, r, SWIZ_MASK)] =
                    (kc < p.kv_len) ? p.k[kv_base + kc * HEAD_DIM + d] : z;
                sV[r * LD + swiz_col(d, r, SWIZ_MASK)] =
                    (kc < p.kv_len) ? p.v[kv_base + kc * HEAD_DIM + d] : z;
            }
        }
        __syncwarp();

        float Sacc[NC8][4];
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

        float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
#pragma unroll
        for (int n8 = 0; n8 < NC8; n8++) {
            int cc = kv0 + n8 * 8 + 2 * tid4;
            bool bc0 = (cc >= p.kv_len) || (has_mask && !p.mask[mask_base + cc]);
            bool bc1 = (cc + 1 >= p.kv_len) || (has_mask && !p.mask[mask_base + cc + 1]);
            bool cz = p.is_causal;
            int off = p.causal_offset;
            bool bad0 = bc0 || (cz && cc > off);
            bool bad1 = bc1 || (cz && (cc + 1) > off);
            float s0 = bad0 ? -FLT_MAX : Sacc[n8][0];
            float s1 = bad1 ? -FLT_MAX : Sacc[n8][1];
            float s2 = bad0 ? -FLT_MAX : Sacc[n8][2];
            float s3 = bad1 ? -FLT_MAX : Sacc[n8][3];
            Sacc[n8][0] = s0; Sacc[n8][1] = s1; Sacc[n8][2] = s2; Sacc[n8][3] = s3;
            rmax0 = fmaxf(rmax0, fmaxf(s0, s1));
            rmax1 = fmaxf(rmax1, fmaxf(s2, s3));
        }
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

        float nm0 = fmaxf(m0, rmax0), nm1 = fmaxf(m1, rmax1);
        float corr0 = (nm0 == -FLT_MAX) ? 1.0f : __expf(m0 - nm0);
        float corr1 = (nm1 == -FLT_MAX) ? 1.0f : __expf(m1 - nm1);

        float rsum0 = 0.0f, rsum1 = 0.0f;
#pragma unroll
        for (int n8 = 0; n8 < NC8; n8++) {
            float p0 = (Sacc[n8][0] == -FLT_MAX) ? 0.0f : __expf(Sacc[n8][0] - nm0);
            float p1 = (Sacc[n8][1] == -FLT_MAX) ? 0.0f : __expf(Sacc[n8][1] - nm0);
            float p2 = (Sacc[n8][2] == -FLT_MAX) ? 0.0f : __expf(Sacc[n8][2] - nm1);
            float p3 = (Sacc[n8][3] == -FLT_MAX) ? 0.0f : __expf(Sacc[n8][3] - nm1);
            Sacc[n8][0] = p0; Sacc[n8][1] = p1; Sacc[n8][2] = p2; Sacc[n8][3] = p3;
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
        __syncwarp();
    }

    // ---- write UN-normalised partials for this split ----
#pragma unroll
    for (int dn8 = 0; dn8 < DN8; dn8++) {
        int d = dn8 * 8 + 2 * tid4;
        int r0 = gid, r1 = gid + 8;
        if (r0 < G) {
            int hh = q_head0 + r0;
            float* op = o_part + ((size_t)(batch * p.q_head + hh) * num_splits + split) * HEAD_DIM;
            op[d] = Oacc[dn8][0];
            op[d + 1] = Oacc[dn8][1];
        }
        if (r1 < G) {
            int hh = q_head0 + r1;
            float* op = o_part + ((size_t)(batch * p.q_head + hh) * num_splits + split) * HEAD_DIM;
            op[d] = Oacc[dn8][2];
            op[d + 1] = Oacc[dn8][3];
        }
    }
    if (tid4 == 0) {
        int r0 = gid, r1 = gid + 8;
        if (r0 < G) {
            float* mp = ml_part + ((size_t)(batch * p.q_head + q_head0 + r0) * num_splits + split) * 2;
            mp[0] = m0; mp[1] = l0;
        }
        if (r1 < G) {
            float* mp = ml_part + ((size_t)(batch * p.q_head + q_head0 + r1) * num_splits + split) * 2;
            mp[0] = m1; mp[1] = l1;
        }
    }
}

// Reduce split-K partials into the final bf16 output. One block per (batch,
// q_head); each thread owns one head_dim element and folds across all splits
// with a numerically-stable online rescale.
__global__ void gqa_decode_combine_kernel(const float* __restrict__ o_part,
                                          const float* __restrict__ ml_part,
                                          bf16* __restrict__ out,
                                          int num_splits, int head_dim) {
    int bh = blockIdx.x;  // batch * q_head + head
    int d = threadIdx.x;
    if (d >= head_dim) return;

    const float* mlp = ml_part + (size_t)bh * num_splits * 2;
    float mstar = -FLT_MAX;
    for (int s = 0; s < num_splits; s++)
        mstar = fmaxf(mstar, mlp[s * 2]);

    float lstar = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        float mi = mlp[s * 2];
        if (mi > -FLT_MAX) lstar += mlp[s * 2 + 1] * __expf(mi - mstar);
    }

    const float* op = o_part + (size_t)bh * num_splits * head_dim;
    float acc = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        float mi = mlp[s * 2];
        if (mi > -FLT_MAX) acc += op[s * head_dim + d] * __expf(mi - mstar);
    }
    float inv = (lstar > 1e-20f) ? (1.0f / lstar) : 0.0f;
    out[(size_t)bh * head_dim + d] = __float2bfloat16(acc * inv);
}