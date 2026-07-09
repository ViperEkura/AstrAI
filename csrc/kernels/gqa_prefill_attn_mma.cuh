#pragma once
#include "gqa_common.cuh"
#include "gqa_mma_utils.cuh"

// Tensor-core prefill flash attention (raw mma.sync PTX).
// One warp owns BR=16 query rows. S = Q@K^T and O = P@V run on bf16 tensor
// cores via mma.sync.m16n8k16 (f32 accumulate). Q is staged in static shared
// memory (sQ) and reloaded per tile via ldmatrix — this avoids keeping KD*4
// fragment registers resident across the tile loop, cutting ~32 regs for
// HEAD_DIM=128 and enabling 4 blocks/SM (33% occupancy, up from 25%). S, O,
// and the online-softmax stats (m, l) live in registers. Shared memory is
// statically sized via template parameters — no dynamic allocation. The mma
// fragment layout is used directly: the S accumulator (f32) maps element-
// for-element onto the P matrix_a (bf16) operand, so softmax needs no shuffle
// repack; row reductions fold across the 4-lane thread group. Templated on
// <HEAD_DIM, WARPS, BC, MIN_BLOCKS> with BC a multiple of 16.
//
// Optimizations: shared sQ staging (single area, serialized per-warp load) with
// per-tile reload → cuts registers; pre-scale Q by attention scale during Q
// load; cp.async global→shared for K/V; scalar fallback only for the last
// partial tile; causal tile skipping (block-level early break + warp-level
// skip); XOR swizzle (swiz_col) → eliminates ldmatrix bank conflicts without
// LD padding (LD=HEAD_DIM).

template <int HEAD_DIM, int WARPS, int BC, int MIN_BLOCKS>
__global__ __launch_bounds__(WARPS * 32, MIN_BLOCKS)
void gqa_prefill_attn_mma_kernel(GQAParams p) {
    constexpr int BR = 16;
    constexpr int KD = HEAD_DIM / 16;  // Q/K k-tiles
    constexpr int NC8 = BC / 8;        // S n-tiles (N=8 each)
    constexpr int KT2 = BC / 16;       // P k-tiles (K=16 each)
    constexpr int DN8 = HEAD_DIM / 8;  // O n-tiles (N=8 each)
    constexpr int LD = HEAD_DIM;   // XOR swizzle (swiz_col) handles bank conflicts
    constexpr int SWIZ_MASK = (HEAD_DIM >= 64) ? 7 : (HEAD_DIM / 8 - 1);  // chunk bits, stay within LD

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gid = lane >> 2;   // 0..7  → rows gid, gid+8
    const int tid4 = lane & 3;   // 0..3
    const int nthreads = WARPS * 32;

    const int q_head = blockIdx.y;
    const int batch = blockIdx.z;
    const int kv_head = q_head / (p.q_head / p.kv_head);
    const int qrow0 = (blockIdx.x * WARPS + warp) * BR;

    // Static shared memory — sized by template parameters at compile time.
    // No extern __shared__ / cudaFuncSetAttribute needed.
    __shared__ __align__(16) bf16 sK[BC * LD];
    __shared__ __align__(16) bf16 sV[BC * LD];
    __shared__ __align__(16) bf16 sQ[BR * LD];

    // Load Q into sQ with pre-scaling (staged per-warp to avoid smem conflicts).
    // Pre-scale by attention scale so softmax doesn't need to multiply later.
    // Q fragments are NOT kept resident — reloaded from sQ each tile via
    // ldmatrix to cut ~KD*4 registers.
    const int q_base = ((batch * p.q_head + q_head) * p.q_len) * HEAD_DIM;
    bf16 scale_bf16 = __float2bfloat16(p.scale);
    int qrow_l = (lane & 7) + (lane & 8);       // 0..15
    int qcol_l = (lane & 16) ? 8 : 0;
    for (int w = 0; w < WARPS; w++) {
        if (warp == w) {
            for (int i = lane; i < BR * HEAD_DIM; i += 32) {
                int r = i / HEAD_DIM, d = i % HEAD_DIM;
                int qr = qrow0 + r;
                bf16 qv = (qr < p.q_len) ? p.q[q_base + qr * HEAD_DIM + d]
                                         : __float2bfloat16(0.0f);
                sQ[r * LD + swiz_col(d, r, SWIZ_MASK)] = __hmul(qv, scale_bf16);
            }
            __syncwarp();
        }
        __syncthreads();  // prevent next warp from overwriting sQ prematurely
    }

    float Oacc[DN8][4];
#pragma unroll
    for (int j = 0; j < DN8; j++)
        Oacc[j][0] = Oacc[j][1] = Oacc[j][2] = Oacc[j][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;

    const int kv_base = ((batch * p.kv_head + kv_head) * p.kv_len) * HEAD_DIM;
    const int tiles = (p.kv_len + BC - 1) / BC;
    const int qr0 = qrow0 + gid;      // row for c0/c1
    const int qr1 = qrow0 + gid + 8;  // row for c2/c3

    // Causal tile-skip bounds (no-op when is_causal == 0)
    const int use_skip = p.is_causal;
    const int max_kv = qrow0 + BR - 1 + p.causal_offset;
    const int block_max_kv =
        blockIdx.x * WARPS * BR + WARPS * BR - 1 + p.causal_offset;
    const int has_mask = p.use_mask && p.mask;
    const int mb = batch * p.kv_len;

    for (int ti = 0; ti < tiles; ti++) {
        int kv0 = ti * BC;

        // Block-level causal early break
        if (use_skip && kv0 > block_max_kv) break;

        // ---- load K/V tile to shared memory (cp.async on full tiles) ----
        bool full_tile = (kv0 + BC <= p.kv_len);
        if (full_tile) {
            constexpr int VEC = 8;  // bf16 per cp.async unit (16 bytes)
            int total = BC * HEAD_DIM;
#pragma unroll
            for (int i = threadIdx.x * VEC; i < total; i += nthreads * VEC) {
                int r = i / HEAD_DIM;
                int d = i % HEAD_DIM;
                int kc = kv0 + r;
                cp_async_16(&sK[r * LD + swiz_col(d, r, SWIZ_MASK)], &p.k[kv_base + kc * HEAD_DIM + d]);
                cp_async_16(&sV[r * LD + swiz_col(d, r, SWIZ_MASK)], &p.v[kv_base + kc * HEAD_DIM + d]);
            }
            cp_async_commit();
            cp_async_wait_all();
        } else {
            for (int i = threadIdx.x; i < BC * HEAD_DIM; i += nthreads) {
                int r = i / HEAD_DIM, d = i % HEAD_DIM;
                int kc = kv0 + r;
                bf16 z = __float2bfloat16(0.0f);
                sK[r * LD + swiz_col(d, r, SWIZ_MASK)] = (kc < p.kv_len)
                                     ? p.k[kv_base + kc * HEAD_DIM + d] : z;
                sV[r * LD + swiz_col(d, r, SWIZ_MASK)] = (kc < p.kv_len)
                                     ? p.v[kv_base + kc * HEAD_DIM + d] : z;
            }
        }
        __syncthreads();

        // Warp-level causal skip
        if (!use_skip || kv0 <= max_kv) {

        // Reload Q fragments from sQ each tile — saves KD*4 resident registers.
        // ldmatrix.sync is warp-cooperative; all 32 lanes execute it together
        // (the if condition is uniform per warp).
        unsigned Qa[KD][4];
#pragma unroll
        for (int kt = 0; kt < KD; kt++)
            ldmatrix_x4(Qa[kt], &sQ[qrow_l * LD + swiz_col(kt * 16 + qcol_l, qrow_l, SWIZ_MASK)]);

        // S = Q @ K^T  → Sacc[n8][0..3]   (n8: 8 kv cols each)
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

        // ---- online softmax (in registers) ----
        // Q is pre-scaled, so Sacc already includes the attention scale.
        int maxc0 = p.is_causal ? min(p.kv_len, qr0 + p.causal_offset + 1)
                                : p.kv_len;
        int maxc1 = p.is_causal ? min(p.kv_len, qr1 + p.causal_offset + 1)
                                : p.kv_len;
        float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
#pragma unroll
        for (int n8 = 0; n8 < NC8; n8++) {
            int cc = kv0 + n8 * 8 + 2 * tid4;
            int c1 = cc + 1;
            bool b0 = (cc >= maxc0) || (has_mask && !p.mask[mb + cc]);
            bool b1 = (c1 >= maxc0) || (has_mask && !p.mask[mb + c1]);
            bool b2 = (cc >= maxc1) || (has_mask && !p.mask[mb + cc]);
            bool b3 = (c1 >= maxc1) || (has_mask && !p.mask[mb + c1]);
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
        float corr0 = (nm0 == -FLT_MAX) ? 1.0f : __expf(m0 - nm0);
        float corr1 = (nm1 == -FLT_MAX) ? 1.0f : __expf(m1 - nm1);

        float rsum0 = 0.0f, rsum1 = 0.0f;
#pragma unroll
        for (int n8 = 0; n8 < NC8; n8++) {
            float p0 = (Sacc[n8][0] == -FLT_MAX) ? 0.0f
                                                 : __expf(Sacc[n8][0] - nm0);
            float p1 = (Sacc[n8][1] == -FLT_MAX) ? 0.0f
                                                 : __expf(Sacc[n8][1] - nm0);
            float p2 = (Sacc[n8][2] == -FLT_MAX) ? 0.0f
                                                 : __expf(Sacc[n8][2] - nm1);
            float p3 = (Sacc[n8][3] == -FLT_MAX) ? 0.0f
                                                 : __expf(Sacc[n8][3] - nm1);
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

        // rescale O accumulator by per-row correction
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
        }  // if active (warp-level causal skip)
        __syncthreads();
    }

    // ---- write output ----
    float rl0 = (l0 > 1e-20f) ? (1.0f / l0) : 0.0f;
    float rl1 = (l1 > 1e-20f) ? (1.0f / l1) : 0.0f;
    const int o_base = ((batch * p.q_head + q_head) * p.q_len) * HEAD_DIM;
#pragma unroll
    for (int dn8 = 0; dn8 < DN8; dn8++) {
        int d = dn8 * 8 + 2 * tid4;
        if (qr0 < p.q_len) {
            p.o[o_base + qr0 * HEAD_DIM + d] =
                __float2bfloat16(Oacc[dn8][0] * rl0);
            p.o[o_base + qr0 * HEAD_DIM + d + 1] =
                __float2bfloat16(Oacc[dn8][1] * rl0);
        }
        if (qr1 < p.q_len) {
            p.o[o_base + qr1 * HEAD_DIM + d] =
                __float2bfloat16(Oacc[dn8][2] * rl1);
            p.o[o_base + qr1 * HEAD_DIM + d + 1] =
                __float2bfloat16(Oacc[dn8][3] * rl1);
        }
    }
}
