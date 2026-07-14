#pragma once
#include <cfloat>
#include <cuda_bf16.h>
#include "attn_common.h"
#include "attn_mma_utils.cuh"

using bf16 = __nv_bfloat16;

// Split-K (FlashDecoding) tensor-core decode via GQA head-packing.
//
// Decode has q_len == 1, so S = q @ K^T is a GEMV per head — no tensor-core work
// on its own. But GQA gives us G = q_head / kv_head query heads that all share
// one kv_head. We pack those G heads into the M=16 rows of mma.sync.m16n8k16,
// turning G independent GEMVs into a single GEMM that reuses each loaded K/V tile
// across all G heads (K/V load is the decode bottleneck, so the reuse is the win,
// not the flops). The KV sequence is partitioned across gridDim.z blocks so that
// a decode with only batch*kv_head independent tasks can fill all SMs. Each
// (batch, kv_head, split) block computes an UN-normalised partial (Oacc, m, l)
// over its KV slice; the combine kernel below reduces across splits. Fixes the
// "grid too small" bottleneck (0.04 waves/SM → many blocks) for long-context,
// small-batch decode.
//
// Partial layout (float, contiguous):
//   o_part : [batch, q_head, num_splits, HEAD_DIM]
//   ml_part: [batch, q_head, num_splits, 2]  (m, l)
//
// Optimizations:
//   - cp.async global→shared for K/V (bypasses registers, cuts instruction count)
//   - XOR swizzle (swiz_col): LD=HEAD_DIM, zero waste, no bank conflicts
//   - Q loaded directly from global into mma A-operand registers (no sQ staging,
//     no prologue syncwarp) — frees shared memory for double-buffering
//   - Double-buffered KV (STAGES=2): next tile's cp.async overlaps current
//     tile's MMA compute — hides global load latency / boosts bandwidth
//     utilization for small-batch (low-occupancy) decode
//   - Predicated cp.async (cp_async_16_pred) for full AND partial tiles on one
//     uniform path — eliminates the scalar fallback branch
//
// Smem footprint (BC=32): STAGES=2 → 2*(sK+sV) = 2*2*32*HEAD_DIM*2 bytes.
//   D=128: 16 KB (fits 48 KB static cap).  D=256: 32 KB (also fits).
// STAGES=1 fallback (4/8 KB) for smem-constrained configs.
template <int HEAD_DIM, int BC, int STAGES = 2>
__global__ void attn_decode_split_kv_mma_kernel(AttentionParams<bf16> p) {
    constexpr int BR = 16;
    constexpr int KD = HEAD_DIM / 16;
    constexpr int NC8 = BC / 8;
    constexpr int KT2 = BC / 16;
    constexpr int DN8 = HEAD_DIM / 8;
    constexpr int LD = HEAD_DIM;
    constexpr int SWIZ_MASK = (HEAD_DIM >= 64) ? 7 : (HEAD_DIM / 8 - 1);
    constexpr int VEC = 8;
    constexpr int TOTAL = BC * HEAD_DIM;

    const int lane = threadIdx.x;
    const int gid = lane >> 2;
    const int tid4 = lane & 3;

    const int kv_head = blockIdx.x;
    const int batch = blockIdx.y;
    const int split = blockIdx.z;
    const int G = p.q_head / p.kv_head;
    const int q_head0 = kv_head * G;

    // Double-buffered shared memory for K/V (no sQ needed — Q goes direct
    // from global to registers).
    __shared__ __align__(16) bf16 sK[STAGES * BC * LD];
    __shared__ __align__(16) bf16 sV[STAGES * BC * LD];

    // ---- Load Q directly from global into mma A-operand registers ----
    // Same layout as prefill: frag[0]/[2] = row gid, frag[1]/[3] = row gid+8
    // cols kt*16 + tid4*2 + {0,1} / +{8,9}. pau[0]=cols c,c+1; pau[4]=c+8,c+9.
    // Stride-based: Q is [batch, q_head, q_len=1, head_dim]
    const int q_base = batch * p.q_stride_b + q_head0 * p.q_stride_h;
    const int qra = gid;
    const int qrb = gid + 8;
    const bool va = qra < G, vb = qrb < G;
    unsigned Qa[KD][4];
#pragma unroll
    for (int kt = 0; kt < KD; kt++) {
        int c = kt * 16 + tid4 * 2;
        const unsigned* pau = reinterpret_cast<const unsigned*>(
            &p.q[q_base + qra * p.q_stride_h + c * p.q_stride_d]);
        const unsigned* pbu = reinterpret_cast<const unsigned*>(
            &p.q[q_base + qrb * p.q_stride_h + c * p.q_stride_d]);
        Qa[kt][0] = va ? pau[0] : 0u;
        Qa[kt][1] = vb ? pbu[0] : 0u;
        Qa[kt][2] = va ? pau[4] : 0u;
        Qa[kt][3] = vb ? pbu[4] : 0u;
    }

    float Oacc[DN8][4];
#pragma unroll
    for (int j = 0; j < DN8; j++)
        Oacc[j][0] = Oacc[j][1] = Oacc[j][2] = Oacc[j][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;

    // KV: stride-based base — [batch, kv_head, kv_len, head_dim]
    const int kv_base = batch * p.kv_stride_b + kv_head * p.kv_stride_h;
    const int mask_batch_base = batch * p.mask_b_stride;
    const int tiles_total = (p.kv_len + BC - 1) / BC;
    const int tiles_per_split = (tiles_total + p.num_splits - 1) / p.num_splits;
    const int ti_begin = split * tiles_per_split;
    const int ti_end = min(tiles_total, ti_begin + tiles_per_split);
    const int has_mask = p.use_mask && p.mask;

    // ---- Load tile lambda: predicated cp.async, unified full/partial ----
    auto load_tile = [&](int ti, int buf) {
        int kv0 = ti * BC;
        bf16* dK = sK + buf * BC * LD;
        bf16* dV = sV + buf * BC * LD;
#pragma unroll
        for (int i = lane * VEC; i < TOTAL; i += 32 * VEC) {
            int r = i / HEAD_DIM, d = i % HEAD_DIM;
            int kc = kv0 + r;
            bool valid = kc < p.kv_len;
            int off = r * LD + swiz_col(d, r, SWIZ_MASK);
            // KV stride-based: contiguous within head_dim (stride_d == 1 typically)
            int g_off = kv_base + kc * p.kv_stride_l + d * p.kv_stride_d;
            cp_async_16_pred(&dK[off], &p.k[g_off], valid);
            cp_async_16_pred(&dV[off], &p.v[g_off], valid);
        }
        cp_async_commit();
    };

    // ---- Prologue: issue first tile load ----
    if (ti_begin < ti_end) {
        load_tile(ti_begin, 0);
    }

    for (int ti = ti_begin; ti < ti_end; ti++) {
        constexpr int BUF_MASK = (STAGES > 1) ? (STAGES - 1) : 0;
        int buf = (ti - ti_begin) & BUF_MASK;

        // Wait for current tile, then issue next tile's prefetch (overlaps
        // with this tile's compute). Single syncwarp covers both hazards.
        // When STAGES==1, no prefetch — load happens at end of prior iter.
        cp_async_wait_group<0>();
        __syncwarp();
        if constexpr (STAGES > 1) {
            if (ti + 1 < ti_end)
                load_tile(ti + 1, (ti + 1 - ti_begin) & BUF_MASK);
        }

        const bf16* bK = sK + buf * BC * LD;
        const bf16* bV = sV + buf * BC * LD;
        int kv0 = ti * BC;

        float Sacc[NC8][4];
        mma_compute_scores<KD, NC8>(Qa, bK, LD, SWIZ_MASK, lane, Sacc);

        #pragma unroll
        for (int n8 = 0; n8 < NC8; n8++)
            Sacc[n8][0] *= p.scale, Sacc[n8][1] *= p.scale,
            Sacc[n8][2] *= p.scale, Sacc[n8][3] *= p.scale;

        // Decode: q_len=1, so qrow0=qrow1=0, mask_q_stride irrelevant
        int maxc = (p.causal_offset >= 0) ? min(p.kv_len, p.causal_offset + 1) : p.kv_len;
        mma_softmax_tile<NC8, DN8>(kv0, maxc, maxc,
                                    0, 0,
                                    mask_batch_base, 0,
                                    batch,
                                    p.mask, has_mask,
                                    Sacc, Oacc, m0, m1, l0, l1, lane);

        mma_pv_accumulate<DN8, KT2>(Sacc, bV, LD, SWIZ_MASK, lane, Oacc);
        __syncwarp();

        if constexpr (STAGES == 1) {
            if (ti + 1 < ti_end)
                load_tile(ti + 1, 0);
        }
    }

    // ---- write UN-normalised partials for this split ----
    auto split_slot = [&](int h) -> size_t {
        size_t bh = (size_t)batch * p.q_head + h;
        return bh * p.num_splits + split;
    };
#pragma unroll
    for (int dn8 = 0; dn8 < DN8; dn8++) {
        int d = dn8 * 8 + 2 * tid4;
        int r0 = gid, r1 = gid + 8;
        if (r0 < G) {
            int h = q_head0 + r0;
            float* op = p.o_part + split_slot(h) * HEAD_DIM;
            op[d] = Oacc[dn8][0];
            op[d + 1] = Oacc[dn8][1];
        }
        if (r1 < G) {
            int h = q_head0 + r1;
            float* op = p.o_part + split_slot(h) * HEAD_DIM;
            op[d] = Oacc[dn8][2];
            op[d + 1] = Oacc[dn8][3];
        }
    }
    if (tid4 == 0) {
        int r0 = gid, r1 = gid + 8;
        if (r0 < G) {
            int h = q_head0 + r0;
            float* mp = p.ml_part + split_slot(h) * 2;
            mp[0] = m0; mp[1] = l0;
        }
        if (r1 < G) {
            int h = q_head0 + r1;
            float* mp = p.ml_part + split_slot(h) * 2;
            mp[0] = m1; mp[1] = l1;
        }
    }
}
