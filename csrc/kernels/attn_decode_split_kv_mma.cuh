#pragma once
#include <cfloat>
#include <cuda_bf16.h>
#include "attn_common.h"
#include "attn_mma_utils.cuh"

// Split-K (FlashDecoding) tensor-core decode via GQA head-packing.
// Decode has q_len == 1, so we pack G = q_head/kv_head query heads into the
// M=16 rows of mma.sync.m16n8k16, turning G independent GEMVs into a single
// GEMM that reuses each loaded K/V tile across all G heads.
//
// IsCausal and HasMask are compile-time bools — no runtime branch in the
// inner compute loop.
//
// Traits = KernelTraits<HEAD_DIM, BC=32, WARPS=1, STAGES=<2 or 1>>.
template <typename Traits, bool IsCausal, bool HasMask>
__global__ void attn_decode_split_kv_mma_kernel(AttentionParams<bf16> p) {
    const int lane = threadIdx.x;
    const int gid = lane >> 2;
    const int tid4 = lane & 3;

    const int kv_head = blockIdx.x;
    const int batch = blockIdx.y;
    const int split = blockIdx.z;
    const int G = p.q_head / p.kv_head;
    const int q_head0 = kv_head * G;

    // Double-buffered shared memory for K/V (no sQ needed)
    __shared__ __align__(16) bf16 sK[Traits::STAGES * Traits::BC * Traits::LD];
    __shared__ __align__(16) bf16 sV[Traits::STAGES * Traits::BC * Traits::LD];

    // Load Q directly from global into mma A-operand registers.
    // stride_row = p.q_stride_h for decode (q_len=1).
    const int q_base = batch * p.q_stride_b + q_head0 * p.q_stride_h;
    const int qra = gid;
    const int qrb = gid + 8;
    const bool va = qra < G, vb = qrb < G;
    unsigned Qa[Traits::KD][4];
    load_q_mma_frags<Traits::KD>(p.q + q_base, p.q_stride_h, p.q_stride_d,
                                  qra, qrb, va, vb, tid4, Qa);

    float Oacc[Traits::DN8][4];
    #pragma unroll
    for (int j = 0; j < Traits::DN8; j++)
        Oacc[j][0] = Oacc[j][1] = Oacc[j][2] = Oacc[j][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;

    const int kv_base = batch * p.kv_stride_b + kv_head * p.kv_stride_h;
    const int tiles_total = (p.kv_len + Traits::BC - 1) / Traits::BC;
    const int tiles_per_split = (tiles_total + p.num_splits - 1) / p.num_splits;
    const int ti_begin = split * tiles_per_split;
    const int ti_end = min(tiles_total, ti_begin + tiles_per_split);

    // ---- Load tile lambda: predicated cp.async ----
    auto load_tile = [&](int ti, int buf) {
        int kv0 = ti * Traits::BC;
        bf16* dK = sK + buf * Traits::BC * Traits::LD;
        bf16* dV = sV + buf * Traits::BC * Traits::LD;
        #pragma unroll
        for (int i = lane * Traits::VEC; i < Traits::TOTAL;
             i += Traits::NUM_THREADS * Traits::VEC) {
            int r = i / Traits::HEAD_DIM, d = i % Traits::HEAD_DIM;
            int kc = kv0 + r;
            bool valid = kc < p.kv_len;
            int off = r * Traits::LD + swiz_col(d, r, Traits::SWIZ_MASK);
            int g_off = kv_base + kc * p.kv_stride_l + d * p.kv_stride_d;
            cp_async_16_pred(&dK[off], &p.k[g_off], valid);
            cp_async_16_pred(&dV[off], &p.v[g_off], valid);
        }
        cp_async_commit();
    };

    constexpr int BUF_MASK = (Traits::STAGES > 1) ? (Traits::STAGES - 1) : 0;

    // Prologue
    if (ti_begin < ti_end) {
        load_tile(ti_begin, 0);
    }

    for (int ti = ti_begin; ti < ti_end; ti++) {
        int buf = (ti - ti_begin) & BUF_MASK;

        cp_async_wait_group<0>();
        __syncwarp();
        if constexpr (Traits::STAGES > 1) {
            if (ti + 1 < ti_end)
                load_tile(ti + 1, (ti + 1 - ti_begin) & BUF_MASK);
        }

        const bf16* bK = sK + buf * Traits::BC * Traits::LD;
        const bf16* bV = sV + buf * Traits::BC * Traits::LD;
        int kv0 = ti * Traits::BC;

        float Sacc[Traits::NC8][4];
        mma_compute_scores<Traits>(Qa, bK, lane, Sacc);

        #pragma unroll
        for (int n8 = 0; n8 < Traits::NC8; n8++)
            Sacc[n8][0] *= p.scale, Sacc[n8][1] *= p.scale,
            Sacc[n8][2] *= p.scale, Sacc[n8][3] *= p.scale;

        // Decode: q_len=1, so qrow0=qrow1=0
        int maxc = IsCausal ? min(p.kv_len, p.causal_offset + 1) : p.kv_len;
        mma_softmax_tile<Traits, HasMask>(kv0, maxc, maxc,
                                           0, 0,
                                           p.mask_b_stride, 0,
                                           batch,
                                           p.mask,
                                           Sacc, Oacc, m0, m1, l0, l1, lane);

        mma_pv_accumulate<Traits>(Sacc, bV, lane, Oacc);
        __syncwarp();

        if constexpr (Traits::STAGES == 1) {
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
    for (int dn8 = 0; dn8 < Traits::DN8; dn8++) {
        int d = dn8 * 8 + 2 * tid4;
        int r0 = gid, r1 = gid + 8;
        if (r0 < G) {
            int h = q_head0 + r0;
            float* op = p.o_part + split_slot(h) * Traits::HEAD_DIM;
            op[d] = Oacc[dn8][0];
            op[d + 1] = Oacc[dn8][1];
        }
        if (r1 < G) {
            int h = q_head0 + r1;
            float* op = p.o_part + split_slot(h) * Traits::HEAD_DIM;
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
