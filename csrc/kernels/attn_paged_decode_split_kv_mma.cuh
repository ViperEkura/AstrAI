#pragma once
#include <cfloat>
#include <cuda_bf16.h>
#include "attn_common.h"
#include "attn_mma_utils.cuh"

using bf16 = __nv_bfloat16;

// Paged split-KV tensor-core decode via GQA head-packing.
// Identical algorithm to attn_decode_split_kv_mma_kernel but reads K/V
// directly from the page pool through a page table, eliminating the gather
// copy.  Each tile (BC=32) fits within a single page (page_size >= 32), so
// the page-table lookup happens once per tile for cp.async.

template <int HEAD_DIM, int BC, int STAGES = (HEAD_DIM <= 128) ? 2 : 1>
__global__ void paged_attn_decode_split_kv_mma_kernel(PagedAttentionParams<bf16> p) {
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

    __shared__ __align__(16) bf16 sK[STAGES * BC * LD];
    __shared__ __align__(16) bf16 sV[STAGES * BC * LD];

    // ---- Load Q directly from global into mma A-operand registers ----
    const int q_base = batch * p.q_stride_b + q_head0 * p.q_stride_h;
    const int qra = gid;
    const int qrb = gid + 8;
    const bool va = qra < G, vb = qrb < G;
    unsigned Qa[KD][4];
    load_q_mma_frags<KD>(p.q + q_base, p.q_stride_h, p.q_stride_d,
                          qra, qrb, va, vb, tid4, Qa);

    float Oacc[DN8][4];
#pragma unroll
    for (int j = 0; j < DN8; j++)
        Oacc[j][0] = Oacc[j][1] = Oacc[j][2] = Oacc[j][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;

    const int mask_batch_base = batch * p.mask_b_stride;
    const int tiles_total = (p.kv_len + BC - 1) / BC;
    const int tiles_per_split = (tiles_total + p.num_splits - 1) / p.num_splits;
    const int ti_begin = split * tiles_per_split;
    const int ti_end = min(tiles_total, ti_begin + tiles_per_split);
    const int has_mask = p.use_mask && p.mask;

    // Paged strides (constant for the block)
    const int64_t page_stride = (int64_t)p.page_size * p.kv_head * HEAD_DIM;
    const int64_t pos_stride  = (int64_t)p.kv_head * HEAD_DIM;
    const int64_t head_off    = (int64_t)kv_head * HEAD_DIM;

    // ---- Load tile lambda: predicated cp.async, paged addressing ----
    auto load_tile = [&](int ti, int buf) {
        int kv0 = ti * BC;
        bf16* dK = sK + buf * BC * LD;
        bf16* dV = sV + buf * BC * LD;
        int logical_page = kv0 / p.page_size;
        int phys_page = p.page_table[batch * p.max_pages + logical_page];
        bool page_valid = (phys_page >= 0);
#pragma unroll
        for (int i = lane * VEC; i < TOTAL; i += 32 * VEC) {
            int r = i / HEAD_DIM, d = i % HEAD_DIM;
            int kc = kv0 + r;
            bool valid = (kc < p.kv_len) && page_valid;
            int page_off = kc % p.page_size;
            int64_t gmem_base = (int64_t)phys_page * page_stride
                              + (int64_t)page_off * pos_stride
                              + head_off;
            int off = r * LD + swiz_col(d, r, SWIZ_MASK);
            cp_async_16_pred(&dK[off], &p.k_cache[gmem_base + d], valid);
            cp_async_16_pred(&dV[off], &p.v_cache[gmem_base + d], valid);
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
