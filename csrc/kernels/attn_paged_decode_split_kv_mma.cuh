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
template <int HEAD_DIM, int BC>
__global__ void paged_attn_decode_split_kv_mma_kernel(PagedAttentionParams<bf16> p) {
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

    const int kv_head_idx = blockIdx.x;
    const int batch = blockIdx.y;
    const int split = blockIdx.z;
    const int G = p.q_head / p.kv_head;
    const int q_head0 = kv_head_idx * G;

    __shared__ __align__(16) bf16 sK[BC * HEAD_DIM];
    __shared__ __align__(16) bf16 sV[BC * HEAD_DIM];
    __shared__ __align__(16) bf16 sQ[BR * HEAD_DIM];

    // ---- load Q into registers via ldmatrix ----
    for (int i = lane; i < BR * HEAD_DIM; i += 32) {
        int r = i / HEAD_DIM, d = i % HEAD_DIM;
        bf16 val = __float2bfloat16(0.0f);
        if (r < G) {
            int qh = q_head0 + r;
            val = p.q[(batch * p.q_head + qh) * HEAD_DIM + d];
        }
        sQ[r * LD + swiz_col(d, r, SWIZ_MASK)] = val;
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

    const int mask_base = batch * p.kv_len;
    const int tiles_total = (p.kv_len + BC - 1) / BC;
    const int tiles_per_split = (tiles_total + p.num_splits - 1) / p.num_splits;
    const int ti_begin = split * tiles_per_split;
    const int ti_end = min(tiles_total, ti_begin + tiles_per_split);
    const int has_mask = p.use_mask && p.mask;

    // Paged strides (constant for the block)
    const int64_t page_stride = (int64_t)p.page_size * p.kv_head * HEAD_DIM;
    const int64_t pos_stride  = (int64_t)p.kv_head * HEAD_DIM;
    const int64_t head_off    = (int64_t)kv_head_idx * HEAD_DIM;

    for (int ti = ti_begin; ti < ti_end; ti++) {
        int kv0 = ti * BC;

        // phys_page is constant for the whole tile (BC <= page_size).
        int logical_page = kv0 / p.page_size;
        int phys_page = p.page_table[batch * p.max_pages + logical_page];
        bool page_valid = (phys_page >= 0);

        bool full_tile = page_valid && (kv0 + BC <= p.kv_len);
        if (full_tile) {
            constexpr int VEC = 8;
            int total = BC * HEAD_DIM;
            #pragma unroll
            for (int i = lane * VEC; i < total; i += 32 * VEC) {
                int r = i / HEAD_DIM, d = i % HEAD_DIM;
                int kc = kv0 + r;
                int page_off = kc % p.page_size;
                int64_t gmem_base = (int64_t)phys_page * page_stride
                                  + (int64_t)page_off * pos_stride
                                  + head_off;
                cp_async_16(&sK[r * LD + swiz_col(d, r, SWIZ_MASK)],
                            &p.k_cache[gmem_base + d]);
                cp_async_16(&sV[r * LD + swiz_col(d, r, SWIZ_MASK)],
                            &p.v_cache[gmem_base + d]);
            }
            cp_async_commit();
            cp_async_wait_all();
        } else {
            for (int i = lane; i < BC * HEAD_DIM; i += 32) {
                int r = i / HEAD_DIM, d = i % HEAD_DIM;
                int kc = kv0 + r;
                bf16 z = __float2bfloat16(0.0f);
                if (kc < p.kv_len && page_valid) {
                    int page_off = kc % p.page_size;
                    int64_t gmem_base = (int64_t)phys_page * page_stride
                                      + (int64_t)page_off * pos_stride
                                      + head_off;
                    sK[r * LD + swiz_col(d, r, SWIZ_MASK)] = p.k_cache[gmem_base + d];
                    sV[r * LD + swiz_col(d, r, SWIZ_MASK)] = p.v_cache[gmem_base + d];
                } else {
                    sK[r * LD + swiz_col(d, r, SWIZ_MASK)] = z;
                    sV[r * LD + swiz_col(d, r, SWIZ_MASK)] = z;
                }
            }
        }
        __syncwarp();

        float Sacc[NC8][4];
        mma_compute_scores<KD, NC8>(Qa, sK, LD, SWIZ_MASK, lane, Sacc);

        #pragma unroll
        for (int n8 = 0; n8 < NC8; n8++)
            Sacc[n8][0] *= p.scale, Sacc[n8][1] *= p.scale,
            Sacc[n8][2] *= p.scale, Sacc[n8][3] *= p.scale;

        int maxc = p.is_causal ? min(p.kv_len, p.causal_offset + 1) : p.kv_len;
        mma_softmax_tile<NC8, DN8>(kv0, maxc, maxc,
                                    mask_base, p.mask, has_mask,
                                    Sacc, Oacc, m0, m1, l0, l1, lane);

        mma_pv_accumulate<DN8, KT2>(Sacc, sV, LD, SWIZ_MASK, lane, Oacc);
        __syncwarp();
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
