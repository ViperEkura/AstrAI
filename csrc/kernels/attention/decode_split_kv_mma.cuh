#pragma once
#include <cfloat>
#include <cuda_bf16.h>
#include "common.h"
#include "layout_policies.cuh"
#include "mma_utils.cuh"

namespace astrai {
namespace attention {

// Split-K (FlashDecoding) tensor-core decode via GQA head-packing, unified
// across contiguous and paged (SGLang flat-pool) K/V via the KV template
// parameter.  Decode has q_len == 1, so we pack G = q_head/kv_head query
// heads into the M=16 rows of mma.sync.m16n8k16, turning G independent GEMVs
// into a single GEMM that reuses each loaded K/V tile across all G heads.
//
// KV = ContigKV (dense tensors) or PagedKV (flat pool + req_to_token).
// IsCausal and HasMask are compile-time bools — no runtime branch in the
// inner compute loop.
//
// Traits = KernelTraits<HEAD_DIM, BC=16, WARPS=1, STAGES=2>.
template <typename Traits, typename KV, bool IsCausal, bool HasMask>
__global__ void attn_decode_split_kv_mma_kernel(AttentionParams<bf16> p) {
    const int lane = threadIdx.x;
    const int gid = lane >> 2;
    const int tid4 = lane & 3;

    const int pass = blockIdx.x / p.kv_head;
    const int kv_head = blockIdx.x % p.kv_head;
    const int batch = blockIdx.y;
    const int split = blockIdx.z;

    constexpr int MAX_G = 16;
    const int G_total = p.q_head / p.kv_head;
    const int g_begin = pass * MAX_G;
    const int G = min(MAX_G, G_total - g_begin);
    const int q_head0 = kv_head * G_total + g_begin;

    // Per-request seq_len (paged reads kv_indptr; contig uses p.kv_len).
    const int seq_len = KV::kv_len(p, batch);
    const KVContext kctx = KV::template make_ctx<Traits::HEAD_DIM>(p, batch, kv_head);

    // Double-buffered shared memory for K/V (no sQ needed)
    __shared__ __align__(16) bf16 sK[Traits::STAGES * Traits::BC * Traits::LD];
    __shared__ __align__(16) bf16 sV[Traits::STAGES * Traits::BC * Traits::LD];

    // Load Q directly from global into mma A-operand registers.
    const int q_base = KV::q_decode_base(p, batch, q_head0);
    const int qra = gid;
    const int qrb = gid + 8;
    const bool va = qra < G, vb = qrb < G;
    unsigned Qa[Traits::KD][4];
    load_q_mma_frags<Traits::KD>(p.q_ptr + q_base, p.q_h_stride, p.q_d_stride,
                                  qra, qrb, va, vb, tid4, Qa);

    float Oacc[Traits::DN8][4];
    #pragma unroll
    for (int j = 0; j < Traits::DN8; j++)
        Oacc[j][0] = Oacc[j][1] = Oacc[j][2] = Oacc[j][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;

    const int tiles_total = (seq_len + Traits::BC - 1) / Traits::BC;
    const int tiles_per_split = (tiles_total + p.num_splits - 1) / p.num_splits;
    const int ti_begin = split * tiles_per_split;
    const int ti_end = min(tiles_total, ti_begin + tiles_per_split);

    // ---- Load tile lambda: predicated cp.async (addressing via KV policy;
    // only the first GQA pass persists new K/V to the pool) ----
    auto load_tile = [&](int ti, int buf) {
        load_kv_tile<Traits>(sK, sV, ti, buf, seq_len,
            [&](int kc, int d, bool valid) {
                return KV::template decode_addr<Traits::VEC>(
                    p, kctx, batch, kv_head, kc, d, valid, pass == 0);
            });
    };

    // ---- Multi-stage cp.async pipeline ----
    // Prologue loads STAGES tiles; each loop iteration waits only for the
    // oldest outstanding group (wait_group<STAGES-1>) so the STAGES-1 newer
    // tile loads stay in flight and overlap with the current tile's compute.
    constexpr int STAGES = Traits::STAGES;
    const int ntiles = ti_end - ti_begin;

    auto process_tile = [&](int it, int buf) {
        const bf16* bK = sK + buf * Traits::BC * Traits::LD;
        const bf16* bV = sV + buf * Traits::BC * Traits::LD;
        int kv0 = (ti_begin + it) * Traits::BC;

        float Sacc[Traits::NC8][4];
        mma_compute_scores<Traits>(Qa, bK, p.scale, lane, Sacc);

        // Decode: q_len=1, so qrow0=qrow1=0.  Paged treats [0, seq_len) as
        // the causal range (query is the last token); contig clips to the
        // causal_offset bound.  Dead code eliminated when IsCausal == false.
        int maxc = IsCausal ? KV::decode_attend_len(p, batch) : seq_len;
        mma_softmax_tile<Traits, HasMask>(kv0, maxc, maxc,
                                           0, 0,
                                            p.mask_b_stride, p.mask_h_stride, p.mask_l_stride,
                                            batch, q_head0 + gid, q_head0 + gid + 8,
                                            p.mask,
                                            va, vb,
                                            Sacc, Oacc, m0, m1, l0, l1, lane);

        mma_pv_accumulate<Traits>(Sacc, bV, lane, Oacc);
    };

    if (ntiles >= STAGES) {
        #pragma unroll
        for (int i = 0; i < STAGES; i++)
            load_tile(ti_begin + i, i);

        for (int it = 0; it < ntiles; it++) {
            if (it + 1 == ntiles)
                astrai::cp_async_wait_group<0>();
            else
                astrai::cp_async_wait_group<STAGES - 1>();
            __syncwarp();
            process_tile(it, it & (STAGES - 1));
            __syncwarp();
            if (it + STAGES < ntiles)
                load_tile(ti_begin + it + STAGES, (it + STAGES) & (STAGES - 1));
        }
    } else {
        // Fewer tiles than stages: load all, wait for all, process.
        for (int i = 0; i < ntiles; i++)
            load_tile(ti_begin + i, i);
        astrai::cp_async_wait_all();
        __syncwarp();
        for (int it = 0; it < ntiles; it++)
            process_tile(it, it);
    }

    // ---- write UN-normalised partials for this split ----
    auto split_slot = [&](int h) -> size_t {
        size_t bh = (size_t)batch * p.q_head + h;
        return bh * MAX_SPLITS + split;
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

}  // namespace attention
}  // namespace astrai
