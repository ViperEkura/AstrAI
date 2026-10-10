#pragma once

#include <cfloat>
#include <cuda_bf16.h>

#include <api/attention_common.h>
#include <arith/softmax.cuh>
#include <kernel/attention/mma.cuh>
#include <memory/layout_policies.cuh>

namespace astrai {
namespace attention {

/*
 * Tensor-core prefill flash attention, unified across contiguous and paged
 * K/V via the KV template parameter; S = Q@K^T and O = P@V run on
 * mma.sync.m16n8k16 (f32 accumulate), one warp owning 16 packed rows.
 *
 * PackGQA head folding (FA3-style): the block's row space is the packed
 * (head, row) space of one kv-head group — packed idx in [0, G*rows) with
 * h = idx % G, m = idx / G. Every block covers BLOCK_M = BR*WARPS packed
 * rows of one host tile, K/V tiles loaded once per block and shared by all
 * G heads' rows; any G packs with zero idle warps (G=6 no longer wastes 25%
 * of every block). Rows past a head's q_len tail are masked per-row; G=1
 * (MHA) degenerates to the unfolded layout.
 *
 * KV = ContigKV<T> or PagedKV<T> (T = Traits::Elem); IsCausal/HasMask are
 * compile-time bools — dead branches eliminated in the compute loop.
 * Traits = KernelTraits<HEAD_DIM, BC, WARPS=4, STAGES=2, Elem>.
 */
template <typename Traits,
          typename QSchedule,
          typename KV,
          bool IsCausal,
          bool HasMask,
          bool MaskCoversShape = false>
__global__ void attn_prefill_split_q_mma_kernel(const AttentionParams p) {
    using T = typename Traits::Elem;
    using Mma = AttentionMma<Traits>;
    using Layout = typename Traits::FragmentLayout;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gid = Layout::row(lane);         // 0..7
    const int tid4 = Layout::column(lane) / 2; // 0..3

    constexpr int BLOCK_M = Traits::BR * Traits::WARPS; // packed rows per block

    const int G = p.q_head / p.kv_head;
    const int kv_head = blockIdx.y;
    /*
     * scale * log2(e): the exp2 base-change factor folded into every
     * softmax exponent (see arith/softmax.cuh).
     */
    const float scale_log2 = p.scale * LOG2E;

    int batch, packed0;
    QSchedule::map_packed_block(p, BLOCK_M, batch, packed0);

    /*
     * Warp w folds packed rows [warp*BR, (warp+1)*BR) of the block; each mma
     * row maps to (head, row-within-head) = (idx % G, idx / G) over the
     * GLOBAL packed index (a block-local decode would shift the head phase
     * when BLOCK_M % G != 0).
     */
    const int ia = packed0 + warp * Traits::BR + gid;
    const int ib = ia + 8;
    const int h0 = ia % G;
    const int h1 = ib % G;
    const int mra = ia / G; // row within the head
    const int mrb = ib / G;

    // Per-request dims (from KV policy — paged reads kv_indptr/qo_indptr).
    const int seq_len = KV::kv_len(p, batch);
    const int q_len = QSchedule::q_len(p, batch);
    const int query_start = seq_len - q_len;
    const KVContext kctx = KV::template make_ctx<Traits::HEAD_DIM>(p, batch, kv_head);

    /*
     * Static shared memory: double-buffered K/V (Q goes straight to
     * registers in mma A-operand layout).
     */
    __shared__ __align__(16) T sK[Traits::STAGES * Traits::BC * Traits::LD];
    __shared__ __align__(16) T sV[Traits::STAGES * Traits::BC * Traits::LD];

    /*
     * Load Q fragments straight from global into mma A-operand layout.
     * Row b loads through row a's base when both rows sit in the same head
     * (always true when 8 | G); a BR straddling a head boundary (G not a
     * multiple of 8) loads row b through its own base instead.
     */
    const T* __restrict__ q_gmem = static_cast<const T*>(p.q_ptr);
    const bool va = mra < q_len, vb = mrb < q_len;
    const T* qb = (h1 == h0) ? q_gmem + QSchedule::q_base(p, batch, kv_head * G + h0)
                             : q_gmem + QSchedule::q_base(p, batch, kv_head * G + h1);
    typename Traits::QueryFragment Qa;
    Mma::load_query(q_gmem + QSchedule::q_base(p, batch, kv_head * G + h0), qb, p.q_d_stride,
                    mra * p.q_l_stride, mrb * p.q_l_stride, va, vb, tid4, Qa);

    typename Traits::OutputFragment Oacc;
    Mma::clear(Oacc);
    WarpSoftmax<Traits> softmax;
    const float& m0 = softmax.rows[0].m;
    const float& m1 = softmax.rows[1].m;
    const float& l0 = softmax.rows[0].l;
    const float& l1 = softmax.rows[1].l;

    // Causal and explicit-mask bounds are fixed for the warp's query rows.
    const int maxc0 = IsCausal ? min(seq_len, query_start + mra + 1) : seq_len;
    const int maxc1 = IsCausal ? min(seq_len, query_start + mrb + 1) : seq_len;
    const MaskView mask_view{
        p.mask,       p.mask_b_stride, p.mask_h_stride,  p.mask_l_stride,  p.mask_k_len,
        p.mask_q_len, batch,           kv_head * G + h0, kv_head * G + h1, mra,
        mrb};
    const AttentionMask<HasMask, MaskCoversShape> mask{mask_view, maxc0, maxc1, va, vb};

    const int tiles = (seq_len + Traits::BC - 1) / Traits::BC;

    /*
     * Causal tile-skip bounds (dead code when IsCausal == false): the
     * warp-uniform sweep end covers the warp's deepest in-head row; the
     * block bound covers the whole block's packed space for the shared loop.
     */
    const int warp_max_m = (packed0 + (warp + 1) * Traits::BR - 1) / G;
    const int block_max_m = (packed0 + BLOCK_M - 1) / G;

    int t_end = tiles - 1;
    if constexpr (IsCausal) {
        int bt = (block_max_m + query_start) / Traits::BC;
        if (bt < t_end)
            t_end = bt;
    }

    // Load tile via predicated cp.async and KV policy
    auto load_tile = [&](int ti, int buf) {
        KVTileLoader<Traits>::load(sK, sV, ti, buf, seq_len, [&](int kc, int d, bool valid) {
            int token = KV::resolve_token(p, kctx, kc, valid);
            return KV::kv_addr_from_token(p, kctx, token, d);
        });
    };

    // Prologue: issue first tile load
    if (t_end >= 0)
        load_tile(0, 0);

    for (int ti = 0; ti <= t_end; ti++) {
        int buf = ti & 1;

        // Wait for current tile, then publish cross-warp + guard buffer reuse.
        astrai::cp_async_wait_group<0>();
        __syncthreads();
        if (ti < t_end)
            load_tile(ti + 1, (ti + 1) & 1);

        const T* bK = sK + buf * Traits::BC * Traits::LD;
        const T* bV = sV + buf * Traits::BC * Traits::LD;
        int kv0 = ti * Traits::BC;

        // Warp-level causal skip (dead branch eliminated when IsCausal == false)
        if (!IsCausal || kv0 <= warp_max_m + query_start) {

            typename Traits::ScoreFragment Sacc;
            Mma::scores(Qa, bK, lane, Sacc);

            softmax.update(kv0, scale_log2, Sacc, Oacc, lane, mask);

            Mma::values(Sacc, bV, lane, Oacc);
        }
    }

    // Write packed element-pair output
    float rl0 = (l0 > 1e-20f) ? (1.0f / l0) : 0.0f;
    float rl1 = (l1 > 1e-20f) ? (1.0f / l1) : 0.0f;
    T* __restrict__ o_gmem = static_cast<T*>(p.o_ptr);
    const int o_base0 = QSchedule::q_base(p, batch, kv_head * G + h0);
    const int o_base1 = (h1 == h0) ? o_base0 : QSchedule::q_base(p, batch, kv_head * G + h1);
#pragma unroll
    for (int dn8 = 0; dn8 < Traits::DN8; dn8++) {
        int d = dn8 * 8 + 2 * tid4;
        if (va) {
            astrai::store2<T>(o_gmem + o_base0 + mra * p.q_l_stride + d * p.q_d_stride,
                              Oacc[dn8][0] * rl0, Oacc[dn8][1] * rl0);
        }
        if (vb) {
            astrai::store2<T>(o_gmem + o_base1 + mrb * p.q_l_stride + d * p.q_d_stride,
                              Oacc[dn8][2] * rl1, Oacc[dn8][3] * rl1);
        }
    }
}

} // namespace attention
} // namespace astrai
