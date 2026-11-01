#pragma once

#include <cfloat>
#include <cuda_bf16.h>

#include <api/native/attention.h>
#include <arith/softmax.cuh>
#include <kernel/attention/kernel.cuh>
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
template <typename Tile,
          typename QSchedule,
          typename KV,
          bool IsCausal,
          bool HasMask,
          bool MaskCoversShape = false>
struct PrefillMainloop {
    using Traits = Tile;
    static DEVICE_FORCEINLINE bool
    run(const AttentionParams& p, AttentionStorage<Traits>& storage, AttentionTile<Traits>& tile) {
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
        // Fold scale * log2(e) into the softmax exponent.
        const float scale_log2 = p.scale * LOG2E;

        const int q_block = QSchedule::template order_work<IsCausal>(p, blockIdx.x, gridDim.x);
        int batch, packed0;
        QSchedule::map_packed_block(p, BLOCK_M, q_block, batch, packed0);

        // Map packed GQA rows to their request-local head and query positions.
        const int ia = packed0 + warp * Traits::BR + gid;
        const int ib = ia + 8;
        const int h0 = ia % G;
        const int h1 = ib % G;
        const int mra = ia / G; // row within the head
        const int mrb = ib / G;

        // Per-request dims (from KV policy — paged reads kv_indptr/qo_indptr).
        const int seq_len = KV::kv_len(p, batch);
        const int q_len = QSchedule::q_len(p, batch);
        if constexpr (KV::kPaged) {
            // The host tile map pads the last query tile of each request.
            if (packed0 >= G * q_len)
                return false;
        }
        const int query_start = seq_len - q_len;
        const auto kctx = KV::template make_ctx<Traits::HEAD_DIM>(p, batch, kv_head);

        // Q stays in registers; K/V share a double buffer.
        T* sK = storage.k;
        T* sV = storage.v;

        // Reuse a Q base within one head; GQA row pairs may cross a head boundary.
        const T* __restrict__ q_gmem = static_cast<const T*>(p.q_ptr);
        const bool va = mra < q_len, vb = mrb < q_len;
        const T* qb = (h1 == h0) ? q_gmem + QSchedule::q_base(p, batch, kv_head * G + h0)
                                 : q_gmem + QSchedule::q_base(p, batch, kv_head * G + h1);
        typename Traits::QueryFragment Qa;
        Mma::load_query(q_gmem + QSchedule::q_base(p, batch, kv_head * G + h0), qb, p.q_d_stride,
                        mra * p.q_l_stride, mrb * p.q_l_stride, va, vb, tid4, Qa);

        auto& Oacc = tile.output;
        Mma::clear(Oacc);
        auto& softmax = tile.softmax;

        // Causal and explicit-mask bounds are fixed for the warp's query rows.
        const int maxc0 = IsCausal ? min(seq_len, query_start + mra + 1) : seq_len;
        const int maxc1 = IsCausal ? min(seq_len, query_start + mrb + 1) : seq_len;
        const AttentionRows rows{batch, kv_head * G + h0, kv_head * G + h1, mra, mrb};
        const AttentionMask<HasMask, MaskCoversShape> mask{p, rows, maxc0, maxc1, va, vb};

        const int tiles = (seq_len + Traits::BC - 1) / Traits::BC;

        // Bound the shared loop by the block's deepest visible query.
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

        tile.rows = rows;
        tile.valid0 = va;
        tile.valid1 = vb;
        return true;
    }
};

/*
 * Split-K (FlashDecoding) tensor-core decode, unified across contiguous
 * and paged K/V via the KV template parameter. Decode has q_len == 1, so
 * the G = q_head/kv_head query heads pack into the M=16 rows of
 * mma.sync.m16n8k16, turning G independent GEMVs into one GEMM reusing
 * each K/V tile across all G heads (head-packing details in PrefillMainloop above).
 *
 * KV = ContigKV or PagedKV; HasMask is compile-time. Traits =
 * KernelTraits<HEAD_DIM, BC=16, WARPS=1, STAGES=2, Elem>.
 */
template <typename Tile, typename KV, bool HasMask, bool SkipEmptySplits = false>
struct DecodeMainloop {
    using Traits = Tile;
    static DEVICE_FORCEINLINE bool
    run(const AttentionParams& p, AttentionStorage<Traits>& storage, AttentionTile<Traits>& tile) {
        using T = typename Traits::Elem;
        using Mma = AttentionMma<Traits>;
        using Layout = typename Traits::FragmentLayout;

        const int lane = threadIdx.x;
        const int gid = Layout::row(lane);
        const int tid4 = Layout::column(lane) / 2;

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
        const auto kctx = KV::template make_ctx<Traits::HEAD_DIM>(p, batch, kv_head);
        // Fold scale * log2(e) into the softmax exponent.
        const float scale_log2 = p.scale * LOG2E;

        // Double-buffered shared memory for K/V (no sQ needed)
        T* sK = storage.k;
        T* sV = storage.v;

        // Decode packs GQA query heads into the MMA row dimension.
        const T* __restrict__ q_gmem = static_cast<const T*>(p.q_ptr);
        const int q_base = KV::q_base(p, batch, q_head0);
        const int qra = gid;
        const int qrb = gid + 8;
        const bool va = qra < G, vb = qrb < G;
        typename Traits::QueryFragment Qa;
        Mma::load_query(q_gmem + q_base, q_gmem + q_base, p.q_d_stride, qra * p.q_h_stride,
                        qrb * p.q_h_stride, va, vb, tid4, Qa);

        auto& Oacc = tile.output;
        Mma::clear(Oacc);
        auto& softmax = tile.softmax;

        // Visibility is fixed for these query rows across all K/V tiles.
        const AttentionRows rows{batch, q_head0 + gid, q_head0 + gid + 8, 0, 0};
        const AttentionMask<HasMask> mask{p, rows, seq_len, seq_len, va, vb};

        const int tiles_total = (seq_len + Traits::BC - 1) / Traits::BC;
        const int tiles_per_split = (tiles_total + p.num_splits - 1) / p.num_splits;
        const int ti_begin = split * tiles_per_split;
        const int ti_end = min(tiles_total, ti_begin + tiles_per_split);
        if constexpr (SkipEmptySplits && KV::kPaged) {
            // Skip empty splits only when a graph reuses much less than its capacity.
            if (skip_empty_decode_splits(p, seq_len) && split > 0 && ti_begin >= tiles_total)
                return false;
        }

        // Fetch one page per lane ahead of the K/V copies, then share it across columns.
        auto prefetch_tokens = [&](int ti) {
            if constexpr (KV::kPaged) {
                const int kc = ti * Traits::BC + lane;
                return KV::resolve_token(p, kctx, kc, lane < Traits::BC && kc < seq_len);
            }
            return 0;
        };
        auto load_tile = [&](int ti, int buf, int tokens) {
            KVTileLoader<Traits>::load(sK, sV, ti, buf, seq_len, [&](int kc, int d, bool valid) {
                int token = valid ? kc : -1;
                if constexpr (KV::kPaged)
                    token = __shfl_sync(0xffffffff, tokens, kc % Traits::BC);
                return KV::template decode_addr<Traits::VEC>(p, kctx, batch, kv_head, seq_len, kc,
                                                             d, token, pass == 0);
            });
        };

        // Wait for the oldest copy group while newer loads overlap compute.
        constexpr int STAGES = Traits::STAGES;
        const int ntiles = ti_end - ti_begin;

        auto process_tile = [&](int it, int buf) {
            const T* bK = sK + buf * Traits::BC * Traits::LD;
            const T* bV = sV + buf * Traits::BC * Traits::LD;
            int kv0 = (ti_begin + it) * Traits::BC;

            typename Traits::ScoreFragment Sacc;
            Mma::scores(Qa, bK, lane, Sacc);

            softmax.update(kv0, scale_log2, Sacc, Oacc, lane, mask);

            Mma::values(Sacc, bV, lane, Oacc);
        };

        if (ntiles >= STAGES) {
#pragma unroll
            for (int i = 0; i < STAGES; i++)
                load_tile(ti_begin + i, i, prefetch_tokens(ti_begin + i));

            for (int it = 0; it < ntiles; it++) {
                const int next = ti_begin + it + STAGES;
                const int tokens = next < ti_end ? prefetch_tokens(next) : -1;
                if (it + 1 == ntiles)
                    astrai::cp_async_wait_group<0>();
                else
                    astrai::cp_async_wait_group<STAGES - 1>();
                __syncwarp();
                process_tile(it, it & (STAGES - 1));
                __syncwarp();
                if (it + STAGES < ntiles)
                    load_tile(next, (it + STAGES) & (STAGES - 1), tokens);
            }
        } else {
            // Fewer tiles than stages: load all, wait for all, process.
            for (int i = 0; i < ntiles; i++)
                load_tile(ti_begin + i, i, prefetch_tokens(ti_begin + i));
            astrai::cp_async_wait_all();
            __syncwarp();
            for (int it = 0; it < ntiles; it++)
                process_tile(it, it);
        }

        tile.rows = rows;
        tile.valid0 = va;
        tile.valid1 = vb;
        return true;
    }
};

} // namespace attention
} // namespace astrai
