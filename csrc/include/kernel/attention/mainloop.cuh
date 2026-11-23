#pragma once

#include <cfloat>
#include <cuda_bf16.h>

#include <api/native/attention.h>
#include <arith/softmax.cuh>
#include <kernel/attention/kernel.cuh>
#include <kernel/attention/mma.cuh>
#include <memory/layout_policies.cuh>

namespace astrai {
namespace attention {

// A CTA shares K/V across its warps. Within a KV head, successive packed
// rows span the query heads of one token before advancing to the next token.
template <typename Tile,
          typename QSchedule,
          typename KV,
          bool IsCausal,
          bool HasMask,
          bool MaskCoversShape = false,
          bool FixedSplits = false>
struct PrefillMainloop {
    using Traits = Tile;
    static DEVICE_FORCEINLINE bool
    run(const AttentionParams& p, AttentionStorage<Traits>& storage, AttentionTile<Traits>& tile) {
        using T = typename Traits::Elem;
        using Mma = AttentionMma<Traits>;
        using Layout = typename Traits::FragmentLayout;

        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        constexpr int BLOCK_M = Traits::BR * Traits::WARPS; // packed rows per block

        const int G = p.q_head / p.kv_head;
        const int kv_head = blockIdx.y;
        // Fold scale * log2(e) into the softmax exponent.
        const float scale_log2 = p.scale * LOG2E;

        const int q_block = QSchedule::template order_work<IsCausal>(p, blockIdx.x, gridDim.x);
        int batch, packed0;
        QSchedule::map_packed_block(p, BLOCK_M, q_block, batch, packed0);

        // The storage and query policies supply request-local lengths.
        const int seq_len = KV::kv_len(p, batch);
        const int q_len = QSchedule::q_len(p, batch);
        if constexpr (KV::kPaged || std::is_same_v<QSchedule, VarlenQSchedule>) {
            // The final work tile may extend past this query segment.
            if (packed0 >= G * q_len)
                return false;
        }
        const int warp_begin = packed0 + warp * Traits::BR;
        tile.query =
            QueryRows::prefill<Layout, QSchedule>(p, batch, warp_begin, kv_head, G, q_len, lane);
        typename Traits::QueryFragment query;
        Mma::load_query(p, tile.query, lane, query);

        const VisibleKeys<IsCausal> keys{seq_len, q_len};
        const AttentionMask<Layout, HasMask, MaskCoversShape> mask{p, tile.query, keys, lane};
        const int full_tiles = keys.count(packed0 / G) / Traits::BC;
        const int warp_end = keys.count((warp_begin + Traits::BR - 1) / G);
        const int block_end = keys.count((packed0 + BLOCK_M - 1) / G);
        const int tiles = block_end / Traits::BC + (block_end % Traits::BC != 0);
        const auto kctx = KV::template make_ctx<Traits::HEAD_DIM>(p, batch, kv_head);

        // Load tile via predicated cp.async and KV policy
        auto load_tile = [&](int ti, int buf) {
            KVTileLoader<Traits>::load(storage.k, storage.v, ti, buf, seq_len,
                                       [&](int kc, int d, bool valid) {
                                           int token = KV::resolve_token(p, kctx, kc, valid);
                                           return KV::kv_addr_from_token(p, kctx, token, d);
                                       });
        };

        auto run_tiles = [&](int begin, int end, AttentionAccumulator<Traits>& accum) {
            if (begin >= end)
                return;
            load_tile(begin, begin & 1);
            for (int ti = begin; ti < end; ++ti) {
                const int buf = ti & 1;
                astrai::cp_async_wait_group<0>();
                __syncthreads();
                if (ti + 1 < end)
                    load_tile(ti + 1, (ti + 1) & 1);

                const T* bK = storage.k + buf * Traits::BC * Traits::LD;
                const T* bV = storage.v + buf * Traits::BC * Traits::LD;
                const int kv0 = ti * Traits::BC;
                if (!IsCausal || kv0 < warp_end) {
                    typename Traits::ScoreFragment scores;
                    Mma::scores(query, bK, lane, scores);
                    if (HasMask || ti >= full_tiles)
                        mask.apply(scores, kv0);
                    // Fixed segments reproduce the decode reduction order for replay.
                    constexpr auto reduction =
                        FixedSplits ? SumReduction::PerTile : SumReduction::Deferred;
                    accum.template update_softmax<reduction>(scores, scale_log2);
                    Mma::values(scores, bV, lane, accum.output);
                }
            }
        };

        if constexpr (FixedSplits) {
            const int split_tiles = p.kv_split_size / Traits::BC;
            if (tiles > split_tiles) {
                for (int begin = 0; begin < tiles; begin += split_tiles) {
                    if (begin)
                        __syncthreads(); // Drain readers before reusing the K/V buffers.
                    AttentionAccumulator<Traits> partial;
                    run_tiles(begin, min(begin + split_tiles, tiles), partial);
                    tile.accum.merge(partial, scale_log2);
                }
                return true;
            }
        }
        run_tiles(0, tiles, tile.accum);
        if constexpr (!FixedSplits)
            tile.accum.finish_sum();
        return true;
    }
};

// Each decode CTA processes a KV split for up to one MMA tile of query heads.
template <typename Tile, typename KV, bool HasMask, bool FixedSplits = false>
struct DecodeMainloop {
    using Traits = Tile;
    static DEVICE_FORCEINLINE bool
    run(const AttentionParams& p, AttentionStorage<Traits>& storage, AttentionTile<Traits>& tile) {
        using T = typename Traits::Elem;
        using Mma = AttentionMma<Traits>;
        using Layout = typename Traits::FragmentLayout;

        const int lane = threadIdx.x;
        const int pass = blockIdx.x / p.kv_head;
        const int kv_head = blockIdx.x % p.kv_head;
        const int batch = blockIdx.z;
        const int split = blockIdx.y;

        constexpr int MAX_G = Traits::BR;
        const int G_total = p.q_head / p.kv_head;
        const int g_begin = pass * MAX_G;
        const int G = min(MAX_G, G_total - g_begin);
        const int q_head0 = kv_head * G_total + g_begin;

        // Decode uses the materialized prefix plus the appended token.
        const int seq_len = KV::kv_len(p, batch);
        const auto kctx = KV::template make_ctx<Traits::HEAD_DIM>(p, batch, kv_head);
        // Fold scale * log2(e) into the softmax exponent.
        const float scale_log2 = p.scale * LOG2E;

        // Double-buffered shared memory for K/V (no sQ needed)
        T* sK = storage.k;
        T* sV = storage.v;

        // A group of up to eight heads occupies only the first row fragment per lane.
        const int row_count = G <= Layout::kRowGroup ? 1 : 2;
        tile.query = QueryRows::decode<Layout, KV>(p, batch, q_head0, G, lane);
        typename Traits::QueryFragment query;
        Mma::load_query(p, tile.query, lane, query);
        const AttentionMask<Layout, HasMask> mask{p, tile.query, VisibleKeys<false>{seq_len, 1},
                                                  lane};

        const int tiles_total = (seq_len + Traits::BC - 1) / Traits::BC;
        // A KV tile belongs to the same split across batch shapes and graph capacities.
        const int chunk = FixedSplits ? p.kv_split_size / Traits::BC : 1;
        const int first = split * chunk;
        if (first >= tiles_total)
            return false;
        const int stride = FixedSplits ? 1 : MAX_SPLITS;
        const int ntiles = FixedSplits ? min(chunk, tiles_total - first)
                                       : 1 + (tiles_total - first - 1) / stride;
        auto tile_index = [&](int i) { return first + i * stride; };

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

        auto process_tile = [&](int it, int buf) {
            const T* bK = sK + buf * Traits::BC * Traits::LD;
            const T* bV = sV + buf * Traits::BC * Traits::LD;
            int kv0 = tile_index(it) * Traits::BC;

            typename Traits::ScoreFragment scores;
            Mma::scores(query, bK, lane, scores);
            if (HasMask || seq_len - kv0 < Traits::BC)
                mask.apply(scores, kv0);
            tile.accum.update_softmax(scores, scale_log2, row_count);
            Mma::values(scores, bV, lane, tile.accum.output);
        };

        if (ntiles >= STAGES) {
#pragma unroll
            for (int i = 0; i < STAGES; i++)
                load_tile(tile_index(i), i, prefetch_tokens(tile_index(i)));

            for (int it = 0; it < ntiles; it++) {
                const int next = it + STAGES;
                const int tokens = next < ntiles ? prefetch_tokens(tile_index(next)) : -1;
                if (it + 1 == ntiles)
                    astrai::cp_async_wait_group<0>();
                else
                    astrai::cp_async_wait_group<STAGES - 1>();
                __syncwarp();
                process_tile(it, it & (STAGES - 1));
                __syncwarp();
                if (it + STAGES < ntiles)
                    load_tile(tile_index(next), (it + STAGES) & (STAGES - 1), tokens);
            }
        } else {
            // Fewer tiles than stages: load all, wait for all, process.
            for (int i = 0; i < ntiles; i++)
                load_tile(tile_index(i), i, prefetch_tokens(tile_index(i)));
            astrai::cp_async_wait_all();
            __syncwarp();
            for (int it = 0; it < ntiles; it++)
                process_tile(it, it);
        }

        return true;
    }
};

} // namespace attention
} // namespace astrai
