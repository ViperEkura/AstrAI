#pragma once

#include <cfloat>
#include <cuda_runtime.h>

#include <api/native/attention.h>
#include <arith/softmax.cuh>

namespace astrai::attention {

// Keep the extra split bookkeeping on large graph capacities only. The device
// skips partials when the replayed request uses under one eighth of the table.
constexpr int SKIP_EMPTY_SPLITS_MIN_CAPACITY = 8192;
constexpr int SKIP_EMPTY_SPLITS_CAPACITY_RATIO = 8;

DEVICE_FORCEINLINE bool skip_empty_decode_splits(const AttentionParams& p, int seq_len) {
    return seq_len < p.kv_len / SKIP_EMPTY_SPLITS_CAPACITY_RATIO;
}

// Each lane holds two rows of an m16n8 accumulator. Coordinates are local to the request.
struct QueryRows {
    struct Row {
        int head, token, offset;
        bool valid;
    };
    int batch;
    Row row[2];

    template <class Layout, class Addressing>
    static DEVICE_FORCEINLINE QueryRows prefill(const AttentionParams& p, int batch, int warp_begin,
                                               int kv_head, int group_size, int q_len, int lane) {
        QueryRows q{batch, {}};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int packed = warp_begin + Layout::row(lane, r);
            const int token = packed / group_size;
            const int head = kv_head * group_size + packed % group_size;
            const bool valid = token < q_len;
            const int offset = valid ? Addressing::q_base(p, batch, head) + token * p.q_l_stride : 0;
            q.row[r] = {head, token, offset, valid};
        }
        return q;
    }

    template <class Layout, class Addressing>
    static DEVICE_FORCEINLINE QueryRows decode(const AttentionParams& p, int batch, int first_head,
                                              int head_count, int lane) {
        QueryRows q{batch, {}};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int local_head = Layout::row(lane, r);
            const int head = first_head + local_head;
            const bool valid = local_head < head_count;
            q.row[r] = {head, 0, valid ? Addressing::q_base(p, batch, head) : 0, valid};
        }
        return q;
    }

};

// Bottom-right causality aligns the last query with the last key.
template <bool Causal> struct VisibleKeys {
    int kv_len, q_len;
    DEVICE_FORCEINLINE int count(int token) const {
        if constexpr (Causal) {
            const int later_queries = max(0, q_len - 1 - token);
            return max(0, kv_len - later_queries);
        }
        return kv_len;
    }
};

template <class Layout, bool HasMask, bool MaskCoversShape = false> struct AttentionMask {
    const bool* mask;
    int base[2], end[2], columns[2];

    template <bool Causal>
    DEVICE_FORCEINLINE AttentionMask(const AttentionParams& p, const QueryRows& q,
                                     VisibleKeys<Causal> keys, int lane)
        : mask(p.mask),
          columns{Layout::column(lane), Layout::column(lane, 1)} {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const auto& row = q.row[r];
            end[r] = row.valid ? keys.count(row.token) : 0;
            if constexpr (HasMask) {
                if constexpr (!MaskCoversShape) {
                    end[r] = p.mask_q_size == 1 || row.token < p.mask_q_size
                                 ? min(end[r], p.mask_k_size) : 0;
                }
                base[r] = end[r] ? q.batch * p.mask_b_stride + row.head * p.mask_h_stride +
                                       row.token * p.mask_q_stride : 0;
            }
        }
    }

    template <int N>
    DEVICE_FORCEINLINE void apply(float (&scores)[N][4], int kv0) const {
#pragma unroll
        for (int n = 0; n < N; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int key = kv0 + Layout::kN * n + columns[i & 1];
                const int r = i / 2;
                if (key >= end[r] || (HasMask && !mask[base[r] + key]))
                    scores[n][i] = -FLT_MAX;
            }
        }
    }
};

template <typename Traits> struct AttentionStorage {
    alignas(16) typename Traits::Elem k[Traits::STAGES * Traits::BC * Traits::LD];
    alignas(16) typename Traits::Elem v[Traits::STAGES * Traits::BC * Traits::LD];
};

template <typename Traits> struct AttentionTile {
    QueryRows query;
    AttentionAccumulator<Traits> accum;
};

// Compile-time composition: the mainloop computes a tile; the epilogue stores it.
template <typename Mainloop, typename Epilogue>
__global__ void attention_kernel(const AttentionParams p) {
    using Traits = typename Mainloop::Traits;
    __shared__ AttentionStorage<Traits> storage;
    AttentionTile<Traits> tile;
    if (Mainloop::run(p, storage, tile))
        Epilogue::store(p, tile);
}

} // namespace astrai::attention
