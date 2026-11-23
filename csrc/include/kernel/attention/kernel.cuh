#pragma once

#include <cfloat>
#include <cuda_runtime.h>

#include <api/native/attention.h>
#include <arith/softmax.cuh>

namespace astrai::attention {

// Each lane holds two rows of an m16n8 accumulator. Coordinates are local to the request.
struct QueryRows {
    struct Row {
        int head, token, offset;
        bool valid;
    };
    int batch;
    Row row[2];

    template <class Layout, class Addressing>
    static DEVICE_FORCEINLINE QueryRows prefill(const AttentionParams& p,
                                                int batch,
                                                int warp_begin,
                                                int kv_head,
                                                int group_size,
                                                int q_len,
                                                int lane) {
        QueryRows q{batch, {}};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int packed = warp_begin + Layout::row(lane, r);
            const int token = packed / group_size;
            const int head = kv_head * group_size + packed % group_size;
            const bool valid = token < q_len;
            const int offset =
                valid ? Addressing::q_base(p, batch, head) + token * p.q_l_stride : 0;
            q.row[r] = {head, token, offset, valid};
        }
        return q;
    }

    template <class Layout, class Addressing>
    static DEVICE_FORCEINLINE QueryRows
    decode(const AttentionParams& p, int batch, int first_head, int head_count, int lane) {
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
template <bool Causal>
struct VisibleKeys {
    int kv_len, q_len;
    DEVICE_FORCEINLINE int count(int token) const {
        if constexpr (Causal) {
            const int later_queries = max(0, q_len - 1 - token);
            return max(0, kv_len - later_queries);
        }
        return kv_len;
    }
};

template <class Layout, bool HasMask, bool MaskCoversShape = false>
struct AttentionMask {
    const bool* mask;
    int base[2], end[2], columns[2];

    template <bool Causal>
    DEVICE_FORCEINLINE
    AttentionMask(const AttentionParams& p, const QueryRows& q, VisibleKeys<Causal> keys, int lane)
        : mask(p.mask), columns{Layout::column(lane), Layout::column(lane, 1)} {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const auto& row = q.row[r];
            end[r] = row.valid ? keys.count(row.token) : 0;
            if constexpr (HasMask) {
                if constexpr (!MaskCoversShape) {
                    end[r] = p.mask_q_size == 1 || row.token < p.mask_q_size
                                 ? min(end[r], p.mask_k_size)
                                 : 0;
                }
                base[r] = end[r] ? q.batch * p.mask_b_stride + row.head * p.mask_h_stride +
                                       row.token * p.mask_q_stride
                                 : 0;
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

template <typename Traits>
struct AttentionStorage {
    alignas(16) typename Traits::Elem k[Traits::STAGES * Traits::BC * Traits::LD];
    alignas(16) typename Traits::Elem v[Traits::STAGES * Traits::BC * Traits::LD];
};

template <typename Traits>
struct AttentionTile {
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

template <typename T>
__global__ void append_kv(AttentionParams p) {
    const int token = blockIdx.x;
    int lo = 0, hi = p.batch;
    while (lo + 1 < hi) {
        int mid = lo + (hi - lo) / 2;
        if (p.cu_seqlens_q[mid] <= token)
            lo = mid;
        else
            hi = mid;
    }
    int begin = p.cu_seqlens_q[lo], end = p.cu_seqlens_q[lo + 1];
    if (begin < 0 || begin > token || token >= end || end > p.q_len)
        __trap();
    int past = p.cache_seqlens[lo];
    if (past < 0 || past > p.kv_len - (end - begin))
        __trap();
    int pos = past + (token - begin);
    int page = p.page_table[(int64_t)lo * p.table_stride + pos / p.page_size];
    if (page < 0 || page >= p.num_pages)
        __trap();
    int64_t slot = (int64_t)page * p.page_size + pos % p.page_size;
    for (int i = threadIdx.x; i < p.kv_head * p.head_dim; i += blockDim.x) {
        int64_t src = (int64_t)token * p.new_kv_token_stride +
                      (i / p.head_dim) * p.new_kv_h_stride + i % p.head_dim;
        int64_t dst = slot * p.kv_head * p.head_dim + i;
        static_cast<T*>(const_cast<void*>(p.k_ptr))[dst] = static_cast<const T*>(p.new_k_ptr)[src];
        static_cast<T*>(const_cast<void*>(p.v_ptr))[dst] = static_cast<const T*>(p.new_v_ptr)[src];
    }
}

} // namespace astrai::attention
