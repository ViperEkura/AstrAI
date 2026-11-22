#pragma once

#include <datatype/element.cuh>
#include <kernel/attention/kernel.cuh>

namespace astrai::attention {

// Prefill retains packed stores; decode retains scalar stores and its register footprint.
template <typename Traits, bool Vectorized = false> struct OutputEpilogue {
    static DEVICE_FORCEINLINE void store(const AttentionParams& p,
                                         const AttentionTile<Traits>& tile) {
        using T = typename Traits::Elem;
        const int column = Traits::FragmentLayout::column(threadIdx.x % 32);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (!tile.query.row[r].valid)
                continue;
            T* out = static_cast<T*>(p.o_ptr) + tile.query.row[r].offset;
            const float sum = tile.accum.rows[r].l;
            const float inv = sum > 1e-20f ? 1.0f / sum : 0.0f;
#pragma unroll
            for (int i = 0; i < Traits::DN8; ++i) {
                const int d = (i * Traits::FragmentLayout::kN + column);
                const float x = tile.accum.output[i][2 * r] * inv;
                const float y = tile.accum.output[i][2 * r + 1] * inv;
                if constexpr (Vectorized) {
                    astrai::store2<T>(out + d, x, y);
                } else {
                    out[d] = ElemTrait<T>::from_float(x);
                    out[d + 1] = ElemTrait<T>::from_float(y);
                }
            }
        }
    }
};

// Split-KV keeps FP32 unnormalized accumulators until the combine pass.
template <typename Traits> struct SplitEpilogue {
    static DEVICE_FORCEINLINE void store(const AttentionParams& p,
                                         const AttentionTile<Traits>& tile) {
        const int column = Traits::FragmentLayout::column(threadIdx.x % 32);
#pragma unroll
        for (int row = 0; row < 2; ++row) {
            if (!tile.query.row[row].valid)
                continue;
            const int head = tile.query.row[row].head;
            const size_t slot =
                (size_t(tile.query.batch) * p.q_head + head) * p.split_stride + blockIdx.z;
            float* output = p.o_part + slot * Traits::HEAD_DIM;
#pragma unroll
            for (int i = 0; i < Traits::DN8; ++i) {
                output[i * 8 + column] = tile.accum.output[i][row * 2];
                output[i * 8 + column + 1] = tile.accum.output[i][row * 2 + 1];
            }
            if (column == 0) {
                p.ml_part[slot * 2] = tile.accum.rows[row].m;
                p.ml_part[slot * 2 + 1] = tile.accum.rows[row].l;
            }
        }
    }
};

/*
 * Split-combine: merges the per-split partials (o_part/ml_part) into the
 * final normalised O (KV selects the O addressing and element type).
 */
template <typename KV, int TileK, bool SkipEmptySplits = false, bool FixedSplits = false>
__global__ void split_kv_combine(const AttentionParams p) {
    using T = typename KV::Elem;

    const int bh = blockIdx.x;
    const int d = threadIdx.x;
    const int batch = bh / p.q_head;
    const int q_head = bh % p.q_head;

    const size_t split_base = (size_t)bh * p.split_stride;
    const float* mlp = p.ml_part + split_base * 2;
    const float* op = p.o_part + split_base * p.head_dim;
    const float scale_log2 = p.scale * LOG2E;
    if constexpr (FixedSplits) {
        const int length = KV::kv_len(p, batch);
        const int splits = length > 0 ? (length - 1) / p.kv_split_size + 1 : 0;
        SoftmaxState total;
        float acc = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const SoftmaxState part{mlp[2 * s], mlp[2 * s + 1]};
            if (part.l == 0.0f)
                continue;
            float alpha, beta;
            merge_softmax(total, part, alpha, beta, scale_log2);
            acc = merge_output(acc, op[s * p.head_dim + d], alpha, beta);
        }
        const float inv = total.l > 1e-20f ? 1.0f / total.l : 0.0f;
        const int o_off = KV::q_base(p, batch, q_head) + d;
        static_cast<T*>(p.o_ptr)[o_off] = ElemTrait<T>::from_float(acc * inv);
        return;
    }
    int active_splits = p.num_splits;
    if constexpr (SkipEmptySplits && KV::kPaged) {
        const int seq_len = KV::kv_len(p, batch);
        if (skip_empty_decode_splits(p, seq_len)) {
            const int tiles_total = (seq_len + TileK - 1) / TileK;
            const int tiles_per_split = (tiles_total + p.num_splits - 1) / p.num_splits;
            active_splits =
                tiles_per_split ? (tiles_total + tiles_per_split - 1) / tiles_per_split : 1;
        }
    }

    // Avoid warp reductions when merging only one or two splits.
    if (active_splits <= 2) {
        if (d >= p.head_dim)
            return;
        SoftmaxState st;
        float acc = 0.0f;
        for (int s = 0; s < active_splits; s++) {
            const float mi = mlp[s * 2];
            if (mi <= -FLT_MAX)
                continue;
            const float li = mlp[s * 2 + 1];
            float corr, e;
            softmax_step(st, mi, li, corr, e, scale_log2);
            acc = fmaf(acc, corr, op[s * p.head_dim + d] * e);
        }
        const float inv = st.l > 1e-20f ? 1.0f / st.l : 0.0f;
        const int o_off = KV::q_base(p, batch, q_head) + d;
        static_cast<T*>(p.o_ptr)[o_off] = ElemTrait<T>::from_float(acc * inv);
        return;
    }

    // Each warp reduces the split weights locally, avoiding shared memory and a block barrier.
    const int lane = threadIdx.x % 32;
    const bool valid = lane < active_splits;
    const float mi = valid ? mlp[lane * 2] : -FLT_MAX;
    const float li = valid ? mlp[lane * 2 + 1] : 0.0f;
    float max_m = mi;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        max_m = fmaxf(max_m, __shfl_xor_sync(0xFFFFFFFF, max_m, offset));
    const float weight = mi > -FLT_MAX ? exp2f(mi * scale_log2 - max_m * scale_log2) : 0.0f;
    float sum_l = li * weight;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum_l += __shfl_xor_sync(0xFFFFFFFF, sum_l, offset);
    const float inv_sum = sum_l > 1e-20f ? 1.0f / sum_l : 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < active_splits; s++)
        acc = fmaf(op[s * p.head_dim + d], __shfl_sync(0xFFFFFFFF, weight, s), acc);
    const int o_off = KV::q_base(p, batch, q_head) + d;
    static_cast<T*>(p.o_ptr)[o_off] = ElemTrait<T>::from_float(acc * inv_sum);
}

} // namespace astrai::attention
