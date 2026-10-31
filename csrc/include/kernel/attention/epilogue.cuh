#pragma once

#include <kernel/attention/kernel.cuh>

namespace astrai::attention {

// Prefill retains packed stores; decode retains scalar stores and its register footprint.
template <typename Traits, typename Addressing, bool Vectorized = false> struct OutputEpilogue {
    static DEVICE_FORCEINLINE void store(const AttentionParams& p,
                                         const AttentionTile<Traits>& tile) {
        using T = typename Traits::Elem;
        const auto& output = tile.output;
        const float inv0 = tile.softmax.rows[0].l > 1e-20f ? 1.0f / tile.softmax.rows[0].l : 0.0f;
        const float inv1 = tile.softmax.rows[1].l > 1e-20f ? 1.0f / tile.softmax.rows[1].l : 0.0f;
        const auto rows = tile.rows;
        const int head0 = Addressing::q_base(p, rows.batch, rows.head0);
        const int head1 =
            rows.head1 == rows.head0 ? head0 : Addressing::q_base(p, rows.batch, rows.head1);
        T* out0 = static_cast<T*>(p.o_ptr) + head0 + rows.qrow0 * p.q_l_stride;
        T* out1 = static_cast<T*>(p.o_ptr) + head1 + rows.qrow1 * p.q_l_stride;
        const int column = Traits::FragmentLayout::column(threadIdx.x % 32);
#pragma unroll
        for (int i = 0; i < Traits::DN8; ++i) {
            const int d = (i * 8 + column) * p.q_d_stride;
            if (tile.valid0) {
                if constexpr (Vectorized) {
                    astrai::store2<T>(out0 + d, output[i][0] * inv0, output[i][1] * inv0);
                } else {
                    out0[d] = ElemTrait<T>::from_float(output[i][0] * inv0);
                    out0[d + p.q_d_stride] = ElemTrait<T>::from_float(output[i][1] * inv0);
                }
            }
            if (tile.valid1) {
                if constexpr (Vectorized) {
                    astrai::store2<T>(out1 + d, output[i][2] * inv1, output[i][3] * inv1);
                } else {
                    out1[d] = ElemTrait<T>::from_float(output[i][2] * inv1);
                    out1[d + p.q_d_stride] = ElemTrait<T>::from_float(output[i][3] * inv1);
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
            if (!(row == 0 ? tile.valid0 : tile.valid1))
                continue;
            const int head = row == 0 ? tile.rows.head0 : tile.rows.head1;
            const size_t slot =
                (size_t(tile.rows.batch) * p.q_head + head) * MAX_SPLITS + blockIdx.z;
            float* output = p.o_part + slot * Traits::HEAD_DIM;
#pragma unroll
            for (int i = 0; i < Traits::DN8; ++i) {
                output[i * 8 + column] = tile.output[i][row * 2];
                output[i * 8 + column + 1] = tile.output[i][row * 2 + 1];
            }
            if (column == 0) {
                p.ml_part[slot * 2] = tile.softmax.rows[row].m;
                p.ml_part[slot * 2 + 1] = tile.softmax.rows[row].l;
            }
        }
    }
};

/*
 * Split-combine: merges the per-split partials (o_part/ml_part) into the
 * final normalised O (KV selects the O addressing and element type).
 */
template <typename KV, int TileK, bool SkipEmptySplits = false>
__global__ void split_kv_combine(const AttentionParams p) {
    using T = typename KV::Elem;

    const int bh = blockIdx.x;
    const int d = threadIdx.x;
    const int batch = bh / p.q_head;
    const int q_head = bh % p.q_head;

    const size_t split_base = (size_t)bh * MAX_SPLITS;
    const float* mlp = p.ml_part + split_base * 2;
    const float* op = p.o_part + split_base * p.head_dim;
    const float scale_log2 = p.scale * LOG2E;
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
        const int o_off = KV::q_base(p, batch, q_head) + d * p.q_d_stride;
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
    const int o_off = KV::q_base(p, batch, q_head) + d * p.q_d_stride;
    static_cast<T*>(p.o_ptr)[o_off] = ElemTrait<T>::from_float(acc * inv_sum);
}

} // namespace astrai::attention
