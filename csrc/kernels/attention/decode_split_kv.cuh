#pragma once
#include <cuda_bf16.h>
#include <float.h>
#include "common.h"
#include "common/reduce.cuh"
#include "layout_policies.cuh"
#include "softmax.cuh"

namespace astrai {
namespace attention {

constexpr int DC_CHUNK = 64;

// Scalar split-KV decode (fallback for sm < 80, no tensor cores), unified
// across contiguous and paged (SGLang flat-pool) K/V via the KV template
// parameter.  For decode the query is the last token, so its valid range
// [0, seq_len) IS the causal range; KV::decode_attend_len expresses that
// bound per addressing mode (contig clips to causal_offset, paged = seq_len).
template <int HEAD_DIM, typename KV, bool IsCausal, bool HasMask>
__global__ void attn_decode_split_kv_kernel(AttentionParams<bf16> p) {
    int batch = blockIdx.x / p.kv_head;
    int kv_head = blockIdx.x % p.kv_head;
    int split = blockIdx.z;
    int lane = threadIdx.x;
    int hd_per_thread = p.head_dim / 32;

    const int seq_len = KV::kv_len(p, batch);
    const KVContext kctx = KV::template make_ctx<HEAD_DIM>(p, batch, kv_head);

    extern __shared__ __align__(16) bf16 smem[];
    bf16* k_smem = smem;
    bf16* v_smem = smem + DC_CHUNK * p.head_dim;

    // Split-KV: each split processes a contiguous subset of chunks
    int chunks_total = (seq_len + DC_CHUNK - 1) / DC_CHUNK;
    int chunks_per_split = (chunks_total + p.num_splits - 1) / p.num_splits;
    int ch_begin = split * chunks_per_split;
    int ch_end = min(chunks_total, ch_begin + chunks_per_split);

    const int group_size = p.q_head / p.kv_head;

    // GQA group can exceed blockDim.y (capped at 32 for the 1024-thread
    // limit): loop passes over 32-head slices, mirroring the MMA kernel's
    // block-level GQA passes.  G <= 32 is a single pass.  All threads of a
    // pass — including inactive tail threads — must reach the __syncthreads
    // in the chunk loop, so `active` gates loads/stores, never barriers.
    // Inactive warps compute on zero Q and never store.
    for (int g0 = 0; g0 < group_size; g0 += 32) {
        const int q_head = kv_head * group_size + g0 + threadIdx.y;
        // Clip at the GROUP boundary, not p.q_head: a partial last pass of
        // kv_head g must not spill into heads owned by group g+1.
        const bool active = q_head < (kv_head + 1) * group_size;
        const int safe_head = active ? q_head : 0;

        // Q: [batch, q_head, q_len=1, head_dim] — stride-based
        float q_reg[8] = {0.0f};
        if (active) {
            int q_off = KV::q_decode_base(p, batch, safe_head)
                      + lane * hd_per_thread * p.q_d_stride;
            for (int i = 0; i < hd_per_thread; i++)
                q_reg[i] = __bfloat162float(p.q_ptr[q_off + i * p.q_d_stride]);
        }

        int mask_base = batch * p.mask_b_stride + safe_head * p.mask_h_stride;

        SoftmaxState sm;
        float acc_reg[8] = {0.0f};

        for (int ci = ch_begin; ci < ch_end; ci++) {
            int chunk_start = ci * DC_CHUNK;
            int this_chunk = min(DC_CHUNK, seq_len - chunk_start);

            // Load K and V into shared memory (addressing via KV policy;
            // paged guards empty slots with zero-fill).
            fill_kv_smem(k_smem, v_smem, this_chunk * p.head_dim, p.head_dim,
                         chunk_start, threadIdx.y * 32 + lane,
                         blockDim.x * blockDim.y,
                         [&](int kc, int d) {
                             return KV::template decode_addr<1>(
                                 p, kctx, batch, kv_head, kc, d, true, true);
                         });
            __syncthreads();

            for (int s = 0; s < this_chunk; s++) {
                float partial = 0.0f;
                for (int i = 0; i < hd_per_thread; i++)
                    partial += q_reg[i] * __bfloat162float(
                        k_smem[s * p.head_dim + lane * hd_per_thread + i]);
                partial = warp_reduce_sum(partial) * p.scale;

                int kv_idx = chunk_start + s;
                if constexpr (HasMask) {
                    if (!p.mask[mask_base + kv_idx])
                        partial = -FLT_MAX;
                }
                if constexpr (IsCausal) {
                    if (kv_idx >= KV::decode_attend_len(p, batch))
                        partial = -FLT_MAX;
                }

                float alpha, beta;
                softmax_step(sm, partial, 1.0f, alpha, beta);

                for (int i = 0; i < hd_per_thread; i++) {
                    float vv = __bfloat162float(v_smem[s * p.head_dim + lane * hd_per_thread + i]);
                    acc_reg[i] = fmaf(acc_reg[i], alpha, vv * beta);
                }
            }
            __syncthreads();
        }

        // ---- write UN-normalised partials for this split ----
        if (active) {
            size_t bh = (size_t)batch * p.q_head + q_head;
            size_t slot = bh * MAX_SPLITS + split;
            int d0 = lane * hd_per_thread;
            for (int i = 0; i < hd_per_thread; i++) {
                int dd = d0 + i;
                p.o_part[slot * p.head_dim + dd] = acc_reg[i];
            }
            if (lane == 0) {
                p.ml_part[slot * 2] = sm.m;
                p.ml_part[slot * 2 + 1] = sm.l;
            }
        }
    }
}

// Split-combine: merges the per-split partials (o_part/ml_part) into the
// final normalised O.  KV selects the O addressing (contig batch stride vs
// paged row stride).
template <typename KV>
__global__ void attn_decode_combine_kernel(AttentionParams<bf16> p) {
    int bh = blockIdx.x;
    int d = threadIdx.x;
    if (d >= p.head_dim) return;

    int batch = bh / p.q_head;
    int q_head = bh % p.q_head;

    size_t split_base = (size_t)bh * MAX_SPLITS;
    const float* mlp = p.ml_part + split_base * 2;
    const float* op = p.o_part + split_base * p.head_dim;

    SoftmaxState st;
    float acc = 0.0f;
    for (int s = 0; s < p.num_splits; s++) {
        float mi = mlp[s * 2];
        if (mi <= -FLT_MAX) continue;
        float li = mlp[s * 2 + 1];
        float corr, e;
        softmax_step(st, mi, li, corr, e);
        acc = fmaf(acc, corr, op[s * p.head_dim + d] * e);
    }

    float inv = (st.l > 1e-20f) ? (1.0f / st.l) : 0.0f;
    int o_off = KV::q_decode_base(p, batch, q_head) + d * p.q_d_stride;
    p.o_ptr[o_off] = __float2bfloat16(acc * inv);
}

}  // namespace attention
}  // namespace astrai
