#pragma once
#include <cuda_bf16.h>
#include <float.h>
#include "attn_common.h"

using bf16 = __nv_bfloat16;
constexpr int DC_CHUNK = 64;

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

template <int HEAD_DIM, bool IsCausal, bool HasMask>
__global__ void attn_decode_split_kv_kernel(AttentionParams<bf16> p) {
    int batch = blockIdx.x / p.kv_head;
    int kv_head = blockIdx.x % p.kv_head;
    int split = blockIdx.z;
    int group_size = blockDim.y;
    int q_head = kv_head * group_size + threadIdx.y;
    int lane = threadIdx.x;
    int hd_per_thread = p.head_dim / 32;

    // Q: [batch, q_head, q_len=1, head_dim] — stride-based
    float q_reg[8];
    int q_off = batch * p.q_stride_b + q_head * p.q_stride_h
              + lane * hd_per_thread * p.q_stride_d;
    for (int i = 0; i < hd_per_thread; i++)
        q_reg[i] = __bfloat162float(p.q[q_off + i * p.q_stride_d]);

    // KV: [batch, kv_head, kv_len, head_dim] — stride-based base
    int kv_base = batch * p.kv_stride_b + kv_head * p.kv_stride_h;
    int mask_base = batch * p.mask_b_stride;

    float m = -FLT_MAX, d = 0.0f, acc_reg[8] = {0.0f};

    extern __shared__ __align__(16) bf16 k_smem[];

    // Split-KV: each split processes a contiguous subset of chunks
    int chunks_total = (p.kv_len + DC_CHUNK - 1) / DC_CHUNK;
    int chunks_per_split = (chunks_total + p.num_splits - 1) / p.num_splits;
    int ch_begin = split * chunks_per_split;
    int ch_end = min(chunks_total, ch_begin + chunks_per_split);

    for (int ci = ch_begin; ci < ch_end; ci++) {
        int chunk_start = ci * DC_CHUNK;
        int this_chunk = min(DC_CHUNK, p.kv_len - chunk_start);

        // Load K into shared memory (gather from strided global)
        int total = this_chunk * p.head_dim;
        for (int i = threadIdx.y * 32 + lane; i < total;
             i += blockDim.x * blockDim.y) {
            int s = i / p.head_dim;
            int d_dim = i % p.head_dim;
            int kv_idx = chunk_start + s;
            int g_off = kv_base + kv_idx * p.kv_stride_l + d_dim * p.kv_stride_d;
            k_smem[i] = p.k[g_off];
        }
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
                if (kv_idx > p.causal_offset)
                    partial = -FLT_MAX;
            }

            float new_m = fmaxf(m, partial);
            float alpha = expf(m - new_m);
            float beta  = expf(partial - new_m);
            d = d * alpha + beta;

            int v_off = kv_base + kv_idx * p.kv_stride_l
                        + lane * hd_per_thread * p.kv_stride_d;
            for (int i = 0; i < hd_per_thread; i++)
                acc_reg[i] = fmaf(acc_reg[i], alpha,
                                  __bfloat162float(p.v[v_off + i * p.kv_stride_d]) * beta);
            m = new_m;
        }
        __syncthreads();
    }

    // ---- write UN-normalised partials for this split ----
    size_t bh = (size_t)batch * p.q_head + q_head;
    size_t slot = bh * p.num_splits + split;
    int d0 = lane * hd_per_thread;
    for (int i = 0; i < hd_per_thread; i++) {
        int dd = d0 + i;
        p.o_part[slot * p.head_dim + dd] = acc_reg[i];
    }
    if (lane == 0) {
        p.ml_part[slot * 2] = m;
        p.ml_part[slot * 2 + 1] = d;
    }
}

__global__ void attn_decode_combine_kernel(AttentionParams<bf16> p) {
    int bh = blockIdx.x;
    int d = threadIdx.x;
    if (d >= p.head_dim) return;

    int batch = bh / p.q_head;
    int q_head = bh % p.q_head;

    size_t split_base = (size_t)bh * p.num_splits;
    const float* mlp = p.ml_part + split_base * 2;
    const float* op = p.o_part + split_base * p.head_dim;

    float m = -FLT_MAX, l = 0.0f, acc = 0.0f;
    for (int s = 0; s < p.num_splits; s++) {
        float mi = mlp[s * 2];
        if (mi <= -FLT_MAX) continue;
        float li = mlp[s * 2 + 1];
        float nm = fmaxf(m, mi);
        float corr = expf(m - nm);
        float e = expf(mi - nm);
        acc = fmaf(acc, corr, op[s * p.head_dim + d] * e);
        l = fmaf(l, corr, li * e);
        m = nm;
    }

    float inv = (l > 1e-20f) ? (1.0f / l) : 0.0f;
    int o_off = batch * p.q_stride_b + q_head * p.q_stride_h + d * p.q_stride_d;
    p.o[o_off] = __float2bfloat16(acc * inv);
}
