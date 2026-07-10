#pragma once
#include "attn_common.cuh"

constexpr int DC_CHUNK = 64;

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void attn_decode_split_kv_kernel(AttentionParams p) {
    int batch = blockIdx.x / p.kv_head;
    int kv_head = blockIdx.x % p.kv_head;
    int split = blockIdx.z;
    int group_size = blockDim.y;
    int q_head = kv_head * group_size + threadIdx.y;
    int lane = threadIdx.x;
    int hd_per_thread = p.head_dim / 32;

    float q_reg[8];
    int q_off = ((batch * p.q_head + q_head) * 1) * p.head_dim + lane * hd_per_thread;
    for (int i = 0; i < hd_per_thread; i++)
        q_reg[i] = __bfloat162float(p.q[q_off + i]);

    int kv_base = ((batch * p.kv_head + kv_head) * p.kv_len) * p.head_dim;
    int mask_base = batch * p.kv_len;

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

        int total = this_chunk * p.head_dim;
        for (int i = threadIdx.y * 32 + lane; i < total; i += blockDim.x * blockDim.y)
            k_smem[i] = p.k[kv_base + chunk_start * p.head_dim + i];
        __syncthreads();

        for (int s = 0; s < this_chunk; s++) {
            float partial = 0.0f;
            for (int i = 0; i < hd_per_thread; i++)
                partial += q_reg[i] * __bfloat162float(k_smem[s * p.head_dim + lane * hd_per_thread + i]);
            partial = warp_reduce_sum(partial) * p.scale;

            if (p.use_mask && p.mask && !p.mask[mask_base + chunk_start + s])
                partial = -FLT_MAX;
            if (p.is_causal && (chunk_start + s) > p.causal_offset)
                partial = -FLT_MAX;

            float new_m = fmaxf(m, partial);
            float alpha = expf(m - new_m);
            float beta  = expf(partial - new_m);
            d = d * alpha + beta;

            int v_off = kv_base + (chunk_start + s) * p.head_dim + lane * hd_per_thread;
            for (int i = 0; i < hd_per_thread; i++)
                acc_reg[i] = acc_reg[i] * alpha + __bfloat162float(p.v[v_off + i]) * beta;
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

// Reduce split-K partials into the final bf16 output. One block per (batch,
// q_head); each thread owns one head_dim element and folds across all splits
// with a numerically-stable online rescale.
__global__ void attn_decode_combine_kernel(AttentionParams p) {
    int bh = blockIdx.x;
    int d = threadIdx.x;
    if (d >= p.head_dim) return;

    size_t split_base = (size_t)bh * p.num_splits;

    const float* mlp = p.ml_part + split_base * 2;
    float mstar = -FLT_MAX;
    for (int s = 0; s < p.num_splits; s++)
        mstar = fmaxf(mstar, mlp[s * 2]);

    float lstar = 0.0f;
    for (int s = 0; s < p.num_splits; s++) {
        float mi = mlp[s * 2];
        if (mi > -FLT_MAX) lstar += mlp[s * 2 + 1] * __expf(mi - mstar);
    }

    const float* op = p.o_part + split_base * p.head_dim;
    float acc = 0.0f;
    for (int s = 0; s < p.num_splits; s++) {
        float mi = mlp[s * 2];
        if (mi > -FLT_MAX) acc += op[s * p.head_dim + d] * __expf(mi - mstar);
    }
    float inv = (lstar > 1e-20f) ? (1.0f / lstar) : 0.0f;
    p.o[(size_t)bh * p.head_dim + d] = __float2bfloat16(acc * inv);
}
