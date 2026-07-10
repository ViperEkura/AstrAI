#pragma once
#include "gqa_common.cuh"

constexpr int DC_CHUNK = 64;

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void gqa_decode_attn_kernel(GQAParams p) {
    int batch = blockIdx.x / p.kv_head;
    int kv_head = blockIdx.x % p.kv_head;
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

    for (int chunk_start = 0; chunk_start < p.kv_len; chunk_start += DC_CHUNK) {
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

    int out_off = ((batch * p.q_head + q_head) * 1) * p.head_dim + lane * hd_per_thread;
    for (int i = 0; i < hd_per_thread; i++)
        p.o[out_off + i] = __float2bfloat16(acc_reg[i] / d);
}
