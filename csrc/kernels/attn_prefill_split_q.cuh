#pragma once
#include "attn_common.cuh"

// v9: group-split register blocking. G threads cooperate on one query row,
// each owning HEAD_DIM/G dims of qreg[]/acc[]. Small per-thread footprint keeps
// occupancy high; the S dot product is reduced across the G-lane group with a
// short shuffle chain (log2(G) shuffles) instead of a full 32-lane warp reduce.
// Online (per-kv) softmax — cheap because acc[] is only HEAD_DIM/G long.
// Templated on <HEAD_DIM, G, ROWS, P_BC>. Block = (G, ROWS). G power-of-two,
// G*ROWS a multiple of 32 with groups warp-aligned.

template <int G>
__device__ __forceinline__ float group_reduce_sum(float v, unsigned mask) {
#pragma unroll
    for (int o = G / 2; o > 0; o >>= 1)
        v += __shfl_xor_sync(mask, v, o);
    return v;
}

// load 8 contiguous bf16 from (16-byte aligned) smem as one float4, unpack to
// 8 floats — cuts shared-load instructions 8x vs scalar bf16 loads.
__device__ __forceinline__ void ld8(const bf16* p, float* o) {
    float4 raw = *reinterpret_cast<const float4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        float2 f = __bfloat1622float2(h[j]);
        o[2 * j] = f.x;
        o[2 * j + 1] = f.y;
    }
}

template <int HEAD_DIM, int G, int ROWS, int P_BC>
__global__ void attn_prefill_split_q_kernel_t(AttentionParams p) {
    constexpr int DPT = HEAD_DIM / G;

    int q_tile = blockIdx.x;
    int q_head = blockIdx.y;
    int batch  = blockIdx.z;
    int gpos   = threadIdx.x;  // 0..G-1  (which d-chunk)
    int row    = threadIdx.y;  // 0..ROWS-1
    int q_row  = q_tile * ROWS + row;

    int kv_head = q_head / (p.q_head / p.kv_head);

    __shared__ __align__(16) bf16 sK[P_BC * HEAD_DIM];
    __shared__ __align__(16) bf16 sV[P_BC * HEAD_DIM];

    float qreg[DPT];
    if (q_row < p.q_len) {
        int q_off = ((batch * p.q_head + q_head) * p.q_len + q_row) * HEAD_DIM + gpos * DPT;
#pragma unroll
        for (int i = 0; i < DPT; i++)
            qreg[i] = __bfloat162float(p.q[q_off + i]) * p.scale;
    }

    float m = -FLT_MAX, l = 0.0f;
    float acc[DPT];
#pragma unroll
    for (int i = 0; i < DPT; i++)
        acc[i] = 0.0f;

    int kv_base = ((batch * p.kv_head + kv_head) * p.kv_len) * HEAD_DIM;
    int tiles   = (p.kv_len + P_BC - 1) / P_BC;
    int tt      = G * ROWS;
    int lid     = row * G + gpos;

    // per-group shuffle mask: only the G lanes of this row's group participate,
    // so causal masking (differing loop bounds across rows in a warp) is safe.
    int lane_in_warp = lid & 31;
    unsigned gmask = (G == 32) ? 0xFFFFFFFFu
                               : (((1u << G) - 1u) << (lane_in_warp & ~(G - 1)));

    for (int ti = 0; ti < tiles; ti++) {
        int kv0  = ti * P_BC;
        int tlen = min(P_BC, p.kv_len - kv0);

        for (int i = lid; i < tlen * HEAD_DIM; i += tt) {
            int gidx = kv_base + (kv0 + i / HEAD_DIM) * HEAD_DIM + (i % HEAD_DIM);
            sK[i] = p.k[gidx];
            sV[i] = p.v[gidx];
        }
        __syncthreads();

        int lim = tlen;
        if (p.is_causal && q_row < p.q_len) {
            int ep = q_row + p.causal_offset + 1;
            if (kv0 >= ep)
                lim = 0;
            else if (kv0 + tlen > ep)
                lim = ep - kv0;
        }

        for (int s = 0; s < lim; s++) {
            const bf16* kr = sK + s * HEAD_DIM + gpos * DPT;
            float part = 0.0f;
#pragma unroll
            for (int i = 0; i < DPT; i += 8) {
                float k8[8];
                ld8(kr + i, k8);
#pragma unroll
                for (int j = 0; j < 8; j++)
                    part = fmaf(qreg[i + j], k8[j], part);
            }
            float dot = group_reduce_sum<G>(part, gmask);

            if (p.use_mask && p.mask && !p.mask[batch * p.kv_len + kv0 + s])
                dot = -FLT_MAX;

            float nm = fmaxf(m, dot);
            float al = __expf(m - nm);
            float be = __expf(dot - nm);
            l = l * al + be;

            const bf16* vr = sV + s * HEAD_DIM + gpos * DPT;
#pragma unroll
            for (int i = 0; i < DPT; i += 8) {
                float v8[8];
                ld8(vr + i, v8);
#pragma unroll
                for (int j = 0; j < 8; j++)
                    acc[i + j] = fmaf(v8[j], be, acc[i + j] * al);
            }
            m = nm;
        }
        __syncthreads();
    }

    if (q_row < p.q_len) {
        int o_off = ((batch * p.q_head + q_head) * p.q_len + q_row) * HEAD_DIM + gpos * DPT;
        float rl = (l > 1e-10f) ? (1.0f / l) : 0.0f;
#pragma unroll
        for (int i = 0; i < DPT; i++)
            p.o[o_off + i] = __float2bfloat16(acc[i] * rl);
    }
}
