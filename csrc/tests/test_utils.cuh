#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

inline bf16 f2bf(float x) { return __float2bfloat16(x); }
inline float bf2f(bf16 x) { return __bfloat162float(x); }

inline float randf() { return (float)rand() / (float)RAND_MAX - 0.5f; }

inline double now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

inline int compute_num_splits(int base_blocks, int tiles_total) {
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int n = (2 * sm_count + base_blocks - 1) / base_blocks;
    if (n > tiles_total) n = tiles_total;
    if (n > 32) n = 32;
    if (n < 1) n = 1;
    return n;
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

// Generic CPU reference for multi-query / grouped-query attention.
// Tensor shapes (all float*):
//   Q : [B, Hq, q_len, D]
//   K : [B, Hk, kv_len, D]
//   V : [B, Hk, kv_len, D]
//   O : [B, Hq, q_len, D]
// mask: if q_len == 1, shape is [B, kv_len]; otherwise mask is not supported.
static void cpu_attention_ref(
    const float* Q, const float* K, const float* V, const bool* mask,
    float* O, int B, int Hq, int Hk, int q_len, int kv_len, int D,
    int is_causal, int causal_offset
) {
    float scale = 1.0f / sqrtf((float)D);
    int n_rep = Hq / Hk;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < Hq; h++) {
            int kv_h = h / n_rep;
            for (int qi = 0; qi < q_len; qi++) {
                float mv = -INFINITY, sv = 0.0f;
                float accum[256] = {0.0f};
                int lim = kv_len;
                if (is_causal) {
                    int c = qi + causal_offset + 1;
                    lim = (c < kv_len) ? c : kv_len;
                }
                for (int kj = 0; kj < lim; kj++) {
                    if (mask != nullptr && q_len == 1) {
                        if (!mask[b * kv_len + kj]) continue;
                    }
                    float dot = 0.0f;
                    size_t q_idx = ((size_t)b * Hq + h) * q_len + qi;
                    size_t kv_idx = ((size_t)b * Hk + kv_h) * kv_len + kj;
                    for (int d = 0; d < D; d++)
                        dot += Q[q_idx * D + d] * K[kv_idx * D + d];
                    dot *= scale;
                    float nm = fmaxf(mv, dot);
                    float a = expf(mv - nm);
                    float b_exp = expf(dot - nm);
                    sv = sv * a + b_exp;
                    for (int d = 0; d < D; d++)
                        accum[d] = accum[d] * a + V[kv_idx * D + d] * b_exp;
                    mv = nm;
                }
                float inv = 1.0f / sv;
                size_t o_idx = ((size_t)b * Hq + h) * q_len + qi;
                for (int d = 0; d < D; d++)
                    O[o_idx * D + d] = accum[d] * inv;
            }
        }
    }
}
