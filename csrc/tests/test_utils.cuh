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

struct BenchResult {
    float ms;
    double gbps;
    double tflops;
};

template <typename Fn>
BenchResult bench_kernel(Fn launch, int warmup, int iters,
                         double flops, double bytes) {
    for (int i = 0; i < warmup; i++) launch();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before bench: %s\n", cudaGetErrorString(err));
        return {0, 0, 0};
    }

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) launch();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
    cudaEventDestroy(s); cudaEventDestroy(e);

    return {ms, bytes / (ms * 1e-3) / 1e9, flops / (ms * 1e-3) / 1e12};
}

inline void print_bench_header() {
    printf("%-46s | %10s | %10s | %10s\n",
           "config", "latency", "bandwidth", "throughput");
    printf("---------------------------------------------------------------"
           "----------------------------\n");
}

inline void print_bench_row(const char* cfg, const BenchResult& r) {
    printf("%-46s | %7.4f ms | %7.1f GB/s | %6.2f TFLOP/s\n",
           cfg, r.ms, r.gbps, r.tflops);
}

template <int... Ds>
struct _HeadSwitch;

template <int D>
struct _HeadSwitch<D> {
    template <typename Fn>
    static void call(int hd, Fn&& fn) { if (hd == D) fn.template operator()<D>(); }
};

template <int D, int... Rest>
struct _HeadSwitch<D, Rest...> {
    template <typename Fn>
    static void call(int hd, Fn&& fn) {
        if (hd == D) fn.template operator()<D>();
        else _HeadSwitch<Rest...>::call(hd, fn);
    }
};

// Default set: 32, 64, 128, 256
template <typename Fn>
void dispatch_by_head_dim(int head_dim, Fn&& fn) {
    _HeadSwitch<32, 64, 128, 256>::call(head_dim, fn);
}

// Set default strides for contiguous b h l d layout on AttentionParams.
template<typename P>
inline void set_default_strides(P& p) {
    p.q_stride_b  = p.q_head * p.q_len * p.head_dim;
    p.q_stride_h  = p.q_len * p.head_dim;
    p.q_stride_l  = p.head_dim;
    p.q_stride_d  = 1;
    p.kv_stride_b = p.kv_head * p.kv_len * p.head_dim;
    p.kv_stride_h = p.kv_len * p.head_dim;
    p.kv_stride_l = p.head_dim;
    p.kv_stride_d = 1;
    p.mask_b_stride = p.kv_len;
    p.mask_q_stride = 0;
}

// Set default Q strides for contiguous b h l d layout on PagedAttentionParams.
template<typename P>
inline void set_default_paged_strides(P& p) {
    p.q_stride_b  = p.q_head * p.q_len * p.head_dim;
    p.q_stride_h  = p.q_len * p.head_dim;
    p.q_stride_l  = p.head_dim;
    p.q_stride_d  = 1;
    p.mask_b_stride = p.kv_len;
    p.mask_q_stride = 0;
}

// Generic CPU reference for multi-query / grouped-query attention.
// Tensor shapes (all float*):
//   Q : [B, Hq, q_len, D]
//   K : [B, Hk, kv_len, D]
//   V : [B, Hk, kv_len, D]
//   O : [B, Hq, q_len, D]
// mask: if q_len == 1, shape is [B, kv_len]; otherwise mask is not supported.
// causal_offset: -1 = non-causal; >=0 = absolute position of first Q token.
static void cpu_attention_ref(
    const float* Q, const float* K, const float* V, const bool* mask,
    float* O, int B, int Hq, int Hk, int q_len, int kv_len, int D,
    int causal_offset
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
                if (causal_offset >= 0) {
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
