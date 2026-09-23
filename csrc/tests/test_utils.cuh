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
    double tflops;
};

template <typename Fn>
BenchResult bench_kernel(Fn launch, int warmup, int iters,
                         double flops) {
    for (int i = 0; i < warmup; i++) launch();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before bench: %s\n", cudaGetErrorString(err));
        return {0, 0};
    }

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) launch();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
    cudaEventDestroy(s); cudaEventDestroy(e);

    return {ms, flops / (ms * 1e-3) / 1e12};
}

inline void print_bench_header() {
    printf("%-46s | %10s | %10s\n",
           "config", "latency", "TFLOP/s");
    printf("---------------------------------------------------------------"
           "----------------------------\n");
}

inline void print_bench_row(const char* cfg, const BenchResult& r) {
    printf("%-46s | %7.4f ms | %6.2f\n",
           cfg, r.ms, r.tflops);
}

// ---- validation table (kernel vs CPU reference) ----
inline void print_test_header() {
    printf("%-46s | %11s | %11s | %6s\n",
           "config", "max_abs_err", "max_rel_err", "result");
    printf("----------------------------------------------------------------"
           "----------------------------\n");
}

inline void print_test_row(const char* cfg, float max_abs_err,
                           float max_rel_err, bool pass) {
    printf("%-46s | %11.3e | %11.3e | %s\n",
           cfg, max_abs_err, max_rel_err, pass ? "PASS" : "FAIL");
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
    p.q_b_stride  = p.q_head * p.q_len * p.head_dim;
    p.q_h_stride  = p.q_len * p.head_dim;
    p.q_l_stride  = p.head_dim;
    p.q_d_stride  = 1;
    p.kv_b_stride = p.kv_head * p.kv_len * p.head_dim;
    p.kv_h_stride = p.kv_len * p.head_dim;
    p.kv_l_stride = p.head_dim;
    p.kv_d_stride = 1;
    p.mask_b_stride = p.kv_len;
    p.mask_h_stride = 0;
    p.mask_l_stride = 0;
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
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int h = 0; h < Hq; h++) {
            for (int qi = 0; qi < q_len; qi++) {
                int kv_h = h / n_rep;
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
