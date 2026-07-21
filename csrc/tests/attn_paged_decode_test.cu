// Compile:
//   nvcc -I csrc -arch=sm_89 -O3 --use_fast_math --ptxas-options=-O3 \
//        --extra-device-vectorization csrc/tests/attn_paged_decode_test.cu \
//        -o /tmp/test_paged && /tmp/test_paged

#include <cstring>
#include "test_utils.cuh"
#include "../kernels/attn_dispatchers.cuh"

static void gather_kv_cpu(
    const bf16* h_k_pool, const bf16* h_v_pool,
    const int64_t* h_pt, int B, int Hkv, int kv_len,
    int page_size, int head_dim,
    bf16* h_k, bf16* h_v)
{
    int max_pages = (kv_len + page_size - 1) / page_size;
    size_t page_stride = (size_t)page_size * Hkv * head_dim;
    for (int b = 0; b < B; b++) {
        for (int pos = 0; pos < kv_len; pos++) {
            int log_pg = pos / page_size;
            int pg_off = pos % page_size;
            int phys = (int)h_pt[b * max_pages + log_pg];
            for (int h = 0; h < Hkv; h++) {
                size_t src_base = (size_t)phys * page_stride
                                + (size_t)pg_off * Hkv * head_dim
                                + h * head_dim;
                size_t dst_base = ((size_t)b * Hkv + h) * kv_len * head_dim
                                  + (size_t)pos * head_dim;
                memcpy(h_k + dst_base, h_k_pool + src_base, head_dim * sizeof(bf16));
                memcpy(h_v + dst_base, h_v_pool + src_base, head_dim * sizeof(bf16));
            }
        }
    }
}

template <int HEAD_DIM>
static int run_test(int B, int Hq, int Hkv, int kv_len, int page_size, int causal, int seed) {
    printf("B=%d Hq=%d Hkv=%d kv_len=%d page_sz=%d head_dim=%d causal=%d ... ",
           B, Hq, Hkv, kv_len, page_size, HEAD_DIM, causal);
    fflush(stdout);

    int max_pages = (kv_len + page_size - 1) / page_size;
    int n_phys_pages = B * max_pages;
    int max_splits = 32;

    size_t sz_q  = (size_t)B * Hq * 1 * HEAD_DIM * sizeof(bf16);
    size_t sz_o  = sz_q;
    size_t sz_kv = (size_t)n_phys_pages * page_size * Hkv * HEAD_DIM * sizeof(bf16);
    size_t sz_pt = (size_t)B * max_pages * sizeof(int64_t);
    size_t sz_op = (size_t)B * Hq * max_splits * HEAD_DIM * sizeof(float);
    size_t sz_ml = (size_t)B * Hq * max_splits * 2 * sizeof(float);

    bf16 *d_q, *d_o_paged;
    bf16 *d_k_pool, *d_v_pool;
    int64_t* d_pt;
    float *d_op, *d_ml;

    cudaMalloc(&d_q, sz_q);
    cudaMalloc(&d_o_paged, sz_o);
    cudaMalloc(&d_k_pool, sz_kv);
    cudaMalloc(&d_v_pool, sz_kv);
    cudaMalloc(&d_pt, sz_pt);
    cudaMalloc(&d_op, sz_op);
    cudaMalloc(&d_ml, sz_ml);

    srand(seed);
    auto rnd = [&]() { return (rand() / (float)RAND_MAX) * 2.0f - 1.0f; };

    bf16* h_q = (bf16*)malloc(sz_q);
    for (int i = 0; i < B * Hq * HEAD_DIM; i++)
        h_q[i] = __float2bfloat16(rnd());
    cudaMemcpy(d_q, h_q, sz_q, cudaMemcpyHostToDevice);

    bf16* h_k_pool = (bf16*)malloc(sz_kv);
    bf16* h_v_pool = (bf16*)malloc(sz_kv);
    size_t ps = (size_t)page_size * Hkv * HEAD_DIM;
    for (int pg = 0; pg < n_phys_pages; pg++) {
        for (int off = 0; off < page_size; off++) {
            for (int h = 0; h < Hkv; h++) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    float v = sinf((float)(pg * 7919 + off * 1049 + h * 331 + d));
                    size_t idx = (size_t)pg * ps + (size_t)off * Hkv * HEAD_DIM
                                 + h * HEAD_DIM + d;
                    h_k_pool[idx] = __float2bfloat16(v);
                    h_v_pool[idx] = __float2bfloat16(v * 0.3f);
                }
            }
        }
    }
    cudaMemcpy(d_k_pool, h_k_pool, sz_kv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_pool, h_v_pool, sz_kv, cudaMemcpyHostToDevice);

    int64_t* h_pt = (int64_t*)malloc(sz_pt);
    int next_pg = 0;
    for (int b = 0; b < B; b++)
        for (int p = 0; p < max_pages; p++)
            h_pt[b * max_pages + p] = next_pg++;
    cudaMemcpy(d_pt, h_pt, sz_pt, cudaMemcpyHostToDevice);

    bf16* h_k_cont = (bf16*)malloc((size_t)B * kv_len * Hkv * HEAD_DIM * sizeof(bf16));
    bf16* h_v_cont = (bf16*)malloc((size_t)B * kv_len * Hkv * HEAD_DIM * sizeof(bf16));
    gather_kv_cpu(h_k_pool, h_v_pool, h_pt, B, Hkv, kv_len, page_size, HEAD_DIM, h_k_cont, h_v_cont);

    float* h_q_f = (float*)malloc((size_t)B * Hq * HEAD_DIM * sizeof(float));
    float* h_k_f = (float*)malloc((size_t)B * kv_len * Hkv * HEAD_DIM * sizeof(float));
    float* h_v_f = (float*)malloc((size_t)B * kv_len * Hkv * HEAD_DIM * sizeof(float));
    for (int i = 0; i < B * Hq * HEAD_DIM; i++) h_q_f[i] = bf2f(h_q[i]);
    for (int i = 0; i < B * kv_len * Hkv * HEAD_DIM; i++) {
        h_k_f[i] = bf2f(h_k_cont[i]);
        h_v_f[i] = bf2f(h_v_cont[i]);
    }

    float* h_o_ref = (float*)calloc(B * Hq * HEAD_DIM, sizeof(float));
    cpu_attention_ref(h_q_f, h_k_f, h_v_f, nullptr, h_o_ref, B, Hq, Hkv,
                      1, kv_len, HEAD_DIM, causal ? 0 : -1);

    PagedAttentionParams<bf16> p;
    p.batch = B; p.q_head = Hq; p.kv_head = Hkv; p.q_len = 1;
    p.kv_len = kv_len; p.head_dim = HEAD_DIM;
    p.use_mask = 0; p.causal_offset = causal ? 0 : -1;
    set_default_paged_strides(p);
    p.scale = 1.0f / sqrtf((float)HEAD_DIM);
    p.page_size = page_size; p.max_pages = max_pages;
    p.page_table = d_pt;
    p.k_cache = d_k_pool; p.v_cache = d_v_pool;
    p.q = d_q; p.mask = nullptr; p.o = d_o_paged;
    p.o_part = d_op; p.ml_part = d_ml;

    dispatch_by_head_dim(HEAD_DIM, [&]<int H>() { dispatch_paged_decode<H>(p); });
    cudaDeviceSynchronize();

    bf16* h_o_bf16 = (bf16*)malloc(sz_o);
    cudaMemcpy(h_o_bf16, d_o_paged, sz_o, cudaMemcpyDeviceToHost);
    float* h_o_paged = (float*)malloc(B * Hq * HEAD_DIM * sizeof(float));
    for (int i = 0; i < B * Hq * HEAD_DIM; i++)
        h_o_paged[i] = __bfloat162float(h_o_bf16[i]);

    float max_err = 0.0f;
    int bad_idx = -1;
    for (int i = 0; i < B * Hq * HEAD_DIM; i++) {
        float e = fabsf(h_o_paged[i] - h_o_ref[i]);
        if (e > max_err) { max_err = e; bad_idx = i; }
    }

    bool pass = max_err < 0.02f;

    if (pass) {
        printf("PASS (max_abs_err=%.4e)\n", max_err);
    } else {
        int b = bad_idx / (Hq * HEAD_DIM);
        int h = (bad_idx / HEAD_DIM) % Hq;
        int d = bad_idx % HEAD_DIM;
        printf("FAIL (max_abs_err=%.4e at [%d,%d,%d]: ref=%.4f got=%.4f)\n",
               max_err, b, h, d, h_o_ref[bad_idx], h_o_paged[bad_idx]);
        printf("  ref[0..7]:");
        for (int i = 0; i < 8 && i < HEAD_DIM; i++)
            printf(" %.4f", h_o_ref[i]);
        printf("\n  got[0..7]:");
        for (int i = 0; i < 8 && i < HEAD_DIM; i++)
            printf(" %.4f", h_o_paged[i]);
        printf("\n");
    }

    free(h_q); free(h_k_pool); free(h_v_pool); free(h_pt);
    free(h_k_cont); free(h_v_cont);
    free(h_q_f); free(h_k_f); free(h_v_f);
    free(h_o_ref); free(h_o_bf16); free(h_o_paged);
    cudaFree(d_q); cudaFree(d_o_paged);
    cudaFree(d_k_pool); cudaFree(d_v_pool); cudaFree(d_pt);
    cudaFree(d_op); cudaFree(d_ml);

    return pass ? 0 : 1;
}

struct TestCase {
    int head_dim;
    int B, Hq, Hkv, kv_len, page_size, causal, seed;
};

static const TestCase TESTS[] = {
    {128, 1, 1, 1, 8, 128, 0, 1},
    {128, 1, 4, 4, 128, 128, 0, 2},
    {128, 2, 4, 4, 256, 128, 0, 3},
    {128, 1, 4, 1, 64, 64, 0, 4},
    {128, 1, 8, 2, 64, 128, 0, 5},
    {128, 2, 16, 4, 128, 128, 0, 6},
    {64, 1, 4, 2, 32, 128, 0, 7},
    {256, 1, 2, 1, 16, 128, 0, 8},
    {32, 1, 4, 2, 32, 64, 0, 9},
    {128, 3, 8, 2, 256, 128, 0, 10},
    {128, 2, 32, 8, 512, 128, 0, 11},
    {128, 1, 16, 2, 256, 128, 0, 12},
    {128, 2, 32, 4, 512, 128, 0, 13},
    {128, 2, 8, 2, 128, 128, 1, 14},   // causal
};

static int dispatch_test(const TestCase& tc) {
    int r = 0;
    dispatch_by_head_dim(tc.head_dim, [&]<int D>() {
        r = run_test<D>(tc.B, tc.Hq, tc.Hkv, tc.kv_len, tc.page_size, tc.causal, tc.seed);
    });
    return r;
}

template <int HEAD_DIM>
static void bench_config(int B, int Hq, int Hkv, int kv_len, int page_size) {
    int max_pages = (kv_len + page_size - 1) / page_size;
    int n_phys_pages = B * max_pages;
    int max_splits = 32;

    size_t sz_q  = (size_t)B * Hq * 1 * HEAD_DIM * sizeof(bf16);
    size_t sz_kv = (size_t)n_phys_pages * page_size * Hkv * HEAD_DIM * sizeof(bf16);
    size_t sz_pt = (size_t)B * max_pages * sizeof(int64_t);
    size_t sz_op = (size_t)B * Hq * max_splits * HEAD_DIM * sizeof(float);
    size_t sz_ml = (size_t)B * Hq * max_splits * 2 * sizeof(float);

    bf16 *d_q, *d_o, *d_k_pool, *d_v_pool;
    int64_t* d_pt;
    float *d_op, *d_ml;
    cudaMalloc(&d_q, sz_q); cudaMalloc(&d_o, sz_q);
    cudaMalloc(&d_k_pool, sz_kv); cudaMalloc(&d_v_pool, sz_kv);
    cudaMalloc(&d_pt, sz_pt);
    cudaMalloc(&d_op, sz_op); cudaMalloc(&d_ml, sz_ml);

    bf16* tmp = (bf16*)malloc(sz_kv > sz_q ? sz_kv : sz_q);
    for (size_t i = 0; i < sz_q / sizeof(bf16); i++) tmp[i] = f2bf(randf());
    cudaMemcpy(d_q, tmp, sz_q, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < sz_kv / sizeof(bf16); i++) tmp[i] = f2bf(randf());
    cudaMemcpy(d_k_pool, tmp, sz_kv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_pool, tmp, sz_kv, cudaMemcpyHostToDevice);

    int64_t* h_pt = (int64_t*)malloc(sz_pt);
    int next_pg = 0;
    for (int b = 0; b < B; b++)
        for (int p = 0; p < max_pages; p++)
            h_pt[b * max_pages + p] = next_pg++;
    cudaMemcpy(d_pt, h_pt, sz_pt, cudaMemcpyHostToDevice);
    free(h_pt);

    PagedAttentionParams<bf16> pa;
    pa.batch = B; pa.q_head = Hq; pa.kv_head = Hkv; pa.q_len = 1;
    pa.kv_len = kv_len; pa.head_dim = HEAD_DIM;
    pa.use_mask = 0; pa.causal_offset = -1;
    set_default_paged_strides(pa);
    pa.scale = 1.0f / sqrtf((float)HEAD_DIM);
    pa.page_size = page_size; pa.max_pages = max_pages;
    pa.page_table = d_pt;
    pa.k_cache = d_k_pool; pa.v_cache = d_v_pool;
    pa.q = d_q; pa.mask = nullptr; pa.o = d_o;
    pa.o_part = d_op; pa.ml_part = d_ml;

    const int WARMUP = 10, ITERS = 100;
    auto launch = [&]() {
        dispatch_by_head_dim(HEAD_DIM, [&]<int H>() { dispatch_paged_decode<H>(pa); });
    };
    double flops = 4.0 * B * Hq * (double)kv_len * HEAD_DIM;
    size_t nKV = (size_t)B * Hkv * kv_len * HEAD_DIM;
    double bytes = 2.0 * (2.0 * nKV * sizeof(bf16));
    BenchResult r = bench_kernel(launch, WARMUP, ITERS, flops, bytes);

    char cfg[64];
    snprintf(cfg, sizeof(cfg),
             "B=%2d Hq=%2d Hk=%d q=%4d kv=%4d D=%3d page=%3d",
             B, Hq, Hkv, 1, kv_len, HEAD_DIM, page_size);
    print_bench_row(cfg, r);

    free(tmp);
    cudaFree(d_q); cudaFree(d_o);
    cudaFree(d_k_pool); cudaFree(d_v_pool); cudaFree(d_pt);
    cudaFree(d_op); cudaFree(d_ml);
}

static void bench() {
    printf("\n===== PAGED DECODE BENCH =====\n");
    print_bench_header();
    bench_config<128>(1, 32, 4, 512, 128);
    bench_config<128>(1, 32, 4, 1024, 128);
    bench_config<128>(1, 32, 4, 2048, 128);
    bench_config<128>(1, 32, 4, 4096, 128);
    bench_config<128>(16, 32, 4, 2048, 128);
    bench_config<128>(32, 32, 4, 1024, 128);
}

int main() {
    int n = sizeof(TESTS) / sizeof(TESTS[0]);
    int fail = 0;
    printf("=== Paged Decode vs CPU reference (%d cases) ===\n\n", n);

    for (int i = 0; i < n; i++) {
        fail += dispatch_test(TESTS[i]);
        if (fail) break;
    }

    if (fail) {
        printf("\nFAILED (%d/%d tests failed)\n", fail, n);
        return fail;
    }
    printf("\nAll %d tests passed!\n", n);
    bench();
    return 0;
}
