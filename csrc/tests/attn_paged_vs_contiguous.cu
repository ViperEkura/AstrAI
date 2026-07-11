// Compile:
//   nvcc -I csrc -arch=sm_89 -O3 --use_fast_math --ptxas-options=-O3 \
//        --extra-device-vectorization csrc/tests/attn_paged_vs_contiguous.cu \
//        -o /tmp/test_pv && /tmp/test_pv

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include "../kernels/attn_paged_decode_split_kv.cuh"
#ifndef ASTRAI_NO_MMA
#include "../kernels/attn_paged_decode_split_kv_mma.cuh"
#endif

using bf16 = __nv_bfloat16;

static int num_splits(int base_blocks, int tiles_total) {
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int n = (2 * sm_count + base_blocks - 1) / base_blocks;
    return std::max(1, std::min(n, std::min(tiles_total, 32)));
}

// Copy contiguous K/V from page pool (reference gather)
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
                size_t dst_base = ((size_t)b * kv_len + pos) * Hkv * head_dim + h * head_dim;
                memcpy(h_k + dst_base, h_k_pool + src_base, head_dim * sizeof(bf16));
                memcpy(h_v + dst_base, h_v_pool + src_base, head_dim * sizeof(bf16));
            }
        }
    }
}

template <int HEAD_DIM>
static int run_test(int B, int Hq, int Hkv, int kv_len, int page_size, int seed) {
    printf("B=%d Hq=%d Hkv=%d kv_len=%d page_sz=%d head_dim=%d ... ", B, Hq, Hkv, kv_len, page_size, HEAD_DIM);
    fflush(stdout);

    int G = Hq / Hkv;
    int max_pages = (kv_len + page_size - 1) / page_size;
    int n_phys_pages = B * max_pages;

    // ---- allocate ----
    bf16 *d_q, *d_o_paged, *d_o_ref;
    bf16 *d_k_pool, *d_v_pool;
    int64_t* d_pt;
    float *d_op, *d_ml;

    size_t sz_q  = (size_t)B * Hq * 1 * HEAD_DIM * sizeof(bf16);
    size_t sz_o  = sz_q;
    size_t sz_kv = (size_t)n_phys_pages * page_size * Hkv * HEAD_DIM * sizeof(bf16);
    size_t sz_pt = (size_t)B * max_pages * sizeof(int64_t);
    int max_splits = 32;
    size_t sz_op = (size_t)B * Hq * max_splits * HEAD_DIM * sizeof(float);
    size_t sz_ml = (size_t)B * Hq * max_splits * 2 * sizeof(float);

    cudaMalloc(&d_q, sz_q);
    cudaMalloc(&d_o_paged, sz_o);
    cudaMalloc(&d_o_ref, sz_o);
    cudaMalloc(&d_k_pool, sz_kv);
    cudaMalloc(&d_v_pool, sz_kv);
    cudaMalloc(&d_pt, sz_pt);
    cudaMalloc(&d_op, sz_op);
    cudaMalloc(&d_ml, sz_ml);

    // ---- init: deterministic random using seed ----
    srand(seed);
    auto rnd = [&]() { return (rand() / (float)RAND_MAX) * 2.0f - 1.0f; };

    // Q
    bf16* h_q = (bf16*)malloc(sz_q);
    for (int i = 0; i < B * Hq * HEAD_DIM; i++)
        h_q[i] = __float2bfloat16(rnd());
    cudaMemcpy(d_q, h_q, sz_q, cudaMemcpyHostToDevice);

    // Page pool K/V
    bf16* h_k_pool = (bf16*)malloc(sz_kv);
    bf16* h_v_pool = (bf16*)malloc(sz_kv);
    size_t ps = (size_t)page_size * Hkv * HEAD_DIM;
    for (int pg = 0; pg < n_phys_pages; pg++) {
        for (int off = 0; off < page_size; off++) {
            for (int h = 0; h < Hkv; h++) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    float v = sinf((float)(pg * 7919 + off * 1049 + h * 331 + d));
                    size_t idx = (size_t)pg * ps + (size_t)off * Hkv * HEAD_DIM + h * HEAD_DIM + d;
                    h_k_pool[idx] = __float2bfloat16(v);
                    h_v_pool[idx] = __float2bfloat16(v * 0.3f);
                }
            }
        }
    }
    cudaMemcpy(d_k_pool, h_k_pool, sz_kv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_pool, h_v_pool, sz_kv, cudaMemcpyHostToDevice);

    // Page table
    int64_t* h_pt = (int64_t*)malloc(sz_pt);
    int next_pg = 0;
    for (int b = 0; b < B; b++)
        for (int p = 0; p < max_pages; p++)
            h_pt[b * max_pages + p] = next_pg++;
    cudaMemcpy(d_pt, h_pt, sz_pt, cudaMemcpyHostToDevice);

    // ---- reference: gather contiguous K/V, then run CPU online-softmax ----
    bf16* h_k_cont = (bf16*)malloc((size_t)B * kv_len * Hkv * HEAD_DIM * sizeof(bf16));
    bf16* h_v_cont = (bf16*)malloc((size_t)B * kv_len * Hkv * HEAD_DIM * sizeof(bf16));
    gather_kv_cpu(h_k_pool, h_v_pool, h_pt, B, Hkv, kv_len, page_size, HEAD_DIM, h_k_cont, h_v_cont);

    float* h_o_ref = (float*)calloc(B * Hq * HEAD_DIM, sizeof(float));
    float sscale = 1.0f / sqrtf((float)HEAD_DIM);
    for (int b = 0; b < B; b++) {
        for (int hq = 0; hq < Hq; hq++) {
            int hkv = hq / G;
            size_t q_base = (size_t)b * Hq * HEAD_DIM + (size_t)hq * HEAD_DIM;
            size_t kv_base = ((size_t)b * kv_len) * Hkv * HEAD_DIM + (size_t)hkv * HEAD_DIM;

            float m = -1e30f, d = 0.0f;
            float acc[256] = {0.0f};
            for (int pos = 0; pos < kv_len; pos++) {
                float s = 0.0f;
                for (int dim = 0; dim < HEAD_DIM; dim++)
                    s += __bfloat162float(h_q[q_base + dim])
                       * __bfloat162float(h_k_cont[kv_base + (size_t)pos * Hkv * HEAD_DIM + dim]);
                s *= sscale;

                float nm = fmaxf(m, s);
                float a = expf(m - nm);
                float b = expf(s - nm);
                d = d * a + b;
                for (int dim = 0; dim < HEAD_DIM; dim++)
                    acc[dim] = acc[dim] * a + __bfloat162float(h_v_cont[kv_base + (size_t)pos * Hkv * HEAD_DIM + dim]) * b;
                m = nm;
            }
            for (int dim = 0; dim < HEAD_DIM; dim++)
                h_o_ref[b * Hq * HEAD_DIM + hq * HEAD_DIM + dim] = acc[dim] / d;
        }
    }

    // ---- paged decode kernel ----
    float scale_val = 1.0f / sqrtf((float)HEAD_DIM);
    PagedAttentionParams<bf16, float> p;
    p.batch = B;
    p.q_head = Hq;
    p.kv_head = Hkv;
    p.q_len = 1;
    p.kv_len = kv_len;
    p.head_dim = HEAD_DIM;
    p.use_mask = 0;
    p.is_causal = 0;
    p.causal_offset = 0;
    p.num_splits = 1;
    p.scale = scale_val;
    p.page_size = page_size;
    p.max_pages = max_pages;
    p.page_table = d_pt;
    p.k_cache = d_k_pool;
    p.v_cache = d_v_pool;
    p.q = d_q;
    p.mask = nullptr;
    p.o = d_o_paged;
    p.o_part = d_op;
    p.ml_part = d_ml;

    // Dispatch
#ifndef ASTRAI_NO_MMA
    int G_check = p.q_head / p.kv_head;
    bool use_mma = !p.use_mask && G_check >= 1 && G_check <= 16 && p.page_size >= 32;
    if (use_mma) {
        int tiles_total = (p.kv_len + 32 - 1) / 32;
        p.num_splits = num_splits(p.batch * p.kv_head, tiles_total);
        paged_attn_decode_split_kv_mma_kernel<HEAD_DIM, 32>
            <<<dim3(p.kv_head, p.batch, p.num_splits), 32>>>(p);
    } else
#endif
    {
        int group_sz = p.q_head / p.kv_head;
        int chunks_total = (p.kv_len + PDC_CHUNK - 1) / PDC_CHUNK;
        p.num_splits = num_splits(p.batch * p.kv_head, chunks_total);
        size_t smem = PDC_CHUNK * p.head_dim * sizeof(bf16);
        paged_attn_decode_split_kv_kernel<<<
            dim3(p.batch * p.kv_head, 1, p.num_splits),
            dim3(32, group_sz), smem>>>(p);
    }
    paged_attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
    cudaDeviceSynchronize();

    // Download paged output
    bf16* h_o_bf16 = (bf16*)malloc(sz_o);
    cudaMemcpy(h_o_bf16, d_o_paged, sz_o, cudaMemcpyDeviceToHost);
    float* h_o_paged = (float*)malloc(B * Hq * HEAD_DIM * sizeof(float));
    for (int i = 0; i < B * Hq * HEAD_DIM; i++)
        h_o_paged[i] = __bfloat162float(h_o_bf16[i]);

    // Compare
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
        // Print first 8 dims of first head
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
    free(h_o_ref); free(h_o_bf16); free(h_o_paged);
    cudaFree(d_q); cudaFree(d_o_paged); cudaFree(d_o_ref);
    cudaFree(d_k_pool); cudaFree(d_v_pool); cudaFree(d_pt);
    cudaFree(d_op); cudaFree(d_ml);

    return pass ? 0 : 1;
}

int main() {
    int fail = 0;
    printf("=== Paged Decode vs CPU reference ===\n\n");

    printf("-- scalar (G=1) --\n");
    fail += run_test<128>(1, 1, 1, 8, 128, 1);
    fail += run_test<128>(1, 4, 4, 128, 128, 2);
    fail += run_test<128>(2, 4, 4, 256, 128, 3);
    fail += run_test<128>(1, 4, 1, 64, 64, 4);

    printf("-- scalar (G>1) --\n");
    fail += run_test<128>(1, 8, 2, 64, 128, 5);
    fail += run_test<128>(2, 16, 4, 128, 128, 6);

    printf("-- varying head_dim --\n");
    fail += run_test<64>(1, 4, 2, 32, 128, 7);
    fail += run_test<256>(1, 2, 1, 16, 128, 8);
    fail += run_test<32>(1, 4, 2, 32, 64, 9);

    printf("-- multi-batch --\n");
    fail += run_test<128>(3, 8, 2, 256, 128, 10);
    fail += run_test<128>(2, 32, 8, 512, 128, 11);

#ifndef ASTRAI_NO_MMA
    printf("-- MMA (G>1, sm_80+) --\n");
    fail += run_test<128>(1, 16, 2, 256, 128, 12);
    fail += run_test<128>(2, 32, 4, 512, 128, 13);
#endif

    printf("\n%s (%d/%d failed)\n", fail ? "FAILED" : "ALL PASSED", fail, fail + (13 - fail + 1));
    return fail;
}
