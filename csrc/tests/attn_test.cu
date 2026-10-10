/*
Pure-C test — uses shared dispatcher.  Combines the decode (split-KV) and
prefill (split-Q) correctness checks + benchmarks into one binary; one
harness serves both (decode allocates the split scratch).
nvcc -I csrc/include -arch=sm_89 -O3 \
    --use_fast_math --ptxas-options=-O3 --extra-device-vectorization \
    -Xcompiler -fopenmp csrc/tests/attn_test.cu -o test && ./test
*/

#include "test_utils.cuh"
#include <launcher/attention.cuh>

#include <cstring>

using namespace astrai::attention;

/* Dispatch resolves head_dim from the element type, keeping the harness torch-free. */
using bf16 = astrai::bf16;

// Plan checks exercise the public dispatcher without allocating or launching kernels.
static bool same_decode_plan(const DecodeLaunchPlan& a, const DecodeLaunchPlan& b) {
    return a.grid.x == b.grid.x && a.grid.y == b.grid.y && a.grid.z == b.grid.z &&
           a.direct_output == b.direct_output;
}

template <int D, typename KV, bool HasMask>
static bool decode_type_matches(DecodeKernel<D, KV, HasMask>, int dim, bool has_mask) {
    return D == dim && HasMask == has_mask;
}

template <typename KV>
static int check_decode_query(bool paged, int dim, bool has_mask, bool causal) {
    bool mask = true;
    AttentionParams p{};
    p.batch = 3;
    p.q_head = 66; // 33 query heads per KV head requires three passes.
    p.kv_head = 2;
    p.head_dim = dim;
    p.q_len = 1;
    p.kv_len = 17;
    p.max_context_len = 513;
    p.is_causal = causal;
    p.mask = has_mask ? &mask : nullptr;
    // Planning must neither trust the prior split count nor modify caller state.
    p.num_splits = MAX_SPLITS + 7;
    unsigned char before[sizeof(p)];
    std::memcpy(before, &p, sizeof(p));
    bool pass = false;
    with_decode_kernel<KV>(p, [&](auto kernel) {
        const auto query = kernel.query(p);
        const auto first = kernel.plan(p);
        pass = decode_type_matches(kernel, dim, has_mask) && query.batch == 3 &&
               query.q_heads == 66 && query.kv_heads == 2 && query.head_dim == dim &&
               query.kv_tiles == (paged ? 33 : 2) && query.wave_capacity > 0 && first.grid.x == 6 &&
               first.grid.y == 3 && std::memcmp(before, &p, sizeof(p)) == 0;
        p.num_splits = 0;
        std::memcpy(before, &p, sizeof(p));
        const auto second = kernel.plan(p);
        pass = pass && same_decode_plan(first, second) && std::memcmp(before, &p, sizeof(p)) == 0;
        // Reusing the parameter object for shorter KV must produce a fresh plan.
        p.kv_len = 1;
        p.max_context_len = 1;
        p.num_splits = MAX_SPLITS;
    });
    const auto shorter = with_decode_kernel<KV>(p, [&](auto kernel) { return kernel.plan(p); });
    pass = pass && shorter.grid.z == 1 && shorter.direct_output == (dim <= 128);
    if (!pass)
        printf("FAILED decode query: paged=%d D=%d mask=%d causal=%d\n", paged, dim, has_mask,
               causal);
    return pass ? 0 : 1;
}

static int run_plan_tests() {
    struct Case {
        DecodePlanQuery query;
        unsigned int x, y, splits;
        bool direct;
    };
    const Case cases[] = {
        {{1, 8, 1, 32, 0, 128}, 1, 1, 1, true}, // Empty KV is still one block.
        {{1, 8, 1, 64, 3, 128}, 1, 1, 1, true}, // Need two KV tiles per split.
        {{1, 8, 1, 128, 4, 128}, 1, 1, 2, false},
        {{1, 8, 1, 256, 2, 128}, 1, 1, 1, false}, // D256 retains the combine pass.
        {{1, 8, 1, 128, 65, 1024}, 1, 1, MAX_SPLITS, false},
        {{64, 32, 4, 128, 256, 128}, 4, 64, 1, true}, // Already exceeds one wave.
        {{2, 64, 2, 128, 256, 128}, 4, 2, 16, false}, // Exactly 32 heads per KV.
        {{2, 66, 2, 128, 256, 128}, 6, 2, 10, false}, // 33 heads needs third pass.
        {{1, 64, 4, 128, 256, 7}, 4, 1, 1, true},     // Do not cross a wave boundary.
        {{1, 64, 4, 128, 256, 8}, 4, 1, 2, false},
    };
    int fail = 0;
    for (int i = 0; i < static_cast<int>(sizeof(cases) / sizeof(cases[0])); ++i) {
        const auto& c = cases[i];
        const auto plan = make_decode_plan(c.query);
        if (plan.grid.x != c.x || plan.grid.y != c.y || plan.grid.z != c.splits ||
            plan.direct_output != c.direct) {
            printf("FAILED decode split policy case %d\n", i);
            ++fail;
        }
    }

    for (int dim : {32, 64, 128, 256}) {
        for (bool mask : {false, true}) {
            for (bool causal : {false, true}) {
                fail += check_decode_query<ContigKV<bf16>>(false, dim, mask, causal);
                fail += check_decode_query<PagedKV<bf16>>(true, dim, mask, causal);
            }
        }
    }
    AttentionParams unsupported{};
    unsupported.head_dim = 96;
    bool rejected = false;
    try {
        with_decode_kernel<ContigKV<bf16>>(unsupported, [](auto) {});
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("96") != std::string::npos;
    }
    if (!rejected) {
        printf("FAILED unsupported decode head dimension\n");
        ++fail;
    }
    printf("Attention plan tests: %s\n", fail ? "FAILED" : "passed");
    return fail;
}

// Split-K scratch (torch-free)
struct DecodeScratch {
    float* o_part = nullptr;
    float* ml_part = nullptr;
};

static void setup_scratch(AttentionParams& p, DecodeScratch& sc) {
    int max_splits = 32;
    cudaMalloc(&sc.o_part, (size_t)p.batch * p.q_head * max_splits * p.head_dim * sizeof(float));
    cudaMalloc(&sc.ml_part, (size_t)p.batch * p.q_head * max_splits * 2 * sizeof(float));
}

static void free_scratch(DecodeScratch& sc) {
    cudaFree(sc.o_part);
    cudaFree(sc.ml_part);
}

/* Shared contiguous test harness for decode, prefill, and CPU-reference checks. */
static int run_contig_test(int B, int Hq, int Hk, int ql, int kl, int D, int causal, bool decode) {
    size_t nQ = (size_t)B * Hq * ql * D, nKV = (size_t)B * Hk * kl * D;
    float *hQ = new float[nQ], *hK = new float[nKV], *hV = new float[nKV];
    for (size_t i = 0; i < nQ; i++)
        hQ[i] = randf();
    for (size_t i = 0; i < nKV; i++) {
        hK[i] = randf();
        hV[i] = randf();
    }

    bf16 *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, nQ * 2);
    cudaMalloc(&dK, nKV * 2);
    cudaMalloc(&dV, nKV * 2);
    cudaMalloc(&dO, nQ * 2);
    bf16* tmp = new bf16[nQ > nKV ? nQ : nKV];
    for (size_t i = 0; i < nQ; i++)
        tmp[i] = f2bf(hQ[i]);
    cudaMemcpy(dQ, tmp, nQ * 2, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < nKV; i++)
        tmp[i] = f2bf(hK[i]);
    cudaMemcpy(dK, tmp, nKV * 2, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < nKV; i++)
        tmp[i] = f2bf(hV[i]);
    cudaMemcpy(dV, tmp, nKV * 2, cudaMemcpyHostToDevice);

    AttentionParams p = {};
    p.batch = B;
    p.q_head = Hq;
    p.kv_head = Hk;
    p.q_len = ql;
    p.kv_len = kl;
    p.head_dim = D;
    p.is_causal = causal;
    p.scale = 1.0f / sqrtf((float)D);
    set_default_strides(p);
    p.q_ptr = dQ;
    p.k_ptr = dK;
    p.v_ptr = dV;
    p.mask = nullptr;
    p.o_ptr = dO;

    DecodeScratch sc;
    if (decode) {
        setup_scratch(p, sc);
        p.o_part = sc.o_part;
        p.ml_part = sc.ml_part;
        AttnDispatchDecode<bf16>::run(p, 0);
    } else {
        AttnDispatchPrefill<bf16>::run(p, 0);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA err: %s\n", cudaGetErrorString(err));
        return 1;
    }

    bf16* hOut = new bf16[nQ];
    cudaMemcpy(hOut, dO, nQ * 2, cudaMemcpyDeviceToHost);

    float* ref = new float[nQ];
    cpu_attention_ref(hQ, hK, hV, nullptr, ref, B, Hq, Hk, ql, kl, D, causal);

    float max_abs_err = 0, max_rel_err = 0;
    bool pass = true;
    const float atol = 0.01f, rtol = 0.01f;
    for (size_t i = 0; i < nQ; i++) {
        float err = fabsf(bf2f(hOut[i]) - ref[i]);
        if (err > max_abs_err)
            max_abs_err = err;
        float rel = err / fmaxf(fabsf(ref[i]), 1e-4f);
        if (rel > max_rel_err)
            max_rel_err = rel;
        if (!std::isfinite(bf2f(hOut[i])) || !std::isfinite(ref[i]) ||
            err > atol + rtol * fabsf(ref[i]))
            pass = false;
    }
    char cfg[64];
    snprintf(cfg, sizeof(cfg), "B=%2d Hq=%2d Hk=%d q=%4d kv=%4d D=%3d causal=%d", B, Hq, Hk, ql, kl,
             D, causal);
    print_test_row(cfg, max_abs_err, max_rel_err, pass);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    if (decode)
        free_scratch(sc);
    delete[] hQ;
    delete[] hK;
    delete[] hV;
    delete[] hOut;
    delete[] ref;
    delete[] tmp;
    return pass ? 0 : 1;
}

/* Time the same decode/prefill dispatch used by the correctness harness. */
static void bench_contig(int B, int Hq, int Hk, int ql, int kl, int D, int causal, bool decode) {
    size_t nQ = (size_t)B * Hq * ql * D, nKV = (size_t)B * Hk * kl * D;
    bf16 *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, nQ * 2);
    cudaMalloc(&dK, nKV * 2);
    cudaMalloc(&dV, nKV * 2);
    cudaMalloc(&dO, nQ * 2);
    bf16* tmp = new bf16[nQ > nKV ? nQ : nKV];
    for (size_t i = 0; i < nQ; i++)
        tmp[i] = f2bf(randf());
    cudaMemcpy(dQ, tmp, nQ * 2, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < nKV; i++)
        tmp[i] = f2bf(randf());
    cudaMemcpy(dK, tmp, nKV * 2, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < nKV; i++)
        tmp[i] = f2bf(randf());
    cudaMemcpy(dV, tmp, nKV * 2, cudaMemcpyHostToDevice);
    delete[] tmp;

    AttentionParams p = {};
    p.batch = B;
    p.q_head = Hq;
    p.kv_head = Hk;
    p.q_len = ql;
    p.kv_len = kl;
    p.head_dim = D;
    p.is_causal = causal;
    p.scale = 1.0f / sqrtf((float)D);
    set_default_strides(p);
    p.q_ptr = dQ;
    p.k_ptr = dK;
    p.v_ptr = dV;
    p.mask = nullptr;
    p.o_ptr = dO;

    DecodeScratch sc;
    if (decode) {
        setup_scratch(p, sc);
        p.o_part = sc.o_part;
        p.ml_part = sc.ml_part;
    }

    double flops = 4.0 * B * Hq * (double)ql * kl * D;
    if (causal)
        flops *= 0.5;
    BenchResult r = decode
                        ? bench_kernel([&] { AttnDispatchDecode<bf16>::run(p, 0); }, 3, 10, flops)
                        : bench_kernel([&] { AttnDispatchPrefill<bf16>::run(p, 0); }, 3, 10, flops);

    char cfg[64];
    snprintf(cfg, sizeof(cfg), "B=%2d Hq=%2d Hk=%d q=%4d kv=%4d D=%3d causal=%d", B, Hq, Hk, ql, kl,
             D, causal);
    print_bench_row(cfg, r);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    if (decode)
        free_scratch(sc);
}

int main() {
    int fail = run_plan_tests();
    if (fail)
        return fail;

    // DECODE
    {
        // B, Hq, Hk, seq_len, D, causal
        const int configs[][6] = {
            {1, 2, 1, 64, 32, 0},
            {1, 32, 4, 512, 128, 0},
            {1, 32, 4, 1024, 128, 0},
            {1, 32, 4, 512, 128, 1},
        };
        int n_cfgs = sizeof(configs) / sizeof(configs[0]);
        printf("=== DECODE TESTS ===\n");
        print_test_header();
        for (int ci = 0; ci < n_cfgs; ci++) {
            int B = configs[ci][0], Hq = configs[ci][1], Hk = configs[ci][2];
            int sl = configs[ci][3], D = configs[ci][4], causal = configs[ci][5];
            fail += run_contig_test(B, Hq, Hk, 1, sl, D, causal, /*decode=*/true);
            if (fail)
                break;
        }
        if (fail) {
            printf("FAILED decode tests\n");
            return fail;
        }

        // B, Hq, Hk, seq_len (D=128, non-causal)
        const int bc[][4] = {
            {1, 32, 4, 512},  {1, 32, 4, 1024},  {1, 32, 4, 2048},
            {1, 32, 4, 4096}, {16, 32, 4, 2048}, {32, 32, 4, 1024},
        };
        printf("\n===== DECODE BENCH (warmup=%d iters=%d) =====\n", 3, 10);
        print_bench_header();
        for (int ci = 0; ci < (int)(sizeof(bc) / sizeof(bc[0])); ci++)
            bench_contig(bc[ci][0], bc[ci][1], bc[ci][2], 1, bc[ci][3], 128, 0, /*decode=*/true);
    }

    // PREFILL
    {
        // B, Hq, Hk, q_len, kv_len, D, causal
        const int configs[][7] = {
            {1, 2, 1, 64, 128, 32, 0},    // smallest head_dim D=32
            {1, 4, 2, 256, 256, 32, 1},   // causal D=32 dispatch
            {1, 2, 1, 64, 128, 64, 0},    // tiny: B,Hq,Hk,q,kv,D,causal
            {1, 2, 1, 64, 128, 64, 1},    // Causal queries align to the KV tail.
            {1, 2, 1, 128, 64, 64, 1},    // Leading query rows have no visible keys.
            {1, 4, 2, 256, 256, 64, 1},   // causal D=64 dispatch
            {1, 32, 4, 512, 512, 128, 0}, // standard
            {1, 32, 4, 128, 256, 128, 0}, // medium
            {1, 4, 2, 256, 256, 128, 1},  // causal
        };
        int n_configs = sizeof(configs) / sizeof(configs[0]);
        printf("\n=== PREFILL TESTS ===\n");
        print_test_header();
        for (int ci = 0; ci < n_configs; ci++) {
            int B = configs[ci][0], Hq = configs[ci][1], Hk = configs[ci][2];
            int ql = configs[ci][3], kl = configs[ci][4], D = configs[ci][5];
            int causal = configs[ci][6];
            fail += run_contig_test(B, Hq, Hk, ql, kl, D, causal, /*decode=*/false);
            if (fail)
                break;
        }
        if (fail) {
            printf("FAILED prefill tests\n");
            return fail;
        }

        // B, Hq, Hk, q_len, kv_len, D, causal
        const int bc[][7] = {
            {1, 32, 4, 1024, 1024, 32, 0},  {1, 32, 4, 1024, 1024, 32, 1},
            {1, 32, 4, 4096, 4096, 32, 1},  {1, 32, 4, 1024, 1024, 64, 0},
            {1, 32, 4, 1024, 1024, 64, 1},  {1, 32, 4, 4096, 4096, 64, 1},
            {1, 32, 4, 512, 512, 128, 0},   {1, 32, 4, 1024, 1024, 128, 0},
            {1, 32, 4, 2048, 2048, 128, 0}, {1, 32, 4, 2048, 2048, 128, 1},
            {4, 32, 4, 2048, 2048, 128, 1}, {1, 32, 4, 4096, 4096, 128, 1},
        };
        printf("\n===== PREFILL BENCH (warmup=%d iters=%d) =====\n", 3, 10);
        print_bench_header();
        for (int ci = 0; ci < (int)(sizeof(bc) / sizeof(bc[0])); ci++)
            bench_contig(bc[ci][0], bc[ci][1], bc[ci][2], bc[ci][3], bc[ci][4], bc[ci][5],
                         bc[ci][6], /*decode=*/false);
    }

    printf("\nAll tests passed!\n");
    return 0;
}
