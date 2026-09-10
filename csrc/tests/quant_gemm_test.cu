/*
quant_gemm family test: correctness for every operand dtype combination
(fp8 / int8 / bf16 pairs, all four storage layouts, K tiles, ragged shapes,
per-row / per-channel scales, fp32 output) + a per-combination TFLOPS table
over llama-linear shapes through the production NT route.

nvcc -I csrc/kernels -arch=sm_89 -std=c++17 -O3 csrc/tests/quant_gemm_test.cu \
    -o /tmp/quant_gemm_test && /tmp/quant_gemm_test
*/

#include "test_utils.cuh"

#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

#include "common/launch.cuh"
#include "gemm/gemm.cuh"

using namespace astrai::quant;
using namespace astrai::gemm;

namespace {

// ---------------------------------------------------------------------------
// Shared GEMM correctness harness — upload / reference / launch / compare
// ---------------------------------------------------------------------------

// Element conversions: host fp32 -> ElemT for the upload, ElemT -> fp32 in
// the reference kernel, OutT -> fp32 after the download. fp8 goes through
// the type constructors; bf16 keeps the rounding intrinsics.
template <typename ElemT>
static inline ElemT to_elem(float x) {
    return ElemT(x);
}
template <>
inline int8_t to_elem<int8_t>(float x) {
    return (int8_t)fmaxf(-127.0f, fminf(127.0f, lroundf(x)));
}
template <>
inline __nv_bfloat16 to_elem<__nv_bfloat16>(float x) {
    return f2bf(x);
}
template <typename ElemT>
__device__ __forceinline__ float elem2f(ElemT x) {
    if constexpr (std::is_same_v<ElemT, __nv_bfloat16>)
        return __bfloat162float(x);
    else
        return (float)x;
}
static inline float out2f(__nv_bfloat16 x) { return __bfloat162float(x); }
static inline float out2f(float x) { return x; }

// Naive fp32 reference on the GPU (O(m*n) to check instead of O(m*n*k) to
// compute on the host). a_rm/b_rm select each operand's storage: 1 = row
// major along the output dim, 0 = contract-contiguous (the "colmajor" of
// the canonical [K][N] view).
template <typename ElemA, typename ElemB>
__global__ static void naive_gemm_ref(const ElemA* a, const ElemB* b,
                                      float* out, int m, int n, int k,
                                      int a_ld, int b_ld, int a_rm, int b_rm,
                                      const float* b_col_scale,
                                      const float* a_row_scale) {
    const int i = blockIdx.y * 32 + threadIdx.y;
    const int j = blockIdx.x * 32 + threadIdx.x;
    if (i >= m || j >= n) return;
    const float sc = b_col_scale ? b_col_scale[j] : 1.0f;
    const float sa = a_row_scale ? a_row_scale[i] : 1.0f;
    float acc = 0.f;
    for (int kk = 0; kk < k; ++kk) {
        float av = a_rm ? elem2f(a[i * a_ld + kk]) : elem2f(a[kk * a_ld + i]);
        float bv = b_rm ? elem2f(b[kk * b_ld + j]) : elem2f(b[j * b_ld + kk]);
        acc += av * bv * sc * sa;
    }
    out[i * n + j] = acc;
}

// One correctness case: convert the fp32 host operands to ElemT, upload,
// run the naive reference, launch through `dispatch(GemmParams&)`, compare
// with tolerance `tol * max(|ref|, 1.0)` (the printed max_rel normalizes
// by max(|ref|, 0.5)). ElemT and OutT are independent knobs.
template <typename ElemA, typename ElemB = ElemA, typename OutT = __nv_bfloat16,
          typename Fn>
static bool check_gemm(const float* ha, const float* hb, int m, int n, int k,
                       int a_ld, int b_ld, int a_rm, int b_rm, const char* tag,
                       float tol, Fn&& dispatch,
                       const std::vector<float>& b_col_scale = {},
                       const std::vector<float>& a_row_scale = {}) {
    const size_t na = (size_t)m * k, nb = (size_t)n * k, nout = (size_t)m * n;
    std::vector<ElemA> qa(na);
    std::vector<ElemB> qb(nb);
    for (size_t i = 0; i < na; ++i) qa[i] = to_elem<ElemA>(ha[i]);
    for (size_t i = 0; i < nb; ++i) qb[i] = to_elem<ElemB>(hb[i]);
    ElemA* da;
    ElemB* db;
    OutT* dout;
    float *d_ref, *d_scale;
    float* d_bscale = nullptr;
    float* d_ascale = nullptr;
    cudaMalloc(&da, na * sizeof(ElemA));
    cudaMalloc(&db, nb * sizeof(ElemB));
    cudaMalloc(&dout, nout * sizeof(OutT));
    cudaMalloc(&d_ref, nout * 4);
    cudaMalloc(&d_scale, 4);
    cudaMemcpy(da, qa.data(), na * sizeof(ElemA), cudaMemcpyHostToDevice);
    cudaMemcpy(db, qb.data(), nb * sizeof(ElemB), cudaMemcpyHostToDevice);
    const float one = 1.0f;  // fp8 epilogues read *scale; bf16 ignores it
    cudaMemcpy(d_scale, &one, 4, cudaMemcpyHostToDevice);
    if (!b_col_scale.empty()) {
        cudaMalloc(&d_bscale, b_col_scale.size() * 4);
        cudaMemcpy(d_bscale, b_col_scale.data(), b_col_scale.size() * 4,
                   cudaMemcpyHostToDevice);
    }
    if (!a_row_scale.empty()) {
        cudaMalloc(&d_ascale, a_row_scale.size() * 4);
        cudaMemcpy(d_ascale, a_row_scale.data(), a_row_scale.size() * 4,
                   cudaMemcpyHostToDevice);
    }
    naive_gemm_ref<ElemA, ElemB>
        <<<dim3((n + 31) / 32, (m + 31) / 32), dim3(32, 32)>>>(
            da, db, d_ref, m, n, k, a_ld, b_ld, a_rm, b_rm, d_bscale,
            d_ascale);
    ASTRAI_LAUNCH_CHECK();

    GemmParams p = {};
    p.a_ptr = da;
    p.b_ptr = db;
    p.out_ptr = dout;
    p.a_scale = d_scale;
    if (d_bscale) {
        p.b_scale = d_bscale;
        p.b_scale_n = n;
    }
    if (d_ascale) {
        p.a_scale = d_ascale;
        p.a_scale_m = m;
    }
    p.m = m;
    p.n = n;
    p.k = k;
    p.a_ld = a_ld;
    p.b_ld = b_ld;
    p.out_ld = n;
    dispatch(p);

    const cudaError_t e = cudaDeviceSynchronize();
    bool ok = e == cudaSuccess;
    double max_rel = 0;
    if (!ok) {
        printf("  %-18s CUDA err: %s\n", tag, cudaGetErrorString(e));
    } else {
        std::vector<OutT> hout(nout);
        std::vector<float> href(nout);
        cudaMemcpy(hout.data(), dout, nout * sizeof(OutT),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(href.data(), d_ref, nout * 4, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < nout && ok; ++i) {
            const float err = fabsf(out2f(hout[i]) - href[i]);
            max_rel = fmax(max_rel,
                           (double)(err / fmaxf(fabsf(href[i]), 0.5f)));
            if (err > tol * fmaxf(fabsf(href[i]), 1.0f)) ok = false;
        }
        printf("  %-18s max_rel=%.4f %s\n", tag, max_rel, ok ? "PASS" : "FAIL");
    }
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    cudaFree(d_ref);
    cudaFree(d_scale);
    if (d_bscale) cudaFree(d_bscale);
    if (d_ascale) cudaFree(d_ascale);
    return ok;
}

// ---------------------------------------------------------------------------
// Shared section helpers: storage materialization, int8 scale setup, and
// the four-layout production-dispatch sweep every dtype section reuses.
// ---------------------------------------------------------------------------

// [rows][cols] -> [cols][rows] storage materialization.
static std::vector<float> transpose(const std::vector<float>& x, int rows,
                                    int cols) {
    std::vector<float> t((size_t)rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            t[(size_t)j * rows + i] = x[(size_t)i * cols + j];
    return t;
}

// Symmetric scale setup for one host operand stored [rows][cols]: returns
// the per-row scale (amax / qmax — 127 for int8, 448/57344 for fp8) and
// divides the buffer by it, so the kernel (which re-applies the scale in
// its epilogue) and the naive reference see the same quantized values.
static std::vector<float> div_row_scales(std::vector<float>& x, int rows,
                                         int cols, float qmax = 127.f) {
    std::vector<float> s(rows);
    for (int i = 0; i < rows; ++i) {
        float amax = 1e-6f;
        for (int j = 0; j < cols; ++j)
            amax = fmaxf(amax, fabsf(x[(size_t)i * cols + j]));
        s[i] = amax / qmax;
        for (int j = 0; j < cols; ++j) x[(size_t)i * cols + j] /= s[i];
    }
    return s;
}

// One dtype pair through ALL FOUR storage layouts of the production
// dispatch: NT (congruous, the nn.Linear main path), TN/TT (crosswise A),
// NN (dual N-contiguous — the swap rewrite for symmetric pairs, the direct
// mixed instantiation otherwise). ha/hb are the canonical [M][K]/[N][K]
// storages (post-quantization); transposed variants materialize here.
template <typename ElemA, typename ElemB = ElemA>
static bool check_all_layouts(const std::vector<float>& ha,
                              const std::vector<float>& hb, int m, int n,
                              int k, const char* tag, float tol,
                              const std::vector<float>& b_scale = {},
                              const std::vector<float>& a_scale = {}) {
    const std::vector<float> ha_t = transpose(ha, m, k);  // [K][M]
    const std::vector<float> hb_t = transpose(hb, n, k);  // [K][N]
    struct Lay {
        bool ta, tb;
        int ald, bld;
        const char* name;
    };
    const Lay lays[] = {{false, true, k, k, "NT"}, {true, false, m, n, "TN"},
                        {true, true, m, k, "TT"}, {false, false, k, n, "NN"}};
    bool ok = true;
    for (const Lay& l : lays) {
        char name[32];
        snprintf(name, sizeof(name), "%s %s", tag, l.name);
        ok &= check_gemm<ElemA, ElemB>(
            l.ta ? ha_t.data() : ha.data(), l.tb ? hb.data() : hb.data(), m,
            n, k, l.ald, l.bld, !l.ta, !l.tb, name, tol,
            [&](GemmParams& p) {
                gemm_dispatch<ElemA, ElemB>(p, 0, l.ta, l.tb);
            },
            b_scale, a_scale);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// fp8 e4m3 GEMM — all four operand layouts x K-tiles x production routes
// ---------------------------------------------------------------------------

// Big-CTA policies for the direct-layout cases: kK/Stages vary per case;
// the fast interior loop follows the dual-congruous rule, grouped raster 8
// matches the production dispatch.
template <typename LA, typename LB>
constexpr bool kCaseFast =
    !std::is_same_v<LA, ColMajor> && !std::is_same_v<LB, RowMajor>;
template <typename LA, typename LB, int kK, int Stages>
using CasePolicy = GemmPolicy<
    __nv_fp8_e4m3, __nv_fp8_e4m3, LA, LB,
    GemmTileConfig<Shape<128, 128, kK>, Shape<64, 32>, Stages, kCaseFast<LA, LB>>,
    RowMajor, __nv_bfloat16>;

// fp8 e4m3 layout case: direct big-CTA policy (dispatch=0), the production
// NN-swap route (1) or the production NT route (2).
template <typename LA, typename LB, int kK, int Stages>
static bool run_gemm_case(const float* ha, const float* hb, int m, int n,
                          int k, int a_ld, int b_ld, const char* tag,
                          int dispatch = 0) {
    auto launch = [&](GemmParams& p) {
        if (dispatch == 1)
            // Production route, NN: the dual-N-contiguous problem has no
            // dedicated instantiation — canonicalize_gemm swaps to the
            // transposed <ColMajor, ColMajor> kernel with its out-transposed
            // epilogue (see gemm.cuh).
            gemm_dispatch<__nv_fp8_e4m3, __nv_fp8_e4m3>(p, 0, false, false);
        else if (dispatch == 2)
            // Production route, NT: exercises plan_gemm's small/narrow/big
            // selection for this shape.
            gemm_dispatch<__nv_fp8_e4m3, __nv_fp8_e4m3>(p, 0, false, true);
        else
            launch_policy<CasePolicy<LA, LB, kK, Stages>>(p, 0);
    };
    return check_gemm<__nv_fp8_e4m3, __nv_fp8_e4m3>(
        ha, hb, m, n, k, a_ld, b_ld,
        !std::is_same_v<LA, ColMajor>, !std::is_same_v<LB, ColMajor>, tag,
        0.06f, launch);
}

static bool test_gemm() {
    // Deterministic operand data: the fp8 tolerances below are calibrated
    // against a fixed rand stream (seeded here — never rely on the default
    // first-call seed).
    srand(0);
    struct {
        int m, n, k;
    } cfgs[] = {
        {128, 128, 128}, {256, 128, 256}, {128, 256, 64},
        {100, 130, 96},  {64, 64, 160},   {300, 200, 320},
        {2048, 256, 512}, {1024, 1024, 512},
    };
    bool all = true;
    for (auto& c : cfgs) {
        std::vector<float> ha((size_t)c.m * c.k), hb((size_t)c.k * c.n);
        for (float& v : ha) v = randf();
        for (float& v : hb) v = randf();
        const std::vector<float> hb_col = transpose(hb, c.k, c.n);  // [N][K]
        const std::vector<float> ha_t = transpose(ha, c.m, c.k);    // [K][M]
        printf("%dx%dx%d:\n", c.m, c.n, c.k);
        all &= run_gemm_case<RowMajor, ColMajor, 32, 3>(ha.data(), hb_col.data(),
                                                        c.m, c.n, c.k, c.k, c.k,
                                                        "NT K32");
        all &= run_gemm_case<RowMajor, ColMajor, 64, 2>(ha.data(), hb_col.data(),
                                                        c.m, c.n, c.k, c.k, c.k,
                                                        "NT K64");
        all &= run_gemm_case<RowMajor, RowMajor, 64, 2>(
            ha.data(), hb.data(), c.m, c.n, c.k, c.k, c.n, "NN swap", 1);
        all &= run_gemm_case<RowMajor, ColMajor, 64, 2>(
            ha.data(), hb_col.data(), c.m, c.n, c.k, c.k, c.k, "NT disp", 2);
        all &= run_gemm_case<ColMajor, ColMajor, 32, 3>(ha_t.data(), hb_col.data(),
                                                        c.m, c.n, c.k, c.m, c.k,
                                                        "TN K32");
        all &= run_gemm_case<ColMajor, ColMajor, 64, 2>(ha_t.data(), hb_col.data(),
                                                        c.m, c.n, c.k, c.m, c.k,
                                                        "TN K64");
        all &= run_gemm_case<ColMajor, RowMajor, 64, 2>(ha_t.data(), hb.data(),
                                                        c.m, c.n, c.k, c.m, c.n,
                                                        "TT K64");
    }
    return all;
}

// ---------------------------------------------------------------------------
// Dtype combinations — bf16/int8/fp8 operand pairs (all four layouts),
// per-row / per-channel scales, fp32 output
// ---------------------------------------------------------------------------

static bool test_dtype_combos() {
    bool all = true;
    auto prep = [](std::vector<float>& a, std::vector<float>& b, int m, int n,
                   int k, int seed) {
        srand(seed);
        a.resize((size_t)m * k);
        b.resize((size_t)n * k);
        for (float& v : a) v = randf();
        for (float& v : b) v = randf();
    };

    // W16A16: all four layouts through the production dispatch, plus pinned
    // big/small tile variants (the plan ladder routes 256x256 to the small
    // CTA, so the big CTA needs pinning).
    using Bf16Big =
        GemmPolicy<__nv_bfloat16, __nv_bfloat16, RowMajor, ColMajor, Tile_128x128x64_W64x32_S2_Fast,
                   RowMajor, __nv_bfloat16>;
    using Bf16Small =
        GemmPolicy<__nv_bfloat16, __nv_bfloat16, RowMajor, ColMajor, Tile_64x64x64_W16x32_S3_Fast,
                   RowMajor, __nv_bfloat16>;
    printf("W16A16 (bf16 x bf16, all layouts):\n");
    for (int k : {64, 128, 320, 512}) {
        std::vector<float> ha, hb;
        prep(ha, hb, 256, 256, k, 1234 + k);
        printf(" 256x256x%d:\n", k);
        all &= check_all_layouts<__nv_bfloat16>(ha, hb, 256, 256, k, "w16a16",
                                                0.02f);
        all &= check_gemm<__nv_bfloat16>(
            ha.data(), hb.data(), 256, 256, k, k, k, 1, 0,
            "w16a16 big 128x128", 0.02f,
            [&](GemmParams& p) { launch_policy<Bf16Big>(p, 0); });
        all &= check_gemm<__nv_bfloat16>(
            ha.data(), hb.data(), 256, 256, k, k, k, 1, 0,
            "w16a16 small 64x64", 0.02f,
            [&](GemmParams& p) { launch_policy<Bf16Small>(p, 0); });
    }

    // W8A16 weight-only: bf16 activation x per-channel-scaled int8 weight.
    // The kernel dequantizes B fragments in-register and folds the channel
    // scale into the epilogue. All four storage layouts run through the
    // production dispatch (NN takes the direct mixed instantiation), with
    // pinned tile variants the plan ladder would not route 256x256 to.
    using MixedBig =
        GemmPolicy<__nv_bfloat16, int8_t, RowMajor, ColMajor, Tile_128x128x64_W64x32_S2_Fast,
                   RowMajor, __nv_bfloat16>;
    using MixedSmall =
        GemmPolicy<__nv_bfloat16, int8_t, RowMajor, ColMajor, Tile_64x64x64_W16x32_S3_Fast,
                   RowMajor, __nv_bfloat16>;
    using MixedTT =
        GemmPolicy<__nv_bfloat16, int8_t, ColMajor, ColMajor, Tile_128x128x64_W64x32_S2,
                   RowMajor, __nv_bfloat16>;
    using TTSmallS2 =
        GemmPolicy<__nv_bfloat16, int8_t, ColMajor, ColMajor, Tile_64x64x64_W16x32_S2_Fast,
                   RowMajor, __nv_bfloat16>;
    using TTNarrow =
        GemmPolicy<__nv_bfloat16, int8_t, ColMajor, ColMajor, Tile_128x64x64_W32x32_S2_Fast,
                   RowMajor, __nv_bfloat16>;
    using TTBigFast =
        GemmPolicy<__nv_bfloat16, int8_t, ColMajor, ColMajor, Tile_128x128x64_W64x32_S2_Fast,
                   RowMajor, __nv_bfloat16>;
    using MixedTN =
        GemmPolicy<__nv_bfloat16, int8_t, ColMajor, RowMajor, Tile_128x128x64_W64x32_S2,
                   RowMajor, __nv_bfloat16>;
    using MixedNN =
        GemmPolicy<__nv_bfloat16, int8_t, RowMajor, RowMajor, Tile_128x128x64_W64x32_S2,
                   RowMajor, __nv_bfloat16>;
    printf("W8A16 (bf16 act x int8 weight, all layouts):\n");
    for (int k : {64, 320, 512}) {
        std::vector<float> ha, hb;
        prep(ha, hb, 256, 256, k, 777 + k);
        const std::vector<float> scale = div_row_scales(hb, 256, k);
        printf(" 256x256x%d:\n", k);
        all &= check_all_layouts<__nv_bfloat16, int8_t>(ha, hb, 256, 256, k,
                                                        "w8a16", 0.02f, scale);
        all &= check_gemm<__nv_bfloat16, int8_t>(
            ha.data(), hb.data(), 256, 256, k, k, k, 1, 0,
            "w8a16 big 128x128", 0.02f,
            [&](GemmParams& p) { launch_policy<MixedBig>(p, 0); }, scale);
        all &= check_gemm<__nv_bfloat16, int8_t>(
            ha.data(), hb.data(), 256, 256, k, k, k, 1, 0,
            "w8a16 small 64x64", 0.02f,
            [&](GemmParams& p) { launch_policy<MixedSmall>(p, 0); }, scale);
        if (k == 320) {
            // Crosswise/dual-row-major big CTA pinned (the planner routes
            // 256x256 crosswise to the small CTA).
            const std::vector<float> ha_t = transpose(ha, 256, k);
            const std::vector<float> hb_t = transpose(hb, 256, k);
            all &= check_gemm<__nv_bfloat16, int8_t>(
                ha_t.data(), hb.data(), 256, 256, k, 256, k, 0, 0,
                "w8a16 TT big", 0.02f,
                [&](GemmParams& p) { launch_policy<MixedTT>(p, 0); }, scale);
            all &= check_gemm<__nv_bfloat16, int8_t>(
                ha_t.data(), hb.data(), 256, 256, k, 256, k, 0, 0,
                "w8a16 TT smallS2", 0.02f,
                [&](GemmParams& p) { launch_policy<TTSmallS2>(p, 0); }, scale);
            all &= check_gemm<__nv_bfloat16, int8_t>(
                ha_t.data(), hb.data(), 256, 256, k, 256, k, 0, 0,
                "w8a16 TT narrow", 0.02f,
                [&](GemmParams& p) { launch_policy<TTNarrow>(p, 0); }, scale);
            all &= check_gemm<__nv_bfloat16, int8_t>(
                ha_t.data(), hb.data(), 256, 256, k, 256, k, 0, 0,
                "w8a16 TT bigfast", 0.02f,
                [&](GemmParams& p) { launch_policy<TTBigFast>(p, 0); }, scale);
            all &= check_gemm<__nv_bfloat16, int8_t>(
                ha_t.data(), hb_t.data(), 256, 256, k, 256, 256, 0, 1,
                "w8a16 TN big", 0.02f,
                [&](GemmParams& p) { launch_policy<MixedTN>(p, 0); }, scale);
            all &= check_gemm<__nv_bfloat16, int8_t>(
                ha.data(), hb_t.data(), 256, 256, k, k, 256, 1, 1,
                "w8a16 NN big", 0.02f,
                [&](GemmParams& p) { launch_policy<MixedNN>(p, 0); }, scale);
        }
    }

    // Odd-shape mixed TN: a_ld=100 (200B rows) and b_ld=130 are not
    // 16B-run aligned, forcing the crosswise scalar fallback on both
    // operands; k=96 predicates the second k-tile's contract tail and
    // m=100 exercises the row tail.
    {
        std::vector<float> ha, hb;
        prep(ha, hb, 100, 130, 96, 555);
        const std::vector<float> scale = div_row_scales(hb, 130, 96);
        const std::vector<float> ha_t = transpose(ha, 100, 96);
        const std::vector<float> hb_t = transpose(hb, 130, 96);
        printf("W8A16 odd shape (100x130x96, TN):\n");
        all &= check_gemm<__nv_bfloat16, int8_t>(
            ha_t.data(), hb_t.data(), 100, 130, 96, 100, 130, 0, 1,
            "w8a16 TN odd", 0.02f,
            [&](GemmParams& p) {
                gemm_dispatch<__nv_bfloat16, int8_t>(p, 0, true, false);
            },
            scale);
    }

    // W8A8 dynamic: per-row-scaled int8 activation x per-channel-scaled
    // int8 weight — BOTH operands dequantize in-register to the bf16 mma.
    // All four storage layouts (NN rides the symmetric swap rewrite),
    // plus a pinned big-CTA instantiation at k=320.
    {
        using W8A8Big =
            GemmPolicy<int8_t, int8_t, RowMajor, ColMajor, Tile_128x128x64_W64x32_S2_Fast,
                       RowMajor, __nv_bfloat16>;
        printf("W8A8 (int8 act x int8 weight, all layouts):\n");
        for (int k : {64, 320, 512}) {
            std::vector<float> ha, hb;
            prep(ha, hb, 256, 256, k, 888 + k);
            const std::vector<float> rscale = div_row_scales(ha, 256, k);
            const std::vector<float> cscale = div_row_scales(hb, 256, k);
            printf(" 256x256x%d:\n", k);
            all &= check_all_layouts<int8_t>(ha, hb, 256, 256, k, "w8a8",
                                             0.02f, cscale, rscale);
            if (k == 320) {
                all &= check_gemm<int8_t, int8_t>(
                    ha.data(), hb.data(), 256, 256, k, k, k, 1, 0,
                    "w8a8 big 128x128", 0.02f,
                    [&](GemmParams& p) { launch_policy<W8A8Big>(p, 0); },
                    cscale, rscale);
            }
        }
    }

    // A8W16 (the mirrored mixed pair): int8 activation dequantizes on the
    // A side while the bf16 weight rides the ldmatrix path — exercises the
    // split kSegXorA/kSegXorB addressing across all four layouts.
    printf("A8W16 (int8 act x bf16 weight, all layouts):\n");
    for (int k : {64, 320}) {
        std::vector<float> ha, hb;
        prep(ha, hb, 256, 256, k, 999 + k);
        const std::vector<float> rscale = div_row_scales(ha, 256, k);
        printf(" 256x256x%d:\n", k);
        all &= check_all_layouts<int8_t, __nv_bfloat16>(ha, hb, 256, 256, k,
                                                        "a8w16", 0.02f, {},
                                                        rscale);
    }

    // W-F8A16 weight-only: bf16 activation x per-channel-scaled e4m3
    // weight. The weight dequantizes in-register through the hardware
    // fp8->fp16 widen + exact bf16 rounding; row 0 is seeded with exact
    // zeros and subnormal-magnitude values (0.001 < 2^-6) so those paths
    // are exercised, and the mirrored pair follows. Pinned big CTA at
    // k=320 covers the Tile_128x128x64_W64x32_S2_Fast instantiation the planner would not
    // route 256x256 to.
    {
        using F8Big = GemmPolicy<__nv_bfloat16, __nv_fp8_e4m3, RowMajor,
                                 ColMajor, Tile_128x128x64_W64x32_S2_Fast, RowMajor,
                                 __nv_bfloat16>;
        printf("W-F8A16 (bf16 act x e4m3 weight, all layouts):\n");
        for (int k : {64, 320}) {
            std::vector<float> ha, hb;
            prep(ha, hb, 256, 256, k, 222 + k);
            const std::vector<float> scale = div_row_scales(hb, 256, k, 448.f);
            for (int j = 0; j < 16; ++j)
                hb[j] = j < 8 ? 0.f : 0.001f;  // +0 and e4m3 subnormals
            printf(" 256x256x%d:\n", k);
            all &= check_all_layouts<__nv_bfloat16, __nv_fp8_e4m3>(
                ha, hb, 256, 256, k, "w-f8a16", 0.02f, scale);
            if (k == 320) {
                all &= check_gemm<__nv_bfloat16, __nv_fp8_e4m3>(
                    ha.data(), hb.data(), 256, 256, k, k, k, 1, 0,
                    "w-f8a16 big 128x128", 0.02f,
                    [&](GemmParams& p) { launch_policy<F8Big>(p, 0); },
                    scale);
            }
        }
        printf("A-F8W16 (e4m3 act x bf16 weight, all layouts):\n");
        for (int k : {64, 320}) {
            std::vector<float> ha, hb;
            prep(ha, hb, 256, 256, k, 333 + k);
            const std::vector<float> rscale = div_row_scales(ha, 256, k, 448.f);
            printf(" 256x256x%d:\n", k);
            all &= check_all_layouts<__nv_fp8_e4m3, __nv_bfloat16>(
                ha, hb, 256, 256, k, "a-f8w16", 0.02f, {}, rscale);
        }
    }

    // fp32 output (OutT = float): one fixed narrow-CTA policy and one
    // production-planned route through the dtype-generic dispatch. The
    // narrow tile's 32KB output fits the 36KB reclaimed operand rings
    // (launch_plan compile-time-reroutes the 128x128 CTA for fat outputs).
    using Fp8F32Out =
        GemmPolicy<__nv_fp8_e4m3, __nv_fp8_e4m3, RowMajor, ColMajor,
                   Tile_128x64x64_W32x32_S2_Fast, RowMajor, float>;
    printf("fp8 operands, fp32 output:\n");
    for (int k : {64, 320, 512}) {
        std::vector<float> ha, hb;
        prep(ha, hb, 300, 200, k, 4321 + k);
        char tag[24];
        snprintf(tag, sizeof(tag), "f32-out K%d", k);
        all &= check_gemm<__nv_fp8_e4m3, __nv_fp8_e4m3, float>(
            ha.data(), hb.data(), 300, 200, k, k, k, 1, 0, tag, 0.02f,
            [&](GemmParams& p) { launch_policy<Fp8F32Out>(p, 0); });
        snprintf(tag, sizeof(tag), "f32-out disp K%d", k);
        all &= check_gemm<__nv_fp8_e4m3, __nv_fp8_e4m3, float>(
            ha.data(), hb.data(), 300, 200, k, k, k, 1, 0, tag, 0.02f,
            [&](GemmParams& p) {
                gemm_dispatch<__nv_fp8_e4m3, __nv_fp8_e4m3, float>(p, 0, false, true);
            });
    }
    return all;
}

// ---------------------------------------------------------------------------
// Combination effect bench: each dtype pair through the production NT
// route (the nn.Linear main path) at llama-linear shapes. qa_max/qb_max
// >0 quantizes that operand (127 int8 / 448 fp8) and threads the
// per-row/per-channel scales through the epilogue, matching how the
// strategy layer feeds the kernel.
// ---------------------------------------------------------------------------

using bf16_ = __nv_bfloat16;
template <typename ElemA, typename ElemB>
static void bench_combo(int m, int n, int k, const char* tag, float qa_max,
                        float qb_max) {
    srand(7);
    std::vector<float> ha((size_t)m * k), hb((size_t)n * k);
    for (float& v : ha) v = randf();
    for (float& v : hb) v = randf();
    std::vector<float> sa, sb;
    if (qa_max > 0) sa = div_row_scales(ha, m, k, qa_max);
    if (qb_max > 0) sb = div_row_scales(hb, n, k, qb_max);
    std::vector<ElemA> qa((size_t)m * k);
    std::vector<ElemB> qb((size_t)n * k);
    for (size_t i = 0; i < qa.size(); ++i) qa[i] = to_elem<ElemA>(ha[i]);
    for (size_t i = 0; i < qb.size(); ++i) qb[i] = to_elem<ElemB>(hb[i]);

    ElemA* da;
    ElemB* db;
    bf16_* dout;
    float *d_one, *d_sa = nullptr, *d_sb = nullptr;
    cudaMalloc(&da, qa.size() * sizeof(ElemA));
    cudaMalloc(&db, qb.size() * sizeof(ElemB));
    cudaMalloc(&dout, (size_t)m * n * sizeof(bf16_));
    cudaMalloc(&d_one, 4);
    cudaMemcpy(da, qa.data(), qa.size() * sizeof(ElemA), cudaMemcpyHostToDevice);
    cudaMemcpy(db, qb.data(), qb.size() * sizeof(ElemB), cudaMemcpyHostToDevice);
    const float one = 1.0f;
    cudaMemcpy(d_one, &one, 4, cudaMemcpyHostToDevice);
    if (!sa.empty()) {
        cudaMalloc(&d_sa, sa.size() * 4);
        cudaMemcpy(d_sa, sa.data(), sa.size() * 4, cudaMemcpyHostToDevice);
    }
    if (!sb.empty()) {
        cudaMalloc(&d_sb, sb.size() * 4);
        cudaMemcpy(d_sb, sb.data(), sb.size() * 4, cudaMemcpyHostToDevice);
    }

    GemmParams p = {};
    p.a_ptr = da;
    p.b_ptr = db;
    p.out_ptr = dout;
    p.a_scale = d_one;
    if (d_sa) {
        p.a_scale = d_sa;
        p.a_scale_m = m;
    }
    if (d_sb) {
        p.b_scale = d_sb;
        p.b_scale_n = n;
    }
    p.m = m;
    p.n = n;
    p.k = k;
    p.a_ld = k;
    p.b_ld = k;
    p.out_ld = n;

    double flops = 2.0 * m * n * k;
    BenchResult r = bench_kernel(
        [&] { gemm_dispatch<ElemA, ElemB>(p, 0, false, true); }, 3, 10,
        flops);
    char cfg[64];
    snprintf(cfg, sizeof(cfg), "%-16s %5dx%-5dx%-5d", tag, m, n, k);
    print_bench_row(cfg, r);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    cudaFree(d_one);
    if (d_sa) cudaFree(d_sa);
    if (d_sb) cudaFree(d_sb);
}

static void bench_dtype_combos() {
    printf("\n===== QUANT_GEMM COMBO BENCH (production NT route) =====\n");
    print_bench_header();
    struct Shape { int m, n, k; };
    const Shape shapes[] = {
        {4096, 4096, 4096},   // square
        {2048, 4096, 4096},   // batched linear
        {4096, 14336, 4096},  // llama up_gate
    };
    for (const Shape& s : shapes) {
        bench_combo<bf16_, bf16_>(s.m, s.n, s.k, "W16A16", 0, 0);
        bench_combo<bf16_, int8_t>(s.m, s.n, s.k, "W8A16", 0, 127.f);
        bench_combo<int8_t, int8_t>(s.m, s.n, s.k, "W8A8", 127.f, 127.f);
        bench_combo<bf16_, __nv_fp8_e4m3>(s.m, s.n, s.k, "W-F8A16", 0, 448.f);
        bench_combo<__nv_fp8_e4m3, __nv_fp8_e4m3>(s.m, s.n, s.k, "F8A8 e4m3", 0, 0);
        bench_combo<__nv_fp8_e5m2, __nv_fp8_e5m2>(s.m, s.n, s.k, "F8A8 e5m2", 0, 0);
    }
}

}  // namespace

int main() {
    print_test_header();
    bool ok = test_gemm();
    ok &= test_dtype_combos();
    printf(ok ? "All PASS\n" : "FAILURES\n");
    if (ok) bench_dtype_combos();
    return ok ? 0 : 1;
}
