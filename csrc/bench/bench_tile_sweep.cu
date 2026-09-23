/*
Tile-parameter sweep for the gemm family.

Production dispatch keys a tile on (CTA class, ring depth) only —
dispatch_tile matches tile_class<Tile>() == plan.cta && kStages ==
plan.stages — so the other tile parameters (CTA geometry, warp shape,
k-tile depth, the fast predication-free loop, the ring depth's interaction
with crosswise staging) have never been measured: the manifest is an
assumption, not a result. This bench instantiates tile configurations
directly and times them, bypassing plan_gemm, so the whole parameter space is
reachable. Candidates are organised one parameter at a time around the three
production classes, so a win is attributable to a single axis.

Compiled standalone (no CMake target), like the C tests:

    nvcc -I csrc/kernels -I csrc/tests -arch=sm_89 -std=c++20 -O3 \
        csrc/bench/bench_tile_sweep.cu -o /tmp/tile_sweep && /tmp/tile_sweep
    # also build and sweep the int8 candidate set (~2x compile):
    nvcc ... -DASTRAI_SWEEP_INT8=1 ... --dtype both

    /tmp/tile_sweep [--shapes 4096:4096:4096,...] [--dtype bf16|int8|mixed]
                    [--only SUBSTR] [--list] [--warmup 3] [--iters 20]

Instantiating a candidate costs one kernel build, all in one nvcc process;
the grid is the full legal cross product (every CTA x every warp tiling the
kernel's constraints admit — see WarpGeoms), and the run header counts it.
Compile time scales with that count (~35s for the old 4-warp kk32 grid,
~2min at the full nine-warp kk32/64 space with the 4/8-warp CTA pin).
--list prints the whole
instantiated grid per built dtype family host-side and exits, so coverage
questions are answerable without a compile at all. Widening the space is one
line in Candidates below.

Output: CSV `dtype,shape,tile,prod,ms,tflops,ms_wall,checksum,launch_err`
(infeasible tiles carry `skip,<reason>`); ms is the cuda-event timer and
ms_wall a wall-clock best-of, so a candidate whose two timers disagree is
visible rather than trusted, then a per-shape summary — what plan_gemm picks, the best
production-manifest candidate, the best candidate overall, and the ratio
between them. A candidate whose output disagrees with the first feasible
candidate by more than 1e-3 relative is listed as a MISMATCH: a fast tile
that computes the wrong thing is not a win.
*/

#include "test_utils.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "common/device.cuh"
#include "gemm/gemm.cuh"

using namespace astrai;
using namespace astrai::gemm;

namespace {

template <typename T>
double to_float(T v) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return (double)__bfloat162float(v);
    else
        return (double)v;
}

inline int8_t quant_i8(float v) {
    return (int8_t)std::max(-127.0f, std::min(127.0f, std::round(v * 200.0f)));
}

// ---------------------------------------------------------------------------
// The tile grid, instantiated rather than hand-listed.
//
// A hand-written list has to know the kernel's instantiation constraints by
// heart and gets them wrong silently: a configuration the staging cannot load
// surfaces as a static_assert deep in the load path, one failed build at a
// time. So the axes below are enumerated at compile time and each combination
// is filtered by tile_ok(), which restates — in one place, beside the axes —
// the asserts the kernel carries:
//
//   warp tiling   kWarpM % 16 == 0, kWarpN % 8 == 0, warp tiles tile the CTA
//   k depth       kK % kMmaK == 0, and kK*elem <= 128: ComposedLayout needs
//                 kRowShift = kShift - log2(kChunks) >= 0 with
//                 kChunks = kK*elem/16, so Swizzle<4,3> caps kK at 64 for
//                 2-byte and 128 for 1-byte operands
//   staging       kTileLines*kChunks % kThreads == 0 with a power-of-two
//                 quotient, per operand (load_operand_tile), kTileLines being
//                 the CTA extent on a congruous stage
//   reclaim       bm*bn*sizeof(OutT) <= ring_smem_bytes(...), the epilogue's
//                 reuse of the operand rings for the output tile
//
// What survives is what this dtype pair can actually run. The grid is the
// measured space; the production manifest (manifest_for) is consulted only to
// label which survivors production itself can select.
// ---------------------------------------------------------------------------
template <int M, int N>
struct Geom {
    static constexpr int kM = M;
    static constexpr int kN = N;
};

using CtaGeoms = std::tuple<Geom<64, 64>, Geom<128, 64>, Geom<128, 128>,
                            Geom<64, 128>, Geom<128, 256>>;

// The warp axis: tiles of 16..128 per side at aspect ratios 1:2 .. 2:1 (the
// shapes a warp tiling can sensibly take; the extreme slivers — one m16 or
// n8 strip against a long other side — are out of scope). kWarpN stays a
// multiple of 16 because mainloop.cuh declares the paired-B fragment array
// unconditionally, so kNt == 1 instantiates nowhere. Where a tile does not
// divide the CTA, or the CTA's warp count leaves the pinned 4/8 band
// (tile_ok), the combination drops out. The 2026-09-15 pinning covered
// four of these nine; cuBLAS on this box runs 64x64, which was none of the
// four.
using WarpGeoms = std::tuple<
    Geom<16, 16>, Geom<16, 32>, Geom<32, 16>,
    Geom<32, 32>, Geom<32, 64>, Geom<64, 32>,
    Geom<64, 64>, Geom<64, 128>, Geom<128, 64>>;

// The load bus rule, one spelling with load.cuh's loader: a side's chunks
// either divide the threads (each thread one aligned power-of-two run) or
// under-subscribe the bus (each participating thread one chunk, the
// surplus skips). A side that can do neither has no schedule.
constexpr bool bus_fits(int chunks, int threads) {
    const int cpt = chunks < threads ? 1 : chunks / threads;
    return cpt > 0 && (cpt & (cpt - 1)) == 0 &&
           (chunks % threads == 0 || chunks < threads);
}

template <typename EA, typename EB, typename Cta, typename Warp, int KK, int S>
constexpr bool tile_ok() {
    using MmaT = typename gemm_mma_traits<EA, EB>::MmaT;
    constexpr int kMmaK = astrai::MmaShapeFor<MmaT>::type::kK;
    constexpr int kThreads = (Cta::kM / Warp::kM) * (Cta::kN / Warp::kN) * 32;
    constexpr int kChunksA = Cta::kM * (KK * (int)sizeof(EA) / 16);
    constexpr int kChunksB = Cta::kN * (KK * (int)sizeof(EB) / 16);
    constexpr int kRing = ring_smem_bytes(Cta::kM, Cta::kN, KK, S,
                                          (int)sizeof(EA), (int)sizeof(EB));
    // mainloop.cuh's B-fragment pairing (kPairB), restated: on when B feeds
    // the mma natively (no in-register dequant) and its k-tile holds <= 4
    // 16B chunks. It folds two adjacent n8 cells per load. The paired
    // fragment array is declared unconditionally, so even where the pairing
    // is off a warp tile must span at least two n8 cells to instantiate.
    constexpr bool kPairB =
        !gemm_mma_traits<EA, EB>::kDequantB && KK * (int)sizeof(EB) / 16 <= 4;
    // Warp count per CTA is pinned to 4 or 8 (128/256 threads): the band
    // every production tile except the 16-warp manifest entries sits in
    // (128x128x32_W32x32_S2 on the two-byte/mixed ladder,
    // 128x256x64_W64x32_S2 on the byte ladder — those drop out of the grid
    // and print planned="?").
    constexpr int kWarps = kThreads / 32;
    // load.cuh's line/run split needs a thread's 16B run inside ONE staged
    // line (kCpt <= kChunks); fully subscribed that is "the staged extent
    // <= threads". The static_assert pair there missed it until 2026-09-16:
    // the violators (64x64x32 with a 64x64 warp, 128x256x32 with W64x128 or
    // W128x64 warps) deadlock the kernel at 100% GPU instead of failing to
    // build. Restated here so the grid never names one.
    return (kWarps == 4 || kWarps == 8) && Cta::kM <= kThreads &&
           Cta::kN <= kThreads && Warp::kM % 16 == 0 &&
           Warp::kN % 16 == 0 &&
           Cta::kM % Warp::kM == 0 &&
           Cta::kN % Warp::kN == 0 && KK % kMmaK == 0 &&
           KK * (int)sizeof(EA) <= 128 && KK * (int)sizeof(EB) <= 128 &&
           bus_fits(kChunksA, kThreads) && bus_fits(kChunksB, kThreads) &&
           (!kPairB || (Warp::kN / 8) % 2 == 0) &&
           Cta::kM * Cta::kN * 2 <= kRing;
}

// The grid container is a plain type list, NOT std::tuple: libstdc++
// tuple_cat runs an is_convertible validity check that recurses once per
// element, and grids of this size (a few hundred entries) exceed the
// frontend's instantiation depth. A variadic list concatenates at constant
// depth and iterates as a fold expression.
template <typename... Ts>
struct TileList {};

template <typename EA, typename EB, typename Cta, typename Warp, int KK,
          typename... Done>
constexpr TileList<Done...> add_stages(TileList<Done...> acc) {
    return acc;
}

template <typename EA, typename EB, typename Cta, typename Warp, int KK, int S,
          int... Rest, typename... Done>
constexpr auto add_stages(TileList<Done...> acc) {
    if constexpr (tile_ok<EA, EB, Cta, Warp, KK, S>()) {
        using Tile = GemmTileConfig<Shape<Cta::kM, Cta::kN, KK>,
                                    Shape<Warp::kM, Warp::kN>, S>;
        return add_stages<EA, EB, Cta, Warp, KK, Rest...>(
            TileList<Done..., Tile>{});
    } else {
        return add_stages<EA, EB, Cta, Warp, KK, Rest...>(acc);
    }
}

// The k-tile depth axis: 32/64 (the twins every manifest carries) plus 128,
// which the staging swizzle admits for 1-byte operands only (KK*elem <= 128
// in tile_ok). add_ks was named for this axis but only ever instantiated
// KK=32 — the kk64 half of the production vocabulary was never swept.
template <typename EA, typename EB, typename Cta, typename Warp>
constexpr auto add_ks() {
    return add_stages<EA, EB, Cta, Warp, 128, 2, 3, 4>(
        add_stages<EA, EB, Cta, Warp, 64, 2, 3, 4>(
            add_stages<EA, EB, Cta, Warp, 32, 2, 3, 4>(TileList<>{})));
}

template <typename... Ls>
struct list_cat_all;
template <typename... As, typename... Bs, typename... Rest>
struct list_cat_all<TileList<As...>, TileList<Bs...>, Rest...>
    : list_cat_all<TileList<As..., Bs...>, Rest...> {};
template <typename L>
struct list_cat_all<L> {
    using type = L;
};
template <>
struct list_cat_all<> {
    using type = TileList<>;
};

template <typename EA, typename EB, typename Cta, typename... Warps>
constexpr auto add_warps(std::tuple<Warps...>) {
    return typename list_cat_all<decltype(add_ks<EA, EB, Cta, Warps>())...>::type{};
}

template <typename EA, typename EB, typename Cta>
using cta_grid_t = decltype(add_warps<EA, EB, Cta>(WarpGeoms{}));

template <typename EA, typename EB, typename... Ctas>
constexpr TileList<cta_grid_t<EA, EB, Ctas>...> add_ctas(std::tuple<Ctas...>) {
    return {};
}

template <typename EA, typename EB>
using tile_grid_t = decltype(add_ctas<EA, EB>(CtaGeoms{}));

// Total candidates of the nested grid (the run header's count).
template <typename... Ts>
constexpr int list_len(TileList<Ts...>) {
    return (int)sizeof...(Ts);
}
template <typename Grid>
struct grid_size;
template <typename... Ls>
struct grid_size<TileList<Ls...>> {
    static constexpr int value = (0 + ... + list_len(Ls{}));
};

// Which entries production can select: a membership test over the pair's
// manifest, not a position convention, so the prod column stays honest as the
// manifest grows.
template <typename T, typename Tuple>
struct tuple_contains : std::false_type {};
template <typename T, typename... Ts>
struct tuple_contains<T, std::tuple<Ts...>>
    : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

template <typename EA, typename EB>
struct Space {
    using Tiles = tile_grid_t<EA, EB>;
    template <typename Tile>
    static constexpr bool is_prod() {
        return tuple_contains<Tile,
                             manifest_for<EA, EB, RowMajor, ColMajor>>::value;
    }
};

// The structural token the aliases are spelled with, built from the type's
// own parameters so a printed name cannot drift from the tile.
template <typename Tile>
std::string tile_name() {
    char buf[64];
    std::snprintf(buf, sizeof buf, "Tile_%lldx%lldx%lld_W%lldx%lld_S%d",
                  (long long)Tile::CtaShape::kM, (long long)Tile::CtaShape::kN,
                  (long long)Tile::CtaShape::kK, (long long)Tile::WarpShape::kM,
                  (long long)Tile::WarpShape::kN, (int)Tile::kStages);
    return std::string(buf);
}

// Every candidate of the nested grid, in order; the global index is a
// RUNTIME value (the fold keeps every type-level walk at constant depth; the
// fn contract is operator()<Tile>(int index)).
template <typename... Ts, typename Fn>
void for_each_flat(TileList<Ts...>, int& index, Fn&& fn) {
    ((fn.template operator()<Ts>(index++)), ...);
}

template <typename... Ls, typename Fn>
void for_each_candidate(TileList<Ls...>, Fn&& fn) {
    int index = 0;
    (for_each_flat(Ls{}, index, fn), ...);
}

struct Row {
    std::string tile;
    int index = -1;
    bool prod = false;
    bool ok = false;
    std::string why;
    double ms = 0.0;
    double tflops = 0.0;
    double max_rel = 0.0;  // vs the reference candidate's output
    // Two independent witnesses that the timed work happened and was this
    // candidate: the output checksum (a stale buffer from a launch that never
    // ran repeats the previous candidate's) and a wall-clock best-of that does
    // not share the event timer's assumptions.
    double checksum = 0.0;
    double ms_wall = 0.0;
    int launch_err = 0;
};

// One shape, one dtype pair, every candidate of the grid.
// ---------------------------------------------------------------------------
template <typename EA, typename EB, typename OutT>
std::vector<Row> sweep_shape(int m, int n, int k, bool use_scale, int warmup,
                             int iters, const char* only) {
    const size_t a_elems = (size_t)m * k, b_elems = (size_t)n * k;
    std::vector<float> ha(a_elems), hb(b_elems);
    srand(7);
    for (float& v : ha) v = randf();
    for (float& v : hb) v = randf();

    std::vector<EA> qa(a_elems);
    std::vector<EB> qb(b_elems);
    for (size_t i = 0; i < a_elems; ++i) {
        if constexpr (std::is_same_v<EA, int8_t>)
            qa[i] = quant_i8(ha[i]);
        else
            qa[i] = (EA)ha[i];
    }
    for (size_t i = 0; i < b_elems; ++i) {
        if constexpr (std::is_same_v<EB, int8_t>)
            qb[i] = quant_i8(hb[i]);
        else
            qb[i] = (EB)hb[i];
    }

    EA* da = nullptr;
    EB* db = nullptr;
    OutT* dout = nullptr;
    float* d_one = nullptr;
    CUDA_CHECK(cudaMalloc(&da, a_elems * sizeof(EA)));
    CUDA_CHECK(cudaMalloc(&db, b_elems * sizeof(EB)));
    CUDA_CHECK(cudaMalloc(&dout, (size_t)m * n * sizeof(OutT)));
    CUDA_CHECK(cudaMalloc(&d_one, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, qa.data(), a_elems * sizeof(EA),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, qb.data(), b_elems * sizeof(EB),
                          cudaMemcpyHostToDevice));
    const float one = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_one, &one, sizeof(float), cudaMemcpyHostToDevice));

    const DeviceFacts dev = device_facts();
    const double flops = 2.0 * m * n * k;
    using Tiles = typename Space<EA, EB>::Tiles;
    // Progress goes to stderr (CSV owns stdout): the per-shape rows only
    // print at the end, and a few hundred candidates of checksum loops are
    // minutes of silence without a live counter.
    const int total = (int)grid_size<Tiles>::value;
    int done = 0;
    std::vector<Row> rows;
    std::vector<OutT> reference;
    bool have_reference = false;

    for_each_candidate(Tiles{}, [&]<typename Tile>(int index) {
        // --only: measure just the candidates whose tile name contains the
        // substring (everything still instantiates — the grid is compile
        // time). The bisect tool for a candidate that hangs or MISMATCHes.
        if (only != nullptr &&
            std::strstr(tile_name<Tile>().c_str(), only) == nullptr)
            return;
        Row row;
        row.tile = tile_name<Tile>();
        row.index = index;
        row.prod = Space<EA, EB>::template is_prod<Tile>();

        constexpr int kRing = ring_smem_bytes(
            (int)Tile::CtaShape::kM, (int)Tile::CtaShape::kN,
            (int)Tile::CtaShape::kK, (int)Tile::kStages, (int)sizeof(EA),
            (int)sizeof(EB));
        // The epilogue reclaims the operand rings for the output tile; a
        // configuration that cannot is not a candidate (its kernel would not
        // even instantiate — the launcher static_asserts on it).
        constexpr bool kReclaim =
            (int)Tile::CtaShape::kM * (int)Tile::CtaShape::kN * (int)sizeof(OutT) <=
            kRing;
        // The illegal configurations must not be *named*, not merely skipped:
        // the kernel static_asserts on the reclaim budget, and launch_policy
        // instantiates it, so the guard has to be an if/else chain rather
        // than an early return.
        if constexpr (!kReclaim) {
            row.why = "output exceeds the reclaimed ring";
        } else {
            using Policy = GemmPolicy<EA, EB, RowMajor, ColMajor, Tile, RowMajor,
                                      OutT, false, false>;
            if (Policy::kSmemBytes > dev.smem_max) {
                row.why = "ring over the smem opt-in ceiling";
            } else {
        GemmParams p = {};
        p.a_ptr = da;
        p.b_ptr = db;
        p.out_ptr = dout;
        if (use_scale) {  // int8 dequant scales; 1.0 keeps the math a no-op
            p.a_scale = d_one;
            p.b_scale = d_one;
        }
        p.m = m;
        p.n = n;
        p.k = k;
        p.a_ld = k;
        p.b_ld = k;
        p.out_ld = n;
        p.batch = 1;
        p.out_batch_stride = m * n;
        // Raster is a property of the geometry, and production recomputes it
        // per tile; match that so the comparison is apples-to-apples.
        p.raster = plan_raster(plan_query<EA, EB, RowMajor, ColMajor>(p, dev),
                               (int)Tile::CtaShape::kM, (int)Tile::CtaShape::kN);

        const BenchResult r = bench_kernel(
            [&] { launch_policy<Policy>(p, 0); }, warmup, iters, flops);
        row.ok = true;
        row.ms = r.ms;
        row.tflops = r.tflops;

        for (int w = 0; w < 3; ++w) {  // wall-clock, best of 3
            cudaDeviceSynchronize();
            const double t0 = now_ms();
            for (int i = 0; i < iters; ++i) launch_policy<Policy>(p, 0);
            cudaDeviceSynchronize();
            const double wall = (now_ms() - t0) / iters;
            if (w == 0 || wall < row.ms_wall) row.ms_wall = wall;
        }
        row.launch_err = (int)cudaGetLastError();

        std::vector<OutT> out((size_t)m * n);
        CUDA_CHECK(cudaMemcpy(out.data(), dout, out.size() * sizeof(OutT),
                              cudaMemcpyDeviceToHost));
        double checksum = 0.0;
        for (size_t i = 0; i < out.size(); ++i) checksum += to_float(out[i]);
        row.checksum = checksum;
        if (!have_reference) {
            reference = out;
            have_reference = true;
        } else {
            double worst = 0.0;
            for (size_t i = 0; i < out.size(); ++i) {
                const double ref = to_float(reference[i]);
                const double den = std::fabs(ref) > 1e-6 ? std::fabs(ref) : 1e-6;
                worst = std::max(worst, std::fabs(to_float(out[i]) - ref) / den);
            }
            row.max_rel = worst;
        }
            }
        }
        rows.push_back(row);
        std::fprintf(stderr, "\r  %d/%d %s%s", ++done, total,
                     row.tile.c_str(), row.ok ? "" : " (skip)");
    });
    std::fprintf(stderr, "\n");

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dout));
    CUDA_CHECK(cudaFree(d_one));
    return rows;
}

// The production-manifest entry a decision names. The full grid carries
// several warp twins per CTA class, and the manifest itself splits one class
// across warp shapes (big kk32 s2 is the 16-warp twin, s3 the 8-warp), so
// (cta class, stages) alone matches two entries — the plan's kk decides.
template <typename EA, typename EB>
int planned_candidate_index(const PlanDecision& d) {
    int found = -1;
    for_each_candidate(typename Space<EA, EB>::Tiles{},
        [&]<typename Tile>(int index) {
        if constexpr (Space<EA, EB>::template is_prod<Tile>()) {
            if (found < 0 &&
                tile_class<Tile>() == static_cast<TileClass>(d.recipe.cta) &&
                Tile::kStages == d.recipe.stages &&
                (int)Tile::CtaShape::kK == d.recipe.kk)
                found = index;
        }
    });
    return found;
}

template <typename EA, typename EB>
void report_shape(const char* cfg, const std::vector<Row>& rows, int m, int n,
                  int k, const DeviceFacts& dev, const char* dtype_tag) {
    for (const Row& r : rows) {
        if (r.ok)
            std::printf("%s,%s,%s,%d,%.4f,%.2f,%.4f,%.4f,%d\n", dtype_tag, cfg,
                        r.tile.c_str(), r.prod ? 1 : 0, r.ms, r.tflops,
                        r.ms_wall, r.checksum, r.launch_err);
        else
            std::printf("%s,%s,%s,%d,skip,%s,,,,,\n", dtype_tag, cfg,
                        r.tile.c_str(), r.prod ? 1 : 0, r.why.c_str());
    }

    GemmParams p = {};
    p.m = m;
    p.n = n;
    p.k = k;
    // The sweep only times the NT (fused-linear) route, which is the layout
    // pair the candidates above are built with; the plan derives its own
    // perf class, widths and crosswise count from those same tags.
    const PlanDecision plan =
        plan_dispatch_for<EA, EB, RowMajor, ColMajor>(p);
    const int pi = planned_candidate_index<EA, EB>(plan);
    // Rows may be filtered (--only), so look the planned index up rather
    // than indexing; a filtered-out plan prints as "?".
    const Row* planned_row = nullptr;
    for (const Row& r : rows)
        if (r.index == pi) {
            planned_row = &r;
            break;
        }
    const char* planned = planned_row ? planned_row->tile.c_str() : "?";

    const Row* best_any = nullptr;
    const Row* best_prod = nullptr;
    for (const Row& r : rows) {
        if (!r.ok) continue;
        if (!best_any || r.tflops > best_any->tflops) best_any = &r;
        if (r.prod && (!best_prod || r.tflops > best_prod->tflops))
            best_prod = &r;
    }
    if (!best_any) {
        std::printf("# %s %s: no feasible candidate\n", dtype_tag, cfg);
        return;
    }
    // best_prod is null when --only filtered out every prod row: print a
    // dash rather than dereference.
    const double planned_tflops = planned_row ? planned_row->tflops : 0.0;
    const char* prod_name = best_prod ? best_prod->tile.c_str() : "-";
    const double prod_tflops = best_prod ? best_prod->tflops : 0.0;
    std::printf(
        "# %s %s planned=%s %.1f  best_prod=%s %.1f (%.3fx)  best_any=%s %.1f "
        "(%.3fx)",
        dtype_tag, cfg, planned, planned_tflops, prod_name, prod_tflops,
        planned_tflops > 0 ? prod_tflops / planned_tflops : 0.0,
        best_any->tile.c_str(), best_any->tflops,
        planned_tflops > 0 ? best_any->tflops / planned_tflops : 0.0);
    bool any_mismatch = false;
    for (const Row& r : rows) {
        if (r.ok && r.max_rel > 1e-3) {
            std::printf("%s MISMATCH %s rel=%.2e", any_mismatch ? "," : " ",
                        r.tile.c_str(), r.max_rel);
            any_mismatch = true;
        }
    }
    std::printf("\n");
    (void)dev;
}

// The full combination set, host-only: one line per candidate the grid
// instantiates for a dtype family — the same enumeration a sweep would time —
// so coverage questions are answerable before paying the compile, and a
// --list run doubles as the change record for the grid itself.
template <typename EA, typename EB>
void list_space(const char* tag) {
    using Tiles = typename Space<EA, EB>::Tiles;
    std::printf("# %s candidates=%d\n", tag,
                (int)grid_size<Tiles>::value);
    std::printf("%s,tile,prod,threads,ring_bytes\n", tag);
    for_each_candidate(Tiles{}, [&]<typename Tile>(int) {
        constexpr int kThreads =
            (Tile::CtaShape::kM / Tile::WarpShape::kM) *
            (Tile::CtaShape::kN / Tile::WarpShape::kN) * 32;
        constexpr int kRing = ring_smem_bytes(
            (int)Tile::CtaShape::kM, (int)Tile::CtaShape::kN,
            (int)Tile::CtaShape::kK, (int)Tile::kStages, (int)sizeof(EA),
            (int)sizeof(EB));
        std::printf("%s,%s,%d,%d,%d\n", tag, tile_name<Tile>().c_str(),
                    Space<EA, EB>::template is_prod<Tile>() ? 1 : 0, kThreads,
                    kRing);
    });
}

}  // namespace

int main(int argc, char** argv) {
    const char* shape_arg = nullptr;
    const char* dtype_arg = "bf16";
    const char* only_arg = nullptr;
    int warmup = 3, iters = 20;
    bool list = false;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* {
            return i + 1 < argc ? argv[++i] : nullptr;
        };
        if (!std::strcmp(argv[i], "--shapes")) shape_arg = next();
        else if (!std::strcmp(argv[i], "--dtype")) dtype_arg = next();
        else if (!std::strcmp(argv[i], "--only")) only_arg = next();
        else if (!std::strcmp(argv[i], "--list")) list = true;
        else if (!std::strcmp(argv[i], "--warmup")) warmup = std::atoi(next());
        else if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(next());
        else {
            std::printf("unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    // --list is host-only (no device touch, no shapes needed): it prints the
    // whole instantiated grid per built dtype family and exits.
    if (list) {
        const bool l_bf16 = std::strcmp(dtype_arg, "bf16") == 0 ||
                            std::strcmp(dtype_arg, "both") == 0;
        const bool l_int8 =
            std::strcmp(dtype_arg, "int8") == 0 || std::strcmp(dtype_arg, "both") == 0;
        const bool l_mixed = std::strcmp(dtype_arg, "mixed") == 0;
        if (l_bf16) list_space<__nv_bfloat16, __nv_bfloat16>("bf16");
#ifdef ASTRAI_SWEEP_MIXED
        if (l_mixed) list_space<__nv_bfloat16, int8_t>("mixed");
#endif
#ifdef ASTRAI_SWEEP_INT8
        if (l_int8) list_space<int8_t, int8_t>("int8");
#endif
        if (l_int8 && !l_bf16 && !l_mixed) {
#ifndef ASTRAI_SWEEP_INT8
            std::printf("# int8 candidates are not built; recompile with "
                        "-DASTRAI_SWEEP_INT8=1\n");
#endif
        }
        if (l_mixed) {
#ifndef ASTRAI_SWEEP_MIXED
            std::printf("# mixed candidates are not built; recompile with "
                        "-DASTRAI_SWEEP_MIXED=1\n");
#endif
        }
        return 0;
    }

    // A spread across the regimes the plan table's bands separate: square
    // small/mid/large, a long-K 4k square, a wide-N decode shape, skinny N.
    const char* default_shapes =
        "256:256:256,512:512:512,1024:1024:1024,2048:2048:2048,"
        "4096:4096:4096,4096:4096:1024,64:4096:4096,4096:64:4096";
    const char* shapes = shape_arg ? shape_arg : default_shapes;

    std::vector<std::array<int, 3>> grid;
    for (const char* s = shapes; *s;) {
        int m = 0, n = 0, k = 0;
        if (std::sscanf(s, "%d:%d:%d", &m, &n, &k) != 3) break;
        grid.push_back({m, n, k});
        const char* comma = std::strchr(s, ',');
        if (comma == nullptr) break;
        s = comma + 1;
    }
    if (grid.empty()) {
        std::printf("no shapes parsed\n");
        return 2;
    }

    const DeviceFacts dev = device_facts();
    std::printf("# device cc=%d sms=%d smem_max=%d l2=%lld\n", dev.cc, dev.sms,
                dev.smem_max, (long long)dev.l2_bytes);
    std::printf("# bf16_candidates=%d int8_candidates=%d warmup=%d "
                "iters=%d shapes=%zu\n",
                (int)grid_size<typename Space<__nv_bfloat16, __nv_bfloat16>::Tiles>::value,
                (int)grid_size<typename Space<int8_t, int8_t>::Tiles>::value, warmup,
                iters, grid.size());
#ifdef ASTRAI_SWEEP_MIXED
    std::printf("# mixed_candidates=%d\n",
                (int)grid_size<
                    typename Space<__nv_bfloat16, int8_t>::Tiles>::value);
#endif
    std::printf("dtype,shape,tile,prod,ms,tflops,ms_wall,checksum,launch_err\n");

    const bool want_bf16 =
        std::strcmp(dtype_arg, "int8") != 0 && std::strcmp(dtype_arg, "mixed") != 0;
    const bool want_int8 =
        std::strcmp(dtype_arg, "bf16") != 0 && std::strcmp(dtype_arg, "mixed") != 0;
    const bool want_mixed = std::strcmp(dtype_arg, "mixed") == 0;
#ifndef ASTRAI_SWEEP_INT8
    if (want_int8) {
        std::printf("# int8 candidates are not built; recompile with "
                    "-DASTRAI_SWEEP_INT8=1\n");
        return 2;
    }
#endif
#ifndef ASTRAI_SWEEP_MIXED
    if (want_mixed) {
        std::printf("# mixed (bf16 x int8) candidates are not built; recompile "
                    "with -DASTRAI_SWEEP_MIXED=1\n");
        return 2;
    }
#endif

    for (const auto& shape : grid) {
        const int m = shape[0], n = shape[1], k = shape[2];
        char cfg[64];
        std::snprintf(cfg, sizeof cfg, "%dx%dx%d", m, n, k);
        if (want_bf16) {
            const auto rows =
                sweep_shape<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
                    m, n, k, /*use_scale=*/false, warmup, iters, only_arg);
            report_shape<__nv_bfloat16, __nv_bfloat16>(cfg, rows, m, n, k, dev,
                                                       "bf16");
        }
#ifdef ASTRAI_SWEEP_MIXED
        if (want_mixed) {
            const auto rows =
                sweep_shape<__nv_bfloat16, int8_t, __nv_bfloat16>(
                    m, n, k, /*use_scale=*/true, warmup, iters, only_arg);
            report_shape<__nv_bfloat16, int8_t>(cfg, rows, m, n, k, dev,
                                                "mixed");
        }
#endif
#ifdef ASTRAI_SWEEP_INT8
        if (want_int8) {
            const auto rows = sweep_shape<int8_t, int8_t, __nv_bfloat16>(
                m, n, k, /*use_scale=*/true, warmup, iters, only_arg);
            report_shape<int8_t, int8_t>(cfg, rows, m, n, k, dev, "int8");
        }
#endif
    }
    return 0;
}
