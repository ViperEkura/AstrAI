#pragma once
// GEMM umbrella (bf16 / int8 / fp8): the kernel orchestrator and the launch
// machinery — pure CUDA, no torch; the planner lives in launcher/planning.h
// (one TU per binary, reached through policy.cuh's declarations). Layout
// tags: utils/gemm_common.h.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <tuple>
#include <utility>
#include <type_traits>

#include <memory/pipeline.cuh>
#include <utils/device.cuh>
#include <utils/launch.cuh>
#include <epilogue/writer.cuh>
#include <utils/gemm_common.h>
#include <kernel/gemm_mainloop.cuh>
#include <policy.cuh>
#include <scheduler.cuh>

namespace astrai {
namespace gemm {

// The ONE quantized-GEMM orchestrator (cp.async staging).
template <typename Policy>
__global__ void __launch_bounds__(Policy::kCtaThreads, Policy::kMinCtas)
    gemm_kernel(GemmParams p) {
    using Mainloop = GemmCollectiveMainloop<Policy>;
    using Epilogue = GemmCollectiveEpilogue<Policy>;
    // Stages live in dynamic shared memory so deep pipelines (> 48KB static
    // limit) opt in via cudaFuncSetAttribute in the launcher.
    extern __shared__ __align__(16) char gemm_smem[];

    // Batch slice (grid.z): broadcast operands carry a 0 stride, so the
    // same pointer serves every batch.
    using ElemA = typename Mainloop::ElemA;
    using ElemB = typename Mainloop::ElemB;
    using OutT = typename Policy::OutT;
    const ElemA* a = reinterpret_cast<const ElemA*>(p.a_ptr) +
                  (int64_t)blockIdx.z * p.a_batch_stride;
    const ElemB* b = reinterpret_cast<const ElemB*>(p.b_ptr) +
                  (int64_t)blockIdx.z * p.b_batch_stride;
    auto* out = reinterpret_cast<OutT*>(p.out_ptr) +
                (int64_t)blockIdx.z * p.out_batch_stride;

    static_assert(Mainloop::kOutputReclaimsRings,
                  "output tile must fit the reclaimed operand smem");
    const int2 blk = GemmTileScheduler::tile(blockIdx, gridDim, p.raster);
    Mainloop mainloop(gemm_smem, a, b, p.m, p.n, p.k, p.a_ld, p.b_ld,
                      threadIdx.x, blk);
    typename Mainloop::AccTensor acc = {};  // C cells on the (mt, nt) grid
    mainloop.prologue();
    mainloop.accumulate(acc);
    // Drain the pipeline before the epilogue reclaims the operand rings:
    // cp_async_wait_all drains only the CALLING thread's copies and the
    // last mainloop iteration carries no trailing barrier — without this,
    // a thread racing into the epilogue scatters the output tile over
    // peers' still-in-flight staging writes. One barrier closes both.
    astrai::PipelineSync<Mainloop::kStages>{}.drain();
    Epilogue(gemm_smem, p, blk.x, blk.y, threadIdx.x).run(acc, out);
}


// TMA orchestrator (sm_90+, dual-congruous staging): identical rings,
// layouts and epilogue; the staging discipline changes — one elected
// thread arms a per-slot mbarrier and issues the operand boxes
// (cp.async.bulk.tensor), consumers wait the slot's phase. The rings sit
// on a 1024B-aligned base because TMA swizzles the ABSOLUTE smem address
// (the pad is budgeted in Policy::kSmemBytes), and the mbarriers live
// right past the B ring.
template <typename Policy, bool kRank3A, bool kRank3B>
__global__ void __launch_bounds__(Policy::kCtaThreads, Policy::kMinCtas)
    gemm_kernel_tma(GemmParams p, const __grid_constant__ CUtensorMap tma_a,
                    const __grid_constant__ CUtensorMap tma_b) {
    using Traits = typename Policy::Traits;
    using Mainloop = GemmCollectiveMainloop<Policy>;
    using Epilogue = GemmCollectiveEpilogue<Policy>;
    static_assert(!Mainloop::kDirectA && !Mainloop::kDirectB,
                  "TMA staging requires dual-congruous operands");
    extern __shared__ __align__(16) char gemm_smem[];
    // Round the ring base up to its 1024B pattern period. Two's-complement
    // form: already-aligned bases pad 0 (~p would pad 1023 and misalign).
    char* smem =
        gemm_smem + ((-reinterpret_cast<uintptr_t>(gemm_smem)) & 1023u);

    using OutT = typename Policy::OutT;
    auto* out = reinterpret_cast<OutT*>(p.out_ptr) +
                (int64_t)blockIdx.z * p.out_batch_stride;

    static_assert(Mainloop::kOutputReclaimsRings,
                  "output tile must fit the reclaimed operand smem");

    GemmTmaContext<kRank3A, kRank3B> tma;
    tma.map_a = &tma_a;
    tma.map_b = &tma_b;
    tma.bars = reinterpret_cast<uint64_t*>(
        smem + Mainloop::RingA::Layout::kTotalBytes +
               Mainloop::RingB::Layout::kTotalBytes);
    tma.depth = Mainloop::kARing;
    tma.z = blockIdx.z;
    if (threadIdx.x == 0) {
        for (int s = 0; s < Mainloop::kARing; ++s) {
            astrai::mbarrier_init(tma.full(s), 1);  // producer expect_tx
            astrai::mbarrier_init(tma.empty(s), Policy::kCtaThreads);
        }
    }
    __syncthreads();

    const int2 blk = GemmTileScheduler::tile(blockIdx, gridDim, p.raster);
    Mainloop mainloop(smem, static_cast<const typename Mainloop::ElemA*>(p.a_ptr),
                      static_cast<const typename Mainloop::ElemB*>(p.b_ptr), p.m,
                      p.n, p.k, p.a_ld, p.b_ld, threadIdx.x, blk);
    typename Mainloop::AccTensor acc = {};
    mainloop.prologue(tma);
    mainloop.accumulate(acc, tma);
    // No cp.async groups on this path; the CTA join alone releases the
    // rings for the epilogue's reclaim.
    __syncthreads();
    Epilogue(smem, p, blk.x, blk.y, threadIdx.x).run(acc, out);
}

// ---------------------------------------------------------------------------
// Launchers — pure CUDA, usable from the binding and the C tests. The
// runtime knobs live in GemmConfig (policy.cuh), owned at runtime by
// astrai.extension.plan.
// ---------------------------------------------------------------------------

// Grid for one Policy's tile: N x M blocks, batch on z.
template <typename Traits>
dim3 gemm_grid(const GemmParams& p) {
    return dim3((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
                (p.m + Traits::kBlockM - 1) / Traits::kBlockM, p.batch);
}

// One plan-log line per launch (" mx" marks the block_scale cell).
inline void log_gemm_plan(const GemmParams& p, const dim3& grid, int bm,
                          int bn, int stages, int smem, bool tma,
                          bool mx = false) {
    if (!gemm_plan_log_enabled()) return;
    std::fprintf(stderr,
                 "[gemm-plan] %lldx%lldx%lld b=%d -> tile %dx%d s%d%s%s "
                 "grid %dx%dx%d raster %d smem %d\n",
                 (long long)p.m, (long long)p.n, (long long)p.k, p.batch, bm,
                 bn, stages, tma ? " tma" : "", mx ? " mx" : "", grid.x,
                 grid.y, grid.z, p.raster, smem);
}

// Launch with the smem budget: >48KB opt-ins once per instantiation via
// cudaFuncSetAttribute. Templated on the kernel VALUE (auto NTTP) so
// same-signature kernels never share the armed flag; a failed opt-in arms
// nothing and the launch fails loudly.
template <auto Kernel, typename... Args>
void launch_with_smem(int smem_bytes, dim3 grid, dim3 block,
                      cudaStream_t stream, Args... args) {
    if (smem_bytes > 48 * 1024) {
        static bool armed = false;  // per instantiation
        if (!armed) {
            const cudaError_t err = cudaFuncSetAttribute(
                Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes);
            armed = (err == cudaSuccess);
        }
    }
    Kernel<<<grid, block, smem_bytes, stream>>>(args...);
    ASTRAI_LAUNCH_CHECK();
}

// Dtype-class derivation (returns policy.cuh's GemmPerfClass): the mma
// promotion rule plus operand widths (mixed bf16xfp8 lands with W8A16 —
// same bytes, same promoted bf16 k16 mma). int8 is tested FIRST: it
// promotes to an int8 mma, so the "not bf16 -> fp8" arm would swallow it
// and every W8A8 row would be unreachable.
template <typename ElemA, typename ElemB>
constexpr GemmPerfClass gemm_perf_class() {
    using MmaT = typename gemm_mma_traits<ElemA, ElemB>::MmaT;
    if constexpr (std::is_same_v<ElemA, int8_t> &&
                  std::is_same_v<ElemB, int8_t>) {
        return GemmPerfClass::kW8A8;
    } else if constexpr (!std::is_same_v<MmaT, __nv_bfloat16>) {
        return GemmPerfClass::kF8A8;  // native fp8 symmetric pair
    } else if constexpr (std::is_same_v<ElemA, __nv_bfloat16> &&
                         std::is_same_v<ElemB, __nv_bfloat16>) {
        return GemmPerfClass::kW16A16;
    } else {
        return GemmPerfClass::kW8A16;
    }
}

static_assert(gemm_perf_class<int8_t, int8_t>() == GemmPerfClass::kW8A8,
              "int8 x int8 is its own class (see the ordering note above)");
static_assert(gemm_perf_class<__nv_fp8_e4m3, __nv_fp8_e4m3>() ==
                  GemmPerfClass::kF8A8,
              "fp8 x fp8 keys the F8A8 table");
static_assert(gemm_perf_class<__nv_bfloat16, __nv_bfloat16>() ==
                  GemmPerfClass::kW16A16,
              "bf16 x bf16 keys the W16A16 table");
static_assert(gemm_perf_class<__nv_bfloat16, int8_t>() ==
                  GemmPerfClass::kW8A16,
              "a quantized weight against bf16 activations keys W8A16");



// The one place a GemmParams becomes planner input and the dispatch key is
// derived: perf class, widths and crosswise are computed from the typed call,
// so a caller cannot hand the planner a key that contradicts its own types.
// OutT (default bf16) prices the model cost's output term at real size.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename OutT = __nv_bfloat16>
PlanQuery plan_query(const GemmParams& p, const DeviceFacts& dev) {
    PlanQuery q;
    q.m = p.m;
    q.n = p.n;
    q.k = p.k;
    q.batch = p.batch;
    q.perf_class = (int)gemm_perf_class<ElemA, ElemB>();
    q.crosswise = crosswise_of<LayoutA, LayoutB>();
    q.ba = (int)sizeof(ElemA);
    q.bb = (int)sizeof(ElemB);
    q.out_elem_bytes = (int)sizeof(OutT);
    // The staging the launch that follows will take — launch_plan_impl's
    // predicate minus the descriptor-encodable runtime check: TMA only for
    // the dual-congruous layout pair with descriptor-encodable dtypes on
    // sm_90+ without the kill switch, cp.async otherwise.
    q.tma = crosswise_of<LayoutA, LayoutB>() == 0 && sizeof(ElemA) <= 2 &&
            sizeof(ElemB) <= 2 && dev.cc >= 90 &&
            !gemm_tma_staging_disabled();
    q.dev = dev;
    return q;
}

// Typed dispatch entry: the layout tags as types tie the decision to the
// launch that follows it.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename OutT = __nv_bfloat16>
PlanDecision plan_dispatch_for(const GemmParams& p) {
    return plan_dispatch(
        plan_query<ElemA, ElemB, LayoutA, LayoutB, OutT>(p, device_facts()));
}
template <typename Policy>
void launch_policy(GemmParams p, cudaStream_t stream) {
    using Traits = typename Policy::Traits;
    dim3 grid = gemm_grid<Traits>(p);
    log_gemm_plan(p, grid, Traits::kBlockM, Traits::kBlockN, Traits::kStages,
                  Policy::kSmemBytes, /*tma=*/false, Traits::kMxCell);
    launch_with_smem<gemm_kernel<Policy>>(
        Policy::kSmemBytes, grid, dim3(Traits::kCtaThreads), stream, p);
}

// ---------------------------------------------------------------------------
// TMA staging (sm_90+): descriptor build + the TMA twin of launch_policy;
// descriptors are cached exact-match (tma.cuh), the encode is paid once.
// ---------------------------------------------------------------------------

// Output-reclaim feasibility: a fat output (fp32) can outgrow the ring the
// planner priced — the wide CTA's 128x256 of fp32 is 131072B against the
// 73728B a 1-byte pair leaves. Exactly the launch twins' reclaim
// static_assert, so a new CTA class cannot drift from it.
template <typename Tile, typename ElemA, typename ElemB, typename OutT>
constexpr bool reclaim_fits() {
    return Tile::CtaShape::kM * Tile::CtaShape::kN * sizeof(OutT) <=
           ring_smem_bytes(Tile::CtaShape::kM, Tile::CtaShape::kN,
                           Tile::CtaShape::kK, Tile::kStages,
                           (int)sizeof(ElemA), (int)sizeof(ElemB));
}

// Both operand descriptors for one TMA Policy. Dim/stride in bytes along
// the contract dim; batch is a third dim only when it strides. Swizzle mode,
// box extents and byte scaling derive from the staging layout (tma_spec in
// memory/tma.cuh) — the same instances the fragment readers consume.
template <typename Policy>
bool tma_maps_for(const GemmParams& p, CUtensorMap* ma, CUtensorMap* mb) {
    using Mainloop = GemmCollectiveMainloop<Policy>;
    const auto a = astrai::tma_map_cache().lookup(
        astrai::tma_spec<typename Mainloop::ElemA, typename Mainloop::SmemLayoutA,
                        Mainloop::kBlockM>(p.a_ptr, p.m, p.k, p.a_ld, p.batch,
                                           p.a_batch_stride));
    if (!a) return false;
    *ma = *a;
    const auto b = astrai::tma_map_cache().lookup(
        astrai::tma_spec<typename Mainloop::ElemB, typename Mainloop::SmemLayoutB,
                        Mainloop::kBlockN>(p.b_ptr, p.n, p.k, p.b_ld, p.batch,
                                           p.b_batch_stride));
    if (!b) return false;
    *mb = *b;
    return true;
}

// TMA launch for one Policy; false (nothing launched) on an undescribable
// operand (misaligned base/ld) so the caller falls back to cp.async.
template <typename Policy>
bool launch_policy_tma(const GemmParams& p, cudaStream_t stream) {
    using Traits = typename Policy::Traits;
    // The planner prices rings only; TMA's pad + barriers can tip past
    // the opt-in ceiling on the fattest pair — fall back, not fail.
    if (Policy::kSmemBytes > astrai::device_facts().smem_max) return false;
    CUtensorMap ma{}, mb{};
    if (!tma_maps_for<Policy>(p, &ma, &mb)) return false;
    dim3 grid = gemm_grid<Traits>(p);
    log_gemm_plan(p, grid, Traits::kBlockM, Traits::kBlockN, Traits::kStages,
                  Policy::kSmemBytes, /*tma=*/true, Traits::kMxCell);
    // Rank bits pick the instantiation: strided batch rides the 3D
    // emitters, broadcast keeps the shared 2D map; the per-stage pick
    // compiles away.
    auto launch_rank = [&](auto rank3a, auto rank3b) {
        launch_with_smem<gemm_kernel_tma<Policy, decltype(rank3a)::value,
                                         decltype(rank3b)::value>>(
            Policy::kSmemBytes, grid, dim3(Traits::kCtaThreads), stream, p,
            ma, mb);
    };
    if (p.batch > 1 && p.a_batch_stride > 0 && p.b_batch_stride > 0)
        launch_rank(std::true_type{}, std::true_type{});
    else if (p.batch > 1 && p.a_batch_stride > 0)
        launch_rank(std::true_type{}, std::false_type{});
    else if (p.batch > 1 && p.b_batch_stride > 0)
        launch_rank(std::false_type{}, std::true_type{});
    else
        launch_rank(std::false_type{}, std::false_type{});
    return true;
}

// Manifest dispatch (CUTLASS builder-table style): (CTA class, ring depth,
// k-tile depth) selects one manifest entry; the resolver maps it to a Policy
// and launches.
template <typename Manifest, typename Resolver>
bool dispatch_tile(const PlanDecision& d, const Resolver& resolve) {
    return std::apply(
        [&d, &resolve](auto... tiles) {
            return (... ||
                    (tile_class<decltype(tiles)>() ==
                         static_cast<TileClass>(d.recipe.cta) &&
                     decltype(tiles)::kStages == d.recipe.stages &&
                     (int)decltype(tiles)::CtaShape::kK == d.recipe.kk &&
                     resolve.template run<decltype(tiles)>()));
        },
        Manifest{});
}

// Narrow twin of a big tile for the reclaim fallback, same ring depth as
// the tile it replaces (the planner priced that ring). No kK=32 narrow s3
// exists — that big tile falls to its s2 twin.
template <typename Tile>
using narrow_fallback_t = std::conditional_t<
    Tile::CtaShape::kK == 32, Tile_128x64x32_W32x32_S2,
    std::conditional_t<(Tile::kStages >= 3), Tile_128x64x64_W32x32_S3,
                       Tile_128x64x64_W32x32_S2>>;

// The reclaim chain every instantiation terminates in: output outgrows the
// ring -> narrow twin; the kK=32 narrow (18KB ring) still cannot hold 4B/elem
// -> small CTA (24KB reclaims every output <= 4B/elem). Identity whenever
// the tile fits, so the launchers apply it unconditionally; the dispatch
// names every manifest tile as a potential substitute, so the chain is
// load-bearing even for unplanned geometries.
template <typename Tile, typename ElemA, typename ElemB, typename OutT>
using reclaim_fallback_t = std::conditional_t<
    reclaim_fits<Tile, ElemA, ElemB, OutT>(), Tile,
    std::conditional_t<
        reclaim_fits<narrow_fallback_t<Tile>, ElemA, ElemB, OutT>(),
        narrow_fallback_t<Tile>, Tile_64x64x64_W16x32_S2>>;

// TMA ladder resolver: the launch_plan gate already guarantees
// dual-congruous 1-/2-byte operands, so only a reclaim overflow swaps the
// CTA. conditional_t keeps every alias instantiable (an if-constexpr branch
// still NAMES its dead types — the kernel's static_assert requires it).
template <typename ElemA, typename ElemB, typename LayoutOut, typename OutT,
          bool UseMx = false>
struct TmaLauncher {
    const GemmParams& p;
    cudaStream_t stream;
    template <typename Tile>
    bool run() const {
        // The reclaim chain is applied unconditionally (identity whenever the
        // tile fits): the dispatch walks name every manifest tile, so even
        // the narrow/small classes need an out when an instantiation's output
        // outgrows their rings (the kK=32 narrow vs a 4B/elem output).
        using Widened = warp_widened_t<ElemA, ElemB, Tile>;
        using TileT = reclaim_fallback_t<Widened, ElemA, ElemB, OutT>;
        return launch_policy_tma<GemmPolicy<ElemA, ElemB, RowMajor, ColMajor, TileT, LayoutOut,
                       OutT, false, true, UseMx>>(p, stream);
    }
};

// cp.async ladder resolver: one substitution — a fat output (fp32) that
// cannot reclaim the ring routes to the narrow CTA (same math, lower
// reuse). Narrow and small pass through. (A second substitution — the big
// CTA downgraded to a predicated-loop twin on crosswise staging — measured
// 9-19% SLOWER in interleaved A/B, 2026-09-16; see policy.cuh's
// GemmTileConfig note. Buried.)
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut, typename OutT, bool UseMx = false>
struct CpAsyncLauncher {
    GemmParams p;
    cudaStream_t stream;
    template <typename Tile>
    bool run() const {
        // Warp widening + reclaim chain, both unconditional (identity when
        // the tile fits); the rule lives in policy.cuh's warp_widened_t.
        using Widened = warp_widened_t<ElemA, ElemB, Tile>;
        using TileT = reclaim_fallback_t<Widened, ElemA, ElemB, OutT>;
        launch_policy<GemmPolicy<ElemA, ElemB, LayoutA, LayoutB, TileT,
                                 LayoutOut, OutT, false, false, UseMx>>(
            p, stream);
        return true;
    }
};

// Plan -> Policy: dispatch_tile picks the manifest entry, the launcher
// applies the ladder's substitutions; stages >= 3 selects the deep-ring
// sibling. Params by value — the raster decision lands in the copy the
// kernel receives. UseMx threads the block_scale cell through.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut = RowMajor, typename OutT = __nv_bfloat16,
          bool UseMx = false>
void launch_plan_impl(GemmParams p, const PlanDecision& d,
                      cudaStream_t stream) {
    p.raster = d.raster;
    // Dual-congruous (crosswise 0): the only layout pair TMA can describe —
    // both operands staged as-is, so the descriptors are encodable.
    constexpr bool kCongruous = crosswise_of<LayoutA, LayoutB>() == 0;
    // TMA staging first when the layout pair and dtypes allow it (the
    // planner's stage/tile decisions are shared): sm_90+ device, no
    // kill switch, and every descriptor encodable — else the cp.async
    // twin below runs unchanged.
    if constexpr (kCongruous && sizeof(ElemA) <= 2 && sizeof(ElemB) <= 2) {
        if (!gemm_tma_staging_disabled() && astrai::device_facts().cc >= 90 &&
            dispatch_tile<manifest_for<ElemA, ElemB, RowMajor, ColMajor>>(
                d, TmaLauncher<ElemA, ElemB, LayoutOut, OutT, UseMx>{
                        p, stream}))
            return;
    }
    dispatch_tile<manifest_for<ElemA, ElemB, LayoutA, LayoutB>>(
        d, CpAsyncLauncher<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                           UseMx>{p, stream});
}

// Planner entry: symmetric fp8 rides the sm_120 block_scale cell unless
// knocked out.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut = RowMajor, typename OutT = __nv_bfloat16>
void launch_plan(GemmParams p, const PlanDecision& d, cudaStream_t stream) {
    constexpr bool kMxCell =
        (std::is_same_v<ElemA, __nv_fp8_e4m3> &&
         std::is_same_v<ElemB, __nv_fp8_e4m3>) ||
        (std::is_same_v<ElemA, __nv_fp8_e5m2> &&
         std::is_same_v<ElemB, __nv_fp8_e5m2>);
    if constexpr (kMxCell) {
        // cc is CC-tens (120 = CC 12.0). One half of a contract: CMake
        // emits the sm_120a slice exactly when "120" is in the arch list,
        // so the route fires only where that image exists.
        if (!gemm_mx_cell_disabled() && astrai::device_facts().cc == 120) {
            launch_plan_impl<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                             true>(p, d, stream);
            return;
        }
    }
    launch_plan_impl<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT>(
        p, d, stream);
}

// Pure problem rewrite: dual-N-contiguous (NN) has no instantiation — it
// runs as its transpose E = B^T A^T over swapped operands (CUTLASS-sm90's
// is_swapAB) with a transposed-epilogue scatter into the [M][N] buffer.
// The rare NN path pays a scalar-store scatter for it.
inline void canonicalize_gemm(GemmParams& p, bool& trans_a, bool& trans_b) {
    if (!trans_a && !trans_b) {
        GemmParams s = p;  // E = B^T * A^T: swap roles, M <-> N
        s.m = p.n;
        s.n = p.m;
        s.a_ptr = p.b_ptr;
        s.b_ptr = p.a_ptr;
        s.a_ld = p.b_ld;
        s.b_ld = p.a_ld;
        s.a_batch_stride = p.b_batch_stride;
        s.b_batch_stride = p.a_batch_stride;
        // The caller's [M][N] buffer read as E = B^T A^T: the epilogue
        // walks the caller's rows (kernel n) with the caller's N stride.
        s.out_ld = p.n;
        p = s;
        trans_a = trans_b = true;
    }
}

// The (trans_a, trans_b) -> layout-tag ladder, ONE home for the branch both
// the launch and the probe walk (a probe cannot answer for a layout the
// launch does not take). fn gets three tags; the probe ignores the output
// one. `swapped` marks the symmetric-NN rewrite's TT — the only arm whose
// OUTPUT tag is transposed. A visitor, not a tag-returning function: a tag's
// identity is its TYPE; every arm calls fn, keeping this total.
template <typename ElemA, typename ElemB, typename F>
auto with_layout_tags(bool trans_a, bool trans_b, bool swapped, F&& fn) {
    constexpr bool kSymmetric = std::is_same_v<ElemA, ElemB>;
    if (trans_a && trans_b) {
        if constexpr (kSymmetric) {
            if (swapped) return fn(ColMajor{}, ColMajor{}, ColMajor{});
            return fn(ColMajor{}, ColMajor{}, RowMajor{});
        }
        return fn(ColMajor{}, ColMajor{}, RowMajor{});
    }
    if (trans_b) {
        // NT (the fused-linear shape), the production nn.Linear route.
        return fn(RowMajor{}, ColMajor{}, RowMajor{});
    }
    if (trans_a) return fn(ColMajor{}, RowMajor{}, RowMajor{});
    if constexpr (kSymmetric) {
        // Unreachable: canonicalize_gemm turns a symmetric NN into the TT arm
        // above, so both callers arrive here for a mixed pair only. The arm
        // stays total anyway — the probe returns fn's value and needs no dead
        // fallback — and names the instantiation the rewrite's own TT branch
        // already takes.
        return fn(ColMajor{}, ColMajor{}, RowMajor{});
    } else {
        // Dual row-major: mixed only — symmetric NN was rewritten above
        // into the transposed TT kernel (if constexpr keeps this
        // instantiation out of symmetric builds).
        return fn(RowMajor{}, RowMajor{}, RowMajor{});
    }
}

// Dtype-generic entry: canonicalize, plan, wire the tags. ElemA/ElemB/OutT
// are independent knobs. The one asymmetry is NN: the swap rewrite assumes
// a single element type, so symmetric NN rewrites to TT while mixed NN
// instantiates the dual-row-major shape directly (A congruous, B
// crosswise).
template <typename ElemA, typename ElemB = ElemA, typename OutT = __nv_bfloat16>
void gemm_dispatch(GemmParams p, cudaStream_t stream, bool trans_a,
                   bool trans_b) {
    constexpr bool kSymmetric = std::is_same_v<ElemA, ElemB>;
    bool swapped = false;
    if constexpr (kSymmetric) {
        swapped = !trans_a && !trans_b;  // canonicalize rewrites NN
        canonicalize_gemm(p, trans_a, trans_b);
    }
    // Each branch plans from its OWN layout tags, so the plan and the launch
    // below cannot disagree about the crosswise count, widths or perf class.
    // The tags ride empty tag instances; decltype recovers the types.
    const auto launch = [&](auto la, auto lb, auto lout) {
        launch_plan<ElemA, ElemB, decltype(la), decltype(lb), decltype(lout),
                    OutT>(
            p, plan_dispatch_for<ElemA, ElemB, decltype(la), decltype(lb),
                                 OutT>(p),
            stream);
    };
    with_layout_tags<ElemA, ElemB>(trans_a, trans_b, swapped, launch);
}

// Explicit-instantiation spelling shared by the per-pair TUs (bare) and
// gemm.cu's extern block: one place names the signature.
#define ASTRAI_GEMM_INSTANTIATE(W, A) \
    template void gemm_dispatch<W, A>(GemmParams, cudaStream_t, bool, bool)

// Host-only planner probe (the autotuner's coverage check): the decision
// gemm_dispatch would make, without a launch — the planner is GPU-free.
// The shared tag ladder (NN rewrite included) keeps a probe from
// disagreeing with the real call's branch. Returns the decision plus the
// query it answered.
template <typename ElemA, typename ElemB>
std::pair<PlanDecision, PlanQuery> plan_probe_for(
    int64_t m, int64_t n, int64_t k, int64_t batch,
    bool trans_a, bool trans_b, const DeviceFacts& dev) {
    GemmParams p{};  // the planner reads m/n/k/batch only
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.batch = static_cast<int>(batch);
    bool swapped = false;
    if constexpr (std::is_same_v<ElemA, ElemB>) {
        swapped = !trans_a && !trans_b;
        canonicalize_gemm(p, trans_a, trans_b);  // symmetric NN -> transposed TT
    }
    return with_layout_tags<ElemA, ElemB>(
        trans_a, trans_b, swapped, [&](auto la, auto lb, auto) {
            PlanQuery q =
                plan_query<ElemA, ElemB, decltype(la), decltype(lb)>(p, dev);
            return std::make_pair(plan_dispatch(q), std::move(q));
        });
}

}  // namespace gemm
}  // namespace astrai
