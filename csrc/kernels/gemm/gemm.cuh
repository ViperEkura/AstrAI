#pragma once
// GEMM-family umbrella (bf16 / int8 / fp8): the kernel orchestrator and the
// host-side launch planning. Device layers live in gemm/ (policy / load /
// scheduler / mainloop / epilogue) — pure CUDA, no torch; launchers are
// plain functions shared by the torch binding and the C tests. Layout tags
// and the NN swap semantics live in common.h and
// docs/developer/cuda_kernels.md.

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <tuple>
#include <type_traits>

#include "common/pipeline.cuh"
#include "common/device.cuh"
#include "common/launch.cuh"
#include "epilogue.cuh"
#include "gemm/common.h"
#include "gemm/plan_table.h"
#include "mainloop.cuh"
#include "policy.cuh"
#include "scheduler.cuh"

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

    static_assert(Mainloop::kBlockM * Mainloop::kBlockN * sizeof(OutT) <=
                  Mainloop::RingA::Layout::kTotalBytes +
                  Mainloop::RingB::Layout::kTotalBytes,
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

    static_assert(Mainloop::kBlockM * Mainloop::kBlockN * sizeof(OutT) <=
                  Mainloop::RingA::Layout::kTotalBytes +
                  Mainloop::RingB::Layout::kTotalBytes,
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
// Launchers — pure CUDA (no torch), usable from the binding and pure C tests.
// ---------------------------------------------------------------------------

// ASTR_GEMM_PLAN=1: read-only launch log (shape -> recipe / grid / raster)
// from launch_policy. The env is re-read on every call, like the table source:
// a sweep toggles it around one launch to learn which source served that
// launch, which neither a cached read nor a log left on through the timed loop
// can do. Nothing here can change the launch.
inline bool gemm_plan_log() {
    return std::getenv("ASTR_GEMM_PLAN") != nullptr;
}

// Experiment/debug knobs, one getenv at first use:
//   ASTR_GEMM_NO_TMA=1 forces the cp.async staging everywhere.
inline bool gemm_tma_disabled() {
    static const bool off = std::getenv("ASTR_GEMM_NO_TMA") != nullptr;
    return off;
}

// ASTR_GEMM_NO_MX=1 keeps symmetric fp8 on the plain cell — the A/B knob
// for the sm_120 block_scale cell (MxMmaOp; env read once per process).
inline bool gemm_mx_disabled() {
    static const bool off = std::getenv("ASTR_GEMM_NO_MX") != nullptr;
    return off;
}

// Grid for one Policy's tile: N x M block count, batch on z.
template <typename Traits>
dim3 gemm_grid(const GemmParams& p) {
    return dim3((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
                (p.m + Traits::kBlockM - 1) / Traits::kBlockM, p.batch);
}

// One read-only plan-log line per launch (ASTR_GEMM_PLAN=1; " mx" marks the
// block_scale cell).
inline void log_gemm_plan(const GemmParams& p, const dim3& grid, int bm,
                          int bn, int stages, int smem, bool tma,
                          bool mx = false) {
    if (!gemm_plan_log()) return;
    std::fprintf(stderr,
                 "[gemm-plan] %lldx%lldx%lld b=%d -> tile %dx%d s%d%s%s "
                 "grid %dx%dx%d raster %d smem %d\n",
                 (long long)p.m, (long long)p.n, (long long)p.k, p.batch, bm,
                 bn, stages, tma ? " tma" : "", mx ? " mx" : "", grid.x,
                 grid.y, grid.z, p.raster, smem);
}

// Launch one kernel instantiation with its shared-memory budget: budgets
// beyond the 48KB static limit opt in once per instantiation via
// cudaFuncSetAttribute. Templated on the kernel *value* (auto NTTP) so
// every instantiation owns its own armed flag — same-signature kernels
// must not share it. A failed opt-in arms nothing, so the launch below
// fails loudly through the caller's error checks.
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

// Launch configuration — a pure function of the problem (unit-testable
// without a GPU). Raster order is a plan field picked by the aspect
// heuristic (plan_raster); a manual p.raster=0 keeps plain raster
// reachable for experiments.
struct GemmPlan {
    // The CTA class is the tile manifest's dispatch key (policy.cuh).
    using Cta = TileClass;
    Cta cta;
    // Ring depth (kStages) this launch runs. The manifest default is 2;
    // the planner raises it to 3 where the dtype pair's thinner operands
    // leave smem headroom under the same 96KB budget (humming's
    // _fit_num_stages rule: deepest ring that fits).
    int stages;
    int raster;  // GemmParams::raster value this launch runs
    // Ring K (kK), the third axis of the manifest key: the k-tile-depth twins
    // are separate tiles, so a plan that named only the CTA class and depth
    // could not reach them.
    int kk;
};

// One [gemm-plan] line naming the planner's decision (ASTR_GEMM_PLAN = 1):
// the AOT row table or the last-resort degraded band that produced the plan.
inline void log_plan_decision(const char* src, const GemmParams& p,
                              const GemmPlan& plan) {
    if (!gemm_plan_log()) return;
    std::fprintf(stderr,
                 "[gemm-plan] %s m%lld n%lld k%lld b=%d -> cta%d s%d "
                 "raster %d\n",
                 src, (long long)p.m, (long long)p.n, (long long)p.k, p.batch,
                 (int)plan.cta, plan.stages, plan.raster);
}

// Raster order. Direction follows the tile aspect (walk the dimension with
// more tiles fastest, CUTLASS's rule): the N-side mirrored group keeps the
// measured width 8. The M-side group width is humming's L2-budget rule
// (tune/raster.py) instead of a fixed width: a group's A tiles are reused
// across its whole N sweep, so the group is sized to keep them L2-resident
// while B streams through the remainder — B already L2-resident means no
// grouping pays (g = 1, plain raster); otherwise reserve a B-streaming
// fraction of L2 (fatter B traffic than A reserves more), cap the group so
// the A side fits, and floor it at enough M rows to keep every SM busy
// within one group sweep.
inline int plan_raster(const GemmParams& p, int bm, int bn, int ba, int bb,
                       const DeviceFacts& dev) {
    const int64_t m_tiles = (p.m + bm - 1) / bm;
    const int64_t n_tiles = (p.n + bn - 1) / bn;
    if (m_tiles < n_tiles) return -8;
    if (p.n * p.k * (int64_t)bb <= dev.l2_bytes * 7 / 10) return 1;
    const double reserve = 0.12 + 0.28 * (double)bb / (double)ba;
    const double budget = (1.0 - std::min(reserve, 0.5)) * (double)dev.l2_bytes;
    const int64_t ub = (int64_t)(budget / ((double)bm * (double)p.k * ba));
    const int64_t lb = (dev.sms + n_tiles - 1) / n_tiles;
    int64_t g = std::min(ub, m_tiles);
    if (ub >= lb) g = std::min(std::max(g, lb), m_tiles);
    return (int)std::max(g, (int64_t)1);
}

// Dtype-class ids the plan-table rows key on.
enum class GemmPerfClass : int { kW16A16 = 0, kW8A16, kW8A8, kF8A8 };

// Compile-time dtype-class derivation from the operand pair (the mma
// promotion rule plus operand widths; mixed bf16xfp8 lands with the 2B x
// 1B class — same bytes and same promoted bf16 k16 mma as W8A16).
template <typename ElemA, typename ElemB>
constexpr GemmPerfClass gemm_perf_class() {
    using MmaT = typename gemm_mma_traits<ElemA, ElemB>::MmaT;
    if constexpr (!std::is_same_v<MmaT, __nv_bfloat16>) {
        return GemmPerfClass::kF8A8;  // native fp8 symmetric pair
    } else if constexpr (std::is_same_v<ElemA, __nv_bfloat16> &&
                         std::is_same_v<ElemB, __nv_bfloat16>) {
        return GemmPerfClass::kW16A16;
    } else if constexpr (std::is_same_v<ElemA, int8_t> &&
                         std::is_same_v<ElemB, int8_t>) {
        return GemmPerfClass::kW8A8;
    } else {
        return GemmPerfClass::kW8A16;
    }
}



// A row is a plan: its CTA class names the manifest geometry (resolved at
// launch through dispatch_tile), the ring depth rides on the row, raster 0
// means "plan_raster for this row's geometry". A row whose ring exceeds
// the smem opt-in ceiling is a stale tuning artifact — no plan from that
// row, the caller tries the next source.
inline std::optional<GemmPlan> plan_from_row(const TableRow& row,
                                             const GemmParams& p, int ba,
                                             int bb, const DeviceFacts& dev,
                                             int crosswise = 0) {
    int bm, bn;
    plan_row_geometry(row.cta, bm, bn);
    // A row can name a geometry or a depth that this operand pair has no tile
    // for: the wide CTA only exists on the 1-byte manifest, and the manifests
    // carry kK 32 and 64. Such a row would match no tile in dispatch_tile and
    // launch nothing at all, so it is rejected here — the next source, or the
    // degraded bands, serves the shape instead. The failure is silent (the
    // launcher just does not fire), so this gate is the only thing standing
    // between a stale row and an uninitialized output tile.
    if (row.cta == TileClass::kWide128x256 && (ba != 1 || bb != 1))
        return std::nullopt;
    if (!row_k_supported(row.kk)) return std::nullopt;
    // Only the dual-2-byte ladder carries the kK=32 twins (policy.cuh): a
    // 1-byte line holds half as many 16B chunks, so no kK=32 tile divides its
    // load path. A row naming one for such a pair matches no tile either, and
    // is rejected on the same terms as the width rule above.
    if (row.kk == 32 && (ba != 2 || bb != 2)) return std::nullopt;
    // Only the 64x64 kK=64 geometry carries the deep s4/s5 rings (policy.cuh),
    // and only it has the smem room for them: every other class or depth would
    // dispatch to nothing. The ring check below cannot catch this — a deep
    // kK=32 ring is only 40-80KB, well inside the ceiling — so the rule has to
    // be explicit here. Likewise the wide CTA is a lone s2 entry on the 1-byte
    // ladder, deeper than the ring check expects: 128x256 s3 is 96KB, which
    // fits, and still names no tile.
    if (row.stages > 3 &&
        (row.cta != TileClass::kSmall64 || row.kk != kTableRowK))
        return std::nullopt;
    if (row.cta == TileClass::kWide128x256 && row.stages != 2)
        return std::nullopt;
    // Crosswise staging runs the conservative ladder, which carries kK 64 and
    // no wide CTA; a row naming more than that would match no tile there.
    if (crosswise != 0 && (row.kk != kTableRowK || row.cta == TileClass::kWide128x256))
        return std::nullopt;
    if (ring_smem_bytes(bm, bn, row.kk, row.stages, ba, bb) > dev.smem_max)
        return std::nullopt;
    return GemmPlan{row.cta, row.stages,
                    row.raster != 0 ? row.raster
                                    : plan_raster(p, bm, bn, ba, bb, dev),
                    row.kk};
}

// crosswise_ops counts the operands taking the direct crosswise load
// (A ColMajor / B RowMajor storage): 0 = dual-congruous NT, 1 = TT and
// the NN swap, 2 = TN. ba / bb are the operand element sizes; perf is
// the dtype class the table rows key on.
inline GemmPlan plan_gemm(const GemmParams& p, int ba, int bb,
                          GemmPerfClass perf, int crosswise_ops = 0) {
    const DeviceFacts dev = device_facts();
    // Table-only dispatch, one source: the AOT rows (override file, then the
    // compiled-in ones). plan_from_row smem-gates the row, so a stale tuning
    // ring falls through instead of failing a launch. The original cost model
    // is deleted from the codebase — a miss falls to the degraded bands (open
    // on M, -1 keys; always match, so planning stays a total function);
    // ASTR_GEMM_TABLE=- skips the rows entirely.
    if (auto row = table_row(p, (int)perf, crosswise_ops); row) {
        if (auto plan = plan_from_row(*row, p, ba, bb, dev, crosswise_ops)) {
            log_plan_decision("table", p, *plan);
            return *plan;
        }
    }
    // The degraded bands are open on N with -1 keys, so a row always
    // matches and planning stays a total function (degraded_row_for
    // covers the degenerate m=0); the smem gate cannot demote them — the
    // s2 64x64 ring is the floor every supported device fits.
    GemmPlan plan = *plan_from_row(degraded_row_for(p.m), p, ba, bb, dev);
    log_plan_decision("degraded (no table)", p, plan);
    return plan;
}

// Grid + launch for one concrete Policy — the only place a GEMM kernel
// goes to the wire.
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
// TMA staging (sm_90+): descriptor build + the TMA twin of launch_policy.
// The descriptors are cached exact-match (tma.cuh), so steady-state calls
// with unchanged tensors and tile pay the encode once.
// ---------------------------------------------------------------------------

// Output-reclaim feasibility of one tile: the epilogue scatters the output
// tile into the reclaimed operand rings, so a fat output (fp32, 4B/elem) can
// outgrow the ring the planner priced — the wide CTA's 128x256 of fp32 is
// 131072B against the 73728B a 1-byte pair leaves it. Keyed on the tile's own
// geometry, which makes this exactly the predicate the launch twins' reclaim
// static_assert states, so a new CTA class or ring depth cannot drift from the
// assert that guards it. One definition serves both dispatch ladders.
template <typename Tile, typename ElemA, typename ElemB, typename OutT>
constexpr bool reclaim_fits() {
    return Tile::CtaShape::kM * Tile::CtaShape::kN * sizeof(OutT) <=
           ring_smem_bytes(Tile::CtaShape::kM, Tile::CtaShape::kN,
                           Tile::CtaShape::kK, Tile::kStages,
                           (int)sizeof(ElemA), (int)sizeof(ElemB));
}

// Build both operand descriptors for one TMA Policy's geometry. Dim/stride
// units are bytes along the contract dim; the batch encodes as a third
// dimension only when it strides (a broadcast operand shares one 2D map's
// coordinates across grid.z). The swizzle mode, box extents and byte
// scaling all derive from the operand's declared staging layout (the
// TmaSwizzleOf / tma_spec trait layer in common/tma.cuh) — the same
// instances the fragment readers consume, so the map cannot drift from
// the staging.
template <typename Policy>
bool tma_maps_for(const GemmParams& p, const CUtensorMap** ma,
                  const CUtensorMap** mb) {
    using Mainloop = GemmCollectiveMainloop<Policy>;
    *ma = astrai::tma_map_cache().lookup(
        astrai::tma_spec<typename Mainloop::ElemA, typename Mainloop::SmemLayoutA,
                        Mainloop::kBlockM>(p.a_ptr, p.m, p.k, p.a_ld, p.batch,
                                           p.a_batch_stride));
    *mb = astrai::tma_map_cache().lookup(
        astrai::tma_spec<typename Mainloop::ElemB, typename Mainloop::SmemLayoutB,
                        Mainloop::kBlockN>(p.b_ptr, p.n, p.k, p.b_ld, p.batch,
                                           p.b_batch_stride));
    return *ma != nullptr && *mb != nullptr;
}

// TMA launch for one Policy; false (nothing launched) when an operand
// cannot be described — misaligned base/ld — so the caller falls back to
// the cp.async twin.
template <typename Policy>
bool launch_policy_tma(const GemmParams& p, cudaStream_t stream) {
    using Traits = typename Policy::Traits;
    // The planner's feasibility gate prices rings only; the TMA budget
    // adds the alignment pad + barriers and can tip past the opt-in
    // ceiling on the fattest pair — fall back rather than fail.
    if (Policy::kSmemBytes > astrai::device_facts().smem_max) return false;
    const CUtensorMap *ma = nullptr, *mb = nullptr;
    if (!tma_maps_for<Policy>(p, &ma, &mb)) return false;
    dim3 grid = gemm_grid<Traits>(p);
    log_gemm_plan(p, grid, Traits::kBlockM, Traits::kBlockN, Traits::kStages,
                  Policy::kSmemBytes, /*tma=*/true, Traits::kMxCell);
    // Rank bits pick the kernel instantiation: a strided batch rides the
    // 3D emitters, a broadcast operand keeps its shared 2D map — the
    // per-stage 2D/3D issue pick compiles away either way.
    auto launch_rank = [&](auto rank3a, auto rank3b) {
        launch_with_smem<gemm_kernel_tma<Policy, decltype(rank3a)::value,
                                         decltype(rank3b)::value>>(
            Policy::kSmemBytes, grid, dim3(Traits::kCtaThreads), stream, p,
            *ma, *mb);
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

// Manifest dispatch (CUTLASS builder-table style): the plan's (CTA class,
// ring depth, k-tile depth) selects exactly one manifest entry — the ||
// short-circuits — and the resolver maps its tile onto a concrete Policy and
// launches.
template <typename Manifest, typename Resolver>
bool dispatch_tile(const GemmPlan& plan, const Resolver& resolve) {
    return std::apply(
        [&plan, &resolve](auto... tiles) {
            return (... ||
                    (tile_class<decltype(tiles)>() == plan.cta &&
                     decltype(tiles)::kStages == plan.stages &&
                     (int)decltype(tiles)::CtaShape::kK == plan.kk &&
                     resolve.template run<decltype(tiles)>()));
        },
        Manifest{});
}

// Narrow twin of a big tile for the output-reclaim fallback, carrying the
// same ring depth as the tile it replaces (the planner priced that ring, so a
// deeper substitute could overflow the smem opt-in). There is no kK=32 narrow
// s3, so a deep kK=32 big tile falls back to its s2 twin.
template <typename Tile>
using narrow_fallback_t = std::conditional_t<
    Tile::CtaShape::kK == 32, Tile_128x64x32_W32x32_S2_Fast,
    std::conditional_t<(Tile::kStages >= 3), Tile_128x64x64_W32x32_S3_Fast,
                       Tile_128x64x64_W32x32_S2_Fast>>;

// TMA ladder resolver. The gate in launch_plan already guarantees
// dual-congruous 1-/2-byte operands, so the fast tile stays; only an
// output-reclaim overflow swaps the CTA for its narrow twin. std::conditional_t
// keeps every alias instantiable, which the kernel's reclaim static_assert
// requires (an if-constexpr branch still NAMES its dead types).
template <typename ElemA, typename ElemB, typename LayoutOut, typename OutT,
          bool UseMx = false>
struct TmaLauncher {
    const GemmParams& p;
    cudaStream_t stream;
    template <typename Tile>
    bool run() const {
        // Only the geometries whose output tile can outgrow their own rings
        // carry a substitution; the narrow and small classes always fit.
        constexpr bool kReclaimGated = tile_class<Tile>() == TileClass::kBig128 ||
                                       tile_class<Tile>() == TileClass::kWide128x256;
        using TileT = std::conditional_t<
            kReclaimGated && !reclaim_fits<Tile, ElemA, ElemB, OutT>(),
            narrow_fallback_t<Tile>, Tile>;
        return launch_policy_tma<GemmPolicy<ElemA, ElemB, RowMajor, ColMajor, TileT, LayoutOut,
                       OutT, false, true, UseMx>>(p, stream);
    }
};

// cp.async ladder resolver. Two substitutions, both on the CTA classes whose
// output tile can outgrow their own rings: the fast loop exists only for
// dual-congruous staging (crosswise operands take the predicated generic loop
// — the NonFast twin), and a fat output (fp32, 4B) that cannot reclaim the
// ring routes to the narrow CTA — same math at lower reuse. Narrow and small
// entries pass through.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut, typename OutT, bool kBigFast, bool UseMx = false>
struct CpAsyncLauncher {
    GemmParams p;
    cudaStream_t stream;
    template <typename Tile>
    bool run() const {
        using NonFast = GemmTileConfig<typename Tile::CtaShape,
                                       typename Tile::WarpShape, Tile::kStages,
                                       false>;
        constexpr bool kFits = reclaim_fits<Tile, ElemA, ElemB, OutT>();
        constexpr bool kBig = tile_class<Tile>() == TileClass::kBig128;
        constexpr bool kWide = tile_class<Tile>() == TileClass::kWide128x256;
        using TileT = std::conditional_t<
            kBig, std::conditional_t<
                      kFits, std::conditional_t<kBigFast, Tile, NonFast>,
                      narrow_fallback_t<Tile>>,
            std::conditional_t<kWide, std::conditional_t<
                                          kFits, Tile, narrow_fallback_t<Tile>>,
                               Tile>>;
        launch_policy<GemmPolicy<ElemA, ElemB, LayoutA, LayoutB, TileT,
                                 LayoutOut, OutT, false, false, UseMx>>(
            p, stream);
        return true;
    }
};

// Plan -> Policy: compose the operand facts with one manifest tile
// (dispatch_tile); CpAsyncLauncher applies this ladder's substitutions.
// plan.stages >= 3 selects the deep-ring sibling of the same geometry (the
// planner only raises it where the operand pair's ring fits the smem
// opt-in ceiling). Takes the params by value: the plan's raster decision
// lands in the copy the kernel receives (callers keep theirs). UseMx threads
// the block_scale cell through both ladders (launch_plan routes it).
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut = RowMajor, typename OutT = __nv_bfloat16,
          bool UseMx = false>
void launch_plan_impl(GemmParams p, const GemmPlan& plan,
                      cudaStream_t stream) {
    p.raster = plan.raster;
    constexpr bool kBigFast = !std::is_same_v<LayoutA, ColMajor> &&
                              !std::is_same_v<LayoutB, RowMajor>;
    // TMA staging first when the layout pair and dtypes allow it (the
    // planner's stage/tile decisions are shared): sm_90+ device, no
    // kill switch, and every descriptor encodable — else the cp.async
    // twin below runs unchanged.
    if constexpr (kBigFast && sizeof(ElemA) <= 2 && sizeof(ElemB) <= 2) {
        if (!gemm_tma_disabled() && astrai::device_facts().cc >= 90 &&
            dispatch_tile<manifest_for<ElemA, ElemB, RowMajor, ColMajor>>(
                plan, TmaLauncher<ElemA, ElemB, LayoutOut, OutT, UseMx>{
                          p, stream}))
            return;
    }
    dispatch_tile<manifest_for<ElemA, ElemB, LayoutA, LayoutB>>(
        plan, CpAsyncLauncher<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                              kBigFast, UseMx>{p, stream});
}

// The planner entry: symmetric fp8 rides the sm_120 block_scale cell unless
// ASTR_GEMM_NO_MX knocks it out.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut = RowMajor, typename OutT = __nv_bfloat16>
void launch_plan(GemmParams p, const GemmPlan& plan, cudaStream_t stream) {
    constexpr bool kMxCell =
        (std::is_same_v<ElemA, __nv_fp8_e4m3> &&
         std::is_same_v<ElemB, __nv_fp8_e4m3>) ||
        (std::is_same_v<ElemA, __nv_fp8_e5m2> &&
         std::is_same_v<ElemB, __nv_fp8_e5m2>);
    if constexpr (kMxCell) {
        // cc is the CC-tens runtime form (120 = CC 12.0 — the convention
        // table lives at csrc/CMakeLists.txt's arch-level comment). This
        // gate is one half of a contract: the CMake side emits the
        // sm_120a SASS slice exactly when "120" is in the arch list, so
        // the route fires only where that image exists.
        if (!gemm_mx_disabled() && astrai::device_facts().cc == 120) {
            launch_plan_impl<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                             true>(p, plan, stream);
            return;
        }
    }
    launch_plan_impl<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT>(
        p, plan, stream);
}

// Pure problem rewrite: the dual-N-contiguous problem (trans_a/trans_b both
// false) has no dedicated instantiation — it runs as its transpose
// E[N][M] = B^T @ A^T (CUTLASS-sm90's is_swapAB) over swapped operands,
// with the geometry-derived transposed epilogue staging scattering into
// [M][N] row-major buffer. The rewritten trans flags become the layout tags
// the launcher instantiates; the NN path pays a scalar-store scatter, which
// its rare usage makes the right trade.
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

// Dtype-generic entry point: canonicalize the problem, plan the launch,
// wire the layout tags through. ElemA / ElemB / OutT are independent
// knobs; the fp8 pairs enter with their element types directly.
//
// Symmetric and mixed dtypes share this fan-out; the one asymmetry is NN
// (dual row-major storage): the swap rewrite exchanges operand roles and
// so assumes a single element type — symmetric operands rewrite to the
// transposed TT kernel, mixed operands instantiate the dual-row-major
// shape directly (A congruous, B crosswise) instead.
template <typename ElemA, typename ElemB = ElemA, typename OutT = __nv_bfloat16>
void gemm_dispatch(GemmParams p, cudaStream_t stream, bool trans_a,
                   bool trans_b) {
    constexpr bool kSymmetric = std::is_same_v<ElemA, ElemB>;
    bool swapped = false;
    if constexpr (kSymmetric) {
        swapped = !trans_a && !trans_b;  // canonicalize rewrites NN
        canonicalize_gemm(p, trans_a, trans_b);
    }
    // Crosswise operand count for the plan: transposed-A storage
    // (ColMajor) and plain-B storage (RowMajor) both take the direct
    // crosswise load.
    const int crosswise = (trans_a ? 1 : 0) + (trans_b ? 0 : 1);
    const GemmPlan plan = plan_gemm(p, (int)sizeof(ElemA), (int)sizeof(ElemB),
                                    gemm_perf_class<ElemA, ElemB>(),
                                    crosswise);
    if (trans_a && trans_b) {
        // The swap computes the transposed problem; its (rewritten TT)
        // branch instantiates the column-major-output epilogue through
        // LayoutOut. Mixed never swaps, so its output stays row-major.
        if constexpr (kSymmetric) {
            if (swapped)
                launch_plan<ElemA, ElemB, ColMajor, ColMajor, ColMajor, OutT>(p, plan, stream);
            else
                launch_plan<ElemA, ElemB, ColMajor, ColMajor, RowMajor, OutT>(p, plan, stream);
        } else {
            launch_plan<ElemA, ElemB, ColMajor, ColMajor, RowMajor, OutT>(p, plan, stream);
        }
    } else if (trans_b) {
        // NT (the fused-linear shape).
        launch_plan<ElemA, ElemB, RowMajor, ColMajor, RowMajor, OutT>(p, plan, stream);
    } else if (trans_a) {
        launch_plan<ElemA, ElemB, ColMajor, RowMajor, RowMajor, OutT>(p, plan, stream);
    } else {
        // Dual row-major: mixed only — symmetric NN was rewritten above
        // into the transposed TT kernel (if constexpr keeps this
        // instantiation out of symmetric builds).
        if constexpr (!kSymmetric)
            launch_plan<ElemA, ElemB, RowMajor, RowMajor, RowMajor, OutT>(p, plan, stream);
    }
}

}  // namespace gemm
}  // namespace astrai
