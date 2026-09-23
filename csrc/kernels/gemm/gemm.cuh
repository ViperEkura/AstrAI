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
#include <functional>
#include <optional>
#include <tuple>
#include <utility>
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
// Launchers — pure CUDA (no torch), usable from the binding and pure C tests.
// The runtime knobs (plan log, planner rank, staging A/B switches, table
// mode) live in plan_table.h's GemmConfig, seeded once from the deprecated
// ASTR_GEMM_* variables and owned at runtime by astrai.extension.plan.
// ---------------------------------------------------------------------------

// Grid for one Policy's tile: N x M block count, batch on z.
template <typename Traits>
dim3 gemm_grid(const GemmParams& p) {
    return dim3((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
                (p.m + Traits::kBlockM - 1) / Traits::kBlockM, p.batch);
}

// One read-only plan-log line per launch (plan.set_log; " mx" marks the
// block_scale cell).
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
inline int plan_raster(const PlanQuery& q, int bm, int bn) {
    const int64_t m_tiles = (q.m + bm - 1) / bm;
    const int64_t n_tiles = (q.n + bn - 1) / bn;
    if (m_tiles < n_tiles) return -8;
    if (q.n * q.k * (int64_t)q.bb <= q.dev.l2_bytes * 7 / 10) return 1;
    const double reserve = 0.12 + 0.28 * (double)q.bb / (double)q.ba;
    const double budget = (1.0 - std::min(reserve, 0.5)) * (double)q.dev.l2_bytes;
    const int64_t ub = (int64_t)(budget / ((double)bm * (double)q.k * q.ba));
    const int64_t lb = (q.dev.sms + n_tiles - 1) / n_tiles;
    int64_t g = std::min(ub, m_tiles);
    if (ub >= lb) g = std::min(std::max(g, lb), m_tiles);
    return (int)std::max(g, (int64_t)1);
}

// Dtype-class ids the plan-table rows key on.
enum class GemmPerfClass : int { kW16A16 = 0, kW8A16, kW8A8, kF8A8 };


// Compile-time dtype-class derivation from the operand pair (the mma
// promotion rule plus operand widths; mixed bf16xfp8 lands with the 2B x
// 1B class — same bytes and same promoted bf16 k16 mma as W8A16).
//
// The int8 pair is tested first: it promotes to an int8 mma, not to bf16, so
// the "not bf16 -> fp8 pair" arm below would otherwise swallow it and every
// W8A8 row would be unreachable (int8 dispatched on the F8A8 rows, and the
// plan log showed an int8 problem resolving to a class-3 row).
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



// ---------------------------------------------------------------------------
// Recipe vocabulary: ONE spelling of a launchable tile configuration. The
// manifest types are the compiled truth, GemmRecipe is their runtime form,
// and every consumer — row tables, the analytical model, the launchers,
// the tile_vocabulary binding — names tiles through it.
// ---------------------------------------------------------------------------

struct GemmRecipe {
    int cta;      // TileClass ordinal — the row-file serialization key
    int stages;   // ring depth
    int kk;       // k-tile depth (the kK twins are separate recipes)
    int bm, bn;   // CTA geometry
    int wm, wn;   // warp tiling (the recipe name's W<x>x<y>)
    int threads;  // the manifest entry's warp tiling (first match wins)
    int smem;     // ring bytes at this staging pair's operand widths
};

// One manifest tile -> its recipe row, priced against the operand widths
// the caller asks about (smem is pair-specific). append_recipe and
// recipe_scan below share this — the vector form and the scan form must
// never disagree about a tile's geometry.
template <typename Tile>
inline GemmRecipe recipe_for_tile(int ba, int bb) {
    return GemmRecipe{
        (int)tile_class<Tile>(), Tile::kStages, (int)Tile::CtaShape::kK,
        Tile::CtaShape::kM, Tile::CtaShape::kN,
        Tile::WarpShape::kM, Tile::WarpShape::kN,
        (Tile::CtaShape::kM / Tile::WarpShape::kM) *
            (Tile::CtaShape::kN / Tile::WarpShape::kN) * 32,
        ring_smem_bytes(Tile::CtaShape::kM, Tile::CtaShape::kN,
                        Tile::CtaShape::kK, Tile::kStages, ba, bb)};
}

// Deduped on the dispatch key: two tiles sharing (class, stages, kK) — the
// 16-warp small CTA behind its 32-warp twin — are one candidate, because
// dispatch_tile takes the first manifest match.
template <typename Tile>
inline void append_recipe(std::vector<GemmRecipe>& out, int ba, int bb) {
    const GemmRecipe r = recipe_for_tile<Tile>(ba, bb);
    for (const GemmRecipe& have : out)
        if (have.cta == r.cta && have.stages == r.stages && have.kk == r.kk)
            return;
    out.push_back(r);
}

template <typename Manifest>
inline void collect_recipes(std::vector<GemmRecipe>& out, int ba, int bb) {
    std::apply(
        [&out, ba, bb](auto... tiles) {
            (append_recipe<decltype(tiles)>(out, ba, bb), ...);
        },
        Manifest{});
}

// manifest_kind -> THE one manifest type list that ladder instantiates —
// the runtime half of manifest_for's rule, spelled once: the vocabulary
// builder and the scan oracle below dispatch through it, so they can never
// disagree about which ladder serves a staging pair.
template <typename F>
inline auto with_manifest(bool crosswise_staging, int ba, int bb, F&& fn) {
    switch (manifest_kind(crosswise_staging, ba, bb)) {
        case ManifestKind::kTwoByte:
        case ManifestKind::kMixed:  // the congruous ladder carries the
                                    // mixed bus on the predicated skip
            return fn(TileManifest{});
        case ManifestKind::kByte:
            return fn(TileManifestByte{});
        default:  // kCrosswise is the fallback kind, manifest_for included
            return fn(TileManifestCross{});
    }
}

// Every recipe the ladders instantiate for one staging pair.
inline std::vector<GemmRecipe> gemm_recipes_for(bool crosswise_staging,
                                                int ba, int bb) {
    std::vector<GemmRecipe> out;
    with_manifest(crosswise_staging, ba, bb, [&](auto manifest) {
        collect_recipes<decltype(manifest)>(out, ba, bb);
    });
    return out;
}

// The instantiation oracle: does this staging pair's ladder carry a tile
// for (class, stages, kK)? This is the ONE gate a row's recipe fields must
// pass — the wide CTA's 1-byte-only rule, ring depths past s3, the
// crosswise ladder's conservative set — because a row naming a
// non-instantiable combination would match no tile in dispatch_tile and
// launch nothing at all.
//
// Scanning the manifest type list directly (no std::vector): this runs once
// per plan, and building the vocabulary's vector here cost ~2.5us of the
// ~2.9us a dispatch took — the row scan itself is ~100ns (measured; the
// vector form stays for tile_vocabulary, which wants a list).
template <typename Manifest>
inline bool recipe_scan(int cta, int stages, int kk, int ba, int bb,
                        GemmRecipe& out) {
    bool found = false;
    auto consider = [&](auto tile) {
        using T = decltype(tile);
        if (found) return;
        if ((int)tile_class<T>() != cta || (int)T::kStages != stages ||
            (int)T::CtaShape::kK != kk)
            return;
        out = recipe_for_tile<T>(ba, bb);
        found = true;
    };
    std::apply([&](auto... tiles) { (consider(tiles), ...); }, Manifest{});
    return found;
}

inline std::optional<GemmRecipe> recipe_of(int cta, int stages, int kk,
                                           bool crosswise, int ba, int bb) {
    GemmRecipe out{};
    const bool found = with_manifest(crosswise, ba, bb, [&](auto manifest) {
        return recipe_scan<decltype(manifest)>(cta, stages, kk, ba, bb, out);
    });
    if (!found) return std::nullopt;
    return out;
}

// One dispatch decision: the recipe, the resolved raster (a row's literal,
// or plan_raster at the recipe's geometry), and the planner that made it.
// source is that planner's own name — the log line and the probe dict
// report it verbatim.
struct PlanDecision {
    GemmRecipe recipe;
    int raster;
    const char* source;
};

// The [gemm-plan] decision line (plan.set_log gates it; gen_plan_table's
// tag regex reads it).
inline void log_dispatch(const PlanQuery& q, const PlanDecision& d) {
    if (!gemm_plan_log_enabled()) return;
    std::fprintf(stderr,
                 "[gemm-plan] %s m%lld n%lld k%lld b=%d -> cta%d s%d "
                 "raster %d\n",
                 d.source, (long long)q.m, (long long)q.n, (long long)q.k,
                 (int)q.batch, d.recipe.cta, d.recipe.stages, d.raster);
}

// ---------------------------------------------------------------------------
// Planners: one plan strategy per dispatch source, composed into a chain
// (first planner to answer wins). A new source is one class and one chain
// entry; the mode knob (GemmConfig::planner) only picks which chain.
// ---------------------------------------------------------------------------
struct GemmPlanner {
    virtual ~GemmPlanner() = default;
    virtual const char* name() const = 0;
    virtual std::optional<PlanDecision> plan(const PlanQuery& q) const = 0;
};

// Rows to a decision: match (band + wave gates, all inside plan_row_for),
// then the instantiation oracle, then the smem ceiling — a stale tuning
// row falls through to the next planner instead of failing a launch.
// Raster 0 on the row resolves through plan_raster at the recipe's
// geometry (a bare 0 would be GemmParams' PLAIN raster, which costs ~14%
// on the M<<N shapes).
class RowSetPlanner final : public GemmPlanner {
public:
    using RowFn =
        std::function<std::optional<TableRow>(const PlanQuery&)>;
    RowSetPlanner(const char* source, RowFn rows, bool respects_table_off)
        : source_(source), rows_(std::move(rows)),
          respects_table_off_(respects_table_off) {}
    const char* name() const override { return source_; }
    std::optional<PlanDecision> plan(const PlanQuery& q) const override {
        if (respects_table_off_ && gemm_table_off()) return std::nullopt;
        const std::optional<TableRow> row = rows_(q);
        if (!row) return std::nullopt;
        const std::optional<GemmRecipe> recipe =
            recipe_of((int)row->cta, row->stages, row->kk, q.crosswise > 0,
                      q.ba, q.bb);
        if (!recipe || recipe->smem > q.dev.smem_max) return std::nullopt;
        return PlanDecision{
            *recipe,
            row->raster != 0 ? row->raster
                             : plan_raster(q, recipe->bm, recipe->bn),
            source_};
    }

private:
    const char* source_;
    RowFn rows_;
    bool respects_table_off_;
};

// The analytical planner — a port of DeepGEMM's config search
// (get_best_configs), with the one term DeepGEMM can leave out restored.
//
// DeepGEMM ranks candidates by wave COUNT. That is only a valid proxy when
// every candidate's wave costs the same, which holds for it because its
// blocks are pinned to instruction shapes and its tiles fill an SM; here a
// 64x64 CTA's wave carries a quarter of a 128x128's work, so counting
// waves systematically prefers the coarse tile. Measured over the ten
// production cells (csrc/bench/bench_tile_sweep.cu, 7 shapes, sm_89): the
// fewest-wave cell is the WORST on the narrow-N shapes — at 512x1536 the
// one-wave cells measure 49.4-51.6 TFLOPS against 63.9 for the two-wave
// ones, because 48 blocks over 92 SMs x 2 resident is a 26%-full machine,
// not an efficient one. Wave count ranks those shapes at rho -0.90 against
// the measurement.
//
// So the model does not rank by wave COUNT, and it does not price waves
// either: written with a fractional last wave the makespan is
// (blocks/slots) * concurrency * solo, `solo` grows with bm*bn while
// `blocks` shrinks with it, and the product is the same MN for every
// candidate of a problem. What is left to choose between is the resource
// the ring buys — how many CTAs it keeps resident per SM, how deep it can
// prefetch — and, where two candidates tie on both, the tile's own traffic
// (cost_of below).
//
// Two earlier claims in this comment did not survive measurement on the
// saved sweeps and are recorded here so they are not re-derived: (a) "every
// production cell lands within 1.13-1.38x of every other on a given shape"
// holds only from m >= 256 (26-65% spread) — at m <= 64 the within-cell
// spread reaches 143-203%, and that is exactly the band the model decides
// worst; (b) "the L1/L2 terms change no decision at all" is false within a
// cell, where the per-block byte count and the wave fill are the two
// strongest correlates of the measured time (rank rho -0.255 and +0.282
// against the resource keys' +0.134/+0.169). Across cells they do not
// order anything, which is what the original claim was about.
//
// The per-CTA efficiency that separates the classes at a given (M, N) is
// the one axis no such formula reaches; it is the measured-not-modeled
// axis the row tables own, and the reason the hybrid chain keeps rows
// first.
//
// kK is not a model axis (DeepGEMM fixes block_k) and falls out of the
// same residency rule: the kK=32 twin's 48KB ring holds two CTAs where
// kK=64's 96KB holds one.
class ModelPlanner final : public GemmPlanner {
public:
    const char* name() const override { return "model"; }

    std::optional<PlanDecision> plan(const PlanQuery& q) const override {
        if (q.dev.sms <= 0 || q.m <= 0 || q.n <= 0 || q.k <= 0)
            return std::nullopt;
        const std::vector<GemmRecipe> recipes =
            gemm_recipes_for(q.crosswise > 0, q.ba, q.bb);
        const GemmRecipe* best = nullptr;
        std::int64_t best_cost = 0;
        for (const GemmRecipe& r : recipes) {
            const int resident = resident_of(r, q);
            if (resident <= 0) continue;  // ring cannot be resident
            const std::int64_t cost = cost_of(r, q, resident);
            // every width pair ranks on the cost alone; a tie keeps the
            // candidate seen first, the manifest's own order
            if (!best || cost < best_cost) {
                best = &r;
                best_cost = cost;
            }
        }
        if (!best) return std::nullopt;
        return PlanDecision{*best, plan_raster(q, best->bm, best->bn),
                            name()};
    }

private:
    // Both constants are 2026-09-16 RTX 5090 grid fits — re-fit per box.
    // kKTileIssueBytes: per-k-tile overhead (ring-refill barrier, mma
    // issue, load scheduling) priced as output-cell-bytes per k-iteration
    // — the term that makes kK a model axis. Byte pairs take none: their
    // ladder is all kK=64, the term degenerates to an output re-weight.
    static constexpr std::int64_t kKTileIssueBytes = 8;
    // kMmaArmBytesPerInstr: the tensor-pipe arm — one mma instruction per
    // 16x8xkMmaK cell (kMmaK 32 on byte pairs, 16 otherwise; mma.cuh's
    // 256-bit A-fragment invariant), priced at this many bytes per
    // instruction.
    static constexpr std::int64_t kMmaArmBytesPerInstr = 64;

    static int resident_of(const GemmRecipe& r, const PlanQuery& q) {
        if (q.dev.smem_per_sm <= 0 || q.dev.regs_per_sm <= 0) return 0;
        return std::min(q.dev.smem_per_sm / r.smem,
                        min_ctas_for_ring(r.smem));
    }

    // cost = max(memory-side bytes, mma arm) * W_eff, per CTA, on the TMA
    // staging. Loads and mma run on independent hardware and overlap, so
    // the arms take max — a non-binding arm must not tax the ranking.
    // W_eff keeps the coarse resident-scaled waves on two-byte pairs only:
    // a byte/mixed ring is half-size for the same tile, those winners sit
    // at resident=2, and the resident-scaled phantom tail misprices them
    // (resident-blind measured ahead on all three of those grids,
    // 2026-09-16). cp.async staging prices differently (below): the
    // software ring is the ONLY latency hiding there, so residency divides
    // the makespan instead of sharing bandwidth — the axis flips with
    // staging, and each form loses badly on the other's grids.
    static std::int64_t cost_of(const GemmRecipe& r, const PlanQuery& q,
                                int resident) {
        const std::int64_t blocks =
            q.batch * ((std::int64_t)((q.m + r.bm - 1) / r.bm) *
                       (std::int64_t)((q.n + r.bn - 1) / r.bn));
        if (!q.tma) {
            // cp.async (the zero-constant L20 form, 2026-09-16): raw-floor
            // residency in the wave denominator, the k-tail priced whole,
            // no fitted constants. On the cp.async grid this measures
            // 0.9552 against the TMA form's 0.8445 (and the reverse on
            // every TMA grid).
            const std::int64_t operand =
                ((q.k + r.kk - 1) / r.kk) * (std::int64_t)r.kk *
                (r.bm * q.ba + r.bn * q.bb);
            const std::int64_t mu = q.dev.smem_per_sm / r.smem;
            const std::int64_t slots = (std::int64_t)q.dev.sms * mu;
            const std::int64_t waves =
                slots > 0 ? (blocks + slots - 1) / slots : 1;
            return (operand +
                    (std::int64_t)q.out_elem_bytes * r.bm * r.bn) * waves;
        }
        const bool byte_pair = q.ba == 1 && q.bb == 1;
        const std::int64_t operand =
            (std::int64_t)q.k * (r.bm * q.ba + r.bn * q.bb);
        const std::int64_t output =
            (std::int64_t)q.out_elem_bytes * r.bm * r.bn;
        const std::int64_t issue = byte_pair
            ? 0
            : kKTileIssueBytes * (std::int64_t)r.bm * r.bn *
                  ((q.k + r.kk - 1) / r.kk);
        const std::int64_t mma_arm = kMmaArmBytesPerInstr *
            (std::int64_t)r.bm * r.bn * q.k / (128 * (byte_pair ? 32 : 16));
        const std::int64_t per_cta =
            std::max(operand + output + issue, mma_arm);
        const std::int64_t slots = (std::int64_t)q.dev.sms * resident;
        const std::int64_t waves =
            slots > 0 ? (blocks + slots - 1) / slots : 1;
        const std::int64_t w_eff = q.ba == 2 && q.bb == 2
            ? waves * resident
            : (blocks + q.dev.sms - 1) / q.dev.sms;
        return per_cta * w_eff;
    }
};

// The chain: planners in rank order, first to answer wins. Composition is
// the planner mode — "table" runs the row tiers then the degraded tail,
// "hybrid" inserts the model between them, "model" trusts the model
// alone. Every chain ends in the degraded rows (their bands are open on N
// with -1 keys, and the m=0 degenerate case falls to the first row), so
// dispatch is a total function.
inline PlanDecision plan_dispatch(const PlanQuery& q) {
    static const RowSetPlanner override_planner(
        "override",
        [](const PlanQuery& query) {
            return plan_table_override_source().lookup(query);
        },
        /*respects_table_off=*/true);
    static const RowSetPlanner injected_planner(
        "injected",
        [](const PlanQuery& query) {
            return plan_table_injected_source().lookup(query);
        },
        /*respects_table_off=*/true);
    static const RowSetPlanner builtin_planner(
        "builtin",
        [](const PlanQuery& query) {
            // Compiled-in rows are calibrated to the part they were
            // measured on (see the GENERATED block); anywhere else the
            // tier is inert and the model answers instead.
            if (!builtin_rows_match_device(query.dev))
                return std::optional<TableRow>{};
            int count = 0;
            const TableRow* rows =
                builtin_plan_table(query.perf_class, count);
            if (rows == nullptr) return std::optional<TableRow>{};
            const TableRow* row = plan_row_for(rows, count, query);
            if (row == nullptr) return std::optional<TableRow>{};
            return std::optional<TableRow>{*row};
        },
        /*respects_table_off=*/true);
    static const RowSetPlanner degraded_planner(
        "degraded",
        [](const PlanQuery& query) {
            // The M band alone decides: the degraded rows are open on N
            // and K with -1 keys and carry no gate. The n of 1 stands for
            // "some real n" — the matcher's band test is "strictly past
            // the min", and an n of 0 sits ON the open bound.
            PlanQuery m_only;
            m_only.m = query.m;
            m_only.n = 1;
            const TableRow* row =
                plan_row_for(kDegradedPlanRows, 3, m_only);
            if (row == nullptr)
                return std::optional<TableRow>{kDegradedPlanRows[0]};
            return std::optional<TableRow>{*row};
        },
        /*respects_table_off=*/false);
    static const ModelPlanner model_planner;

    static constexpr const GemmPlanner* kChainTable[] = {
        &override_planner, &injected_planner, &builtin_planner,
        &degraded_planner};
    static constexpr const GemmPlanner* kChainHybrid[] = {
        &override_planner, &injected_planner, &builtin_planner,
        &model_planner, &degraded_planner};
    static constexpr const GemmPlanner* kChainModel[] = {&model_planner,
                                                         &degraded_planner};

    const int mode = gemm_planner_mode();
    const GemmPlanner* const* chain =
        mode == 2 ? kChainModel : mode == 1 ? kChainHybrid : kChainTable;
    const int chain_len = mode == 2 ? 2 : mode == 1 ? 5 : 4;
    for (int i = 0; i < chain_len; ++i)
        if (std::optional<PlanDecision> d = chain[i]->plan(q)) {
            log_dispatch(q, *d);
            return *d;
        }
    // The chain above returns for every query that has device facts
    // (the degraded row function has the m=0 fallback and resolves for
    // any width pair). It cannot answer only when the query carries no
    // usable device (smem_max 0 fails the ring gate) or a non-positive
    // dim, so this tail is the no-facts answer: the first degraded row,
    // the historical choice for degenerate shapes.
    const TableRow& row = kDegradedPlanRows[0];
    PlanDecision d{*recipe_of((int)row.cta, row.stages, row.kk, false, 2, 2),
                   0, "degraded"};
    log_dispatch(q, d);
    return d;
}

// The one place a GemmParams becomes planner input, and the one place the
// dispatch key is derived: perf class, operand widths and the crosswise count
// are all functions of the typed call, so they are computed rather than passed
// in and a caller cannot hand the planner a key that contradicts its own types
// and layouts. OutT (default bf16, the fused-linear convention) prices the
// model cost's output term at its real element size.
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

// The typed dispatch entry: problem in, decision out. Taking the layout
// tags as types is what ties the decision to the launch that follows it —
// every derived field comes from the same tags the launcher instantiates
// with.
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
    CUtensorMap ma{}, mb{};
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

// Manifest dispatch (CUTLASS builder-table style): the plan's (CTA class,
// ring depth, k-tile depth) selects exactly one manifest entry — the ||
// short-circuits — and the resolver maps its tile onto a concrete Policy and
// launches.
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

// Narrow twin of a big tile for the output-reclaim fallback, carrying the
// same ring depth as the tile it replaces (the planner priced that ring, so a
// deeper substitute could overflow the smem opt-in). There is no kK=32 narrow
// s3, so a deep kK=32 big tile falls back to its s2 twin.
template <typename Tile>
using narrow_fallback_t = std::conditional_t<
    Tile::CtaShape::kK == 32, Tile_128x64x32_W32x32_S2,
    std::conditional_t<(Tile::kStages >= 3), Tile_128x64x64_W32x32_S3,
                       Tile_128x64x64_W32x32_S2>>;

// The full reclaim chain a dispatch instantiation must terminate in: a tile
// whose output outgrows its ring falls to its narrow twin, and the kK=32
// narrow twin (18KB ring) still cannot hold a 4B/elem output — that ends at
// the small CTA, whose 24KB ring reclaims every output the dispatch
// instantiates (<= 4B/elem). Identity whenever the tile itself fits, so the
// launchers can apply it unconditionally; the dispatch walks name every
// manifest tile as a potential substitute, which makes the chain load-bearing
// even for geometries production never plans to.
template <typename Tile, typename ElemA, typename ElemB, typename OutT>
using reclaim_fallback_t = std::conditional_t<
    reclaim_fits<Tile, ElemA, ElemB, OutT>(), Tile,
    std::conditional_t<
        reclaim_fits<narrow_fallback_t<Tile>, ElemA, ElemB, OutT>(),
        narrow_fallback_t<Tile>, Tile_64x64x64_W16x32_S2>>;

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

// cp.async ladder resolver. One substitution, on the CTA classes whose
// output tile can outgrow their own rings: a fat output (fp32, 4B) that
// cannot reclaim the ring routes to the narrow CTA — same math at lower
// reuse. (A second substitution once downgraded the big CTA to a
// predicated-loop twin on crosswise staging; interleaved A/B 2026-09-16
// measured that twin 9-19% SLOWER on this part and the planner never
// routed big on the crosswise ladder anyway — see policy.cuh's
// GemmTileConfig note for the burial record.) Narrow and small entries
// pass through.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut, typename OutT, bool UseMx = false>
struct CpAsyncLauncher {
    GemmParams p;
    cudaStream_t stream;
    template <typename Tile>
    bool run() const {
        // The small CTA's warp widening: the 8-warp 64x64 twin starves the
        // tensor pipe on two-byte operands (measured 1.36-1.66x for the
        // 16-warp form, parity to -3% on the thinnest shapes). The rule and
        // its arms live in policy.cuh's warp_widened_t. The reclaim chain
        // rides unconditionally (identity whenever the tile itself fits —
        // same reasoning as TmaLauncher's).
        using Widened = warp_widened_t<ElemA, ElemB, Tile>;
        using TileT = reclaim_fallback_t<Widened, ElemA, ElemB, OutT>;
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

// The planner entry: symmetric fp8 rides the sm_120 block_scale cell unless
// ASTR_GEMM_NO_MX knocks it out.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB,
          typename LayoutOut = RowMajor, typename OutT = __nv_bfloat16>
void launch_plan(GemmParams p, const PlanDecision& d, cudaStream_t stream) {
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
        if (!gemm_mx_cell_disabled() && astrai::device_facts().cc == 120) {
            launch_plan_impl<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT,
                             true>(p, d, stream);
            return;
        }
    }
    launch_plan_impl<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT>(
        p, d, stream);
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

// The (trans_a, trans_b) -> layout-tag ladder: ONE home for the branch both
// the launch below and the planner probe walk, so a probe cannot answer for
// a layout the launch then does not take. fn gets the three tags (operand A,
// operand B, output); the probe ignores the output one, which never reaches
// the planner. `swapped` marks the symmetric-NN rewrite's own TT — the only
// arm whose OUTPUT tag is transposed, the rewrite having computed the
// transposed problem into the caller's [M][N] buffer.
//
// A visitor rather than a tag-returning function because a tag's identity is
// its TYPE: the value cannot vary per arm, the instantiations can. Every arm
// calls fn, which keeps this total for a caller that returns fn's value.
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

// The explicit-instantiation spelling shared by the per-pair TUs (bare, the
// definition form) and gemm.cu's declaration block (`extern`-prefixed): one
// place names the signature, so the stub units and the extern table cannot
// drift apart.
#define ASTRAI_GEMM_INSTANTIATE(W, A) \
    template void gemm_dispatch<W, A>(GemmParams, cudaStream_t, bool, bool)

// Host-only planner probe (the Python autotuner's coverage check): the
// decision gemm_dispatch would make for this problem, without a launch —
// the planner is GPU-free by design. The tag selection is the shared ladder
// (with_layout_tags), symmetric-NN rewrite included, so a probe cannot
// disagree with the branch the real call takes; the output tag the ladder
// hands over is ignored here, LayoutOut never reaching the planner.
// The probe returns the decision plus the query it answered (the binding
// reports perf_class/crosswise from the query, source/recipe from the
// decision).
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
