#pragma once
// The host planning half of the GEMM dispatch: the planner chain, the
// recipe vocabulary it decides on, and the runtime knobs (config state and
// row tables) it reads. Split out of kernel/gemm.cuh so the per-dtype
// kernel TUs compile the device stack and reference the planner through the
// declarations in policy.cuh — plan_table.h's 620 lines stop being dragged
// through nvcc once per dtype pair.
//
// SINGLE-INCLUSION: plan_dispatch is defined NON-inline here, so exactly ONE
// TU per binary includes this header (gemm.cu, or a standalone harness) — a
// second includer is a multiple-definition link error, which is the
// enforcement. The planner is GPU-free by design and never launches.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <tuple>
#include <vector>

#include <launcher/plan_table.h>
#include <policy.cuh>

namespace astrai {
namespace gemm {

// Raster order: walk the dimension with more tiles fastest (CUTLASS's
// rule; the N-side mirrored group keeps the measured width 8). M-side group
// width is humming's L2-budget rule: keep a group's A tiles L2-resident
// while B streams (B already resident => g=1 plain raster); otherwise
// reserve a B-streaming fraction of L2, cap at what fits, floor at enough
// M rows to keep every SM busy per sweep.
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

// ---------------------------------------------------------------------------
// Recipe vocabulary: GemmRecipe is the runtime form of a manifest tile; row
// tables, the model, the launchers and the binding all name tiles through it.
// ---------------------------------------------------------------------------

// One manifest tile -> its recipe row, priced at the caller's operand
// widths (smem is pair-specific); the vector and scan forms share this and
// must never disagree.
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

// Deduped on (class, stages, kK): dispatch_tile takes the first manifest
// match, so the 16-warp small CTA behind its 32-warp twin is one candidate.
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

// manifest_kind -> THE one manifest type list that ladder instantiates;
// the vocabulary builder and the scan oracle both dispatch through it.
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

// The instantiation oracle: does this ladder carry a tile for (class,
// stages, kK)? The ONE gate a row's fields must pass — a row naming a
// non-instantiable combination matches no tile and launches nothing.
// Scans the manifest list directly (no vector): the vector build cost
// ~2.5us of the ~2.9us a dispatch took, the scan ~100ns (measured).
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

// The [gemm-plan] decision line (gen_plan_table's tag regex reads it).
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

// Rows to a decision: band + wave gates, then the instantiation oracle,
// then the smem ceiling — a stale row falls through to the next planner,
// never fails a launch. Raster 0 resolves through plan_raster at the
// recipe's geometry (a bare 0 is PLAIN raster, ~14% off on M<<N).
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

// The analytical planner — a port of DeepGEMM's get_best_configs with the
// one term DeepGEMM leaves out restored.
//
// DeepGEMM ranks by wave COUNT; valid only when every wave costs the same,
// which fails here — a 64x64 wave carries a quarter of a 128x128's work,
// so wave count prefers the coarse tile. Measured over the ten production
// cells (bench_tile_sweep.cu, 7 shapes, sm_89): fewest-wave is WORST on
// narrow-N — at 512x1536 one-wave cells measure 49.4-51.6 TFLOPS vs 63.9
// for two-wave (48 blocks over 92 SMs x 2 resident = a 26%-full machine);
// wave count ranks rho -0.90 against measurement there.
//
// So the model ranks by the resource the ring buys — resident CTAs per SM,
// prefetch depth — and on ties the tile's own traffic (cost_of). Two
// buried claims that did not survive the saved sweeps: (a) the 1.13-1.38x
// within-shape cell spread holds only from m >= 256 (at m <= 64 it reaches
// 143-203%, the band the model decides worst); (b) "L1/L2 terms change no
// decision" is false within a cell (per-block bytes and wave fill rank
// rho -0.255/+0.282 vs the resource keys' +0.134/+0.169) though true
// across cells.
//
// The per-CTA efficiency separating classes at a given (M, N) is the axis
// no formula reaches — the measured-not-modeled axis the row tables own,
// why the hybrid chain keeps rows first. kK falls out of the residency
// rule: the kK=32 twin's 48KB ring holds two CTAs where kK=64's 96KB
// holds one.
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
    // 2026-09-16 RTX 5090 grid fits — re-fit per box. kKTileIssueBytes
    // prices the per-k-tile overhead (barrier, mma issue, load scheduling)
    // as output-cell-bytes per k-iteration — the term that makes kK a
    // model axis; byte pairs take none (all kK=64, degenerates to a
    // re-weight). kMmaArmBytesPerInstr prices the tensor-pipe arm (one mma
    // per 16x8xkMmaK cell; mma.cuh's 256-bit A-fragment invariant).
    static constexpr std::int64_t kKTileIssueBytes = 8;
    static constexpr std::int64_t kMmaArmBytesPerInstr = 64;

    static int resident_of(const GemmRecipe& r, const PlanQuery& q) {
        if (q.dev.smem_per_sm <= 0 || q.dev.regs_per_sm <= 0) return 0;
        return std::min(q.dev.smem_per_sm / r.smem,
                        min_ctas_for_ring(r.smem));
    }

    // cost = max(memory bytes, mma arm) * W_eff per CTA on TMA: the arms
    // overlap on independent hardware, so max — a non-binding arm must not
    // tax the ranking. W_eff's resident-scaled waves apply to two-byte
    // pairs only (byte/mixed rings are half-size, winners at resident=2;
    // resident-blind measured ahead on all three grids, 2026-09-16).
    // cp.async prices differently: the software ring is the only latency
    // hiding, so residency DIVIDES the makespan instead of sharing
    // bandwidth — the axis flips with staging.
    static std::int64_t cost_of(const GemmRecipe& r, const PlanQuery& q,
                                int resident) {
        const std::int64_t blocks =
            q.batch * ((std::int64_t)((q.m + r.bm - 1) / r.bm) *
                       (std::int64_t)((q.n + r.bn - 1) / r.bn));
        if (!q.tma) {
            // cp.async (zero-constant L20 form, 2026-09-16): raw-floor
            // residency in the denominator, k-tail priced whole. Measures
            // 0.9552 vs the TMA form's 0.8445 on the cp.async grid, the
            // reverse on every TMA grid.
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

// The chain: rank order, first to answer wins; the mode picks the chain —
// "table" (rows then degraded), "hybrid" (+model between), "model" alone.
// Every chain ends in the degraded rows (open bands, m=0 fallback), so
// dispatch is total. Non-inline: the static chain members are one set per
// process.
PlanDecision plan_dispatch(const PlanQuery& q) {
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

}  // namespace gemm
}  // namespace astrai
