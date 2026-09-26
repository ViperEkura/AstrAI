#pragma once
// The host planning half of the GEMM dispatch: the planner chain, the
// recipe vocabulary it decides on, and the runtime knobs (config state and
// row tables) it reads. Split out of kernel/gemm.cuh so the per-dtype
// kernel TUs compile the device stack and reference the planner through the
// declarations in policy.cuh — plan_table.h's 750 lines stop being dragged
// through nvcc once per dtype pair.
//
// SINGLE-INCLUSION DISCIPLINE: plan_dispatch (the planner entry declared in
// policy.cuh) is defined here NON-inline, so this header is included by
// exactly ONE translation unit per binary — the module's gemm.cu, or the
// standalone harness itself. A second includer is a multiple-definition
// link error, which is the enforcement. Everything else here is inline and
// safe to reach; the three runtime knobs stay inline in plan_table.h.
//
// The planner is GPU-free by design: it answers for a problem, not a device
// state, and never launches (the probe and the model harness build on that).

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

// ---------------------------------------------------------------------------
// Recipe vocabulary: ONE spelling of a launchable tile configuration. The
// manifest types are the compiled truth, GemmRecipe is their runtime form,
// and every consumer — row tables, the analytical model, the launchers,
// the tile_vocabulary binding — names tiles through it.
// ---------------------------------------------------------------------------

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
// dispatch is a total function. Non-inline (this header's one compiled
// definition), so the static chain members are one set per process.
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
