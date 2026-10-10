/* GEMM host planning, config, and recipe vocabulary. */
#include <launcher/gemm_cost.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "plan_table.h"

#include <api/gemm.h>
#include <policy/manifest.cuh>

namespace astrai {
namespace gemm {

int plan_raster(const PlanQuery& q, int bm, int bn) { return geometry_raster(q, bm, bn); }

// Build the same recipe for introspection and model scans.
namespace {

template <typename Tile> inline GemmRecipe recipe_for_tile(int ba, int bb) {
    return GemmRecipe{(int)tile_class<Tile>(),
                      Tile::kStages,
                      (int)Tile::kTile,
                      Tile::CtaShape::kM,
                      Tile::CtaShape::kN,
                      Tile::WarpShape::kM,
                      Tile::WarpShape::kN,
                      (Tile::CtaShape::kM / Tile::WarpShape::kM) *
                          (Tile::CtaShape::kN / Tile::WarpShape::kN) * 32,
                      ring_smem_bytes(Tile::CtaShape::kM, Tile::CtaShape::kN, Tile::kTile,
                                      Tile::kStages, ba, bb)};
}

template <typename Tile> constexpr int recipe_key() {
    return (int)tile_class<Tile>() | (Tile::kStages << 8) | ((int)Tile::kTile << 16);
}

template <typename Manifest> constexpr bool unique_recipe_keys() {
    std::array<int, std::tuple_size_v<Manifest>> keys{};
    std::size_t count = 0;
    std::apply([&](auto... tiles) { ((keys[count++] = recipe_key<decltype(tiles)>()), ...); },
               Manifest{});
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = i + 1; j < count; ++j)
            if (keys[i] == keys[j])
                return false;
    return true;
}

template <typename Manifest, typename F> inline void for_each_recipe(int ba, int bb, F&& fn) {
    static_assert(unique_recipe_keys<Manifest>(), "manifest dispatch keys must be unique");
    std::apply([&](auto... tiles) { (fn(recipe_for_tile<decltype(tiles)>(ba, bb)), ...); },
               Manifest{});
}

// Match the manifests compiled by the launch ladder.
template <typename F> inline auto with_manifest(bool crosswise_staging, int ba, int bb, F&& fn) {
    switch (manifest_kind(crosswise_staging, ba, bb)) {
    case ManifestKind::kTwoByte:
    case ManifestKind::kMixed:
        return fn(TileManifest{});
    case ManifestKind::kByte:
        return fn(TileManifestByte{});
    default:
        return fn(TileManifestCross{});
    }
}

} // namespace

// A row is eligible only if its exact (class, k_stages, K) recipe was instantiated.
namespace {

template <typename Manifest>
inline bool recipe_scan(int cta, int k_stages, int k_tile, int ba, int bb, GemmRecipe& out) {
    bool found = false;
    auto consider = [&](auto tile) {
        using T = decltype(tile);
        if (found)
            return;
        if ((int)tile_class<T>() != cta || (int)T::kStages != k_stages || (int)T::kTile != k_tile)
            return;
        out = recipe_for_tile<T>(ba, bb);
        found = true;
    };
    std::apply([&](auto... tiles) { (consider(tiles), ...); }, Manifest{});
    return found;
}

inline std::optional<GemmRecipe>
recipe_of(int cta, int k_stages, int k_tile, bool crosswise, int ba, int bb) {
    GemmRecipe out{};
    const bool found = with_manifest(crosswise, ba, bb, [&](auto manifest) {
        return recipe_scan<decltype(manifest)>(cta, k_stages, k_tile, ba, bb, out);
    });
    if (!found)
        return std::nullopt;
    return out;
}

// The [gemm-plan] decision line (gen_plan_table's tag regex reads it).
inline void log_dispatch(const PlanQuery& q, const PlanDecision& d) {
    if (!gemm_plan_log_enabled())
        return;
    std::fprintf(stderr,
                 "[gemm-plan] %s m%lld n%lld k%lld b=%d -> cta%d s%d "
                 "raster %d\n",
                 d.source, (long long)q.m, (long long)q.n, (long long)q.k, (int)q.batch,
                 d.recipe.cta, d.recipe.k_stages, d.raster);
}

// Validate table recipes against the compiled manifest and device limits.
std::optional<PlanDecision>
row_plan(const PlanQuery& q, std::optional<TableRow> row, const char* source) {
    if (!row)
        return std::nullopt;
    const auto recipe =
        recipe_of((int)row->cta, row->k_stages, row->k_tile, q.crosswise > 0, q.ba, q.bb);
    if (!recipe || plan_resident_ctas(row->cta, row->k_stages, row->k_tile, q) <= 0)
        return std::nullopt;
    return PlanDecision{
        *recipe, row->raster != 0 ? row->raster : plan_raster(q, recipe->bm, recipe->bn), source};
}

// RTX 5090 fitted mainloop and MMA costs in equivalent bytes, not TMA instruction counts.
constexpr std::int64_t kMainloopBytesPerCellTile = 8;
constexpr std::int64_t kMmaArmBytesPerInstr = 64;

int resident_of(const GemmRecipe& r, const PlanQuery& q) {
    return plan_resident_ctas(static_cast<TileClass>(r.cta), r.k_stages, r.k_tile, q);
}

// TMA overlaps copy and compute; cp.async hides copy latency through residency.
// Resident-scaled waves apply only to two-byte pairs.
std::int64_t cost_of(const GemmRecipe& r, const PlanQuery& q, int resident) {
    const std::int64_t blocks = q.batch * ((std::int64_t)((q.m + r.bm - 1) / r.bm) *
                                           (std::int64_t)((q.n + r.bn - 1) / r.bn));
    if (!q.tma) {
        // Price complete K tiles and divide waves by resident CTA capacity.
        const std::int64_t operand = ((q.k + r.k_tile - 1) / r.k_tile) * (std::int64_t)r.k_tile *
                                     (r.bm * q.ba + r.bn * q.bb);
        const std::int64_t mu = q.dev.smem_per_sm / r.smem;
        const std::int64_t slots = (std::int64_t)q.dev.sms * mu;
        const std::int64_t waves = slots > 0 ? (blocks + slots - 1) / slots : 1;
        return (operand + (std::int64_t)q.out_elem_bytes * r.bm * r.bn) * waves;
    }
    const bool byte_pair = q.ba == 1 && q.bb == 1;
    const std::int64_t operand = (std::int64_t)q.k * (r.bm * q.ba + r.bn * q.bb);
    const std::int64_t output = (std::int64_t)q.out_elem_bytes * r.bm * r.bn;
    const std::int64_t k_tiles = (q.k + r.k_tile - 1) / r.k_tile;
    const std::int64_t loop_penalty =
        byte_pair ? 0 : kMainloopBytesPerCellTile * (std::int64_t)r.bm * r.bn * k_tiles;
    const std::int64_t mma_instructions = k_tiles * (r.k_tile / q.mma_k) * (r.bm / 16) * (r.bn / 8);
    const std::int64_t mma_arm = kMmaArmBytesPerInstr * mma_instructions;
    const std::int64_t per_cta = std::max(operand + output + loop_penalty, mma_arm);
    const std::int64_t slots = (std::int64_t)q.dev.sms * resident;
    const std::int64_t waves = slots > 0 ? (blocks + slots - 1) / slots : 1;
    const std::int64_t w_eff =
        q.ba == 2 && q.bb == 2 ? waves * resident : (blocks + q.dev.sms - 1) / q.dev.sms;
    return per_cta * w_eff;
}

// geom_cta: rank five work proxies in log space. Smaller is better.
// This is a heuristic ordering, not a prediction of execution time.
double heuristic_cost(const GemmRecipe& named, const PlanQuery& q, int fallback_resident) {
    KernelResources resource{};
    if (q.resources) {
        resource = q.resources(named, q);
    } else {
        // Standalone host callers may have no typed CUDA kernel resolver.
        resource.effective = named;
        resource.resident = fallback_resident;
        if (q.ba + q.bb >= 3 && named.bm == 64 && named.bn == 64 && named.k_tile == 64) {
            resource.effective.wn = 16;
            resource.effective.threads = 512;
        }
    }
    if (resource.resident <= 0)
        return std::numeric_limits<double>::infinity();
    const auto& r = resource.effective;
    auto ceil_div = [](int64_t n, int d) { return n / d + (n % d != 0); };
    const double blocks =
        (double)q.batch * (double)ceil_div(q.m, r.bm) * (double)ceil_div(q.n, r.bn);
    return geometry_cost(r, q, resource.resident, blocks, (double)q.out_elem_bytes * r.bm * r.bn);
}

bool same_model_query(const PlanQuery& a, const PlanQuery& b) {
    const auto& x = a.dev;
    const auto& y = b.dev;
    return a.m == b.m && a.n == b.n && a.k == b.k && a.batch == b.batch &&
           a.perf_class == b.perf_class && a.crosswise == b.crosswise && a.ba == b.ba &&
           a.bb == b.bb && a.out_elem_bytes == b.out_elem_bytes && a.mma_k == b.mma_k &&
           a.tma == b.tma && a.resources == b.resources && a.rank3a == b.rank3a &&
           a.rank3b == b.rank3b && a.contiguous == b.contiguous &&
           x.threads_per_sm == y.threads_per_sm && x.ordinal == y.ordinal && x.sms == y.sms &&
           x.smem_max == y.smem_max && x.smem_per_sm == y.smem_per_sm &&
           x.regs_per_sm == y.regs_per_sm && x.l2_bytes == y.l2_bytes && x.cc == y.cc;
}

std::optional<PlanDecision> model_plan(const PlanQuery& q, bool heuristic = false) {
    if (q.dev.sms <= 0 || q.m <= 0 || q.n <= 0 || q.k <= 0 || q.batch <= 0 || q.mma_k <= 0)
        return std::nullopt;
    struct LastModelPlan {
        PlanQuery query{};
        PlanDecision decision{};
        bool heuristic = false;
        bool valid = false;
    };
    // A layer alternates among several GEMM shapes; retain their exact queries.
    struct ModelPlanCache {
        std::array<LastModelPlan, 8> entries{};
        std::size_t next = 0;
    };
    static thread_local ModelPlanCache cache;
    for (const auto& entry : cache.entries)
        if (entry.valid && entry.heuristic == heuristic && same_model_query(entry.query, q))
            return entry.decision;
    std::optional<GemmRecipe> best;
    std::optional<GemmRecipe> short_k_best;
    double best_cost = 0;
    double short_k_cost = 0;
    int shallow_k_stages = std::numeric_limits<int>::max();
    with_manifest(q.crosswise > 0, q.ba, q.bb, [&](auto manifest) {
        for_each_recipe<decltype(manifest)>(q.ba, q.bb, [&](GemmRecipe recipe) {
            const int resident = resident_of(recipe, q);
            if (resident <= 0)
                return;
            if (recipe.k_tile == 64)
                shallow_k_stages = std::min(shallow_k_stages, recipe.k_stages);
            const double cost = heuristic ? heuristic_cost(recipe, q, resident)
                                          : (double)cost_of(recipe, q, resident);
            if (!std::isfinite(cost))
                return;
            if (!heuristic && recipe.k_tile == 32 && (!short_k_best || cost < short_k_cost)) {
                short_k_best = recipe;
                short_k_cost = cost;
            }
            if (!best || cost < best_cost) {
                best = recipe;
                best_cost = cost;
            }
        });
    });
    if (!best)
        return std::nullopt;
    // Before the shallowest compiled 64-deep pipeline reaches steady state,
    // its prologue and ring footprint may outweigh the shorter K-tile. Limit this
    // measured RTX 5090 BF16 correction to the contiguous NT, batch-one domain;
    // other devices and paths retain the legacy ranking until they have sweeps.
    if (!heuristic && q.dev.cc == 120 && q.dev.sms == 170 && q.dev.smem_per_sm == 100 * 1024 &&
        q.tma && q.contiguous && q.batch == 1 && q.ba == 2 && q.bb == 2 && q.out_elem_bytes == 2 &&
        q.m >= 128 && q.n >= 128 && q.k >= 128 && short_k_best &&
        shallow_k_stages != std::numeric_limits<int>::max() &&
        q.k <= (std::int64_t)shallow_k_stages * best->k_tile)
        best = short_k_best;
    const PlanDecision decision{*best, plan_raster(q, best->bm, best->bn),
                                heuristic ? "heuristic" : "model"};
    cache.entries[cache.next] = {q, decision, heuristic, true};
    cache.next = (cache.next + 1) % cache.entries.size();
    return decision;
}

PlanDecision select_plan(const PlanQuery& q) {
    if (q.m <= 0 || q.n <= 0 || q.k <= 0)
        throw std::invalid_argument("GEMM planner: M, N and K must be greater than zero");
    const int mode = gemm_planner_mode();
    if ((mode == 0 || mode == 1) && !gemm_table_off()) {
        if (auto d = row_plan(q, plan_override_row(q), "override"))
            return *d;
        if (auto d = row_plan(q, plan_injected_row(q), "injected"))
            return *d;
        if (auto d = row_plan(q, plan_builtin_row(q), "builtin"))
            return *d;
    }
    if (mode == 1 || mode == 2 || mode == 3)
        if (auto d = model_plan(q, mode != 2))
            return *d;

    throw std::runtime_error(
        "GEMM planner: no eligible recipe for the selected mode, shape and device");
}

} // namespace

PlanDecision plan_dispatch(const PlanQuery& q) {
    const PlanDecision decision = select_plan(q);
    log_dispatch(q, decision);
    return decision;
}

/* Export instantiated recipes in manifest order for the tuner. */
std::vector<std::vector<int>> tile_vocabulary() {
    const std::pair<int, int> widths[] = {{2, 2}, {2, 1}, {1, 1}};
    std::vector<std::vector<int>> out;
    for (int crosswise = 0; crosswise <= 1; ++crosswise)
        for (const auto& [ba, bb] : widths)
            with_manifest(crosswise != 0, ba, bb, [&](auto manifest) {
                for_each_recipe<decltype(manifest)>(ba, bb, [&](const GemmRecipe& r) {
                    out.push_back({crosswise, ba, bb, r.cta, r.k_stages, r.k_tile, r.bm, r.bn, r.wm,
                                   r.wn, r.threads, r.smem});
                });
            });
    return out;
}

/* Names for serialized CTA classes, in enum order. */
std::vector<const char*> tile_class_names() {
    static constexpr const char* kNames[] = {"kSmall64", "kNarrow128x64", "kBig128", "kWide128x256",
                                             "kTall64x128"};
    static_assert((int)TileClass::kTall64x128 == (int)(sizeof(kNames) / sizeof(kNames[0])) - 1,
                  "kNames is indexed by TileClass: keep it in enum order");
    return std::vector<const char*>(kNames, kNames + sizeof(kNames) / sizeof(kNames[0]));
}

} // namespace gemm
} // namespace astrai
