#pragma once
/* Manifest tile selection and TMA/cp.async policy resolution. */
#include <tuple>
#include <type_traits>

#include <launcher/gemm_launch.cuh>
#include <launcher/kernel_resources.cuh>

namespace astrai {
namespace gemm {

/*
 * Manifest dispatch (CUTLASS builder-table style): (CTA class,
 * k_stages, k_tile) selects one manifest entry; the resolver maps it to a Policy
 * and launches.
 */
template <typename Manifest, typename Resolver>
bool dispatch_tile(const GemmRecipe& recipe, const Resolver& resolve) {
    return std::apply(
        [&recipe, &resolve](auto... tiles) {
            return (... || (tile_class<decltype(tiles)>() == static_cast<TileClass>(recipe.cta) &&
                            decltype(tiles)::kStages == recipe.k_stages &&
                            (int)decltype(tiles)::kTile == recipe.k_tile &&
                            resolve.template run<decltype(tiles)>()));
        },
        Manifest{});
}

/*
 * Narrow twin of a big tile for the reclaim fallback, same ring depth as
 * the tile it replaces (the planner priced that ring). No kK=32 narrow s3
 * exists — that big tile falls to its s2 twin.
 */
template <typename Tile>
using narrow_fallback_t = std::conditional_t<
    Tile::kTile == 32,
    Tile_128x64x32_W32x32_S2,
    std::conditional_t<(Tile::kStages >= 3), Tile_128x64x64_W32x32_S3, Tile_128x64x64_W32x32_S2>>;

/*
 * The reclaim chain every instantiation terminates in: output outgrows the
 * ring -> narrow twin; the kK=32 narrow (18KB ring) still cannot hold 4B/elem
 * -> small CTA (24KB reclaims every output <= 4B/elem). Identity whenever
 * the tile fits, so the launchers apply it unconditionally; the dispatch
 * names every manifest tile as a potential substitute, so the chain is
 * load-bearing even for unplanned geometries.
 */
template <typename Tile, typename ElemA, typename ElemB, typename OutT>
using reclaim_fallback_t = std::conditional_t<
    reclaim_fits<Tile, ElemA, ElemB, OutT>(),
    Tile,
    std::conditional_t<reclaim_fits<narrow_fallback_t<Tile>, ElemA, ElemB, OutT>(),
                       narrow_fallback_t<Tile>,
                       Tile_64x64x64_W16x32_S2>>;

template <typename Schedule, typename ElemA, typename ElemB, typename LayoutA, typename LayoutB>
inline constexpr bool tma_eligible_v =
    Schedule::kTma && crosswise_of<LayoutA, LayoutB>() == 0 && sizeof(ElemA) <= 2 &&
    sizeof(ElemB) <= 2;

/* Bind operand types, layouts, and schedule once; a recipe selects the concrete policy. */
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename LayoutOut,
          typename OutT,
          typename Schedule>
struct GemmTileDispatch {
    template <bool UseTma>
    using StagingA = std::conditional_t<UseTma, RowMajor, LayoutA>;
    template <bool UseTma>
    using StagingB = std::conditional_t<UseTma, ColMajor, LayoutB>;
    template <bool UseTma>
    using Manifest = manifest_for<ElemA, ElemB, StagingA<UseTma>, StagingB<UseTma>>;
    template <typename Tile>
    using ResolvedTile =
        reclaim_fallback_t<warp_widened_t<ElemA, ElemB, Tile>, ElemA, ElemB, OutT>;
    template <bool UseTma, typename Tile>
    using Policy = GemmPolicy<ElemA, ElemB, StagingA<UseTma>, StagingB<UseTma>,
                              ResolvedTile<Tile>, LayoutOut, OutT,
                              PlannedGemmOptions<Schedule, UseTma>>;

    template <bool UseTma>
    struct Launcher {
        const GemmParams& p;
        cudaStream_t stream;
        int smem_max;

        template <typename Tile> bool run() const {
            using P = Policy<UseTma, Tile>;
            if constexpr (UseTma) {
                return launch_policy_tma<P>(p, stream, smem_max);
            } else {
                launch_policy<P>(p, stream);
                return true;
            }
        }
    };

    template <bool UseTma>
    struct ResourceQuery {
        const PlanQuery& q;
        KernelResources& result;

        template <typename Tile> bool run() const {
            using P = Policy<UseTma, Tile>;
            using T = typename P::Tile;
            if constexpr (UseTma) {
                result = with_tma_ranks(q.rank3a, q.rank3b, [&](auto a, auto b) {
                    return kernel_resources<
                        gemm_kernel_tma<P, decltype(a)::value, decltype(b)::value>, P>(q);
                });
            } else {
                result = kernel_resources<gemm_kernel<P>, P>(q);
            }
            result.effective = {static_cast<int>(tile_class<T>()),
                                T::kStages,
                                T::kTile,
                                T::CtaShape::kM,
                                T::CtaShape::kN,
                                T::WarpShape::kM,
                                T::WarpShape::kN,
                                P::Traits::kCtaThreads,
                                P::kSmemBytes};
            return true;
        }
    };

    static KernelResources resources(const GemmRecipe& recipe, const PlanQuery& q) {
        KernelResources result{};
        if constexpr (tma_eligible_v<Schedule, ElemA, ElemB, LayoutA, LayoutB>) {
            if (q.tma) {
                dispatch_tile<Manifest<true>>(recipe, ResourceQuery<true>{q, result});
                return result;
            }
        }
        dispatch_tile<Manifest<false>>(recipe, ResourceQuery<false>{q, result});
        return result;
    }

    static void launch(GemmParams p, const LaunchPlan& selected, cudaStream_t stream,
                       const DeviceFacts& dev) {
        const PlanDecision& decision = selected.decision;
        p.raster = decision.raster;
        // Descriptor rejection falls through to the cp.async policy.
        if constexpr (tma_eligible_v<Schedule, ElemA, ElemB, LayoutA, LayoutB>) {
            if (selected.tma &&
                dispatch_tile<Manifest<true>>(
                    decision.recipe, Launcher<true>{p, stream, dev.smem_max}))
                return;
        }
        dispatch_tile<Manifest<false>>(decision.recipe, Launcher<false>{p, stream, dev.smem_max});
    }
};

} // namespace gemm
} // namespace astrai
