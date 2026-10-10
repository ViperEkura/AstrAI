#pragma once
/* Tile vocabulary, CTA classes, and staging-specific manifests. */
#include <tuple>
#include <type_traits>
#include <utility>

#include <policy/traits.cuh>

namespace astrai {
namespace gemm {

/*
 * Named CTA/warp/stage recipe; adding a tile needs one alias and planner branch.
 * CTAs use the predication-free copy only when aligned, interior, and K-tail-free.
 * The crosswise big-CTA downgrade tested 9-19% slower on sm_120 (TT, cp.async;
 * 2026-09-16), and the planner never selects that downgrade.
 */
template <typename CtaShape_, typename WarpShape_, int PipelineStages_> struct GemmTileConfig {
    using CtaShape = CtaShape_;
    using WarpShape = WarpShape_;
    static constexpr int kTile = CtaShape_::kK;
    static constexpr int kStages = PipelineStages_;
};

/*
 * Tile recipes, named Tile_<cta M>x<N>x<kK>_W<warp M>x<N>_S<k_stages>: the
 * CTA Shape, the warp Shape and the pipeline depth, in the order GemmTileConfig
 * carries them.
 *
 * kK is capped at 128/elem_bytes by the staging swizzle: ComposedLayout
 * requires kRowShift = kShift - log2(kChunks) >= 0 with kChunks =
 * kK*elem/16, so Swizzle<4,3> tops 2-byte operands out at kK 64 while
 * 1-byte operands reach 128. Similarly the load path needs
 * kTileLines*kChunks % kThreads == 0 with a power-of-two chunks-per-thread,
 * so a 1-byte line holds half as many 16B chunks as a 2-byte one and the
 * widest warp tilings do not instantiate for 1-byte operands.
 */
using Tile_128x128x64_W64x32_S2 = GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 2>;
using Tile_128x128x64_W64x32_S3 = GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 3>;
using Tile_128x64x64_W32x32_S2 = GemmTileConfig<Shape<128, 64, 64>, Shape<32, 32>, 2>;
using Tile_128x64x64_W32x32_S3 = GemmTileConfig<Shape<128, 64, 64>, Shape<32, 32>, 3>;
using Tile_64x64x64_W16x32_S2 = GemmTileConfig<Shape<64, 64, 64>, Shape<16, 32>, 2>;
using Tile_64x64x64_W16x32_S3 = GemmTileConfig<Shape<64, 64, 64>, Shape<16, 32>, 3>;
/*
 * Deep-ring s4/s5 twins of this geometry were removed: the sweep measured
 * them a wash against s2..s3 (within 1-2% at this tile) and no compiled-in
 * row reaches past s3. The planner rejects k_stages > 3 outright, so a stale
 * row file naming one falls to the next source instead of silently
 * launching nothing (row_plan in gemm/planning.cpp).
 * 16 warps per CTA on the small geometry (16x16 warp tiles, 512 threads):
 * the 8-warp small tile starves the tensor pipe on two-byte operands —
 * measured 1.36-1.66x for this twin on 11 of 13 shapes (NT and both
 * crosswise layouts), parity or -3% on the two thinnest. The launch
 * resolvers substitute it for the 8-warp entries (small_16w_t below); the
 * ladder keeps naming the 8-warp tile, and a 1-byte operand's thinner load
 * bus no longer blocks the substitution — the surplus threads take the
 * predicated skip (load_async.cuh) and the dequant fragments each lane owes
 * halve with the 16-wide N partition.
 */
template <typename Tile>
using small_16w_t = GemmTileConfig<typename Tile::CtaShape, Shape<16, 16>, Tile::kStages>;

/*
 * kK=32 twins: the measured k-tile-depth winner on most shapes, since a
 * 64-deep k-tile spends ring budget and issue slots the short K loop cannot
 * use.
 */
using Tile_64x64x32_W16x32_S2 = GemmTileConfig<Shape<64, 64, 32>, Shape<16, 32>, 2>;
using Tile_64x64x32_W16x32_S3 = GemmTileConfig<Shape<64, 64, 32>, Shape<16, 32>, 3>;
/* Tall 64x128 won wide-N sweep and congruous square, up to 1.089x over prior best. */
using Tile_64x128x32_W32x32_S2 = GemmTileConfig<Shape<64, 128, 32>, Shape<32, 32>, 2>;
using Tile_64x128x32_W32x32_S3 = GemmTileConfig<Shape<64, 128, 32>, Shape<32, 32>, 3>;
using Tile_128x64x32_W32x32_S2 = GemmTileConfig<Shape<128, 64, 32>, Shape<32, 32>, 2>;
using Tile_128x128x32_W64x32_S3 = GemmTileConfig<Shape<128, 128, 32>, Shape<64, 32>, 3>;
/*
 * 16-warp 128x128x32 keeps the 48KB ring and avoids tensor-pipe stalls. It
 * gained 7-14% on 12/13 fused-linear cases; 512x1536x1536 (-1.2%) uses 64x64.
 * Two CTAs fit the register budget: 512 threads need <=64 registers each.
 */
using Tile_128x128x32_W32x32_S2 = GemmTileConfig<Shape<128, 128, 32>, Shape<32, 32>, 2>;
/*
 * 1-byte operands only: the ring is 147KB for a 2-byte pair (past the smem
 * opt-in ceiling) against 74KB for a 1-byte one.
 */
using Tile_128x256x64_W64x32_S2 = GemmTileConfig<Shape<128, 256, 64>, Shape<64, 32>, 2>;
using Tile_128x128x128_W64x32_S2 = GemmTileConfig<Shape<128, 128, 128>, Shape<64, 32>, 2>;

/*
 * CTA class of a tile config, derived from its CTA geometry — one axis of
 * the dispatch key the launch ladders select on (PlanDecision in launcher/plan_types.h).
 */
enum class TileClass { kSmall64, kNarrow128x64, kBig128, kWide128x256, kTall64x128 };

template <typename Tile> constexpr TileClass tile_class() {
    if constexpr (Tile::CtaShape::kM == 128 && Tile::CtaShape::kN == 256)
        return TileClass::kWide128x256;
    else if constexpr (Tile::CtaShape::kM == 128 && Tile::CtaShape::kN == 128)
        return TileClass::kBig128;
    else if constexpr (Tile::CtaShape::kM == 128 && Tile::CtaShape::kN == 64)
        return TileClass::kNarrow128x64;
    else if constexpr (Tile::CtaShape::kM == 64 && Tile::CtaShape::kN == 128)
        return TileClass::kTall64x128;
    else
        return TileClass::kSmall64;
}

/* Widen only the small kK=64 tile when either operand is 2-byte. The kK=32
 * form wastes lanes; byte pairs keep the 8-warp tile. Tests pin this rule.
 */
template <typename ElemA, typename ElemB, typename Tile>
using warp_widened_t =
    std::conditional_t<sizeof(ElemA) + sizeof(ElemB) >= 3 &&
                           tile_class<Tile>() == TileClass::kSmall64 && Tile::CtaShape::kK == 64,
                       small_16w_t<Tile>,
                       Tile>;
/*
 * CTA geometry per dispatch class — the inverse of tile_class, and the one
 * home for the class -> (M, N) numbers the host plan table prices rows with
 * (plan_row_geometry in gemm/plan_table.cpp). Indexed by TileClass, enum order.
 */
inline constexpr int kTileClassCta[][2] = {
    {64, 64},   // kSmall64
    {128, 64},  // kNarrow128x64
    {128, 128}, // kBig128
    {128, 256}, // kWide128x256
    {64, 128},  // kTall64x128
};
static_assert((int)TileClass::kSmall64 == 0 && (int)TileClass::kNarrow128x64 == 1 &&
                  (int)TileClass::kBig128 == 2 && (int)TileClass::kWide128x256 == 3 &&
                  (int)TileClass::kTall64x128 == 4,
              "kTileClassCta is indexed by TileClass: keep the enum in table order");

/* One geometry row per TileClass, in enum order; plan-table cta values use
 * the same ordinals after bounds checking.
 */
inline constexpr int kTileClassCount = (int)(sizeof(kTileClassCta) / sizeof(kTileClassCta[0]));

/*
 * A class is a function of its CTA shape alone, so one representative tile per
 * class pins the table to the tiles the ladders actually instantiate.
 */
template <typename Tile> constexpr bool cta_matches_class() {
    return kTileClassCta[(int)tile_class<Tile>()][0] == Tile::CtaShape::kM &&
           kTileClassCta[(int)tile_class<Tile>()][1] == Tile::CtaShape::kN;
}
static_assert(cta_matches_class<Tile_64x64x64_W16x32_S2>() &&
                  cta_matches_class<Tile_128x64x64_W32x32_S2>() &&
                  cta_matches_class<Tile_128x128x64_W64x32_S2>() &&
                  cta_matches_class<Tile_128x256x64_W64x32_S2>() &&
                  cta_matches_class<Tile_64x128x32_W32x32_S2>(),
              "kTileClassCta must mirror the tiles' CTA shapes");

/*
 * Tuple concatenation, so a manifest reads as "the shared ladder plus my own
 * additions" instead of re-listing the shared entries — the prefix relationship
 * between the ladders is then structural, not a copy that can drift.
 */
template <typename... Ts> using tuple_cat_t = decltype(std::tuple_cat(std::declval<Ts>()...));

/*
 * The shared six recipes appear in every staging ladder. Crosswise 1-byte
 * kK=32 loads skip inactive lanes, so bus width does not constrain
 * membership. The wide CTA is byte-only; 16-warp small is resolver-only.
 */
using TileManifestCross = std::tuple<Tile_128x128x64_W64x32_S2,
                                     Tile_128x128x64_W64x32_S3,
                                     Tile_128x64x64_W32x32_S2,
                                     Tile_128x64x64_W32x32_S3,
                                     Tile_64x64x64_W16x32_S2,
                                     Tile_64x64x64_W16x32_S3>;

/*
 * Ordered recipes selected by CTA class, k_stages, and kK; first match wins.
 * Width-specific ladders enforce load-divisibility and output-reclaim limits.
 * Byte omits big kK=32 S2 (24KB ring < 32KB output) and 16-warp small;
 * two-byte omits 128x256 (its ring fits only byte operands). Big entries use
 * fast copies; crosswise cp.async resolves to the non-fast twin. Congruous adds kK=32
 * twins; 16-warp widening remains a resolver substitution, not a manifest row.
 */
using TileManifest = tuple_cat_t<TileManifestCross,
                                 std::tuple<Tile_64x64x32_W16x32_S2,
                                            Tile_64x64x32_W16x32_S3,
                                            Tile_128x64x32_W32x32_S2,
                                            Tile_128x128x32_W32x32_S2,
                                            Tile_128x128x32_W64x32_S3,
                                            Tile_64x128x32_W32x32_S2,
                                            Tile_64x128x32_W32x32_S3>>;

/*
 * Byte ladder adds wide-N, kK=128 and kK=32 big tiles. Exclude crosswise kK=32
 * narrow: half-bus loads measured 8-34% slower than kK=64, and the planner
 * mis-picked it on unseen bands. S3's 32KB ring reclaims its output; S2's
 * 24KB ring requires direct-store epilogue. 16-warp small remains resolver-only.
 */
using TileManifestByte = tuple_cat_t<
    TileManifestCross,
    std::tuple<Tile_128x256x64_W64x32_S2, Tile_128x128x128_W64x32_S2, Tile_128x128x32_W64x32_S3>>;

/*
 * How many operands take that direct path (0 = dual-congruous NT). The
 * planner's crosswise field and the launcher's ladder selection are this one
 * number, asked of the layout tags rather than re-derived from trans flags.
 */
template <typename LayoutA, typename LayoutB> constexpr int crosswise_of() {
    return (direct_a<LayoutA>() ? 1 : 0) + (direct_b<LayoutB>() ? 1 : 0);
}

/*
 * One rule drives type-level aliases and runtime plan lookup. Byte pairs use
 * their ladder on either staging path; only the big S3 is a kK=32 entry. The
 * wide CTA has a 72KB ring and packed carries of 256/512 units. Crosswise
 * 2-byte and mixed pairs use the shared ladder; congruous mixed pairs use
 * theirs. Predicated skips handle the 1-byte side; the small CTA widens, and
 * kK=32 reclaim fits (big: 32KB output in a 48KB ring).
 */
enum class ManifestKind { kCrosswise, kTwoByte, kMixed, kByte };

constexpr ManifestKind manifest_kind(bool crosswise_staging, int ba, int bb) {
    if (ba == 1 && bb == 1)
        return ManifestKind::kByte;
    if (crosswise_staging)
        return ManifestKind::kCrosswise;
    if (ba == 2 && bb == 2)
        return ManifestKind::kTwoByte;
    if (ba + bb == 3)
        return ManifestKind::kMixed;
    return ManifestKind::kCrosswise;
}

template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB>
constexpr ManifestKind manifest_kind_of() {
    return manifest_kind(crosswise_of<LayoutA, LayoutB>() != 0, (int)sizeof(ElemA),
                         (int)sizeof(ElemB));
}

/*
 * The manifest a given operand pair and staging selects over: the kind above,
 * mapped to its ladder (kCrosswise is the fallback, so it needs no arm).
 */
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB>
using manifest_for = std::conditional_t<
    manifest_kind_of<ElemA, ElemB, LayoutA, LayoutB>() == ManifestKind::kTwoByte ||
        manifest_kind_of<ElemA, ElemB, LayoutA, LayoutB>() == ManifestKind::kMixed,
    TileManifest,
    std::conditional_t<manifest_kind_of<ElemA, ElemB, LayoutA, LayoutB>() == ManifestKind::kByte,
                       TileManifestByte,
                       TileManifestCross>>;

} // namespace gemm
} // namespace astrai
