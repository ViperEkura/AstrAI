#pragma once
// Kernel policy layer: shared-memory budget, occupancy hint and the single
// Policy type the kernel takes (CUTLASS-style consolidation of traits +
// layout tags + scheduling knobs). Dtype-generic via gemm_elem_traits.

#include <cuda_fp8.h>
#include <tuple>
#include <type_traits>
#include <utility>

#include "common/mma.cuh"
#include "common/tensor.cuh"
#include "gemm/common.h"

namespace astrai {
namespace gemm {

// Compile-time tile configuration (CTA tile + warp tiling + pipeline depth).
// ElemA/ElemB are independent operand types; the MMA runs on the promoted
// MmaT (gemm_mma_traits): W16A16 passes through, symmetric fp8/int8 keep
// their native mma (fp32/int32 accumulators), a lone 8-bit side against
// bf16 dequantizes in-register (kDequantA/B mark those inserts per side).
// UseMx swaps the symmetric-fp8 cell for the sm_120 block_scale cell.
template <typename ElemA_, typename ElemB_, typename CtaShape_,
          typename WarpShape_, int Stages, bool UseMx = false>
struct GemmTraits {
    using ElemA = ElemA_;
    using ElemB = ElemB_;
    using MmaPair = gemm_mma_traits<ElemA_, ElemB_>;
    using MmaT = typename MmaPair::MmaT;
    // The exact mma cell <MmaT, MmaT, shape> (common/mma.cuh): one type
    // carries the instruction's K extent and accumulator type (fp32 for the
    // float families, s32 for the s8 pair).
    static constexpr bool kMxCell =
        UseMx && (std::is_same_v<MmaT, __nv_fp8_e4m3> ||
                  std::is_same_v<MmaT, __nv_fp8_e5m2>);
    using MmaOp = std::conditional_t<
        kMxCell, astrai::MxMmaOp<MmaT>,
        astrai::MmaOp<MmaT, MmaT, typename astrai::MmaShapeFor<MmaT>::type>>;
    using AccT = typename MmaOp::AccT;
    using ElemTraitsA = gemm_elem_traits<ElemA_>;
    using ElemTraitsB = gemm_elem_traits<ElemB_>;

    using CtaShape = CtaShape_;
    using WarpShape = WarpShape_;
    static constexpr int kBlockM = CtaShape_::kM;
    static constexpr int kBlockN = CtaShape_::kN;
    static constexpr int kK = CtaShape_::kK;
    static constexpr int kStages = Stages;
    static constexpr int kWarpM = WarpShape_::kM;
    static constexpr int kWarpN = WarpShape_::kN;

    static constexpr int kElemBytesA = ElemTraitsA::kBytes;
    static constexpr int kElemBytesB = ElemTraitsB::kBytes;
    // MMA shape follows the promoted compute type; dequantized fragments
    // are brought to it in-register (dequant.cuh).
    static constexpr int kMmaK = astrai::MmaShapeFor<MmaT>::type::kK;
    static constexpr bool kDequantA = MmaPair::kDequantA;
    static constexpr bool kDequantB = MmaPair::kDequantB;

    // Derived geometry: warp tiles tile the CTA; the smem budget is
    // layout-aware, so it lives in GemmSmem below.
    static constexpr int kWarpsM = kBlockM / kWarpM;
    static constexpr int kWarpsN = kBlockN / kWarpN;
    static constexpr int kCtaThreads = kWarpsM * kWarpsN * 32;
    static_assert(kWarpsM * kWarpM == kBlockM && kWarpsN * kWarpN == kBlockN,
                  "warp tiles must exactly tile the CTA");
    static_assert(kWarpM % 16 == 0 && kWarpN % 8 == 0,
                  "warp tile must be a multiple of the m16n8 MMA shape");
    // Warp accumulator geometry, one definition for mainloop and epilogue:
    // m16n8 mma cells on the (kMt, kNt) warp tile grid.
    static constexpr int kMt = kWarpM / 16;
    static constexpr int kNt = kWarpN / 8;
    using AccTensor =
        Tensor<ArrayEngine<typename MmaOp::CFrag, kMt * kNt>,
               CellLayout<kNt>>;
};

// Ring-budget formula, one source for GemmSmem, launch_plan's epilogue
// reclaim check and the host planner's recipe feasibility gate (gemm.cuh):
// every operand ring holds kStages+1 buffers of k * (bm*ba + bn*bb) bytes.
constexpr int ring_smem_bytes(int bm, int bn, int k, int stages,
                              int ba, int bb) {
    return (stages + 1) * k * (bm * ba + bn * bb);
}

// Resident-CTA hint for a ring of `bytes` bytes — __launch_bounds__'s second
// argument, and so the per-thread register budget every kernel of that
// geometry is compiled to (regs_per_sm / (threads * hint)). One source:
// GemmSmem states it to the compiler and the planner's residency model
// (plan_table.h) prices the same rule, so a ring the planner counts as two
// CTAs per SM is one the compiler also fitted.
//
// The 48KB watermark is a preference, not a device limit — well under every
// supported part's per-block opt-in ceiling, it reads "a ring that fits in
// half of Ada's smem is worth splitting the register file for". It is
// therefore NOT the smem term of residency: that term is priced from
// DeviceFacts::smem_per_sm, because a part with more smem per SM packs more
// CTAs than a watermark fixed at one device's figure allows.
constexpr int min_ctas_for_ring(int bytes) {
    return bytes <= 48 * 1024 ? 2 : 1;
}

// Which storages take the DIRECT (crosswise) staging path. A stored [K][M]
// (ColMajor) and B stored [K][N] (RowMajor) each keep the tile's rows along K,
// so their load walks memory crosswise; the other two are congruous and stage
// as-is. The operand role rides the function name because a tag alone does not
// say which side it is: the canonical A is [M][K] and B is [K][N], so
// ColMajor means crosswise for A and congruous for B. Everything else in this
// directory spells the predicate from these two — crosswise_of() sums them,
// GemmSmem reads them per operand, the ladder picks its kind from them.
template <typename Layout>
constexpr bool direct_a() {
    return std::is_same_v<Layout, ColMajor>;
}
template <typename Layout>
constexpr bool direct_b() {
    return std::is_same_v<Layout, RowMajor>;
}

// Layout-aware shared-memory budget and occupancy hint. Every operand ring
// holds kStages+1 buffers: the load for tile i+kStages targets slot
// (i-1)%(kStages+1) — already consumed — so neither load path needs a
// post-compute barrier (one __syncthreads per k-tile). The register-budget
// hint comes from min_ctas_for_ring below.
template <typename Traits, typename LayoutA, typename LayoutB>
struct GemmSmem {
    // Crosswise (direct-load) operands: A ColMajor storage, B RowMajor
    // storage (B's tag is relative to the canonical [K][N]).
    static constexpr bool kDirectA = direct_a<LayoutA>();
    static constexpr bool kDirectB = direct_b<LayoutB>();
    static constexpr int kRingDepth = Traits::kStages + 1;
    static constexpr int kBytes =
        ring_smem_bytes(Traits::kBlockM, Traits::kBlockN, Traits::kK,
                        Traits::kStages, Traits::kElemBytesA,
                        Traits::kElemBytesB);
    static constexpr int kMinCtas = min_ctas_for_ring(kBytes);
};

// Tile recipe (CUTLASS-style configuration type): one named bundle of CTA
// shape, warp tiling and pipeline depth. Extending the launch ladder = one
// alias here + one planner branch (never re-spelled positional ints).
//
// There is no loop-specialization axis: every tile carries both mainloop
// copies and the per-CTA runtime verdict (use_interior_copy, mainloop.cuh)
// picks one — interior, 16B-aligned, K-tail-free CTAs take the predication-
// free copy. A compile-time kFastLoop flag once spelled the "fast twin"
// here, but its one remaining effect (the big CTA downgraded to the
// predicated twin on crosswise staging) measured BACKWARDS on this part:
// interleaved A/B 2026-09-16 (sm_120, TT route, cp.async) has the
// predication-free copy 9-19% FASTER on both big kk twins, and the
// downgrade was unreachable through the planner anyway (the model's
// residency rule never picks big on the crosswise ladder).
template <typename CtaShape_, typename WarpShape_, int Stages_>
struct GemmTileConfig {
    using CtaShape = CtaShape_;
    using WarpShape = WarpShape_;
    static constexpr int kStages = Stages_;
};

// Tile recipes, named Tile_<cta M>x<N>x<kK>_W<warp M>x<N>_S<stages>: the
// CTA Shape, the warp Shape and the ring depth, in the order GemmTileConfig
// carries them.
//
// kK is capped at 128/elem_bytes by the staging swizzle: ComposedLayout
// requires kRowShift = kShift - log2(kChunks) >= 0 with kChunks =
// kK*elem/16, so Swizzle<4,3> tops 2-byte operands out at kK 64 while
// 1-byte operands reach 128. Similarly the load path needs
// kTileLines*kChunks % kThreads == 0 with a power-of-two chunks-per-thread,
// so a 1-byte line holds half as many 16B chunks as a 2-byte one and the
// widest warp tilings do not instantiate for 1-byte operands.
using Tile_128x128x64_W64x32_S2 =
    GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 2>;
using Tile_128x128x64_W64x32_S3 =
    GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 3>;
using Tile_128x64x64_W32x32_S2 =
    GemmTileConfig<Shape<128, 64, 64>, Shape<32, 32>, 2>;
using Tile_128x64x64_W32x32_S3 =
    GemmTileConfig<Shape<128, 64, 64>, Shape<32, 32>, 3>;
using Tile_64x64x64_W16x32_S2 =
    GemmTileConfig<Shape<64, 64, 64>, Shape<16, 32>, 2>;
using Tile_64x64x64_W16x32_S3 =
    GemmTileConfig<Shape<64, 64, 64>, Shape<16, 32>, 3>;
// Deep-ring s4/s5 twins of this geometry were removed: the sweep measured
// them a wash against s2..s3 (within 1-2% at this tile) and no compiled-in
// row reaches past s3. The planner rejects stages > 3 outright, so a stale
// row file naming one falls to the next source instead of silently
// launching nothing (plan_from_row in gemm.cuh).
// 16 warps per CTA on the small geometry (16x16 warp tiles, 512 threads):
// the 8-warp small tile starves the tensor pipe on two-byte operands —
// measured 1.36-1.66x for this twin on 11 of 13 shapes (NT and both
// crosswise layouts), parity or -3% on the two thinnest. The launch
// resolvers substitute it for the 8-warp entries (small_16w_t below); the
// ladder keeps naming the 8-warp tile, and a 1-byte operand's thinner load
// bus no longer blocks the substitution — the surplus threads take the
// predicated skip (load.cuh) and the dequant fragments each lane owes
// halve with the 16-wide N partition.
template <typename Tile>
using small_16w_t = GemmTileConfig<typename Tile::CtaShape, Shape<16, 16>,
                                   Tile::kStages>;

// kK=32 twins: the measured k-tile-depth winner on most shapes, since a
// 64-deep k-tile spends ring budget and issue slots the short K loop cannot
// use.
using Tile_64x64x32_W16x32_S2 =
    GemmTileConfig<Shape<64, 64, 32>, Shape<16, 32>, 2>;
using Tile_64x64x32_W16x32_S3 =
    GemmTileConfig<Shape<64, 64, 32>, Shape<16, 32>, 3>;
// The tall 64x128 CTA (N:M = 2:1): the tile sweep's champion at wide N and
// at the parity square on the congruous ladder (up to 1.089x over the best
// previously reachable recipe), so it joins the manifest rather than living
// in the sweep grid only.
using Tile_64x128x32_W32x32_S2 =
    GemmTileConfig<Shape<64, 128, 32>, Shape<32, 32>, 2>;
using Tile_64x128x32_W32x32_S3 =
    GemmTileConfig<Shape<64, 128, 32>, Shape<32, 32>, 3>;
using Tile_128x64x32_W32x32_S2 =
    GemmTileConfig<Shape<128, 64, 32>, Shape<32, 32>, 2>;
using Tile_128x128x32_W64x32_S3 =
    GemmTileConfig<Shape<128, 128, 32>, Shape<64, 32>, 3>;
// 16 warps per CTA on the 128x128x32 ring (32x32 warp tiles, 512 threads):
// same CTA geometry, same 48KB ring, twice the warps. The 8-warp twin above
// leaves the tensor pipe waiting at every fragment boundary; doubling the warps
// per partition is worth 7-14% on every large fused-linear shape and costs
// nothing on the small ones (13 shapes measured, only 512x1536x1536 gives up
// 1.2%, and that shape is served by the 64x64 rows). The residency budget
// still holds: 512 threads x 2 CTAs needs <= 64 registers, which the smaller
// 32x32 warp tile's accumulator (32 fp32 cells) leaves room for.
using Tile_128x128x32_W32x32_S2 =
    GemmTileConfig<Shape<128, 128, 32>, Shape<32, 32>, 2>;
// 1-byte operands only: the ring is 147KB for a 2-byte pair (past the smem
// opt-in ceiling) against 74KB for a 1-byte one.
using Tile_128x256x64_W64x32_S2 =
    GemmTileConfig<Shape<128, 256, 64>, Shape<64, 32>, 2>;

// CTA class of a tile config, derived from its CTA geometry — one axis of
// the dispatch key the launch ladders select on (GemmPlan in gemm.cuh).
enum class TileClass { kSmall64, kNarrow128x64, kBig128, kWide128x256,
                       kTall64x128 };

template <typename Tile>
constexpr TileClass tile_class() {
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

// The one rule the launch resolvers ask: does this tile take the widening?
// Any pair with a 2-byte operand (a 1-byte x 1-byte pair keeps the 8-warp
// tile — its ladder is out of scope here, and W8A8's own cell is not
// issue-starved), the small CTA only, and kK=64 only (a kK=32 16-warp form
// would leave half the threads idle on the 2-byte side and three quarters
// on the 1-byte side; the skip makes it legal, not worthwhile). Named
// rather than inlined in the resolvers so the tests can pin all three arms
// without a launch.
template <typename ElemA, typename ElemB, typename Tile>
using warp_widened_t =
    std::conditional_t<sizeof(ElemA) + sizeof(ElemB) >= 3 &&
                           tile_class<Tile>() == TileClass::kSmall64 &&
                           Tile::CtaShape::kK == 64,
                       small_16w_t<Tile>, Tile>;
// CTA geometry per dispatch class — the inverse of tile_class, and the one
// home for the class -> (M, N) numbers the host plan table prices rows with
// (plan_row_geometry in plan_table.h). Indexed by TileClass, enum order.
inline constexpr int kTileClassCta[][2] = {
    {64, 64},    // kSmall64
    {128, 64},   // kNarrow128x64
    {128, 128},  // kBig128
    {128, 256},  // kWide128x256
    {64, 128},   // kTall64x128
};
static_assert((int)TileClass::kSmall64 == 0 &&
                  (int)TileClass::kNarrow128x64 == 1 &&
                  (int)TileClass::kBig128 == 2 &&
                  (int)TileClass::kWide128x256 == 3 &&
                  (int)TileClass::kTall64x128 == 4,
              "kTileClassCta is indexed by TileClass: keep the enum in table order");

// One row per class is kTileClassCta's contract (cta_matches_class pins it to
// the tiles the ladders instantiate), so its extent is the class count. The
// row file's cta column (plan_table.h) is bounds-checked against this and then
// read as the TileClass ordinal: the file's numbering and the enum's are one
// fact, not two tables to keep in sync.
inline constexpr int kTileClassCount =
    (int)(sizeof(kTileClassCta) / sizeof(kTileClassCta[0]));

// A class is a function of its CTA shape alone, so one representative tile per
// class pins the table to the tiles the ladders actually instantiate.
template <typename Tile>
constexpr bool cta_matches_class() {
    return kTileClassCta[(int)tile_class<Tile>()][0] == Tile::CtaShape::kM &&
           kTileClassCta[(int)tile_class<Tile>()][1] == Tile::CtaShape::kN;
}
static_assert(cta_matches_class<Tile_64x64x64_W16x32_S2>() &&
                  cta_matches_class<Tile_128x64x64_W32x32_S2>() &&
                  cta_matches_class<Tile_128x128x64_W64x32_S2>() &&
                  cta_matches_class<Tile_128x256x64_W64x32_S2>() &&
                  cta_matches_class<Tile_64x128x32_W32x32_S2>(),
              "kTileClassCta must mirror the tiles' CTA shapes");

// Tuple concatenation, so a manifest reads as "the shared ladder plus my own
// additions" instead of re-listing the shared entries — the prefix relationship
// between the ladders is then structural, not a copy that can drift.
template <typename... Ts>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<Ts>()...));

// The shared ladder: the geometries every staging path instantiates.
// load_operand_tile stages a crosswise operand as kK lines of (M or N)*elem/16
// chunks; an under-subscribed bus (fewer chunks than threads, the 1-byte
// side of a kK=32 twin) is a predicated skip there, so membership is a
// choice, not a bus constraint — the shared six is simply what every
// staging path wants; the wide CTA is 1-byte-only besides. The 16-warp
// small is a resolver substitution, never a manifest entry, so the bus
// never gated it either way. Every other manifest contains these.
using TileManifestCross = std::tuple<
    Tile_128x128x64_W64x32_S2, Tile_128x128x64_W64x32_S3,
    Tile_128x64x64_W32x32_S2, Tile_128x64x64_W32x32_S3,
    Tile_64x64x64_W16x32_S2, Tile_64x64x64_W16x32_S3>;

// The dispatch manifests (CUTLASS builder-table style): every recipe the
// launch ladders select over, keyed by the plan's CTA class, ring depth and
// k-tile depth. Split by operand width because the reclaim budget
// (bm*bn*sizeof(OutT) <= ring) and the load-path divisibility both bind
// harder on the narrowest ring: the byte manifest cannot carry the kK=32
// big CTA (32KB of output over a 24KB ring) — the 16-warp small is a
// resolver substitution, not an entry, and stays off a byte pair by rule —
// and the two-byte manifest has no use for the
// 128x256 CTA, whose ring only fits a 1-byte pair. The big entries carry the
// fast variant; the cp.async ladder downgrades to the non-fast twin for
// crosswise staging at its resolver.
//
// The congruous ladder: the shared six plus the kK=32 twins, swept on the
// dual-congruous (NT) route. Order is load-bearing — dispatch_tile takes the
// first entry whose (class, stages, kK) matches. The small CTA's 16-warp
// widening is not an entry here: it is a resolver substitution (small_16w_t)
// so the same key stays legal on 1-byte operands.
using TileManifest = tuple_cat_t<
    TileManifestCross,
    std::tuple<Tile_64x64x32_W16x32_S2, Tile_64x64x32_W16x32_S3,
               Tile_128x64x32_W32x32_S2, Tile_128x128x32_W32x32_S2,
               Tile_128x128x32_W64x32_S3, Tile_64x128x32_W32x32_S2,
               Tile_64x128x32_W32x32_S3>>;

// The 1-byte ladder: the shared six plus the wide CTA at kK 64, and the
// 32-deep-ring kK=32 big CTA. The kK=32 crosswise feed runs its packed grid
// at half bus, and the kK=32 narrow twin measured that penalty losing
// everywhere (-8..-34% vs its own kK=64 twin across m=8192/16384, and the
// model mis-picked it on unmeasured bands) — it stays off this ladder; the
// 128x128x32 S3, whose deeper 32KB ring exactly reclaims its output, is the
// one kK=32 point that pays. The S2 big twin (24KB ring) still cannot
// reclaim its 32KB output — it waits on a direct-store epilogue. The
// 16-warp small substitution stays off this ladder by rule (see
// warp_widened_t).
using TileManifestByte =
    tuple_cat_t<TileManifestCross,
                std::tuple<Tile_128x256x64_W64x32_S2,
                           Tile_128x128x32_W64x32_S3>>;

// How many operands take that direct path (0 = dual-congruous NT). The
// planner's crosswise field and the launcher's ladder selection are this one
// number, asked of the layout tags rather than re-derived from trans flags.
template <typename LayoutA, typename LayoutB>
constexpr int crosswise_of() {
    return (direct_a<LayoutA>() ? 1 : 0) + (direct_b<LayoutB>() ? 1 : 0);
}

// Which of the ladders a (staging path, operand widths) pair selects —
// the one rule behind both the type-level alias and the planner's runtime
// lookup, so the two cannot disagree about which tiles a plan may reach.
// 1-byte pairs keep their own ladder regardless of staging — the wide CTA
// rides it for the congruous route, and crosswise staging measures into it
// too (its ring is (stages+1)*kK*(bm+bn) = 72KB at 512 threads, and both
// direct sides fit the packed carry: 256 units for the 128-row side, an
// exact 512 for the 256-row one); no kK=32 tile pays on a 1-byte pair (see
// TileManifestByte). Crosswise staging keeps the conservative six for the
// 2-byte and mixed widths (different staging budget); a mixed width pair
// rides the congruous ladder — the predicated skip carries its thinner
// 1-byte bus, the small CTA widens on it, and every kK=32 twin's ring and
// reclaim budget hold at the mixed widths (the big twin's 32KB output
// against its 48KB ring included).
enum class ManifestKind { kCrosswise, kTwoByte, kMixed, kByte };

constexpr ManifestKind manifest_kind(bool crosswise_staging, int ba, int bb) {
    if (ba == 1 && bb == 1) return ManifestKind::kByte;
    if (crosswise_staging) return ManifestKind::kCrosswise;
    if (ba == 2 && bb == 2) return ManifestKind::kTwoByte;
    if (ba + bb == 3) return ManifestKind::kMixed;
    return ManifestKind::kCrosswise;
}

template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB>
constexpr ManifestKind manifest_kind_of() {
    return manifest_kind(crosswise_of<LayoutA, LayoutB>() != 0,
                         (int)sizeof(ElemA), (int)sizeof(ElemB));
}

// The manifest a given operand pair and staging selects over: the kind above,
// mapped to its ladder (kCrosswise is the fallback, so it needs no arm).
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB>
using manifest_for = std::conditional_t<
    manifest_kind_of<ElemA, ElemB, LayoutA, LayoutB>() ==
            ManifestKind::kTwoByte ||
        manifest_kind_of<ElemA, ElemB, LayoutA, LayoutB>() ==
            ManifestKind::kMixed,
    TileManifest,
    std::conditional_t<
        manifest_kind_of<ElemA, ElemB, LayoutA, LayoutB>() == ManifestKind::kByte,
        TileManifestByte, TileManifestCross>>;

template <typename ElemA_, typename ElemB_, typename LayoutA_, typename LayoutB_,
          typename Tile_, typename LayoutOut_ = RowMajor,
          typename OutT_ = __nv_bfloat16, bool StreamOut_ = false,
          bool UseTma_ = false, bool UseMxMma_ = false,
          bool StoreWriteThrough_ = false>
struct GemmPolicy {
    using Tile = Tile_;
    using Traits = GemmTraits<ElemA_, ElemB_, typename Tile_::CtaShape,
                              typename Tile_::WarpShape, Tile_::kStages,
                              UseMxMma_>;
    using LayoutTagA = LayoutA_;
    using LayoutTagB = LayoutB_;
    // Output orientation (CUTLASS LayoutC): direction in the type, stride
    // in GemmParams::out_ld. OutT: bf16 (fused-linear convention) or fp32
    // (accumulated outputs, e.g. training dX/dW).
    using LayoutTagOut = LayoutOut_;
    using OutT = OutT_;
    static constexpr bool kStreamOut = StreamOut_;
    // __stwt write-through store: the fused-linear output is read-once, so
    // keeping it out of L2 reserves the cache for reused weights/activations.
    // Separate from kStreamOut (__stcs, evict-first); write-through wins
    // when both are set.
    static constexpr bool kStoreWriteThrough = StoreWriteThrough_;
    // TMA staging (sm_90+): congruous-only by construction — these policies
    // are instantiated solely for dual-congruous layout pairs with aligned
    // operands; staging layouts and fragment addressing are identical, only
    // the load/wait discipline changes (tma.cuh).
    static constexpr bool kUseTma = UseTma_;
    static_assert(!UseTma_ || (sizeof(ElemA_) <= 2 && sizeof(ElemB_) <= 2),
                  "TMA staging covers the 1-/2-byte congruous dtypes");
    using Smem = GemmSmem<Traits, LayoutA_, LayoutB_>;
    // TMA budgets the 1024B ring-base alignment pad plus the full/empty
    // mbarrier pair per ring slot (tma.cuh); the residency hint stays
    // ring-based.
    static constexpr int kTmaExtra =
        UseTma_ ? 1024 + 2 * (Tile_::kStages + 1) * 8 : 0;
    // Flattened for __launch_bounds__, which takes no dependent type names.
    static constexpr int kCtaThreads = Traits::kCtaThreads;
    static constexpr int kMinCtas = Smem::kMinCtas;
    static constexpr int kSmemBytes = Smem::kBytes + kTmaExtra;
};

}  // namespace gemm
}  // namespace astrai
