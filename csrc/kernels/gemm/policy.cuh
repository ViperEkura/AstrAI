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

// Layout-aware shared-memory budget and occupancy hint. Every operand ring
// holds kStages+1 buffers: the load for tile i+kStages targets slot
// (i-1)%(kStages+1) — already consumed — so neither load path needs a
// post-compute barrier (one __syncthreads per k-tile). The 48KB static
// watermark picks the resident-CTA hint for __launch_bounds__.
template <typename Traits, typename LayoutA, typename LayoutB>
struct GemmSmem {
    // Crosswise (direct-load) operands: A ColMajor storage, B RowMajor
    // storage (B's tag is relative to the canonical [K][N]).
    static constexpr bool kDirectA = std::is_same_v<LayoutA, ColMajor>;
    static constexpr bool kDirectB = std::is_same_v<LayoutB, RowMajor>;
    static constexpr int kRingDepth = Traits::kStages + 1;
    static constexpr int kBytes =
        ring_smem_bytes(Traits::kBlockM, Traits::kBlockN, Traits::kK,
                        Traits::kStages, Traits::kElemBytesA,
                        Traits::kElemBytesB);
    static constexpr int kMinCtas = kBytes <= 48 * 1024 ? 2 : 1;
};

// Tile recipe (CUTLASS-style configuration type): one named bundle of CTA
// shape, warp tiling, pipeline depth and loop specialization. Extending the
// launch ladder = one alias here + one planner branch (never re-spelled
// positional ints).
template <typename CtaShape_, typename WarpShape_, int Stages_, bool FastLoop_>
struct GemmTileConfig {
    using CtaShape = CtaShape_;
    using WarpShape = WarpShape_;
    static constexpr int kStages = Stages_;
    static constexpr bool kFastLoop = FastLoop_;
};

// Tile recipes, named Tile_<cta M>x<N>x<kK>_W<warp M>x<N>_S<stages>[_Fast]:
// the CTA Shape, the warp Shape, the ring depth and the loop specialization,
// in the order GemmTileConfig carries them. An absent _Fast is the
// non-fast (predicated) twin, which crosswise staging needs.
//
// kK is capped at 128/elem_bytes by the staging swizzle: ComposedLayout
// requires kRowShift = kShift - log2(kChunks) >= 0 with kChunks =
// kK*elem/16, so Swizzle<4,3> tops 2-byte operands out at kK 64 while
// 1-byte operands reach 128. Similarly the load path needs
// kTileLines*kChunks % kThreads == 0 with a power-of-two chunks-per-thread,
// so a 1-byte line holds half as many 16B chunks as a 2-byte one and the
// widest warp tilings do not instantiate for 1-byte operands.
using Tile_128x128x64_W64x32_S2 =
    GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 2, false>;
using Tile_128x128x64_W64x32_S2_Fast =
    GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 2, true>;
using Tile_128x128x64_W64x32_S3_Fast =
    GemmTileConfig<Shape<128, 128, 64>, Shape<64, 32>, 3, true>;
using Tile_128x64x64_W32x32_S2_Fast =
    GemmTileConfig<Shape<128, 64, 64>, Shape<32, 32>, 2, true>;
using Tile_128x64x64_W32x32_S3_Fast =
    GemmTileConfig<Shape<128, 64, 64>, Shape<32, 32>, 3, true>;
using Tile_64x64x64_W16x32_S2_Fast =
    GemmTileConfig<Shape<64, 64, 64>, Shape<16, 32>, 2, true>;
using Tile_64x64x64_W16x32_S3_Fast =
    GemmTileConfig<Shape<64, 64, 64>, Shape<16, 32>, 3, true>;
// Deep-ring twins. The 64x64 ring is the only class where s4/s5 survive the
// per-block smem opt-in ceiling on a 2-byte pair (80KB / 96KB against 99KB;
// 128x64 tops out at s3 and 128x128 at s2), so the deep ring is testable
// here and nowhere else. The compiled-in rows all carry s2/s3 and the sweep
// measured the deep rings a wash (s2..s5 within 1-2% at this tile), so only
// the row-file route reaches one.
using Tile_64x64x64_W16x32_S4_Fast =
    GemmTileConfig<Shape<64, 64, 64>, Shape<16, 32>, 4, true>;
using Tile_64x64x64_W16x32_S5_Fast =
    GemmTileConfig<Shape<64, 64, 64>, Shape<16, 32>, 5, true>;
// 16 warps per CTA on the small geometry (16x16 warp tiles, 512 threads):
// more parallel slack over the same 64x64x64 ring, for the underfed shapes.
using Tile_64x64x64_W16x16_S2_Fast =
    GemmTileConfig<Shape<64, 64, 64>, Shape<16, 16>, 2, true>;
using Tile_64x64x64_W16x16_S3_Fast =
    GemmTileConfig<Shape<64, 64, 64>, Shape<16, 16>, 3, true>;
// kK=32 twins: the measured k-tile-depth winner on most shapes, since a
// 64-deep k-tile spends ring budget and issue slots the short K loop cannot
// use.
using Tile_64x64x32_W16x32_S2_Fast =
    GemmTileConfig<Shape<64, 64, 32>, Shape<16, 32>, 2, true>;
using Tile_64x64x32_W16x32_S3_Fast =
    GemmTileConfig<Shape<64, 64, 32>, Shape<16, 32>, 3, true>;
using Tile_128x64x32_W32x32_S2_Fast =
    GemmTileConfig<Shape<128, 64, 32>, Shape<32, 32>, 2, true>;
using Tile_128x128x32_W64x32_S2_Fast =
    GemmTileConfig<Shape<128, 128, 32>, Shape<64, 32>, 2, true>;
using Tile_128x128x32_W64x32_S3_Fast =
    GemmTileConfig<Shape<128, 128, 32>, Shape<64, 32>, 3, true>;
// 1-byte operands only: the ring is 147KB for a 2-byte pair (past the smem
// opt-in ceiling) against 74KB for a 1-byte one.
using Tile_128x256x64_W64x32_S2_Fast =
    GemmTileConfig<Shape<128, 256, 64>, Shape<64, 32>, 2, true>;

// CTA class of a tile config, derived from its CTA geometry — one axis of
// the dispatch key the launch ladders select on (GemmPlan in gemm.cuh).
enum class TileClass { kSmall64, kNarrow128x64, kBig128, kWide128x256 };

template <typename Tile>
constexpr TileClass tile_class() {
    if constexpr (Tile::CtaShape::kM == 128 && Tile::CtaShape::kN == 256)
        return TileClass::kWide128x256;
    else if constexpr (Tile::CtaShape::kM == 128 && Tile::CtaShape::kN == 128)
        return TileClass::kBig128;
    else if constexpr (Tile::CtaShape::kM == 128 && Tile::CtaShape::kN == 64)
        return TileClass::kNarrow128x64;
    else
        return TileClass::kSmall64;
}

// CTA geometry per dispatch class — the inverse of tile_class, and the one
// home for the class -> (M, N) numbers the host plan table prices rows with
// (plan_row_geometry in plan_table.h). Indexed by TileClass, enum order.
inline constexpr int kTileClassCta[][2] = {
    {64, 64},    // kSmall64
    {128, 64},   // kNarrow128x64
    {128, 128},  // kBig128
    {128, 256},  // kWide128x256
};
static_assert((int)TileClass::kSmall64 == 0 &&
                  (int)TileClass::kNarrow128x64 == 1 &&
                  (int)TileClass::kBig128 == 2 &&
                  (int)TileClass::kWide128x256 == 3,
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
static_assert(cta_matches_class<Tile_64x64x64_W16x32_S2_Fast>() &&
                  cta_matches_class<Tile_128x64x64_W32x32_S2_Fast>() &&
                  cta_matches_class<Tile_128x128x64_W64x32_S2_Fast>() &&
                  cta_matches_class<Tile_128x256x64_W64x32_S2_Fast>(),
              "kTileClassCta must mirror the tiles' CTA shapes");

// Tuple concatenation, so a manifest reads as "the shared ladder plus my own
// additions" instead of re-listing the shared entries — the prefix relationship
// between the ladders is then structural, not a copy that can drift.
template <typename... Ts>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<Ts>()...));

// The shared ladder: the geometries every staging path instantiates, plus the
// 64x64 deep-ring twins (see the alias above — they are here rather than in
// one ladder so that every manifest carries them and a row naming s4/s5
// always finds a tile, whatever the staging path).
// load_operand_tile stages a crosswise operand as kK lines of (M or N)*elem/16
// chunks and needs that product to divide its thread count with a
// power-of-two quotient, which the kK=32 twins and the 16-warp small CTA both
// miss once the other operand is 1-byte; the wide CTA is 1-byte-only besides.
// Every other manifest contains these, so this is the base.
using TileManifestCross = std::tuple<
    Tile_128x128x64_W64x32_S2_Fast, Tile_128x128x64_W64x32_S3_Fast,
    Tile_128x64x64_W32x32_S2_Fast, Tile_128x64x64_W32x32_S3_Fast,
    Tile_64x64x64_W16x32_S2_Fast, Tile_64x64x64_W16x32_S3_Fast,
    Tile_64x64x64_W16x32_S4_Fast, Tile_64x64x64_W16x32_S5_Fast>;

// The dispatch manifests (CUTLASS builder-table style): every recipe the
// launch ladders select over, keyed by the plan's CTA class, ring depth and
// k-tile depth. Split by operand width because the reclaim budget
// (bm*bn*sizeof(OutT) <= ring) and the load-path divisibility both bind
// harder on the narrowest ring: the byte manifest cannot carry the kK=32
// big CTA (32KB of output over a 24KB ring) or the 16-warp small CTA (512
// threads against 256 chunks), and the two-byte manifest has no use for the
// 128x256 CTA, whose ring only fits a 1-byte pair. The big entries carry the
// fast variant; the cp.async ladder downgrades to the non-fast twin for
// crosswise staging at its resolver.
//
// The congruous ladder: the shared six plus the kK=32 twins and the 16-warp
// small CTA, swept on the dual-congruous (NT) route. Order is load-bearing —
// dispatch_tile takes the first entry whose (class, stages, kK) matches, and
// the 16-warp small CTA shares that key with the 32-warp one above it, so it
// is reached only when the 32-warp twin's resolver declines.
using TileManifest = tuple_cat_t<
    TileManifestCross,
    std::tuple<Tile_64x64x64_W16x16_S2_Fast, Tile_64x64x64_W16x16_S3_Fast,
               Tile_64x64x32_W16x32_S2_Fast, Tile_64x64x32_W16x32_S3_Fast,
               Tile_128x64x32_W32x32_S2_Fast, Tile_128x128x32_W64x32_S2_Fast,
               Tile_128x128x32_W64x32_S3_Fast>>;

// The 1-byte ladder: the shared six plus the wide CTA, every one of them at
// kK 64. No kK=32 tile survives a 1-byte operand — a line then holds half as
// many 16B chunks, so kTileLines*kChunks falls below the thread count (a
// 64x64x32 tile has 128 chunks against 256 threads) and the load path cannot
// divide them; the one kK=32 geometry that does divide, 128x128x32, cannot
// reclaim its own 32KB output tile from a 24KB ring.
using TileManifestByte =
    tuple_cat_t<TileManifestCross, std::tuple<Tile_128x256x64_W64x32_S2_Fast>>;

// The manifest a given operand pair and staging selects over. The widening
// ladder is only legal where it was measured, so everything else keeps the
// conservative six: crosswise staging (the staging budget differs), a mixed
// width pair (one 1-byte operand halves the chunk count the same way),
// 1-byte pairs (no kK=32 tile divides, above), and 2-byte pairs, which get
// the full ladder.
template <typename ElemA, typename ElemB, typename LayoutA, typename LayoutB>
using manifest_for = std::conditional_t<
    std::is_same_v<LayoutA, ColMajor> || std::is_same_v<LayoutB, RowMajor>,
    TileManifestCross,
    std::conditional_t<sizeof(ElemA) == 2 && sizeof(ElemB) == 2, TileManifest,
                       std::conditional_t<sizeof(ElemA) == 1 &&
                                              sizeof(ElemB) == 1,
                                          TileManifestByte,
                                          TileManifestCross>>>;

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
    static constexpr bool kFastLoop = Tile_::kFastLoop;
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
