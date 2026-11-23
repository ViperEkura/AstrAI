#pragma once

#include <datatype/dequant.cuh>
#include <mma/ldmatrix.cuh>
#include <mma/mma.cuh>
#include <policy/gemm.cuh>
#include <utils/tensor.cuh>

namespace astrai {
namespace gemm {

template <typename Policy> struct GemmTileLayout {
    using Traits = typename Policy::Traits;
    using LayoutA = typename Policy::LayoutTagA;
    using LayoutB = typename Policy::LayoutTagB;
    using Smem = GemmSmem<Traits, LayoutA, LayoutB>;
    static constexpr bool kUseTma = Policy::kUseTma;
    static_assert(!kUseTma || (!Smem::kDirectA && !Smem::kDirectB),
                  "TMA staging is congruous-only");
    /*
   * The mma runs on the promoted MmaT (policy/gemm/traits.cuh); a lone int8/fp8
   * side expands in-register between smem read and mma (kDequant* marks the
   * insert, dequant.cuh). AccT rides the mma cell: fp32, int32 for s8.
   */
    using ElemA = typename Traits::ElemA;
    using ElemB = typename Traits::ElemB;
    using MmaT = typename Traits::MmaT;
    using MmaOp = typename Traits::MmaOp;
    using AccT = typename Traits::AccT;
    static constexpr bool kDequantA = Traits::kDequantA;
    static constexpr bool kDequantB = Traits::kDequantB;
    using DequantA = quant::DequantPair<ElemA, MmaT>;
    using DequantB = quant::DequantPair<ElemB, MmaT>;
    static constexpr int kBlockM = Traits::kBlockM;
    static constexpr int kBlockN = Traits::kBlockN;
    static constexpr int kTile = Traits::kTile;
    static constexpr int kStages = Traits::kStages;
    static constexpr int kCtaThreads = Traits::kCtaThreads;
    static constexpr bool kDirectA = Smem::kDirectA;
    static constexpr bool kDirectB = Smem::kDirectB;
    /*
   * Crosswise staging splits by width: 16-bit cp.async into the transposed
   * [kTile][rows] tile (ldmatrix.trans, pipelineable); 8-bit keeps the
   * synchronous LDG+PRMT path into the canonical tile (no trans ldmatrix).
   */
    static constexpr bool kSyncA = kDirectA && sizeof(ElemA) == 1;
    static constexpr bool kTransA = kDirectA && sizeof(ElemA) == 2;
    static constexpr bool kSyncB = kDirectB && sizeof(ElemB) == 1;
    static constexpr bool kTransB = kDirectB && sizeof(ElemB) == 2;
    /*
   * Dequant inserts are int8-storage only; a dequantized side never rides
   * the 16-bit-only trans staging — kTrans* is already false there.
   */
    static_assert(!kDequantA || sizeof(ElemA) == 1, "in-register dequant targets 1-byte storage");
    static_assert(!kDequantB || sizeof(ElemB) == 1, "in-register dequant targets 1-byte storage");
    static_assert(kStages >= 1 && kStages <= 8, "FP8 GEMM k_stages must be in [1, 8]");
    /*
   * CTA = (BlockM/WarpM) x (BlockN/WarpN) warps, each warp computing
   * kMt x kNt m16n8k{kMmaK} MMAs. Rings rotate kStages+1 buffers (see
   * GemmSmem) — one __syncthreads per k-tile.
   */
    static constexpr int kMt = Traits::kMt;             // 16-row MMA tiles per warp
    static constexpr int kNt = Traits::kNt;             // 8-col MMA tiles per warp
    static constexpr int kSegs = kTile / Traits::kMmaK; // mma-sized k segments
    static constexpr int kARing = Smem::kRingDepth;
    static constexpr int kBRing = Smem::kRingDepth;

    /* Staging layout types also define the fragment address calculations. */
    static constexpr int kChunksA = kTile / (16 / (int)sizeof(ElemA));
    static constexpr int kChunksB = kTile / (16 / (int)sizeof(ElemB));
    static constexpr int kChunksAT = kBlockM / (16 / (int)sizeof(ElemA));
    static constexpr int kChunksBT = kBlockN / (16 / (int)sizeof(ElemB));
    using SmemLayoutA =
        decltype(composition(Swizzle<log2_const<kChunksA>::value, 3>{},
                             Layout<Shape<kBlockM, kChunksA>, Stride<kChunksA, 1>>{}));
    using SmemLayoutB =
        decltype(composition(Swizzle<log2_const<kChunksB>::value, 3>{},
                             Layout<Shape<kBlockN, kChunksB>, Stride<kChunksB, 1>>{}));
    using SmemLayoutATrans = decltype(composition(
        Swizzle < log2_const<kChunksAT<8 ? kChunksAT : 8>::value, log2_const<kChunksAT>::value>{},
        Layout<Shape<kTile, kChunksAT>, Stride<kChunksAT, 1>>{}));
    using SmemLayoutBTrans = decltype(composition(
        Swizzle < log2_const<kChunksBT<8 ? kChunksBT : 8>::value, log2_const<kChunksBT>::value>{},
        Layout<Shape<kTile, kChunksBT>, Stride<kChunksBT, 1>>{}));

    /* Packed crosswise bytes use the 16-bit transposed fragment mapping. */
    static constexpr int kPackChunksA = kBlockM / 8; // 16B chunks per row
    static constexpr int kPackChunksB = kBlockN / 8;
    using SmemLayoutAPack =
        decltype(composition(Swizzle < log2_const<kPackChunksA<8 ? kPackChunksA : 8>::value,
                                                  log2_const<kPackChunksA>::value>{},
                             Layout<Shape<kTile / 2, kPackChunksA>, Stride<kPackChunksA, 1>>{}));
    using SmemLayoutBPack =
        decltype(composition(Swizzle < log2_const<kPackChunksB<8 ? kPackChunksB : 8>::value,
                                                  log2_const<kPackChunksB>::value>{},
                             Layout<Shape<kTile / 2, kPackChunksB>, Stride<kPackChunksB, 1>>{}));

    /*
   * Packed staging needs: a 1-byte crosswise operand the dequant readers do
   * not own (they read the canonical tile), even kTile, power-of-two chunks
   * >= 8 (the swizzle row budget: 8 rows per ldmatrix matrix).
   */
    static constexpr bool kPackOkA = sizeof(ElemA) == 1 && (kTile % 2 == 0) && kPackChunksA >= 8 &&
                                     (kPackChunksA & (kPackChunksA - 1)) == 0;
    static constexpr bool kPackOkB = sizeof(ElemB) == 1 && (kTile % 2 == 0) && kPackChunksB >= 8 &&
                                     (kPackChunksB & (kPackChunksB - 1)) == 0;
    static constexpr bool kPackA = kDirectA && !kDequantA && kPackOkA;
    static constexpr bool kPackB = kDirectB && !kDequantB && kPackOkB;

    /* Ring types bind the staging layout, element type, and pointer. */
    using StagedLayoutA =
        std::conditional_t<kPackA,
                           SmemLayoutAPack,
                           std::conditional_t<kTransA, SmemLayoutATrans, SmemLayoutA>>;
    using StagedLayoutB =
        std::conditional_t<kPackB,
                           SmemLayoutBPack,
                           std::conditional_t<kTransB, SmemLayoutBTrans, SmemLayoutB>>;
    using RingA = Tensor<PtrEngine<ElemA>, RingLayout<StagedLayoutA, kARing>>;
    using RingB = Tensor<PtrEngine<ElemB>, RingLayout<StagedLayoutB, kBRing>>;
    using TileA = Tensor<PtrEngine<ElemA>, StagedLayoutA>;
    using TileB = Tensor<PtrEngine<ElemB>, StagedLayoutB>;

    /*
   * The epilogue scatters the output tile into the reclaimed operand rings
   * — the tile must fit them; both orchestrators static_assert this (the
   * launchers' reclaim_fits prices the same rule).
   */
    static constexpr bool kOutputReclaimsRings =
        kBlockM * kBlockN * sizeof(typename Policy::OutT) <=
        RingA::Layout::kTotalBytes + RingB::Layout::kTotalBytes;

    /*
   * The warp's accumulator: typed C cells on a (mt, nt) grid — semantic
   * coordinates all the way to the mma (no pointer decay at the fma seam).
   */
    using AccTensor = typename Traits::AccTensor;

    // Per-stage byte strides.
    static constexpr int kAStageBytes = RingA::Layout::kStageBytes;
    static constexpr int kBStageBytes = RingB::Layout::kStageBytes;
};

template <typename Policy> struct GemmWarpFragments : GemmTileLayout<Policy> {
    using Layout = GemmTileLayout<Policy>;
    using Traits = typename Layout::Traits;
    using ElemA = typename Layout::ElemA;
    using ElemB = typename Layout::ElemB;
    using MmaOp = typename Layout::MmaOp;
    using DequantA = typename Layout::DequantA;
    using DequantB = typename Layout::DequantB;
    using SmemLayoutA = typename Layout::SmemLayoutA;
    using SmemLayoutB = typename Layout::SmemLayoutB;
    using SmemLayoutATrans = typename Layout::SmemLayoutATrans;
    using SmemLayoutBTrans = typename Layout::SmemLayoutBTrans;
    using SmemLayoutAPack = typename Layout::SmemLayoutAPack;
    using SmemLayoutBPack = typename Layout::SmemLayoutBPack;
    using TileA = typename Layout::TileA;
    using TileB = typename Layout::TileB;
    using AccTensor = typename Layout::AccTensor;
    using Layout::kBlockM;
    using Layout::kBlockN;
    using Layout::kDequantA;
    using Layout::kDequantB;
    using Layout::kMt;
    using Layout::kNt;
    using Layout::kPackA;
    using Layout::kPackB;
    using Layout::kTile;
    using Layout::kTransA;
    using Layout::kTransB;
    int a_row0;
    int b_row0;
    DEVICE_FORCEINLINE GemmWarpFragments(int a, int b) : a_row0(a), b_row0(b) {}
    /*
   * One ldmatrix.x4 payload covering two adjacent n8 B fragments (the
   * kPairB fold): cell(i) picks the fragment nt consumes — the pairing
   * lives in the type, not fma-seam pointer arithmetic. (Not "half":
   * nvcc reserves that name for the fp16 type.)
   */
    struct BFragPair : ArrayEngine<unsigned, 4> {
        DEVICE_FORCEINLINE typename MmaOp::BFrag cell(int i) const {
            return {storage[2 * i + 0], storage[2 * i + 1]};
        }
    };

    /* Lane offsets derive from staging layouts and feed ldmatrix byte addresses.
   */
    template <typename SmemLayoutT, typename ElemT>
    static DEVICE_FORCEINLINE unsigned canonical_lane_off(int64_t row, int chunk_half, int lane) {
        constexpr int kChunkShift = log2_const<16 / sizeof(ElemT)>::value;
        const unsigned lswz =
            static_cast<unsigned>(((lane & 7) >> SmemLayoutT::kRowShift) & SmemLayoutT::kMask);
        return static_cast<unsigned>((row * kTile + ((chunk_half ^ lswz) << kChunkShift)) *
                                     sizeof(ElemT));
    }

    /* Crosswise 16-bit fragments use K rows and XOR-swizzled M/N windows. */
    template <typename SmemLayoutT, typename ElemT>
    static DEVICE_FORCEINLINE unsigned trans_lane_off(int krow, int col, int block_extent) {
        const unsigned lswz =
            static_cast<unsigned>((krow >> SmemLayoutT::kRowShift) & SmemLayoutT::kMask);
        return static_cast<unsigned>(
            ((int64_t)krow * block_extent + (((col >> 3) ^ lswz) << 3) + (col & 7)) *
            sizeof(ElemT));
    }

    static constexpr int kChunkElems = 16 / sizeof(ElemA);
    static constexpr int kChunkShift = log2_const<kChunkElems>::value;
    DEVICE_FORCEINLINE unsigned a_lane_off(int lane) const {
        /*
     * Stage-relative, loop-invariant per-lane base; A's fragment row
     * carries the +8-row (rh8) and +1-chunk (rh16) halves.
     */
        return canonical_lane_off<SmemLayoutA, ElemA>(a_row0 + ((lane >> 3) & 1) * 8 + (lane & 7),
                                                      lane >> 4, lane);
    }
    DEVICE_FORCEINLINE unsigned b_lane_off(int lane) const {
        // ldmatrix (non-dequant) B addressing: byte offsets in ElemB units.
        return canonical_lane_off<SmemLayoutB, ElemB>(b_row0 + (lane & 7), (lane >> 3) & 1, lane);
    }
    /* Paired B x4 needs a positive swizzle row shift; K=128 byte tiles use x2. */
    static constexpr unsigned kMtStep = 16u * kTile * sizeof(ElemA); // m-tile row step
    static constexpr unsigned kNtStep = 8u * kTile * sizeof(ElemB);  // n-tile row step
    static constexpr unsigned kSegXorA = (unsigned)Traits::kMmaK * sizeof(ElemA);
    static constexpr unsigned kSegXorB = (unsigned)Traits::kMmaK * sizeof(ElemB);
    static constexpr bool kPairB = !kDequantB && !kPackB && SmemLayoutB::kRowShift > 0;
    static_assert(!kPairB || kTile * sizeof(ElemB) <= 64,
                  "paired B x4 is unsupported for the kTile=128 byte layout");
    static_assert(!kPairB || kNt % 2 == 0, "B pairing needs even kNt");
    static_assert(!kPairB || !kTransB, "2-byte crosswise B never pairs (chunk budget)");
    static constexpr unsigned kPairStep = 16u * kTile * sizeof(ElemB); // nt-pair row step
    DEVICE_FORCEINLINE unsigned b4_lane_off(int lane) const {
        return b_lane_off(lane) + (lane >> 4) * kPairStep / 2;
    }

    static constexpr unsigned kMtXor = 32u; // m16 = 2 chunks
    static constexpr unsigned kNtXor = 16u; // n8 = 1 chunk
    static constexpr unsigned kTransSegA = (unsigned)Traits::kMmaK * kBlockM * sizeof(ElemA);
    static constexpr unsigned kTransSegB = (unsigned)Traits::kMmaK * kBlockN * sizeof(ElemB);
    DEVICE_FORCEINLINE unsigned a_trans_lane_off(int lane) const {
        /*
     * x4 matrix order must match the mma's A-register order (m+8 rides
     * reg1, k+8 reg2): lanes 8-15 step the m+8 chunk, lanes 16-31 the
     * k+8 row half.
     */
        return trans_lane_off<SmemLayoutATrans, ElemA>((lane & 7) + ((lane >> 4) << 3),
                                                       a_row0 + (((lane >> 3) & 1) << 3), kBlockM);
    }
    DEVICE_FORCEINLINE unsigned b_trans_lane_off(int lane) const {
        // nt windows step by kNtXor at call sites (col stays at b_row0).
        return trans_lane_off<SmemLayoutBTrans, ElemB>((lane & 7) + (((lane >> 3) & 1) << 3),
                                                       b_row0, kBlockN);
    }

    /*
   * Packed-grid lane offsets: the trans formula one width down (a packed
   * row is 16-bit units, 8 per 16B chunk; the row extent is in UNITS). The
   * lane contract, XOR steps and per-k-segment ADD are the trans reader's:
   * 16 packed rows = one mma k-segment.
   */
    DEVICE_FORCEINLINE unsigned a_pack_lane_off(int lane) const {
        return trans_lane_off<SmemLayoutAPack, unsigned short>(
            (lane & 7) + ((lane >> 4) << 3), a_row0 + (((lane >> 3) & 1) << 3), kBlockM);
    }
    DEVICE_FORCEINLINE unsigned b_pack_lane_off(int lane) const {
        return trans_lane_off<SmemLayoutBPack, unsigned short>(
            (lane & 7) + (((lane >> 3) & 1) << 3), b_row0, kBlockN);
    }

    /* Convert packed INT8 A pairs to BF16 in ldmatrix register order. */
    DEVICE_FORCEINLINE void
    load_a_frags_at(typename MmaOp::AFrag& frag, TileA stage, int k_seg, int mt, int lane) const {
        const int q = lane >> 2, c2 = (lane & 3) * 2;
        const int row = a_row0 + mt * 16 + q;
        const ElemA* p0 = stage(row, k_seg * 16 + c2);
        const ElemA* p8 = stage(row + 8, k_seg * 16 + c2);
        frag[0] = DequantA::pair(*(const unsigned short*)p0);
        frag[1] = DequantA::pair(*(const unsigned short*)p8);
        frag[2] = DequantA::pair(*(const unsigned short*)(p0 + 8));
        frag[3] = DequantA::pair(*(const unsigned short*)(p8 + 8));
    }

    /*
   * One k_seg's B-fragment loads, shared by the initial fill and the
   * double-buffer's next-seg fill; the unused frag2/frag4 is never touched.
   */
    DEVICE_FORCEINLINE void load_b_frags(typename MmaOp::BFrag (&frag2)[kNt],
                                         BFragPair (&frag4)[kNt / 2],
                                         unsigned seg_base) const {
#pragma unroll
        for (int p = 0; p < kNt / 2; ++p) {
            if constexpr (kTransB || kPackB) {
                /*
         * Trans/packed: each x2.trans reads 16 k rows at one n chunk;
         * nt windows step one XORed chunk.
         */
                astrai::ldmatrix_x2_lane<true>(frag2[p * 2], seg_base ^ (unsigned)(p * 2 * kNtXor));
                astrai::ldmatrix_x2_lane<true>(frag2[p * 2 + 1],
                                               seg_base ^ (unsigned)((p * 2 + 1) * kNtXor));
            } else if constexpr (kPairB) {
                astrai::ldmatrix_x4_lane(frag4[p], seg_base + p * kPairStep);
            } else {
                astrai::ldmatrix_x2_lane(frag2[p * 2], seg_base + p * 2 * kNtStep);
                astrai::ldmatrix_x2_lane(frag2[p * 2 + 1], seg_base + (p * 2 + 1) * kNtStep);
            }
        }
    }

    /* Convert packed INT8 B pairs to BF16 in ldmatrix register order. */
    DEVICE_FORCEINLINE void load_b_frags_at(typename MmaOp::BFrag (&frag2)[kNt],
                                            BFragPair (&frag4)[kNt / 2],
                                            TileB stage,
                                            int k_seg,
                                            unsigned seg_base,
                                            int lane) const {
        if constexpr (kDequantB) {
            const int q = lane >> 2, c2 = (lane & 3) * 2;
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int row = b_row0 + nt * 8 + q;
                const ElemB* p0 = stage(row, k_seg * 16 + c2);
                const ElemB* p1 = stage(row, k_seg * 16 + c2 + 8);
                frag2[nt][0] = DequantB::pair(*(const unsigned short*)p0);
                frag2[nt][1] = DequantB::pair(*(const unsigned short*)p1);
            }
        } else {
            load_b_frags(frag2, frag4, seg_base);
        }
    }

    DEVICE_FORCEINLINE void accumulate_segment(AccTensor& acc,
                                               TileA a_tile,
                                               int k_seg,
                                               unsigned a_addr,
                                               int lane,
                                               const typename MmaOp::BFrag (&b_frag)[kNt],
                                               const BFragPair (&b_frag4)[kNt / 2]) const {
        /* Load the next A fragment ahead; dequantized A needs all row fragments
     * first. */
        typename MmaOp::AFrag a_frag[kMt + 1];
        if constexpr (kDequantA) {
#pragma unroll
            for (int mt = 0; mt < kMt; ++mt)
                load_a_frags_at(a_frag[mt], a_tile, k_seg, mt, lane);
        } else if constexpr (kTransA || kPackA) {
            astrai::ldmatrix_x4_lane<true>(a_frag[0], a_addr);
        } else {
            astrai::ldmatrix_x4_lane(a_frag[0], a_addr);
        }
#pragma unroll
        for (int mt = 0; mt < kMt; ++mt) {
            if constexpr (!kDequantA) {
                if (mt + 1 < kMt) {
                    const unsigned a_next = (kTransA || kPackA)
                                                ? (a_addr ^ (unsigned)((mt + 1) * kMtXor))
                                                : (a_addr + (mt + 1) * kMtStep);
                    if constexpr (kTransA || kPackA)
                        astrai::ldmatrix_x4_lane<true>(a_frag[mt + 1], a_next);
                    else
                        astrai::ldmatrix_x4_lane(a_frag[mt + 1], a_next);
                }
            }
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                if constexpr (kPairB)
                    MmaOp::fma(*acc(mt, nt), a_frag[mt], b_frag4[nt >> 1].cell(nt & 1),
                               *acc(mt, nt));
                else
                    MmaOp::fma(*acc(mt, nt), a_frag[mt], b_frag[nt], *acc(mt, nt));
            }
        }
    }
};

} // namespace gemm
} // namespace astrai
