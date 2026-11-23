#pragma once
/*
 * Collective mainloop: smem stage rings, cp.async/TMA stage loads,
 * crosswise LDG+PRMT, per-lane ldmatrix addressing, software-pipelined
 * mma.sync. Addressing scheme + fast-loop peel rationale:
 * docs/developer/kernels/gemm.md.
 */

#include <type_traits>

#include <api/native/gemm.h>
#include <datatype/dequant.cuh>
#include <kernel/gemm/operand.cuh>
#include <memory/pipeline.cuh>
#include <memory/tma.cuh>
#include <mma/ldmatrix.cuh>
#include <mma/mma.cuh>
#include <policy/gemm.cuh>
#include <utils/define.cuh>
#include <utils/tensor.cuh>

namespace astrai {
namespace gemm {

/* TMA full barriers mark ready copies; empty barriers protect slots until consumers finish. */
template <bool kRank3A = false, bool kRank3B = false> struct GemmTmaContext {
    const void* map_a = nullptr;
    const void* map_b = nullptr;
    uint64_t* bars = nullptr;
    int depth = 0; // ring slots (kStages + 1): 2*depth barriers
    int z = 0;     // batch coordinate (rank-3 descriptors)

    DEVICE_FORCEINLINE uint64_t* full(int slot) const { return bars + slot; }
    DEVICE_FORCEINLINE uint64_t* empty(int slot) const { return bars + depth + slot; }
};

template <typename Policy> struct GemmCollectiveMainloop {
    using Traits = typename Policy::Traits;
    using Operands = GemmOperands<Policy>;
    using A = typename Operands::A;
    using B = typename Operands::B;
    static constexpr bool kUseTma = Policy::kUseTma;
    /*
     * The mma runs on the promoted MmaT (policy/gemm/traits.cuh); a lone int8/fp8 side
     * expands in-register between smem read and mma (kDequant* marks the
     * insert, dequant.cuh). AccT rides the mma cell: fp32, int32 for s8.
     */
    using ElemA = typename Traits::ElemA;
    using ElemB = typename Traits::ElemB;
    using MmaT = typename Traits::MmaT;
    using MmaOp = typename Traits::MmaOp;
    static constexpr bool kDequantA = Traits::kDequantA;
    static constexpr bool kDequantB = Traits::kDequantB;
    using DequantA = quant::DequantPair<ElemA, MmaT>;
    using DequantB = quant::DequantPair<ElemB, MmaT>;
    static constexpr int kBlockM = Traits::kBlockM;
    static constexpr int kBlockN = Traits::kBlockN;
    static constexpr int kTile = Traits::kTile;
    static constexpr int kStages = Traits::kStages;
    static constexpr int kMt = Traits::kMt;
    static constexpr int kNt = Traits::kNt;
    static constexpr int kSegs = kTile / Traits::kMmaK;
    static_assert(kStages >= 1 && kStages <= 8);
    /*
     * The epilogue scatters the output tile into the reclaimed operand rings
     * — the tile must fit them; both orchestrators static_assert this (the
     * launchers' reclaim_fits prices the same rule).
     */
    static constexpr bool kOutputReclaimsRings =
        kBlockM * kBlockN * sizeof(typename Policy::OutT) <= A::kBytes + B::kBytes;

    /*
     * The warp's accumulator: typed C cells on a (mt, nt) grid — semantic
     * coordinates all the way to the mma (no pointer decay at the fma seam).
     */
    using AccTensor = typename Traits::AccTensor;

    const typename A::Ring ring_a; // A's stage ring; B carves right past its end
    const typename B::Ring ring_b;
    const ElemA* const a;
    const ElemB* const b;
    const int64_t m, n, k, a_ld, b_ld;
    const int tid;
    const int64_t block_m, block_n;
    const int warp_m, warp_n;
    const int a_row0; // + mt * 16 in the loop
    const int b_row0; // + nt * 8
    const int64_t tile_count;
    /* Aligned interior tiles use unpredicated copies. */
    const bool use_interior_copy;

    __device__ GemmCollectiveMainloop(char* smem,
                                      const ElemA* a,
                                      const ElemB* b,
                                      int64_t m,
                                      int64_t n,
                                      int64_t k,
                                      int64_t a_ld,
                                      int64_t b_ld,
                                      int tid,
                                      int2 block)
        : ring_a(astrai::make_ring<ElemA, typename A::SharedLayout, A::kDepth>(smem)),
          ring_b(astrai::make_ring<ElemB, typename B::SharedLayout, B::kDepth>(smem + A::kBytes)),
          a(a), b(b), m(m), n(n), k(k), a_ld(a_ld), b_ld(b_ld), tid(tid), block_m(block.x),
          block_n(block.y), warp_m((tid >> 5) / Traits::kWarpsN),
          warp_n((tid >> 5) % Traits::kWarpsN), a_row0(warp_m * Traits::kWarpM),
          b_row0(warp_n * Traits::kWarpN), tile_count((k + kTile - 1) / kTile),
          use_interior_copy(!A::kRegisterLoad && !B::kRegisterLoad &&
                            ((int64_t)block.x * kBlockM + kBlockM <= m) &&
                            ((int64_t)block.y * kBlockN + kBlockN <= n) && A::aligned(a, a_ld) &&
                            B::aligned(b, b_ld) && (k % kTile) == 0) {}

    template <bool Interior = false>
    DEVICE_FORCEINLINE void
    copy_async(typename A::Tile tile_a, typename B::Tile tile_b, int64_t k_base) const {
        A::template copy_async<Interior>(tile_a, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
        B::template copy_async<Interior>(tile_b, b, n, k, b_ld, tid, k_base, block_n * kBlockN);
    }

    /*
     * One TMA stage issue, elected-thread only: gate on the slot's empty
     * barrier (skipped on the ring's first sweep), arm the full barrier's
     * byte count, issue both operand boxes (rank rides the context type).
     */
    template <bool kRank3A, bool kRank3B>
    DEVICE_FORCEINLINE void tma_issue_stage(const GemmTmaContext<kRank3A, kRank3B>& tma,
                                            int tile) const {
        const int slot = tile % A::kDepth;
        if (tile >= A::kDepth)
            astrai::mbarrier_wait_parity(tma.empty(slot), (uint32_t)(((tile / A::kDepth) - 1) & 1));
        uint64_t* bar = tma.full(slot);
        mbarrier_arrive_expect_tx(bar, (uint32_t)((uint64_t)A::kStageBytes + B::kStageBytes));
        const int x = (int)((int64_t)tile * kTile);
        astrai::tma_load<kRank3A>(tma.map_a, bar, astrai::stage_of(ring_a, tile).engine.ptr,
                                  x * (int)sizeof(ElemA), (int)(block_m * kBlockM), tma.z);
        astrai::tma_load<kRank3B>(tma.map_b, bar, astrai::stage_of(ring_b, tile).engine.ptr,
                                  x * (int)sizeof(ElemB), (int)(block_n * kBlockN), tma.z);
    }

    /* Prime kStages groups, including empty groups, to keep wait indexing stable. */
    template <bool kRank3A = false, bool kRank3B = false>
    DEVICE_FORCEINLINE void prologue(const GemmTmaContext<kRank3A, kRank3B>& tma = {}) const {
        if constexpr (kUseTma) {
            if (tid == 0) {
#pragma unroll
                for (int stage = 0; stage < kStages; ++stage) {
                    if (stage < tile_count)
                        tma_issue_stage(tma, stage);
                }
            }
            return;
        }
        const astrai::PipelineSync<kStages> pipe;
#pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
            if (stage < tile_count) {
                const auto tile_a = stage_of(ring_a, stage);
                const auto tile_b = stage_of(ring_b, stage);
                const int64_t k_base = (int64_t)stage * kTile;
                if (use_interior_copy)
                    copy_async<true>(tile_a, tile_b, k_base);
                else
                    copy_async(tile_a, tile_b, k_base);
                A::prime_registers(tile_a, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
                B::prime_registers(tile_b, b, n, k, b_ld, tid, k_base, block_n * kBlockN);
            }
            pipe.producer_commit();
        }
    }

    /* Overlap loads with MMA; TMA uses full/empty barriers. */
    template <bool kInterior, bool kTma = false, bool kRank3A = false, bool kRank3B = false>
    DEVICE_FORCEINLINE void run_loop(AccTensor& acc,
                                     const GemmTmaContext<kRank3A, kRank3B>& tma = {}) const {
        const astrai::PipelineSync<kStages> pipe;
        const int lane = tid & 31;
        /* Carry staged load and read pointers across k tiles. */
        typename A::template Prefetch<!kTma> carry_a(ring_a, a, a_ld, block_m * kBlockM, tid,
                                                     kStages);
        typename B::template Prefetch<!kTma> carry_b(ring_b, b, b_ld, block_n * kBlockN, tid,
                                                     kStages);
        typename A::RegisterCarry sync_a;
        typename B::RegisterCarry sync_b;
        const unsigned a_rd0 =
            __cvta_generic_to_shared(ring_a.engine.ptr) +
            (A::kPacked ? a_pack_lane_off(lane)
                        : (A::kTranspose ? a_trans_lane_off(lane) : a_lane_off(lane)));
        const unsigned b_rd0 =
            __cvta_generic_to_shared(ring_b.engine.ptr) +
            (B::kPacked ? b_pack_lane_off(lane)
                        : (B::kTranspose ? b_trans_lane_off(lane)
                                         : (kPairB ? b4_lane_off(lane) : b_lane_off(lane))));
        const unsigned a_rd_end = a_rd0 + (unsigned)(A::kDepth * A::kStageBytes);
        const unsigned b_rd_end = b_rd0 + (unsigned)(B::kDepth * B::kStageBytes);
        unsigned a_rd = a_rd0, b_rd = b_rd0;
        for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
            /*
             * Steady state: exactly kStages-1 younger groups in flight when
             * this fires; the tail's unconditional empty commits keep it true.
             */
            const bool prefetch = tile_index + kStages < tile_count;
            /* TMA waits full slots; cp.async joins the CTA before slot reuse. */
            if constexpr (kTma) {
                astrai::mbarrier_wait_parity(tma.full((int)(tile_index % A::kDepth)),
                                             static_cast<uint32_t>((tile_index / A::kDepth) & 1));
            } else {
                pipe.consumer_wait();
            }

            /*
             * Staging for tile i+kStages (its slot = (i-1)'s, released above):
             * elected thread arms + issues both TMA boxes; 8-bit crosswise
             * issues its LDG.128 runs here (transpose + STS follow the MMA
             * phase, so global latency overlaps tensor-pipe work).
             */
            if constexpr (kTma) {
                if (prefetch && tid == 0)
                    tma_issue_stage(tma, (int)(tile_index + kStages));
            } else if (prefetch) {
                if constexpr (A::kRegisterLoad)
                    sync_a.issue(a, m, k, a_ld, tid, (tile_index + kStages) * kTile,
                                 block_m * kBlockM);
                if constexpr (B::kRegisterLoad)
                    sync_b.issue(b, n, k, b_ld, tid, (tile_index + kStages) * kTile,
                                 block_n * kBlockN);
            }

            const unsigned a_addr = a_rd;
            const unsigned b_addr = b_rd;
            /*
             * Per-k_seg base pair (cuBLAS's scheme): seg s = seg-0 base XOR
             * (s * kSegXor) — one LOP3 per extra seg, never per fragment.
             * Trans tiles step a plain ADD (k rows at fixed stride). Every
             * LDSM below addresses [base + immediate].
             */
            unsigned a_seg[kSegs], b_seg[kSegs];
#pragma unroll
            for (int s = 0; s < kSegs; ++s) {
                a_seg[s] = (A::kTranspose || A::kPacked) ? (a_addr + (unsigned)(s * kTransSegA))
                                                         : (a_addr ^ (unsigned)(s * kSegXorA));
                b_seg[s] = (B::kTranspose || B::kPacked) ? (b_addr + (unsigned)(s * kTransSegB))
                                                         : (b_addr ^ (unsigned)(s * kSegXorB));
            }

            /* Prefetch B fragments by K segment and pair adjacent N cells where valid. */
            typename MmaOp::BFrag b_frag[2][kNt];
            BFragPair b_frag4[2][kNt / 2];
            /*
             * Staged tensor hoisted out of the k_seg/mt loops: the slot pick
             * is a modulo and regressed the dequant readers' register budget.
             */
            const auto b_tile = astrai::stage_of(ring_b, tile_index);
            load_b_frags_at(b_frag[0], b_frag4[0], b_tile, 0, b_seg[0], lane);
#pragma unroll
            for (int k_seg = 0; k_seg < kSegs; ++k_seg) {
                const int bcur = k_seg & 1, bnext = bcur ^ 1;
                if (k_seg + 1 < kSegs)
                    load_b_frags_at(b_frag[bnext], b_frag4[bnext], b_tile, k_seg + 1,
                                    b_seg[k_seg + 1], lane);
                /* Load the next A fragment ahead; dequantized A needs all row fragments first. */
                typename MmaOp::AFrag a_frag[kMt + 1];
                if constexpr (kDequantA) {
                    const auto a_tile = astrai::stage_of(ring_a, tile_index);
#pragma unroll
                    for (int mt = 0; mt < kMt; ++mt)
                        load_a_frags_at(a_frag[mt], a_tile, k_seg, mt, lane);
                } else if constexpr (A::kTranspose || A::kPacked) {
                    astrai::ldmatrix_x4_lane<true>(a_frag[0], a_seg[k_seg]);
                } else {
                    astrai::ldmatrix_x4_lane(a_frag[0], a_seg[k_seg]);
                }
#pragma unroll
                for (int mt = 0; mt < kMt; ++mt) {
                    if constexpr (!kDequantA) {
                        if (mt + 1 < kMt) {
                            const unsigned a_next =
                                (A::kTranspose || A::kPacked)
                                    ? (a_seg[k_seg] ^ (unsigned)((mt + 1) * kMtXor))
                                    : (a_seg[k_seg] + (mt + 1) * kMtStep);
                            if constexpr (A::kTranspose || A::kPacked)
                                astrai::ldmatrix_x4_lane<true>(a_frag[mt + 1], a_next);
                            else
                                astrai::ldmatrix_x4_lane(a_frag[mt + 1], a_next);
                        }
                    }
#pragma unroll
                    for (int nt = 0; nt < kNt; ++nt) {
                        if constexpr (kPairB)
                            MmaOp::fma(*acc(mt, nt), a_frag[mt],
                                       b_frag4[bcur][nt >> 1].cell(nt & 1), *acc(mt, nt));
                        else
                            MmaOp::fma(*acc(mt, nt), a_frag[mt], b_frag[bcur][nt], *acc(mt, nt));
                    }
                }
                /*
                 * Next tile's LDGSTS chunks inside the MMA phase: A's after the
                 * first k_seg's MMA batch, B's after the last.
                 */
                if constexpr (kInterior && !kTma) {
                    if (k_seg == 0)
                        carry_a.emit(prefetch);
                    if (k_seg == kSegs - 1)
                        carry_b.emit(prefetch);
                }
            }
            /*
             * Generic loop (no interleaved prefetch): the next tile's
             * predicated loads run after the MMA phase — congruous cp.async
             * chunks, 8-bit crosswise transpose + STS of the issued runs.
             */
            if constexpr (!kInterior && !kTma) {
                if (prefetch) {
                    copy_async(astrai::stage_of(ring_a, tile_index + kStages),
                               astrai::stage_of(ring_b, tile_index + kStages),
                               (tile_index + kStages) * kTile);
                    if constexpr (A::kRegisterLoad)
                        sync_a.commit(astrai::stage_of(ring_a, tile_index + kStages), a, m, k, a_ld,
                                      tid, (tile_index + kStages) * kTile, block_m * kBlockM);
                    if constexpr (B::kRegisterLoad)
                        sync_b.commit(astrai::stage_of(ring_b, tile_index + kStages), b, n, k, b_ld,
                                      tid, (tile_index + kStages) * kTile, block_n * kBlockN);
                }
            }
            /*
             * Unconditional commit: empty in the tail, pads the group sequence
             * so the fixed wait stays correct. TMA commits via tma_issue_stage.
             */
            if constexpr (!kTma)
                pipe.producer_commit();
            /*
             * TMA consumer release: this thread's reads are done; the empty
             * barrier trips once every thread arrives, gating the overwrite.
             */
            if constexpr (kTma)
                astrai::mbarrier_arrive(tma.empty((int)(tile_index % A::kDepth)));
            a_rd += (unsigned)A::kStageBytes;
            if (a_rd == a_rd_end)
                a_rd = a_rd0;
            b_rd += (unsigned)B::kStageBytes;
            if (b_rd == b_rd_end)
                b_rd = b_rd0;
            if constexpr (kInterior && !kTma) {
                carry_a.advance();
                carry_b.advance();
            }
        }
    }

    template <bool kRank3A = false, bool kRank3B = false>
    DEVICE_FORCEINLINE void accumulate(AccTensor& acc,
                                       const GemmTmaContext<kRank3A, kRank3B>& tma = {}) const {
        if constexpr (kUseTma) {
            run_loop<false, true, kRank3A, kRank3B>(acc, tma);
        } else {
            if (use_interior_copy)
                run_loop<true>(acc);
            else
                run_loop<false>(acc);
        }
    }

  private:
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

    /* Lane offsets derive from staging layouts and feed ldmatrix byte addresses. */
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
        return canonical_lane_off<typename A::Canonical, ElemA>(
            a_row0 + ((lane >> 3) & 1) * 8 + (lane & 7), lane >> 4, lane);
    }
    DEVICE_FORCEINLINE unsigned b_lane_off(int lane) const {
        // ldmatrix (non-dequant) B addressing: byte offsets in ElemB units.
        return canonical_lane_off<typename B::Canonical, ElemB>(b_row0 + (lane & 7),
                                                                (lane >> 3) & 1, lane);
    }
    /* Paired B x4 needs a positive swizzle row shift; K=128 byte tiles use x2. */
    static constexpr unsigned kMtStep = 16u * kTile * sizeof(ElemA); // m-tile row step
    static constexpr unsigned kNtStep = 8u * kTile * sizeof(ElemB);  // n-tile row step
    static constexpr unsigned kSegXorA = (unsigned)Traits::kMmaK * sizeof(ElemA);
    static constexpr unsigned kSegXorB = (unsigned)Traits::kMmaK * sizeof(ElemB);
    static constexpr bool kPairB = !kDequantB && !B::kPacked && B::Canonical::kRowShift > 0;
    static_assert(!kPairB || kTile * sizeof(ElemB) <= 64,
                  "paired B x4 is unsupported for the kTile=128 byte layout");
    static_assert(!kPairB || kNt % 2 == 0, "B pairing needs even kNt");
    static_assert(!kPairB || !B::kTranspose, "2-byte crosswise B never pairs (chunk budget)");
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
        return trans_lane_off<typename A::Transposed, ElemA>(
            (lane & 7) + ((lane >> 4) << 3), a_row0 + (((lane >> 3) & 1) << 3), kBlockM);
    }
    DEVICE_FORCEINLINE unsigned b_trans_lane_off(int lane) const {
        // nt windows step by kNtXor at call sites (col stays at b_row0).
        return trans_lane_off<typename B::Transposed, ElemB>((lane & 7) + (((lane >> 3) & 1) << 3),
                                                             b_row0, kBlockN);
    }

    /*
     * Packed-grid lane offsets: the trans formula one width down (a packed
     * row is 16-bit units, 8 per 16B chunk; the row extent is in UNITS). The
     * lane contract, XOR steps and per-k-segment ADD are the trans reader's:
     * 16 packed rows = one mma k-segment.
     */
    DEVICE_FORCEINLINE unsigned a_pack_lane_off(int lane) const {
        return trans_lane_off<typename A::Packed, unsigned short>(
            (lane & 7) + ((lane >> 4) << 3), a_row0 + (((lane >> 3) & 1) << 3), kBlockM);
    }
    DEVICE_FORCEINLINE unsigned b_pack_lane_off(int lane) const {
        return trans_lane_off<typename B::Packed, unsigned short>(
            (lane & 7) + (((lane >> 3) & 1) << 3), b_row0, kBlockN);
    }

    /* Convert packed INT8 A pairs to BF16 in ldmatrix register order. */
    DEVICE_FORCEINLINE void load_a_frags_at(typename MmaOp::AFrag& frag,
                                            typename A::Tile stage,
                                            int k_seg,
                                            int mt,
                                            int lane) const {
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
            if constexpr (B::kTranspose || B::kPacked) {
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
                                            typename B::Tile stage,
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
};

} // namespace gemm
} // namespace astrai
