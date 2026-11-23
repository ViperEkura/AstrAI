#pragma once
/*
 * Collective mainloop: smem stage rings, cp.async/TMA stage loads,
 * crosswise LDG+PRMT, per-lane ldmatrix addressing, software-pipelined
 * mma.sync. Addressing scheme + fast-loop peel rationale:
 * docs/developer/kernels/gemm.md.
 */

#include <type_traits>

#include <api/native/gemm.h>
#include <kernel/gemm/warp_fragments.cuh>
#include <memory/load_async.cuh>
#include <memory/load_crosswise.cuh>
#include <memory/load_crosswise_packed.cuh>
#include <memory/pipeline.cuh>
#include <memory/tma.cuh>
#include <policy/gemm.cuh>
#include <utils/define.cuh>
#include <utils/tensor.cuh>

namespace astrai {
namespace gemm {

/* TMA full barriers mark ready copies; empty barriers protect slots until
 * consumers finish. */
template <bool kRank3A = false, bool kRank3B = false> struct GemmTmaContext {
    const void* map_a = nullptr;
    const void* map_b = nullptr;
    uint64_t* bars = nullptr;
    int depth = 0; // ring slots (kStages + 1): 2*depth barriers
    int z = 0;     // batch coordinate (rank-3 descriptors)

    DEVICE_FORCEINLINE uint64_t* full(int slot) const { return bars + slot; }
    DEVICE_FORCEINLINE uint64_t* empty(int slot) const { return bars + depth + slot; }
};

template <typename Policy> struct GemmCollectiveMainloop : GemmTileLayout<Policy> {
    using Layout = GemmTileLayout<Policy>;
    using Fragments = GemmWarpFragments<Policy>;
    using Traits = typename Layout::Traits;
    using ElemA = typename Layout::ElemA;
    using ElemB = typename Layout::ElemB;
    using MmaOp = typename Layout::MmaOp;
    using StagedLayoutA = typename Layout::StagedLayoutA;
    using StagedLayoutB = typename Layout::StagedLayoutB;
    using RingA = typename Layout::RingA;
    using RingB = typename Layout::RingB;
    using TileA = typename Layout::TileA;
    using TileB = typename Layout::TileB;
    using AccTensor = typename Layout::AccTensor;
    using Layout::kARing;
    using Layout::kAStageBytes;
    using Layout::kBlockM;
    using Layout::kBlockN;
    using Layout::kBRing;
    using Layout::kBStageBytes;
    using Layout::kCtaThreads;
    using Layout::kNt;
    using Layout::kPackA;
    using Layout::kPackB;
    using Layout::kSegs;
    using Layout::kStages;
    using Layout::kSyncA;
    using Layout::kSyncB;
    using Layout::kTile;
    using Layout::kTransA;
    using Layout::kTransB;
    using Layout::kUseTma;

    const RingA ring_a; // A's stage ring; B carves right past its end
    const RingB ring_b;
    const ElemA* const a;
    const ElemB* const b;
    const int64_t m, n, k, a_ld, b_ld;
    const int tid;
    const int64_t block_m, block_n;
    const int warp_m, warp_n;
    const Fragments fragments;
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
        : ring_a(astrai::make_ring<ElemA, StagedLayoutA, kARing>(smem)),
          ring_b(
              astrai::make_ring<ElemB, StagedLayoutB, kBRing>(smem + RingA::Layout::kTotalBytes)),
          a(a), b(b), m(m), n(n), k(k), a_ld(a_ld), b_ld(b_ld), tid(tid), block_m(block.x),
          block_n(block.y), warp_m((tid >> 5) / Traits::kWarpsN),
          warp_n((tid >> 5) % Traits::kWarpsN),
          fragments{warp_m * Traits::kWarpM, warp_n * Traits::kWarpN},
          tile_count((k + kTile - 1) / kTile),
          use_interior_copy(!kSyncA && !kSyncB && ((int64_t)block.x * kBlockM + kBlockM <= m) &&
                            ((int64_t)block.y * kBlockN + kBlockN <= n) &&
                            ((reinterpret_cast<uintptr_t>(a) | (uint64_t)a_ld) & 15) == 0 &&
                            ((reinterpret_cast<uintptr_t>(b) | (uint64_t)b_ld) & 15) == 0 &&
                            (k % kTile) == 0) {}

    /* cp.async handles aligned operands; crosswise bytes use direct LDG and
   * register packing. */
    template <bool kInterior = false, bool kSyncPhase = false>
    DEVICE_FORCEINLINE void load_stage(TileA a_tile, TileB b_tile, int64_t k_base) const {
        if constexpr (kSyncPhase) {
            /*
       * 8-bit crosswise: the k-pair packed grid when the tile admits it,
       * the canonical-tile register carry otherwise.
       */
            if constexpr (kPackA)
                load_crosswise_paired<StagedLayoutA, ElemA, kCtaThreads>(a_tile, a, m, k, a_ld, tid,
                                                                         k_base, block_m * kBlockM);
            else if constexpr (kSyncA)
                load_crosswise_direct<StagedLayoutA, ElemA, kCtaThreads>(a_tile, a, m, k, a_ld, tid,
                                                                         k_base, block_m * kBlockM);
            if constexpr (kPackB)
                load_crosswise_paired<StagedLayoutB, ElemB, kCtaThreads>(b_tile, b, n, k, b_ld, tid,
                                                                         k_base, block_n * kBlockN);
            else if constexpr (kSyncB)
                load_crosswise_direct<StagedLayoutB, ElemB, kCtaThreads>(b_tile, b, n, k, b_ld, tid,
                                                                         k_base, block_n * kBlockN);
        } else {
            /*
       * Everything but the 8-bit crosswise (kSync) staging rides
       * cp.async: congruous goes canonical, 16-bit crosswise goes
       * transposed.
       */
            if constexpr (!kSyncA)
                load_operand_tile<StagedLayoutA, ElemA, kCtaThreads, kTransA, kInterior>(
                    a_tile, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
            if constexpr (!kSyncB)
                load_operand_tile<StagedLayoutB, ElemB, kCtaThreads, kTransB, kInterior>(
                    b_tile, b, n, k, b_ld, tid, k_base, block_n * kBlockN);
        }
    }

    /*
   * One TMA stage issue, elected-thread only: gate on the slot's empty
   * barrier (skipped on the ring's first sweep), arm the full barrier's
   * byte count, issue both operand boxes (rank rides the context type).
   */
    template <bool kRank3A, bool kRank3B>
    DEVICE_FORCEINLINE void tma_issue_stage(const GemmTmaContext<kRank3A, kRank3B>& tma,
                                            int tile) const {
        const int slot = tile % kARing;
        if (tile >= kARing)
            astrai::mbarrier_wait_parity(tma.empty(slot), (uint32_t)(((tile / kARing) - 1) & 1));
        uint64_t* bar = tma.full(slot);
        mbarrier_arrive_expect_tx(bar, (uint32_t)((uint64_t)kAStageBytes + kBStageBytes));
        const int x = (int)((int64_t)tile * kTile);
        astrai::tma_load<kRank3A>(tma.map_a, bar, astrai::stage_of(ring_a, tile).engine.ptr,
                                  x * (int)sizeof(ElemA), (int)(block_m * kBlockM), tma.z);
        astrai::tma_load<kRank3B>(tma.map_b, bar, astrai::stage_of(ring_b, tile).engine.ptr,
                                  x * (int)sizeof(ElemB), (int)(block_n * kBlockN), tma.z);
    }

    /* Prime kStages groups, including empty groups, to keep wait indexing stable.
   */
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
                if (use_interior_copy)
                    load_stage<true>(astrai::stage_of(ring_a, stage),
                                     astrai::stage_of(ring_b, stage), (int64_t)stage * kTile);
                else
                    load_stage(astrai::stage_of(ring_a, stage), astrai::stage_of(ring_b, stage),
                               (int64_t)stage * kTile);
                load_stage<false, true>(astrai::stage_of(ring_a, stage),
                                        astrai::stage_of(ring_b, stage), (int64_t)stage * kTile);
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
        PrefetchCarry<!kSyncA && !kTma, RingA, kCtaThreads, kTransA> carry_a(
            ring_a, a, a_ld, block_m * kBlockM, tid, kStages);
        PrefetchCarry<!kSyncB && !kTma, RingB, kCtaThreads, kTransB> carry_b(
            ring_b, b, b_ld, block_n * kBlockN, tid, kStages);
        /*
     * The 8-bit crosswise operands' register carry: issue() runs the global
     * runs before the MMA phase, commit() the byte-perm and STS after it.
     * Packed-grid operands ride the same seam with the two-run sibling.
     */
        std::conditional_t<kPackA, PairPackCarry<StagedLayoutA, ElemA, kCtaThreads, true>,
                           CrosswiseCarry<StagedLayoutA, ElemA, kCtaThreads, kSyncA>>
            sync_a;
        std::conditional_t<kPackB, PairPackCarry<StagedLayoutB, ElemB, kCtaThreads, true>,
                           CrosswiseCarry<StagedLayoutB, ElemB, kCtaThreads, kSyncB>>
            sync_b;
        const unsigned a_rd0 =
            __cvta_generic_to_shared(ring_a.engine.ptr) +
            (kPackA ? fragments.a_pack_lane_off(lane)
                    : (kTransA ? fragments.a_trans_lane_off(lane) : fragments.a_lane_off(lane)));
        const unsigned b_rd0 =
            __cvta_generic_to_shared(ring_b.engine.ptr) +
            (kPackB ? fragments.b_pack_lane_off(lane)
                    : (kTransB ? fragments.b_trans_lane_off(lane)
                               : (Fragments::kPairB ? fragments.b4_lane_off(lane)
                                                    : fragments.b_lane_off(lane))));
        const unsigned a_rd_end = a_rd0 + (unsigned)(kARing * kAStageBytes);
        const unsigned b_rd_end = b_rd0 + (unsigned)(kBRing * kBStageBytes);
        unsigned a_rd = a_rd0, b_rd = b_rd0;
        for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
            /*
       * Steady state: exactly kStages-1 younger groups in flight when
       * this fires; the tail's unconditional empty commits keep it true.
       */
            const bool prefetch = tile_index + kStages < tile_count;
            /* TMA waits full slots; cp.async joins the CTA before slot reuse. */
            if constexpr (kTma) {
                astrai::mbarrier_wait_parity(tma.full((int)(tile_index % kARing)),
                                             static_cast<uint32_t>((tile_index / kARing) & 1));
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
                if constexpr (kSyncA)
                    sync_a.issue(a, m, k, a_ld, tid, (tile_index + kStages) * kTile,
                                 block_m * kBlockM);
                if constexpr (kSyncB)
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
                a_seg[s] = (kTransA || kPackA) ? (a_addr + (unsigned)(s * Fragments::kTransSegA))
                                               : (a_addr ^ (unsigned)(s * Fragments::kSegXorA));
                b_seg[s] = (kTransB || kPackB) ? (b_addr + (unsigned)(s * Fragments::kTransSegB))
                                               : (b_addr ^ (unsigned)(s * Fragments::kSegXorB));
            }

            /* Prefetch B fragments by K segment and pair adjacent N cells where
       * valid. */
            typename MmaOp::BFrag b_frag[2][kNt];
            typename Fragments::BFragPair b_frag4[2][kNt / 2];
            /*
       * Staged tensor hoisted out of the k_seg/mt loops: the slot pick
       * is a modulo and regressed the dequant readers' register budget.
       */
            const auto b_tile = astrai::stage_of(ring_b, tile_index);
            fragments.load_b_frags_at(b_frag[0], b_frag4[0], b_tile, 0, b_seg[0], lane);
#pragma unroll
            for (int k_seg = 0; k_seg < kSegs; ++k_seg) {
                const int bcur = k_seg & 1, bnext = bcur ^ 1;
                if (k_seg + 1 < kSegs)
                    fragments.load_b_frags_at(b_frag[bnext], b_frag4[bnext], b_tile, k_seg + 1,
                                              b_seg[k_seg + 1], lane);
                fragments.accumulate_segment(acc, astrai::stage_of(ring_a, tile_index), k_seg,
                                             a_seg[k_seg], lane, b_frag[bcur], b_frag4[bcur]);
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
                    load_stage(astrai::stage_of(ring_a, tile_index + kStages),
                               astrai::stage_of(ring_b, tile_index + kStages),
                               (tile_index + kStages) * kTile);
                    if constexpr (kSyncA)
                        sync_a.commit(astrai::stage_of(ring_a, tile_index + kStages), a, m, k, a_ld,
                                      tid, (tile_index + kStages) * kTile, block_m * kBlockM);
                    if constexpr (kSyncB)
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
                astrai::mbarrier_arrive(tma.empty((int)(tile_index % kARing)));
            a_rd += (unsigned)kAStageBytes;
            if (a_rd == a_rd_end)
                a_rd = a_rd0;
            b_rd += (unsigned)kBStageBytes;
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
};

} // namespace gemm
} // namespace astrai
