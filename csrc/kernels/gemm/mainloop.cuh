#pragma once
// Collective mainloop: shared-memory stage rings, the gmem->smem stage
// loads (congruous cp.async / crosswise LDG+PRMT), the per-lane ldmatrix
// fragment addressing and the software-pipelined mma.sync loop. The
// fragment addressing scheme and fast-loop peel rationale live in
// docs/developer/cuda_kernels.md.

#include <type_traits>

#include "common/mma.cuh"
#include "common/pipeline.cuh"
#include "common/tma.cuh"
#include "common/tensor.cuh"
#include "gemm/common.h"
#include "load.cuh"
#include "policy.cuh"
#include "quantize/dequant.cuh"

namespace astrai {
namespace gemm {

// TMA producer context: the operand descriptors, the ring-slot mbarriers
// and the batch coordinate bits. Barriers hold 2*kARing slots — full[0..D)
// (count 1, tripped by the elected thread's expect_tx + the TMA's
// transaction bytes) and empty[D..2D) (count = CTA threads, tripped when
// every consumer finished the slot) — the CUTLASS PipelineTmaAsync
// handshake replacing the per-k-tile __syncthreads: warps skew freely
// across slots and the producer's overwrite gate is the empty barrier
// alone. Each operand's rank is a template bit (strided batch = rank 3,
// broadcast = rank 2), so the 2D/3D issue pick compiles away.
template <bool kRank3A = false, bool kRank3B = false>
struct GemmTmaContext {
    const void* map_a = nullptr;
    const void* map_b = nullptr;
    uint64_t* bars = nullptr;
    int depth = 0;       // ring slots (kStages + 1): 2*depth barriers
    int z = 0;           // batch coordinate (rank-3 descriptors)

    __device__ __forceinline__ uint64_t* full(int slot) const {
        return bars + slot;
    }
    __device__ __forceinline__ uint64_t* empty(int slot) const {
        return bars + depth + slot;
    }
};

template <typename Policy>
struct GemmCollectiveMainloop {
    using Traits = typename Policy::Traits;
    using LayoutA = typename Policy::LayoutTagA;
    using LayoutB = typename Policy::LayoutTagB;
    using Smem = GemmSmem<Traits, LayoutA, LayoutB>;
    static constexpr bool kFastLoop = Policy::kFastLoop;
    static constexpr bool kUseTma = Policy::kUseTma;
    static_assert(!kUseTma || (!Smem::kDirectA && !Smem::kDirectB),
                  "TMA staging is congruous-only");
    // Operands are independently typed; the mma runs on the promoted MmaT
    // (policy.cuh). A lone int8 or fp8 side expands in-register between the
    // smem read and the mma — kDequantA/kDequantB mark the insert per side
    // (dequant.cuh); passthrough pairs leave both false. The accumulator
    // type rides the mma cell: fp32 for float families, int32 for the s8.
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
    static constexpr int kK = Traits::kK;
    static constexpr int kStages = Traits::kStages;
    static constexpr int kCtaThreads = Traits::kCtaThreads;
    static constexpr bool kDirectA = Smem::kDirectA;
    static constexpr bool kDirectB = Smem::kDirectB;
    // Crosswise operands split by element width: 16-bit stages by cp.async
    // into a transposed [kK][rows] tile read through ldmatrix.trans (async,
    // pipelineable); 8-bit has no trans ldmatrix and keeps the synchronous
    // LDG+PRMT path into the canonical tile.
    static constexpr bool kSyncA = kDirectA && sizeof(ElemA) == 1;
    static constexpr bool kTransA = kDirectA && sizeof(ElemA) == 2;
    static constexpr bool kSyncB = kDirectB && sizeof(ElemB) == 1;
    static constexpr bool kTransB = kDirectB && sizeof(ElemB) == 2;
    // Dequant inserts are int8-storage only; a dequantized side never rides
    // the 16-bit-only trans staging — kTrans* is already false there.
    static_assert(!kDequantA || sizeof(ElemA) == 1,
                  "in-register dequant targets 1-byte storage");
    static_assert(!kDequantB || sizeof(ElemB) == 1,
                  "in-register dequant targets 1-byte storage");
    static_assert(kStages >= 1 && kStages <= 8,
                  "FP8 GEMM stages must be in [1, 8]");
    // CTA = (BlockM/WarpM) x (BlockN/WarpN) warps, each warp computing
    // kMt x kNt m16n8k{kMmaK} MMAs. Rings rotate kStages+1 buffers (see
    // GemmSmem) — one __syncthreads per k-tile.
    static constexpr int kMt = Traits::kMt;  // 16-row MMA tiles per warp
    static constexpr int kNt = Traits::kNt;  // 8-col MMA tiles per warp
    static constexpr int kSegs = kK / Traits::kMmaK;  // mma-sized k segments
    static constexpr int kARing = Smem::kRingDepth;
    static constexpr int kBRing = Smem::kRingDepth;

    // Staging layouts, cute-style: one declared instance per staged tile —
    // composition(Swizzle, Layout<Shape, Stride>) over the row-major 16B-
    // chunk grid (common/swizzle.cuh) — shared by the stage loaders, the
    // fragment reads and the folded lane-offset mirrors below. Canonical
    // tiles [rows][kK] serve congruous and 8-bit crosswise-direct staging
    // (2-byte elements give the TMA SWIZZLE_128B pattern, 1-byte
    // SWIZZLE_64B); trans tiles [kK][rows] the 16-bit crosswise cp.async +
    // ldmatrix.trans staging (custom XOR: chunks swizzled by the k-row
    // bits, capped at 8 — the LDSM contract's row budget).
    static constexpr int kChunksA = kK / (16 / (int)sizeof(ElemA));
    static constexpr int kChunksB = kK / (16 / (int)sizeof(ElemB));
    static constexpr int kChunksAT = kBlockM / (16 / (int)sizeof(ElemA));
    static constexpr int kChunksBT = kBlockN / (16 / (int)sizeof(ElemB));
    using SmemLayoutA = decltype(
        composition(Swizzle<log2_const<kChunksA>::value, 3>{},
                    Layout<Shape<kBlockM, kChunksA>, Stride<kChunksA, 1>>{}));
    using SmemLayoutB = decltype(
        composition(Swizzle<log2_const<kChunksB>::value, 3>{},
                    Layout<Shape<kBlockN, kChunksB>, Stride<kChunksB, 1>>{}));
    using SmemLayoutATrans = decltype(
        composition(Swizzle<log2_const<kChunksAT < 8 ? kChunksAT : 8>::value,
                            log2_const<kChunksAT>::value>{},
                    Layout<Shape<kK, kChunksAT>, Stride<kChunksAT, 1>>{}));
    using SmemLayoutBTrans = decltype(
        composition(Swizzle<log2_const<kChunksBT < 8 ? kChunksBT : 8>::value,
                            log2_const<kChunksBT>::value>{},
                    Layout<Shape<kK, kChunksBT>, Stride<kChunksBT, 1>>{}));

    // One ring type per operand (common/tensor.cuh): the staged-layout
    // instance each path addresses — the trans tile when the 16-bit
    // crosswise staging is active, the canonical tile otherwise (congruous
    // staging, 8-bit crosswise-direct staging and the dequant fragment
    // readers all address the canonical tile; both stagings hold the same
    // element count, so one ring stride serves either). The rings carry
    // the slot rotation, the stage/ring byte budget and the typed tile
    // view — the smem carve and every stage consumer below read them off
    // the type instead of re-deriving strides.
    using StagedLayoutA =
        std::conditional_t<kTransA, SmemLayoutATrans, SmemLayoutA>;
    using StagedLayoutB =
        std::conditional_t<kTransB, SmemLayoutBTrans, SmemLayoutB>;
    using RingA = Tensor<PtrEngine<ElemA>, RingLayout<StagedLayoutA, kARing>>;
    using RingB = Tensor<PtrEngine<ElemB>, RingLayout<StagedLayoutB, kBRing>>;
    using TileA = Tensor<PtrEngine<ElemA>, StagedLayoutA>;
    using TileB = Tensor<PtrEngine<ElemB>, StagedLayoutB>;

    // The warp's accumulator: typed C cells on a (mt, nt) grid — indexing
    // by semantic coordinates all the way to the mma (no pointer decay at
    // the fma seam; the epilogue reads the same cells).
    using AccTensor = typename Traits::AccTensor;

    // Per-stage byte strides (the smem carve and the byte-address read
    // carries measure against them).
    static constexpr int kAStageBytes = RingA::Layout::kStageBytes;
    static constexpr int kBStageBytes = RingB::Layout::kStageBytes;

    const RingA ring_a;  // A's stage ring; B carves right past its end
    const RingB ring_b;
    const ElemA* const a;
    const ElemB* const b;
    const int64_t m, n, k, a_ld, b_ld;
    const int tid;
    const int64_t block_m, block_n;
    const int warp_m, warp_n;
    const int a_row0;  // + mt * 16 in the loop
    const int b_row0;  // + nt * 8
    const int64_t tile_count;
    // Interior-CTA peel (kFastLoop instantiations only): whole-CTA,
    // 16B-aligned, K without tail — the mainloop then runs a compile-time
    // specialized copy with no per-chunk predication (measured +4.5..10% on
    // the issue-bound small CTA; the 128x128 CTA regressed, so only the
    // small CTA opts in). The verdict is uniform per CTA.
    const bool fast_cta;

    __device__ GemmCollectiveMainloop(char* smem,
                                     const ElemA* a, const ElemB* b,
                                     int64_t m, int64_t n, int64_t k, int64_t a_ld, int64_t b_ld,
                                     int tid, int2 block)
        : ring_a(astrai::make_ring<ElemA, StagedLayoutA, kARing>(smem)),
          ring_b(astrai::make_ring<ElemB, StagedLayoutB, kBRing>(smem + RingA::Layout::kTotalBytes)),
          a(a), b(b), m(m), n(n), k(k), a_ld(a_ld), b_ld(b_ld), tid(tid),
          block_m(block.x), block_n(block.y),
          warp_m((tid >> 5) / Traits::kWarpsN),
          warp_n((tid >> 5) % Traits::kWarpsN),
          a_row0(warp_m * Traits::kWarpM),
          b_row0(warp_n * Traits::kWarpN),
          tile_count((k + kK - 1) / kK),
          fast_cta(kFastLoop && !kSyncA && !kSyncB &&
                   ((int64_t)block.x * kBlockM + kBlockM <= m) &&
                   ((int64_t)block.y * kBlockN + kBlockN <= n) &&
                   ((reinterpret_cast<uintptr_t>(a) | (uint64_t)a_ld) & 15) == 0 &&
                   ((reinterpret_cast<uintptr_t>(b) | (uint64_t)b_ld) & 15) == 0 &&
                   (k % kK) == 0) {}

    // Stage-load one k-tile, per operand picking its loader from the
    // staging class: congruous and 16-bit crosswise cp.async into the
    // canonical / transposed rings, 8-bit crosswise LDG+PRMT (the tiles
    // arrive typed by each ring's staged layout, so a mismatched
    // loader/tile pairing is a compile error). kFast selects the
    // predication-free interior copy (async phase only — trans staging
    // qualifies: it is cp.async like the congruous path). kSyncPhase picks
    // the call-site phase: true = the synchronous direct loads (8-bit
    // crosswise only; in the steady state this runs right after barrier 1,
    // so the LDG latency and the PRMT transpose overlap the MMA phase
    // instead of stalling the inter-barrier window), false = the async
    // loads (kFast applies, and in the generic loop they run after the
    // MMA phase alongside the commit).
    template <bool kFast = false, bool kSyncPhase = false>
    __device__ __forceinline__ void
    load_stage(TileA a_tile, TileB b_tile,
               int64_t k_base) const {
        if constexpr (kSyncPhase) {
            if constexpr (kSyncA)
                load_crosswise_direct<StagedLayoutA, ElemA, kCtaThreads>(
                    a_tile, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
            if constexpr (kSyncB)
                load_crosswise_direct<StagedLayoutB, ElemB, kCtaThreads>(
                    b_tile, b, n, k, b_ld, tid, k_base, block_n * kBlockN);
        } else {
            // Everything but the 8-bit crosswise (kSync) staging rides
            // cp.async: congruous goes canonical, 16-bit crosswise goes
            // transposed.
            if constexpr (!kSyncA)
                load_operand_tile<StagedLayoutA, ElemA, kCtaThreads, kTransA, kFast>(
                    a_tile, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
            if constexpr (!kSyncB)
                load_operand_tile<StagedLayoutB, ElemB, kCtaThreads, kTransB, kFast>(
                    b_tile, b, n, k, b_ld, tid, k_base, block_n * kBlockN);
        }
    }

    // One TMA stage issue: gate on the slot's empty barrier (its previous
    // occupant fully consumed; skipped for the ring's first sweep), arm
    // the full barrier's byte count, then issue both operand boxes
    // (typed loads: each operand's rank rides its context type).
    // Elected-thread only.
    template <bool kRank3A, bool kRank3B>
    __device__ __forceinline__ void
    tma_issue_stage(const GemmTmaContext<kRank3A, kRank3B>& tma,
                    int tile) const {
        const int slot = tile % kARing;
        if (tile >= kARing)
            astrai::mbarrier_wait_parity(
                tma.empty(slot),
                (uint32_t)(((tile / kARing) - 1) & 1));
        uint64_t* bar = tma.full(slot);
        mbarrier_arrive_expect_tx(
            bar, (uint32_t)((uint64_t)kAStageBytes + kBStageBytes));
        const int x = (int)((int64_t)tile * kK);
        astrai::tma_load<kRank3A>(tma.map_a, bar,
                                  astrai::stage_of(ring_a, tile).engine.ptr,
                                  x * (int)sizeof(ElemA),
                                  (int)(block_m * kBlockM), tma.z);
        astrai::tma_load<kRank3B>(tma.map_b, bar,
                                  astrai::stage_of(ring_b, tile).engine.ptr,
                                  x * (int)sizeof(ElemB),
                                  (int)(block_n * kBlockN), tma.z);
    }

    // Prime the pipeline: kStages committed groups, one per stage slot.
    // The commit is unconditional — when K is shorter than the pipeline the
    // skipped stages commit empty groups, so the group sequence stays
    // tile-indexed and the steady-state wait count never needs a runtime
    // dispatch. The TMA discipline arms only stages that carry a copy: an
    // expect_tx barrier with no transaction never trips, so short-K tiles
    // skip their slots' barriers entirely (and are never waited on).
    template <bool kRank3A = false, bool kRank3B = false>
    __device__ __forceinline__ void
    prologue(const GemmTmaContext<kRank3A, kRank3B>& tma = {}) const {
        if constexpr (kUseTma) {
            if (tid == 0) {
#pragma unroll
                for (int stage = 0; stage < kStages; ++stage) {
                    if (stage < tile_count) tma_issue_stage(tma, stage);
                }
            }
            return;
        }
        const astrai::PipelineSync<kStages> pipe;
#pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
            if (stage < tile_count) {
                if (fast_cta)
                    load_stage<true>(astrai::stage_of(ring_a, stage),
                                     astrai::stage_of(ring_b, stage),
                                     (int64_t)stage * kK);
                else
                    load_stage(astrai::stage_of(ring_a, stage),
                               astrai::stage_of(ring_b, stage),
                               (int64_t)stage * kK);
                load_stage<false, true>(astrai::stage_of(ring_a, stage),
                                        astrai::stage_of(ring_b, stage),
                                        (int64_t)stage * kK);
            }
            pipe.producer_commit();
        }
    }

    // Steady-state mainloop, compile-time specialized on kFast: the fast
    // copy runs predication-free loads with loop-carried read/write
    // pointers; the generic copy keeps full predication. kFastLoop=false
    // instantiates only the generic copy. kTma swaps the staging
    // discipline: the per-thread cp.async chunks and the wait_group+
    // syncthreads consumer fence become one elected-thread TMA issue and
    // an mbarrier phase wait (plus the same CTA barrier, which stays the
    // slot-release guarantee: it proves every thread finished reading
    // tile i-1 before tile i+kStages's boxes overwrite its slot).
    template <bool kFast, bool kTma = false, bool kRank3A = false,
              bool kRank3B = false>
    __device__ __forceinline__ void
    run_loop(AccTensor& acc,
             const GemmTmaContext<kRank3A, kRank3B>& tma = {}) const {
        const astrai::PipelineSync<kStages> pipe;
        const int lane = tid & 31;
        // Fast-path write carries: one per congruous operand (crosswise
        // operands get the empty no-op type), targeting the first
        // prefetched tile (kStages). Steady-state read carries: the LDSM
        // base of the current k-tile's stage with the lane offset folded
        // in, advanced one stage per iteration with an equality wrap —
        // replaces the per-k-tile (tile % ring) * stage_bytes
        // recomputation (a UIMAD.WIDE magic-division ladder in SASS).
        // The carries ride the operand rings, so each one already carries
        // its staging geometry (trans layout on the 16-bit crosswise path,
        // canonical otherwise).
        PrefetchCarry<!kSyncA && !kTma, RingA, kCtaThreads, kTransA>
            carry_a(ring_a, a, a_ld, block_m * kBlockM, tid, kStages);
        PrefetchCarry<!kSyncB && !kTma, RingB, kCtaThreads, kTransB>
            carry_b(ring_b, b, b_ld, block_n * kBlockN, tid, kStages);
        const unsigned a_rd0 = __cvta_generic_to_shared(ring_a.engine.ptr) +
                               (kTransA ? a_trans_lane_off(lane)
                                        : a_lane_off(lane));
        const unsigned b_rd0 =
            __cvta_generic_to_shared(ring_b.engine.ptr) +
            (kTransB ? b_trans_lane_off(lane)
                     : (kPairB ? b4_lane_off(lane) : b_lane_off(lane)));
        const unsigned a_rd_end = a_rd0 + (unsigned)(kARing * kAStageBytes);
        const unsigned b_rd_end = b_rd0 + (unsigned)(kBRing * kBStageBytes);
        unsigned a_rd = a_rd0, b_rd = b_rd0;
        for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        // In the steady state exactly kStages-1 younger groups are in flight
        // when this fires; the tail's unconditional (possibly empty)
        // commits keep that invariant true for every iteration.
        const bool prefetch = tile_index + kStages < tile_count;
        // Steady-state wait: the TMA path waits the slot's full barrier
        // (phase flips once per ring sweep) — no CTA-wide barrier; each
        // warp releases its slot below, after its own last fragment read,
        // and the producer's overwrite gate is the empty barrier alone.
        // The cp.async path drains its group ladder then joins the CTA
        // (the join doubles as the slot release).
        if constexpr (kTma) {
            astrai::mbarrier_wait_parity(
                tma.full((int)(tile_index % kARing)),
                static_cast<uint32_t>((tile_index / kARing) & 1));
        } else {
            pipe.consumer_wait();
        }

        // Staging for tile i+kStages (its slot = (i-1)'s, released by the
        // barrier above): the elected thread arms and issues both TMA
        // boxes; the cp.async path issues its direct chunks (LDG+PRMT,
        // 8-bit crosswise) now so the global-load latency hides behind the
        // MMA phase below.
        if constexpr (kTma) {
            if (prefetch && tid == 0)
                tma_issue_stage(tma, (int)(tile_index + kStages));
        } else if (prefetch) {
            load_stage<false, true>(astrai::stage_of(ring_a, tile_index + kStages),
                                    astrai::stage_of(ring_b, tile_index + kStages),
                                    (tile_index + kStages) * kK);
        }

        const unsigned a_addr = a_rd;
        const unsigned b_addr = b_rd;
        // Per-k_seg base pair (cuBLAS's scheme): seg s lives at the seg-0
        // base XOR (s * kSegXor) — one LOP3 per extra seg per k-tile, never
        // per fragment. Trans tiles keep their k rows at kMmaK * RowsT-byte
        // strides, so their seg step is a plain ADD (no XOR trick). Every
        // LDSM below addresses [base + immediate].
        unsigned a_seg[kSegs], b_seg[kSegs];
#pragma unroll
        for (int s = 0; s < kSegs; ++s) {
            a_seg[s] = kTransA ? (a_addr + (unsigned)(s * kTransSegA))
                               : (a_addr ^ (unsigned)(s * kSegXorA));
            b_seg[s] = kTransB ? (b_addr + (unsigned)(s * kTransSegB))
                               : (b_addr ^ (unsigned)(s * kSegXorB));
        }

        // kNt ldmatrix.x2 (B) + kMt ldmatrix.x4 (A) feed kMt*kNt*2 mma.sync
        // per k_seg — 0.5 load instructions per MMA. B fragments
        // double-buffer across k_segs; kPairB folds the two adjacent nt
        // fragments of one pair into a single x4 (see b4_lane_off). All
        // fragment arrays hold typed cells (MmaOp::BFrag / BFragPair): the
        // loads fill cells, the mma consumes cells by reference — fragment
        // pointer arithmetic has no spelling left.
        typename MmaOp::BFrag b_frag[2][kNt];
        BFragPair b_frag4[2][kNt / 2];
        // This tile's staged tensors, hoisted out of the k_seg/mt loops:
        // the slot pick is a tile-index modulo and must not re-enter the
        // MMA phase (it regressed the dequant readers' register budget).
        const auto b_tile = astrai::stage_of(ring_b, tile_index);
        load_b_frags_at(b_frag[0], b_frag4[0], b_tile, 0,
                        b_seg[0], lane);
#pragma unroll
        for (int k_seg = 0; k_seg < kSegs; ++k_seg) {
            const int bcur = k_seg & 1, bnext = bcur ^ 1;
            if (k_seg + 1 < kSegs)
                load_b_frags_at(b_frag[bnext], b_frag4[bnext], b_tile,
                                k_seg + 1,
                                b_seg[k_seg + 1], lane);
        // Software-pipelined A fragments: the ldmatrix.x4 for row mt+1 is
        // issued before the MMAs consuming row mt, so the LDS latency hides
        // behind tensor-pipe work. Costs 4 extra registers. Trans tiles
        // advance the m window by XOR (two 16B chunks), canonical tiles by
        // the 16-row byte stride. Dequantized A (W8A8) fills ALL m-row
        // fragments upfront through the scalar pair reads — the pipelined
        // ldmatrix would clobber the already-converted fragments with raw
        // 2-byte-layout data, so the fill below is ldmatrix-staging
        // (native/bf16) only.
        typename MmaOp::AFrag a_frag[kMt + 1];
        if constexpr (kDequantA) {
            const auto a_tile = astrai::stage_of(ring_a, tile_index);
#pragma unroll
            for (int mt = 0; mt < kMt; ++mt)
                load_a_frags_at(a_frag[mt], a_tile, k_seg,
                                mt, lane);
        } else if constexpr (kTransA) {
            astrai::ldmatrix_x4_lane<true>(a_frag[0], a_seg[k_seg]);
        } else {
            astrai::ldmatrix_x4_lane(a_frag[0], a_seg[k_seg]);
        }
#pragma unroll
        for (int mt = 0; mt < kMt; ++mt) {
            if constexpr (!kDequantA) {
                if (mt + 1 < kMt) {
                    const unsigned a_next =
                        kTransA ? (a_seg[k_seg] ^ (unsigned)((mt + 1) * kMtXor))
                                : (a_seg[k_seg] + (mt + 1) * kMtStep);
                    if constexpr (kTransA)
                        astrai::ldmatrix_x4_lane<true>(a_frag[mt + 1], a_next);
                    else
                        astrai::ldmatrix_x4_lane(a_frag[mt + 1], a_next);
                }
            }
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                if constexpr (kPairB)
                    MmaOp::fma(*acc(mt, nt), a_frag[mt],
                               b_frag4[bcur][nt >> 1].cell(nt & 1),
                               *acc(mt, nt));
                else
                    MmaOp::fma(*acc(mt, nt), a_frag[mt],
                               b_frag[bcur][nt], *acc(mt, nt));
            }
        }
        // Next tile's LDGSTS chunks inside the MMA phase: A's after the
        // first k_seg's MMA batch, B's after the last.
        if constexpr (kFast && !kTma) {
            if (k_seg == 0) carry_a.emit(prefetch);
            if (k_seg == kSegs - 1) carry_b.emit(prefetch);
        }
        }
        // Generic loop (no interleaved prefetch): the next tile's predicated
        // loads run after the MMA phase.
        if constexpr (!kFast && !kTma) {
            if (prefetch) {
                load_stage(astrai::stage_of(ring_a, tile_index + kStages),
                           astrai::stage_of(ring_b, tile_index + kStages),
                           (tile_index + kStages) * kK);
            }
        }
        // Unconditional commit: empty in the tail, it pads the group
        // sequence so the fixed wait above stays correct. TMA's commit is
        // the arm inside tma_issue_stage — nothing to do here.
        if constexpr (!kTma) pipe.producer_commit();
        // TMA consumer release: this thread's fragment reads of the slot
        // are done; the slot's empty barrier trips once every thread
        // arrives, gating the producer's next overwrite.
        if constexpr (kTma)
            astrai::mbarrier_arrive(tma.empty((int)(tile_index % kARing)));
        a_rd += (unsigned)kAStageBytes;
        if (a_rd == a_rd_end) a_rd = a_rd0;
        b_rd += (unsigned)kBStageBytes;
        if (b_rd == b_rd_end) b_rd = b_rd0;
        if constexpr (kFast && !kTma) {
            carry_a.advance();
            carry_b.advance();
        }
        }
    }

    template <bool kRank3A = false, bool kRank3B = false>
    __device__ __forceinline__ void
    accumulate(AccTensor& acc,
               const GemmTmaContext<kRank3A, kRank3B>& tma = {}) const {
        if constexpr (kUseTma) {
            run_loop<false, true, kRank3A, kRank3B>(acc, tma);
        } else if constexpr (kFastLoop) {
            if (fast_cta)
                run_loop<true>(acc);
            else
                run_loop<false>(acc);
        } else {
            run_loop<false>(acc);
        }
    }

  private:
    // One ldmatrix.x4 payload covering two adjacent n8 B fragments (the
    // kPairB fold): cell(i) selects the fragment mma tile nt consumes —
    // the register pairing lives in the type, not in (+ (nt & 1) * 2)
    // pointer arithmetic at the fma seam. (Not "half": nvcc reserves
    // that name for the fp16 type.)
    struct BFragPair : ArrayEngine<unsigned, 4> {
        __device__ __forceinline__ typename MmaOp::BFrag cell(int i) const {
            return {storage[2 * i + 0], storage[2 * i + 1]};
        }
    };

    // Per-lane ldmatrix fragment addressing (base-pair scheme, mirrored
    // from the cuBLAS SASS; derivation in the design notes): one base
    // register per operand per k_seg, every fragment offset an LDSM
    // immediate — zero address arithmetic inside the MMA phase.
    // Per-lane stage-relative BYTE offsets: the ldmatrix *_lane primitives
    // take raw shared-memory byte addresses, so every offset below is
    // element math scaled by sizeof(ElemT). The swizzle chunk term comes
    // from the declared staging layouts — the same instances the staged
    // tiles apply, so the mirror can never drift.
    static constexpr int kChunkElems = 16 / sizeof(ElemA);
    static constexpr int kChunkShift = log2_const<kChunkElems>::value;
    __device__ __forceinline__ unsigned a_lane_off(int lane) const {
        const int r7 = lane & 7;          // row within the 8-row matrix
        const int rh8 = (lane >> 3) & 1;  // +8 rows (A: lanes 8-15, 24-31)
        const int rh16 = lane >> 4;       // +1 chunk (A: lanes 16-31)
        const unsigned lswz = static_cast<unsigned>(
            (r7 >> SmemLayoutA::kRowShift) & SmemLayoutA::kMask);
        // Stage-relative, loop-invariant per-lane base; A's fragment row
        // carries the +8-row (rh8) and +1-chunk (rh16) halves.
        return static_cast<unsigned>(((a_row0 + rh8 * 8 + r7) * kK +
                                      ((rh16 ^ lswz) << kChunkShift)) *
                                     sizeof(ElemA));
    }
    __device__ __forceinline__ unsigned b_lane_off(int lane) const {
        // ldmatrix (non-dequant) B addressing: byte offsets in ElemB units.
        constexpr int kChunkB = 16 / sizeof(ElemB);
        constexpr int kChunkShiftB = log2_const<kChunkB>::value;
        const int r7 = lane & 7;
        const int rh8 = (lane >> 3) & 1;  // +8 rows (B uses rh8 as its chunk half)
        const unsigned lswz = static_cast<unsigned>(
            (r7 >> SmemLayoutB::kRowShift) & SmemLayoutB::kMask);
        return static_cast<unsigned>(((b_row0 + r7) * kK +
                                      ((rh8 ^ lswz) << kChunkShiftB)) *
                                     sizeof(ElemB));
    }
    // x4-paired B loads: one ldmatrix.x4 feeds the two adjacent nt
    // fragments. Lane contract: lanes 0-7 address rows n0..n7 chunk c,
    // lanes 8-15 rows n0..n7 chunk c+1, lanes 16-23 rows n8..n15 chunk c,
    // lanes 24-31 rows n8..n15 chunk c+1. The +8-row step never reaches
    // the swizzle source bits for kK <= 64; kK=128 swizzles on row[2:0]
    // where +8 flips bits, so that config keeps the x2 loads.
    // Fragment step constants, in BYTES (consumed by the *_lane address
    // math). kSegXor{A,B} = one mma k-segment of each operand's STORAGE
    // type — 32B for every ldmatrix-fed dtype (fp8 k32 x 1B, bf16 k16 x
    // 2B), i.e. two 16B chunks; mixed dtype pairs (A8W16) keep their own
    // step per side. Dequant-fed sides never consume theirs.
    static constexpr unsigned kMtStep = 16u * kK * sizeof(ElemA);  // m-tile row step
    static constexpr unsigned kNtStep = 8u * kK * sizeof(ElemB);   // n-tile row step
    static constexpr unsigned kSegXorA = (unsigned)Traits::kMmaK * sizeof(ElemA);
    static constexpr unsigned kSegXorB = (unsigned)Traits::kMmaK * sizeof(ElemB);
    static constexpr bool kPairB = !kDequantB && kK * sizeof(ElemB) / 16 <= 4;
    static_assert(!kPairB || kNt % 2 == 0, "B pairing needs even kNt");
    static_assert(!kPairB || !kTransB,
                  "2-byte crosswise B never pairs (chunk budget)");
    static constexpr unsigned kPairStep = 16u * kK * sizeof(ElemB);  // nt-pair row step
    __device__ __forceinline__ unsigned b4_lane_off(int lane) const {
        return b_lane_off(lane) + (lane >> 4) * kPairStep / 2;
    }

    // Trans-tile addressing (crosswise 16-bit operands): the LDSM row is a
    // k line, the 16B chunk a window of the non-contract dim, chunks
    // swizzled by the k-row bits (the trans layout instance).
    // ldmatrix.trans lane
    // contract: lanes 0-7 feed k rows 0-7, lanes 8-15 k rows 8-15 (the
    // second k half of the fragment), lanes 16-31 (x4) step one column
    // chunk (the +8 half of the m16/n8 tile); x2 ignores lanes 16-31.
    // kMtXor/kNtXor: one m/n-tile step in chunks (16B each) — an XOR on
    // the chunk field, not an add; kTransSeg*: one mma k-segment = kMmaK
    // k rows.
    static constexpr unsigned kMtXor = 32u;  // m16 = 2 chunks
    static constexpr unsigned kNtXor = 16u;  // n8 = 1 chunk
    static constexpr unsigned kTransSegA = (unsigned)Traits::kMmaK * kBlockM * sizeof(ElemA);
    static constexpr unsigned kTransSegB = (unsigned)Traits::kMmaK * kBlockN * sizeof(ElemB);
    __device__ __forceinline__ unsigned a_trans_lane_off(int lane) const {
        // x4 matrix order must match the mma's A-register order (m+8 rides
        // reg1, k+8 reg2): lanes 8-15 step the m+8 chunk, lanes 16-31 the
        // k+8 row half.
        const int krow = (lane & 7) + ((lane >> 4) << 3);
        const int col = a_row0 + (((lane >> 3) & 1) << 3);
        return (unsigned)(((int64_t)krow * kBlockM +
                           (((col >> 3) ^ ((krow >> SmemLayoutATrans::kRowShift) &
                                           SmemLayoutATrans::kMask)) << 3) +
                           (col & 7)) *
                          sizeof(ElemA));
    }
    __device__ __forceinline__ unsigned b_trans_lane_off(int lane) const {
        const int krow = (lane & 7) + (((lane >> 3) & 1) << 3);
        const int col = b_row0 + 0;  // nt windows step by kNtXor at call sites
        return (unsigned)(((int64_t)krow * kBlockN +
                           (((col >> 3) ^ ((krow >> SmemLayoutBTrans::kRowShift) &
                                           SmemLayoutBTrans::kMask)) << 3) +
                           (col & 7)) *
                          sizeof(ElemB));
    }

    // One k_seg's B-fragment loads, shared by the initial fill and the
    // double-buffer's next-seg fill. frag2/frag4 are one b_frag /
    // b_frag4 buffer (the unused one is never touched) — typed cells, so
    // every load addresses a whole fragment.
    __device__ __forceinline__ void
    load_b_frags(typename MmaOp::BFrag (&frag2)[kNt],
                 BFragPair (&frag4)[kNt / 2],
                 unsigned seg_base) const {
#pragma unroll
        for (int p = 0; p < kNt / 2; ++p) {
            if constexpr (kTransB) {
                // Trans tile: each x2.trans reads 16 k rows at one n
                // chunk; the nt windows step by one XORed chunk.
                astrai::ldmatrix_x2_lane<true>(
                    frag2[p * 2], seg_base ^ (unsigned)(p * 2 * kNtXor));
                astrai::ldmatrix_x2_lane<true>(
                    frag2[p * 2 + 1],
                    seg_base ^ (unsigned)((p * 2 + 1) * kNtXor));
            } else if constexpr (kPairB) {
                astrai::ldmatrix_x4_lane(frag4[p], seg_base + p * kPairStep);
            } else {
                astrai::ldmatrix_x2_lane(frag2[p * 2],
                                         seg_base + p * 2 * kNtStep);
                astrai::ldmatrix_x2_lane(frag2[p * 2 + 1],
                                         seg_base + (p * 2 + 1) * kNtStep);
            }
        }
    }

    // Dequantized B fragments (W8A16 weight / W8A8 weight side): the
    // m16n8k16 B fragment of lane l (quad q = l>>2, r = l&3) holds
    // tile[n = b_row0 + nt*8 + q][k = k_seg*16 + {2r, 2r+1, 2r+8, 2r+9}]
    // as two packed pairs — both u16 reads land inside one 16B swizzle
    // chunk, so plain staged-tile addressing works. The LOP3 expansion
    // (dequant.cuh) is exact for the int8 range.
    __device__ __forceinline__ void
    load_b_frags_at(typename MmaOp::BFrag (&frag2)[kNt],
                    BFragPair (&frag4)[kNt / 2],
                    TileB stage, int k_seg, unsigned seg_base,
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

    // Dequantized A fragments (W8A8 activation side): the m16n8k16 A
    // fragment of lane l (q = l>>2, c2 = (l&3)*2) holds tile
    // [m = a_row0 + mt*16 + q (+8)][k = k_seg*16 + c2 (+8)] — register
    // order (m, m+8, k+8, m+8&k+8), matching the ldmatrix x4 matrix order
    // the non-dequant path produces (see a_trans_lane_off's note). Both
    // u16 reads of one row stay inside one swizzle chunk (c2 <= 6).
    __device__ __forceinline__ void
    load_a_frags_at(typename MmaOp::AFrag& frag, TileA stage, int k_seg,
                    int mt, int lane) const {
        const int q = lane >> 2, c2 = (lane & 3) * 2;
        const int row = a_row0 + mt * 16 + q;
        const ElemA* p0 = stage(row, k_seg * 16 + c2);
        const ElemA* p8 = stage(row + 8, k_seg * 16 + c2);
        frag[0] = DequantA::pair(*(const unsigned short*)p0);
        frag[1] = DequantA::pair(*(const unsigned short*)p8);
        frag[2] = DequantA::pair(*(const unsigned short*)(p0 + 8));
        frag[3] = DequantA::pair(*(const unsigned short*)(p8 + 8));
    }
};

}  // namespace gemm
}  // namespace astrai
