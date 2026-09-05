#pragma once
// Collective mainloop: shared-memory stage rings, the gmem->smem stage loads
// (congruous cp.async / crosswise LDG+PRMT), the per-lane ldmatrix fragment
// addressing and the software-pipelined mma.sync loop. The fragment
// addressing scheme and the fast-loop peel rationale live in
// docs/developer/cuda_kernels.md.

#include <type_traits>

#include "common/mma.cuh"
#include "gemm/common.h"
#include "load.cuh"
#include "policy.cuh"

namespace astrai {
namespace gemm {

template <typename Policy>
struct GemmCollectiveMainloop {
    using Traits = typename Policy::Traits;
    using LayoutA = typename Policy::LayoutTagA;
    using LayoutB = typename Policy::LayoutTagB;
    using Smem = GemmSmem<Traits, LayoutA, LayoutB>;
    static constexpr bool kFastLoop = Policy::kFastLoop;
    // Operands are independently typed: A (activation, also the mma compute
    // type) and B (weight; when the types differ, B fragments dequantize
    // in-register between the smem read and the mma — kDequantB).
    using ElemA = typename Traits::ElemA;
    using ElemB = typename Traits::ElemB;
    static constexpr bool kDequantB = Traits::kNeedsDequantB;
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
    static_assert(kStages >= 1 && kStages <= 8,
                  "FP8 GEMM stages must be in [1, 8]");
    // CTA = (BlockM/WarpM) x (BlockN/WarpN) warps, each warp computing
    // kMt x kNt m16n8k32 MMAs. Rings rotate kStages+1 buffers (see
    // GemmSmem) — one __syncthreads per k-tile.
    static constexpr int kMt = Traits::kWarpM / 16;  // 16-row MMA tiles per warp
    static constexpr int kNt = Traits::kWarpN / 8;   // 8-col MMA tiles per warp
    static constexpr int kSegs = kK / Traits::kMmaK;  // mma-sized k segments
    static constexpr int kARing = Smem::kRingDepth;
    static constexpr int kBRing = Smem::kRingDepth;
    // Stage strides in BOTH units: ElemT* staging arithmetic uses elements,
    // the smem split / byte-address carries / PrefetchCarry use bytes.
    static constexpr int kAStageElems = kBlockM * kK;
    static constexpr int kBStageElems = kBlockN * kK;
    static constexpr int kAStageBytes = kAStageElems * (int)sizeof(ElemA);
    static constexpr int kBStageBytes = kBStageElems * (int)sizeof(ElemB);

    ElemA* const a_base;
    ElemB* const b_base;
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
        : a_base(reinterpret_cast<ElemA*>(smem)),
          b_base(reinterpret_cast<ElemB*>(smem + kARing * kAStageBytes)),
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

    // Stage-slot helpers: rings rotate one slot per k-tile, so callers
    // either compute the slot from the tile index (prologue, generic loop)
    // or carry an advancing pointer (steady-state fast loop).
    __device__ __forceinline__ ElemA* a_stage_of(int64_t tile) const {
        return a_base + (size_t)(tile % kARing) * kAStageElems;
    }
    __device__ __forceinline__ ElemB* b_stage_of(int64_t tile) const {
        return b_base + (size_t)(tile % kBRing) * kBStageElems;
    }
    // Asynchronous loads for one k-tile: congruous operands cp.async into
    // the canonical rings, crosswise 16-bit operands cp.async into the
    // transposed rings (kFast selects the predication-free interior copy —
    // trans staging qualifies: it is cp.async like the congruous path).
    // Called after the post-compute barrier, alongside the commit.
    template <bool kFast = false>
    __device__ __forceinline__ void load_async(ElemA* a_stage, ElemB* b_stage,
                                               int64_t k_base) const {
        if constexpr (kTransA)
            load_operand_tile_trans<ElemA, kBlockM, kK, kCtaThreads, kFast>(
                a_stage, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
        else if constexpr (!kDirectA)
            load_operand_tile<ElemA, kK, kBlockM, kCtaThreads, kFast>(
                a_stage, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
        if constexpr (kTransB)
            load_operand_tile_trans<ElemB, kBlockN, kK, kCtaThreads, kFast>(
                b_stage, b, n, k, b_ld, tid, k_base, block_n * kBlockN);
        else if constexpr (!kDirectB)
            load_operand_tile<ElemB, kK, kBlockN, kCtaThreads, kFast>(
                b_stage, b, n, k, b_ld, tid, k_base, block_n * kBlockN);
    }
    // Synchronous direct loads for one k-tile (8-bit crosswise operands
    // only — 16-bit crosswise rides the async trans staging above). In the
    // steady state this runs right after barrier 1, so the LDG latency and
    // the PRMT transpose overlap the MMA phase instead of stalling the
    // inter-barrier window.
    __device__ __forceinline__ void load_direct(ElemA* a_stage, ElemB* b_stage,
                                                int64_t k_base) const {
        if constexpr (kSyncA)
            load_crosswise_direct<ElemA, kK, kBlockM, kCtaThreads>(
                a_stage, a, m, k, a_ld, tid, k_base, block_m * kBlockM);
        if constexpr (kSyncB)
            load_crosswise_direct<ElemB, kK, kBlockN, kCtaThreads>(
                b_stage, b, n, k, b_ld, tid, k_base, block_n * kBlockN);
    }

    // Prime the pipeline: kStages committed groups, one per stage slot.
    // The commit is unconditional — when K is shorter than the pipeline the
    // skipped stages commit empty groups, so the group sequence stays
    // tile-indexed and the steady-state wait count never needs a runtime
    // dispatch.
    __device__ __forceinline__ void prologue() const {
#pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
            if (stage < tile_count) {
                if (fast_cta)
                    load_async<true>(a_stage_of(stage), b_stage_of(stage),
                                     (int64_t)stage * kK);
                else
                    load_async(a_stage_of(stage), b_stage_of(stage),
                               (int64_t)stage * kK);
                load_direct(a_stage_of(stage), b_stage_of(stage),
                            (int64_t)stage * kK);
            }
            astrai::cp_async_commit_group();
        }
    }

    // Steady-state mainloop, compile-time specialized on kFast: the fast
    // copy runs predication-free loads with loop-carried read/write
    // pointers; the generic copy keeps full predication. kFastLoop=false
    // instantiates only the generic copy.
    template <bool kFast>
    __device__ __forceinline__ void run_loop(float acc[kNt][kMt][4]) const {
        const int lane = tid & 31;
        // Fast-path write carries: one per congruous operand (crosswise
        // operands get the empty no-op type), targeting the first
        // prefetched tile (kStages). Steady-state read carries: the LDSM
        // base of the current k-tile's stage with the lane offset folded
        // in, advanced one stage per iteration with an equality wrap —
        // replaces the per-k-tile (tile % ring) * stage_bytes
        // recomputation (a UIMAD.WIDE magic-division ladder in SASS).
        PrefetchCarry<!kSyncA, ElemA, kK, kBlockM, kCtaThreads, kTransA>
            carry_a(a_base, kARing, kAStageBytes, a, a_ld, block_m * kBlockM,
                    tid, kStages);
        PrefetchCarry<!kSyncB, ElemB, kK, kBlockN, kCtaThreads, kTransB>
            carry_b(b_base, kBRing, kBStageBytes, b, b_ld, block_n * kBlockN,
                    tid, kStages);
        const unsigned a_rd0 = __cvta_generic_to_shared(a_base) +
                               (kTransA ? a_trans_lane_off(lane)
                                        : a_lane_off(lane));
        const unsigned b_rd0 =
            __cvta_generic_to_shared(b_base) +
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
        astrai::cp_async_wait_group<kStages - 1>();
        // Barrier 1: every thread's cp.async for this stage is complete
        // before any thread reads tiles written by other threads.
        __syncthreads();

        // Direct chunks for tile i+kStages: issue LDG+PRMT+STS now so the
        // global-load latency hides behind the MMA phase below.
        if (prefetch)
            load_direct(a_stage_of(tile_index + kStages),
                        b_stage_of(tile_index + kStages),
                        (tile_index + kStages) * kK);

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
                               : (a_addr ^ (unsigned)(s * kSegXor));
            b_seg[s] = kTransB ? (b_addr + (unsigned)(s * kTransSegB))
                               : (b_addr ^ (unsigned)(s * kSegXor));
        }

        // kNt ldmatrix.x2 (B) + kMt ldmatrix.x4 (A) feed kMt*kNt*2 mma.sync
        // per k_seg — 0.5 load instructions per MMA. B fragments
        // double-buffer across k_segs; kPairB folds the two adjacent nt
        // fragments of one pair into a single x4 (see b4_lane_off).
        const ElemB* b_stage = b_stage_of(tile_index);
        unsigned b_frag[2][kNt][2];
        unsigned b_frag4[2][kNt / 2][4];
        load_b_frags_at(b_frag[0][0], b_frag4[0][0], b_stage, 0, b_seg[0],
                        lane);
#pragma unroll
        for (int k_seg = 0; k_seg < kSegs; ++k_seg) {
            const int bcur = k_seg & 1, bnext = bcur ^ 1;
            if (k_seg + 1 < kSegs)
                load_b_frags_at(b_frag[bnext][0], b_frag4[bnext][0], b_stage,
                                k_seg + 1, b_seg[k_seg + 1], lane);
        // Software-pipelined A fragments: the ldmatrix.x4 for row mt+1 is
        // issued before the MMAs consuming row mt, so the LDS latency hides
        // behind tensor-pipe work. Costs 4 extra registers. Trans tiles
        // advance the m window by XOR (two 16B chunks), canonical tiles by
        // the 16-row byte stride.
        unsigned a_frag[kMt + 1][4];
        if constexpr (kTransA)
            astrai::ldmatrix_x4_lane<true>(a_frag[0], a_seg[k_seg]);
        else
            astrai::ldmatrix_x4_lane(a_frag[0], a_seg[k_seg]);
#pragma unroll
        for (int mt = 0; mt < kMt; ++mt) {
            if (mt + 1 < kMt) {
                const unsigned a_next =
                    kTransA ? (a_seg[k_seg] ^ (unsigned)((mt + 1) * kMtXor))
                            : (a_seg[k_seg] + (mt + 1) * kMtStep);
                if constexpr (kTransA)
                    astrai::ldmatrix_x4_lane<true>(a_frag[mt + 1], a_next);
                else
                    astrai::ldmatrix_x4_lane(a_frag[mt + 1], a_next);
            }
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const unsigned* bops =
                    kPairB ? (b_frag4[bcur][nt >> 1] + (nt & 1) * 2)
                           : b_frag[bcur][nt];
                astrai::mma_sync<ElemA>(acc[nt][mt], a_frag[mt], bops,
                                        acc[nt][mt]);
            }
        }
        // Next tile's LDGSTS chunks inside the MMA phase: A's after the
        // first k_seg's MMA batch, B's after the last.
        if constexpr (kFast) {
            if (k_seg == 0) carry_a.emit(prefetch);
            if (k_seg == kSegs - 1) carry_b.emit(prefetch);
        }
        }
        // Generic loop (no interleaved prefetch): the next tile's predicated
        // loads run after the MMA phase.
        if constexpr (!kFast) {
            if (prefetch) {
                load_async(a_stage_of(tile_index + kStages),
                           b_stage_of(tile_index + kStages),
                           (tile_index + kStages) * kK);
            }
        }
        // Unconditional commit: empty in the tail, it pads the group
        // sequence so the fixed wait above stays correct.
        astrai::cp_async_commit_group();
        a_rd += (unsigned)kAStageBytes;
        if (a_rd == a_rd_end) a_rd = a_rd0;
        b_rd += (unsigned)kBStageBytes;
        if (b_rd == b_rd_end) b_rd = b_rd0;
        if constexpr (kFast) {
            carry_a.advance(kAStageBytes);
            carry_b.advance(kBStageBytes);
        }
        }
    }

    __device__ __forceinline__ void accumulate(float acc[kNt][kMt][4]) const {
        if constexpr (kFastLoop) {
            if (fast_cta)
                run_loop<true>(acc);
            else
                run_loop<false>(acc);
        } else {
            run_loop<false>(acc);
        }
    }

  private:
    // Per-lane ldmatrix fragment addressing (base-pair scheme, mirrored
    // from the cuBLAS SASS; derivation in the design notes): one base
    // register per operand per k_seg, every fragment offset an LDSM
    // immediate — zero address arithmetic inside the MMA phase.
    // Per-lane stage-relative BYTE offsets: the ldmatrix *_lane primitives
    // take raw shared-memory byte addresses, so every offset below is
    // element math scaled by sizeof(ElemT). The swizzle chunk term mirrors
    // tile_at (16B chunks = kChunkElems elements).
    static constexpr int kChunkElems = 16 / sizeof(ElemA);
    static constexpr int kChunkShift = log2_const<kChunkElems>::value;
    __device__ __forceinline__ unsigned a_lane_off(int lane) const {
        const int r7 = lane & 7;          // row within the 8-row matrix
        const int rh8 = (lane >> 3) & 1;  // +8 rows (A: lanes 8-15, 24-31)
        const int rh16 = lane >> 4;       // +1 chunk (A: lanes 16-31)
        constexpr int kChunks = kK / kChunkElems;
        constexpr int kShift = 3 - log2_const<kChunks>::value;  // tile_at's shift
        const unsigned lswz =
            static_cast<unsigned>((r7 >> kShift) & (kChunks - 1));
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
        constexpr int kChunks = kK / kChunkB;
        constexpr int kShift = 3 - log2_const<kChunks>::value;
        const unsigned lswz =
            static_cast<unsigned>((r7 >> kShift) & (kChunks - 1));
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
    // math). kSegXor = one mma k-segment = kMmaK elements — 32B for every
    // supported dtype (fp8 k32 x 1B, bf16 k16 x 2B), i.e. two 16B chunks.
    static constexpr unsigned kMtStep = 16u * kK * sizeof(ElemA);  // m-tile row step
    static constexpr unsigned kNtStep = 8u * kK * sizeof(ElemB);   // n-tile row step
    static constexpr unsigned kSegXor = (unsigned)Traits::kMmaK * sizeof(ElemA);
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
    // swizzled by the k-row bits (tile_at_trans). ldmatrix.trans lane
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
        constexpr int kSwz = kBlockM / 8 < 8 ? kBlockM / 8 : 8;
        return (unsigned)(((int64_t)krow * kBlockM +
                           (((col >> 3) ^ (krow & (kSwz - 1))) << 3) +
                           (col & 7)) *
                          sizeof(ElemA));
    }
    __device__ __forceinline__ unsigned b_trans_lane_off(int lane) const {
        const int krow = (lane & 7) + (((lane >> 3) & 1) << 3);
        const int col = b_row0 + 0;  // nt windows step by kNtXor at call sites
        constexpr int kSwz = kBlockN / 8 < 8 ? kBlockN / 8 : 8;
        return (unsigned)(((int64_t)krow * kBlockN +
                           (((col >> 3) ^ (krow & (kSwz - 1))) << 3) +
                           (col & 7)) *
                          sizeof(ElemB));
    }

    // One k_seg's B-fragment loads, shared by the initial fill and the
    // double-buffer's next-seg fill. frag2/frag4 are the flat bases of one
    // b_frag / b_frag4 buffer (the unused one is never touched).
    __device__ __forceinline__ void
    load_b_frags(unsigned* frag2, unsigned* frag4, unsigned seg_base) const {
#pragma unroll
        for (int p = 0; p < kNt / 2; ++p) {
            if constexpr (kTransB) {
                // Trans tile: each x2.trans reads 16 k rows at one n
                // chunk; the nt windows step by one XORed chunk.
                astrai::ldmatrix_x2_lane<true>(
                    frag2 + p * 4, seg_base ^ (unsigned)(p * 2 * kNtXor));
                astrai::ldmatrix_x2_lane<true>(
                    frag2 + p * 4 + 2,
                    seg_base ^ (unsigned)((p * 2 + 1) * kNtXor));
            } else if constexpr (kPairB) {
                astrai::ldmatrix_x4_lane(frag4 + p * 4,
                                         seg_base + p * kPairStep);
            } else {
                astrai::ldmatrix_x2_lane(frag2 + p * 4,
                                         seg_base + p * 2 * kNtStep);
                astrai::ldmatrix_x2_lane(frag2 + p * 4 + 2,
                                         seg_base + (p * 2 + 1) * kNtStep);
            }
        }
    }

    // Dequantized B fragments (weight-only path): the m16n8k16 B fragment
    // of lane l (quad q = l>>2, r = l&3) holds tile[n = b_row0 + nt*8 + q]
    // [k = k_seg*16 + {2r, 2r+1, 2r+8, 2r+9}] as two packed pairs — both
    // u16 reads land inside one 16B swizzle chunk, so plain tile_at
    // addressing works with no offline repack.  int8 -> bf16 is exact
    // (|v| <= 127 fits bf16's integer range).  A Humming-style offline
    // weight permutation would swap these scalar loads for LDSM + PRMT;
    // deferred until int4 needs real bit-unpacking anyway.
    static __device__ __forceinline__ unsigned dequant_i8_pair(unsigned short v) {
        static_assert(std::is_same_v<ElemA, __nv_bfloat16>,
                      "in-register dequant currently targets bf16 only");
        const float lo = (float)(int8_t)(v & 0xff);
        const float hi = (float)(int8_t)(v >> 8);
        __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<unsigned*>(&p);
    }

    __device__ __forceinline__ void
    load_b_frags_at(unsigned* frag2, unsigned* frag4, const ElemB* stage,
                    int k_seg, unsigned seg_base, int lane) const {
        if constexpr (kDequantB) {
            const int q = lane >> 2, r = lane & 3;
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int row = b_row0 + nt * 8 + q;
                const ElemB* p0 =
                    tile_at<kK>(stage, row, k_seg * 16 + 2 * r);
                const ElemB* p1 =
                    tile_at<kK>(stage, row, k_seg * 16 + 2 * r + 8);
                frag2[nt * 2 + 0] =
                    dequant_i8_pair(*(const unsigned short*)p0);
                frag2[nt * 2 + 1] =
                    dequant_i8_pair(*(const unsigned short*)p1);
            }
        } else {
            load_b_frags(frag2, frag4, seg_base);
        }
    }
};

}  // namespace gemm
}  // namespace astrai
