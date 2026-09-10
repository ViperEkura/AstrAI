#pragma once
// Operand loaders: swizzled shared-memory staging for congruous operands
// (cp.async, predicated and interior variants, plus the loop-carried
// prefetch state) and the direct LDG+PRMT path for crosswise operands.
// The staging invariants and the swizzle derivation live in
// docs/developer/cuda_kernels.md.

#include "common/pipeline.cuh"
#include "common/tensor.cuh"

namespace astrai {
namespace gemm {

// Stage-load one operand tile from global memory into its swizzled ring
// slot. Two geometries are the role-swapped mirror of the SAME loop, so one
// template bit composes the whole function instead of a parallel copy:
//
//   kTransposed = false — CONGRUOUS operand (contract-contiguous storage,
//     the only cp.async-able shape) into the canonical [rows][kK] tile;
//   kTransposed = true  — CROSSWISE 16-bit operand (row-contiguous 16B
//     runs) into the transposed [kK][rows] tile, where ldmatrix.trans does
//     the matrix turn at fragment-extraction time (b16-only — 8-bit
//     crosswise operands keep the LDG+PRMT staging).
//
// Under transposed staging the tile's line axis is the contract dim, so the
// predication axes trade places with a canonical run's. The staged tile is
// the TRANS layout instance: chunks swizzled by the k-row bits (custom XOR,
// source is the row field) so the 8 k-rows one ldmatrix.trans matrix
// addresses at a fixed column window land on distinct chunks (conflict-
// free); only the low 3 row bits can join the XOR (LDSM gives 8 rows per
// matrix), so tiles wider than 8 chunks leave the upper bits unswizzled.
//
// The tile arrives as a Tensor over the staged layout: the swizzled address
// is the tensor's operator(), dispatching to the layout op. kInterior drops
// all predication — valid only for a fully interior CTA (whole lines,
// 16B-aligned base|ld, k_base + kK <= contract); the address math then folds
// to one immediate XOR per chunk (see the design notes).
template <typename SmemLayout, typename ElemT,
          int kThreads, bool kTransposed = false, bool kInterior = false>
__device__ __forceinline__ void
load_operand_tile(Tensor<PtrEngine<ElemT>, SmemLayout> tile,
                  const ElemT* __restrict__ operand, int64_t rows,
                  int64_t contract, int64_t ld, int tid, int64_t k_base,
                  int64_t block_row) {
    static_assert(!kTransposed || sizeof(ElemT) == 2,
                  "trans staging is 16-bit only");
    constexpr int kChunkElems = 16 / sizeof(ElemT);
    constexpr int kTileLines = SmemLayout::kRows;  // lines staged per tile
    constexpr int kChunks = SmemLayout::kChunks;   // chunks per line
    static_assert(kTileLines * kChunks % kThreads == 0,
                  "tile chunks must divide evenly across threads");
    constexpr int kCpt = kTileLines * kChunks / kThreads;  // chunks per thread
    // The XOR chunk stepping below (dst ^ (j << 4)) is the swizzle of
    // c0c + j only because a thread's chunks are one aligned power-of-two
    // run inside the line — j's bits never reach c0c's.
    static_assert(kCpt > 0 && (kCpt & (kCpt - 1)) == 0,
                  "XOR chunk stepping needs a power-of-two chunks-per-thread");
    constexpr int kCpr = kChunks / kCpt;  // chunks per line slice
    const int r = tid / kCpr;             // line within the tile
    const int c0 = (tid % kCpr) * kCpt * kChunkElems;
    // The mirror is three axes: the tile line sources from block_row
    // (canonical) or k_base (transposed) rows; the 16B run starts at
    // k_base (canonical) or block_row (transposed); and each axis is cut
    // by the extent that is NOT the one the run walks — the line's own
    // extent, then the other (non-contract vs contract).
    const int64_t line0 = kTransposed ? k_base : block_row;
    const int64_t run0 = kTransposed ? block_row : k_base;
    const int64_t line_ext = kTransposed ? contract : rows;
    const int64_t run_ext = kTransposed ? rows : contract;
    if constexpr (kInterior) {
        const char* src = reinterpret_cast<const char*>(
            operand + (line0 + r) * ld + run0 + c0);
        const uintptr_t dst =
            reinterpret_cast<uintptr_t>(tile(r, c0));
#pragma unroll
        for (int j = 0; j < kCpt; ++j)
            astrai::cp_async_16(reinterpret_cast<ElemT*>(dst ^ (j << 4)),
                                src + j * 16);
    } else {
        const int64_t line = line0 + r;
        const bool line_ok = line < line_ext;
        // line0, run0, c0 and every j step are multiples of 16, so all
        // chunks share the run's alignment verdict (verdicts only differ
        // ACROSS lines, when ld is not 16B — see the scalar fallback).
        const auto* src = operand + line * ld + run0 + c0;
        const bool chunk_aligned = (reinterpret_cast<uintptr_t>(src) & 15) == 0;
        const uintptr_t dst =
            reinterpret_cast<uintptr_t>(tile(r, c0));
#pragma unroll
        for (int j = 0; j < kCpt; ++j) {
            const int c = j * kChunkElems;
            const int64_t col = run0 + c0 + c;
            ElemT* dstj = reinterpret_cast<ElemT*>(dst ^ (unsigned)(j << 4));
            if (chunk_aligned) {
                // CUTLASS-style zero-fill predication: one cp.async whose
                // runtime src-size loads the valid prefix (whole chunk,
                // the extent tail cut or nothing for an OOB line) and the
                // hardware zero-fills the remainder.
                const int64_t room = line_ok ? run_ext - col : 0;
                const int bytes = room >= kChunkElems
                                      ? 16
                                      : room > 0
                                            ? (int)(room * (int64_t)sizeof(ElemT))
                                            : 0;
                astrai::cp_async_16(dstj, src + c, bytes);
            } else {
                // Misaligned base only: element-granular fallback.
# pragma unroll
                for (int i = 0; i < kChunkElems; ++i)
                    dstj[i] = line_ok && col + i < run_ext ? src[c + i]
                                                           : ElemT(0.0f);
            }
        }
    }
}

// Loop-carried prefetch state for one congruous-or-trans operand ring:
// per-thread (r, c0) mapping with the swizzled stage destination and global
// source pointer carried across k-tiles, so each prefetch chunk is one
// LDGSTS issued straight from registers. The geometry rides the operand's
// Ring tensor (slots, stage stride, staged layout). kTrans selects the
// crosswise 16-bit geometry on the SOURCE side: the tile's rows are k
// lines, so the per-tile source advance is kK * ld instead of kK.
// kAsync=false (synchronous 8-bit crosswise operand) is an empty no-op.
template <bool kAsync, typename RingT, int kThreads, bool kTrans = false>
struct PrefetchCarry;

template <typename RingT, int kThreads, bool kTrans>
struct PrefetchCarry<true, RingT, kThreads, kTrans> {
    using ElemT = typename RingT::Elem;  // Ring = Tensor<PtrEngine, RingLayout>
    using SmemLayout = typename RingT::Layout::Stage;  // per-stage layout
    static constexpr int kChunkElems = 16 / sizeof(ElemT);
    // The layout carries the tile geometry (rows x chunks per row),
    // whichever way round the staging runs.
    static constexpr int kCpt =
        SmemLayout::kRows * SmemLayout::kChunks / kThreads;
    static_assert(kCpt > 0 && (kCpt & (kCpt - 1)) == 0,
                  "XOR chunk stepping needs a power-of-two chunks-per-thread");
    static constexpr int kCpr = SmemLayout::kChunks / kCpt;
    // All carried state is in BYTES: the smem write ring and the global
    // source pointer both advance by the byte-sized stage stride.
    static constexpr int kK =
        kTrans ? SmemLayout::kRows : SmemLayout::kChunks * kChunkElems;
    static constexpr unsigned kKBytes = (unsigned)kK * sizeof(ElemT);
    unsigned wr = 0;    // current stage's swizzled destination offset
    unsigned wr0 = 0;   // slot-0 wrap base
    unsigned wrEnd = 0; // one-past-the-ring sentinel
    const char* src = nullptr;      // current tile's global source bytes
    int64_t srcStep = 0;            // per-tile source advance (bytes)

    __device__ __forceinline__ PrefetchCarry(
        const RingT& ring, const ElemT* operand,
        int64_t ld, int64_t blockRow, int tid, int firstTile) {
        const int r = tid / kCpr;
        const int c0 = (tid % kCpr) * kCpt * kChunkElems;
        const ElemT* slot0 = astrai::stage_of(ring, firstTile).engine.ptr;
        const unsigned laneOff = static_cast<unsigned>(
            (const char*)astrai::stage_of(ring, firstTile)(r, c0) -
            (const char*)slot0);
        const unsigned base =
            __cvta_generic_to_shared(ring.engine.ptr) + laneOff;
        wr = base + (unsigned)((int64_t)(firstTile % RingT::Layout::kSlots) *
                               RingT::Layout::kStageBytes);
        wr0 = base;
        wrEnd = base + (unsigned)RingT::Layout::kTotalBytes;
        if constexpr (kTrans) {
            src = reinterpret_cast<const char*>(
                operand + ((int64_t)firstTile * kK + r) * ld + blockRow + c0);
            srcStep = (int64_t)kK * ld * sizeof(ElemT);  // k advances rows
        } else {
            src = reinterpret_cast<const char*>(
                      operand + (blockRow + r) * ld + c0) +
                  (int64_t)firstTile * kKBytes;
            srcStep = kKBytes;  // k advances the contiguous columns
        }
    }

    // Emit this thread's chunks for the current tile; pf false (loop tail)
    // zero-fills into the slot compute(i-1) already released.
    __device__ __forceinline__ void emit(bool pf) const {
#pragma unroll
        for (int j = 0; j < kCpt; ++j)
            astrai::cp_async_16(wr ^ (unsigned)(j << 4), src + j * 16, pf);
    }

    __device__ __forceinline__ void advance() {
        wr += (unsigned)RingT::Layout::kStageBytes;
        if (wr == wrEnd) wr = wr0;
        src += srcStep;
    }
};

template <typename RingT, int kThreads, bool kTrans>
struct PrefetchCarry<false, RingT, kThreads, kTrans> {
    __device__ __forceinline__ PrefetchCarry(
        const RingT&, const typename RingT::Elem*, int64_t, int64_t, int,
        int) {}
    __device__ __forceinline__ void emit(bool) const {}
    __device__ __forceinline__ void advance() {}
};

// Direct (synchronous) crosswise load into a canonical rotating stage:
// LDG.128 runs (4 x 16B of the non-contract dim) + in-register transpose
// (PRMT) + 16 STS.32. Crosswise operands cannot cp.async into the
// canonical tile (a 16B global run holds contract positions for a run of
// the other dim), so they take this path; a staged smem->smem variant
// measured 15-20% slower and was removed (see git history).
//
// One chunk = 64B of global memory staging one 16-row group:
//   1-byte elements: 4 runs of 16 rows x 4 contract positions; the PRMT
//     byte-perm gathers one 32-bit word per row across the four runs;
//   2-byte elements: 2 contract positions x 2 eight-row halves; a 16B run
//     covers only 8 rows, and the transpose selects halfwords — one
//     PRMT per output word (the byte selector already spans both source
//     words: 0x5410 low pair, 0x7632 high pair).
template <typename SmemLayout, typename ElemT, int kThreads>
__device__ __forceinline__ void
load_crosswise_direct(Tensor<PtrEngine<ElemT>, SmemLayout> tile,
                      const ElemT* __restrict__ operand, int64_t rows,
                      int64_t contract, int64_t ld, int tid, int64_t k_base,
                      int64_t block_row) {
    static_assert(sizeof(ElemT) == 1 || sizeof(ElemT) == 2,
                  "crosswise LDG+PRMT staging requires 1- or 2-byte elements");
    constexpr int kRowsTile = SmemLayout::kRows;
    constexpr int kK = SmemLayout::kChunks * (16 / (int)sizeof(ElemT));
    constexpr int kCw = sizeof(ElemT) == 1 ? 4 : 2;  // contract elems per chunk
    constexpr int kSpans = kK / kCw;    // contract spans per tile
    constexpr int kGroups = kRowsTile / 16;
    constexpr int kTChunks = kSpans * kGroups;  // 64B chunks per tile
    // r0 is a multiple of 16 and p*ld preserves alignment whenever ld has
    // it, so every run of a chunk shares one alignment verdict.
    const bool run_aligned =
        ((reinterpret_cast<uintptr_t>(operand) | (ld * (int64_t)sizeof(ElemT))) &
         15) == 0;
    for (int chunk = tid; chunk < kTChunks; chunk += kThreads) {
        const int span = chunk / kGroups;
        const int rg = chunk % kGroups;
        const int64_t r0 = block_row + rg * 16;
        const bool rows_full = r0 + 15 < rows;
        if (rows_full && run_aligned) {
            const int64_t p0 = k_base + span * kCw;
            uint4 v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // Contract tail: a run past k carries zero bytes; they flow
                // through the transpose like any other value.
                if constexpr (sizeof(ElemT) == 1) {
                    // v[s] = 16 rows at contract p0+s (run index i = s).
                    if (p0 + i < contract)
                        v[i] = __ldg(reinterpret_cast<const uint4*>(
                            operand + (p0 + i) * ld + r0));
                    else
                        v[i] = make_uint4(0u, 0u, 0u, 0u);
                } else {
                    // v[s][h] = 8 rows at contract p0+s (flat i = s*2+h).
                    const int s = i >> 1, h = i & 1;
                    if (p0 + s < contract)
                        v[i] = __ldg(reinterpret_cast<const uint4*>(
                            operand + (p0 + s) * ld + r0 + h * 8));
                    else
                        v[i] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            const unsigned* bytes = reinterpret_cast<const unsigned*>(v);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                unsigned w;
                if constexpr (sizeof(ElemT) == 1) {
                    // word i = row r0+i's span: byte i of each of the four
                    // runs [v0.b(i), v1.b(i), v2.b(i), v3.b(i)].
                    const unsigned nib = i & 3;
                    const unsigned sel = nib | ((nib + 4) << 4);
                    const unsigned w01 =
                        __byte_perm(bytes[0 + (i >> 2)], bytes[4 + (i >> 2)], sel);
                    const unsigned w23 =
                        __byte_perm(bytes[8 + (i >> 2)], bytes[12 + (i >> 2)], sel);
                    w = __byte_perm(w01, w23, 0x5410u);
                } else {
                    // word i = row r0+i's element pair (contracts p0, p0+1):
                    // uint4 v[s*2+h] holds 8 rows of contract p0+s (h =
                    // i>>3), word (i>>1)&3, halfword i&1 — one selector
                    // spans both source words: 0x5410 low pair, 0x7632 high.
                    const unsigned* v0 =
                        bytes + ((i >> 3) * 4 + ((i >> 1) & 3));
                    const unsigned* v1 = v0 + 8;  // p0+1 run, same h/j
                    w = __byte_perm(*v0, *v1, (i & 1) ? 0x7632u : 0x5410u);
                }
                *reinterpret_cast<unsigned*>(
                    tile(rg * 16 + i, span * kCw)) = w;
            }
        } else {
            // Row-tail or misaligned chunk: element-granular gather with
            // per-row predication; contract-tail columns zero-fill.
#pragma unroll
            for (int s = 0; s < kCw; ++s) {
                const int col = span * kCw + s;
                if (k_base + col >= contract) {
#pragma unroll
                    for (int i = 0; i < 16; ++i)
                        *tile(rg * 16 + i, col) = ElemT(0.0f);
                    continue;
                }
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int64_t r_idx = r0 + i;
                    *tile(rg * 16 + i, col) =
                        r_idx < rows
                            ? operand[(k_base + col) * ld + r_idx]
                            : ElemT(0.0f);
                }
            }
        }
    }
}

}  // namespace gemm
}  // namespace astrai
